#!/usr/bin/env python
"""
Converts MDTVAgent PyTorch model to CoreML format.
"""
import ast
import os
import sys
import argparse
from pathlib import Path
import logging
import glob
import gc
import time
import numpy as np
import einops
import shutil

from typing import Dict, Any, Optional, List, Tuple, Union
from omegaconf import OmegaConf, DictConfig
import hydra

# Set up logging
logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Version check for torch
try:
    import torch
    import packaging.version

    torch_version = packaging.version.parse(torch.__version__)
    min_torch_version = packaging.version.parse("2.1.0")

    if torch_version < min_torch_version:
        logger.warning(
            f"Warning: CoreMLTools optimize requires PyTorch {min_torch_version} or newer. "
            f"Found PyTorch {torch_version}. Some optimizations may not be available.\n"
            f"Consider upgrading with: pip install torch>=2.1.0"
        )
        HAS_REQUIRED_TORCH = False
    else:
        HAS_REQUIRED_TORCH = True
except ImportError:
    logger.error("PyTorch is not installed. Please install PyTorch 2.1.0 or newer.")
    sys.exit(1)

# Import coremltools with proper error handling
try:
    import coremltools as ct
    from coremltools.converters.mil import register_torch_op
    from coremltools.converters.mil.frontend.torch.ops import _get_inputs
except ImportError:
    logger.error(
        "CoreMLTools is not installed. Please install with: pip install coremltools"
    )
    sys.exit(1)

# Add the parent directory to the path to import mdt modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mdt.models.mdtv_agent import MDTVAgent
from mdt.models.networks.mdtv_transformer import MDTVTransformer
from mdt.models.networks.transformers.perceiver_resampler import PerceiverResampler
from mdt.models.edm_diffusion.gc_sampling import sample_ddim, get_sigmas_exponential

def _get_out_path(output_dir, submodule_name):
    fname = f"MDT_{submodule_name}.mlpackage"
    return os.path.join(output_dir, fname)


def get_config_from_dir(dir):
    dir = Path(dir)
    config_yaml = list(dir.rglob("*.hydra/config.yaml"))[0]
    return OmegaConf.load(config_yaml)


def load_pretrained_model(cfg: DictConfig, use_ema_weights: bool = False, strict_loading: bool = False) -> MDTVAgent:
    """
    Load a pretrained MDTVAgent model from the specified directory.

    Args:
        cfg: Configuration containing model parameters and paths
        use_ema_weights: Whether to use EMA weights
    Returns:
        The loaded MDTVAgent model in evaluation mode
    """
    # load the checkpoint path
    checkpoint_path = glob.glob(str(Path(cfg.train_folder) / "saved_models" / "*.ckpt"))
    assert (
        len(checkpoint_path) == 1
    ), f"Found {len(checkpoint_path)} checkpoint files in {cfg.train_folder / 'saved_models'}"
    checkpoint_path = checkpoint_path[0]

    # merge cfg and train_cfg
    train_cfg_path = Path(cfg.train_folder) / ".hydra/config.yaml"
    def_cfg = OmegaConf.load(train_cfg_path)
    eval_override_cfg = OmegaConf.create(cfg.eval_cfg_overwrite)

    # Fix device interpolation - crucial step!
    device_cfg = OmegaConf.create({"device": "cpu"})
    def_cfg = OmegaConf.merge(def_cfg, device_cfg)
    config_dict = OmegaConf.to_container(def_cfg, resolve=True)
    def_cfg = OmegaConf.create(config_dict)

    merged_cfg = OmegaConf.merge(def_cfg, eval_override_cfg)

    class_name = def_cfg.model.pop("_target_")
    if "_recursive_" in def_cfg.model:
        del def_cfg.model["_recursive_"]
    logger.info(
        f"class_name: {class_name}, device: {def_cfg.device}, visual_goal.device: {def_cfg.model.visual_goal.device}, model.inner_model.device: {def_cfg.model.visual_goal.device}"
    )
    assert (
        class_name == "mdt.models.mdtv_agent.MDTVAgent"
    ), "Only MDTVAgent is supported for now"

    # Create load_cfg with the appropriate settings
    # merge config recursively
    load_cfg = OmegaConf.merge(def_cfg.model, merged_cfg.overwrite_module_cfg, {"map_location": "cpu"})
    logger.info(f"load_cfg: {OmegaConf.to_yaml(load_cfg)}")
    # ori_load_cfg = {
    #     **def_cfg.model,
    #     **merged_cfg.overwrite_module_cfg, # this is not recursively merged
    #     "map_location": "cpu",
    # }
    # logger.info(f"ori_load_cfg: {OmegaConf.to_yaml(ori_load_cfg)}")

    logger.info("All devices explicitly set to CPU for CoreML compatibility")

    logger.info(f"Loading model from {checkpoint_path}")
    model = MDTVAgent.load_from_checkpoint(checkpoint_path, strict=strict_loading, **load_cfg) # strict=False to avoid missing keys

    # use EMA weights if available
    checkpoint_data = torch.load(checkpoint_path, map_location="cpu")
    if (
        "callbacks" in checkpoint_data
        and "EMA" in checkpoint_data["callbacks"]
        and "ema_weights" in checkpoint_data["callbacks"]["EMA"]
        and use_ema_weights
    ):
        ema_weights_list = checkpoint_data["callbacks"]["EMA"]["ema_weights"]

        # Convert list of tensors to a state_dict format
        ema_weights_dict = {
            name: ema_weights_list[i]
            for i, (name, _) in enumerate(model.named_parameters())
        }

        m, u = model.load_state_dict(ema_weights_dict, strict=strict_loading) # strict=False to avoid missing keys
        logger.info(f"m: {m}")
        logger.info(f"u: {u}")
        logger.info("Successfully loaded EMA weights from checkpoint!")
    else:
        if use_ema_weights:
            logger.info("Warning: No EMA weights found in checkpoint!")
        else:
            logger.info("Skipping EMA weights loading")

    logger.info(f"Finished loading model {checkpoint_path}")
    model.eval()  # Set to evaluation mode
    model.freeze()
    return model


def _get_coreml_inputs(sample_inputs):
    return [
        ct.TensorType(
            name=k,
            shape=v.shape,
            dtype=v.numpy().dtype if isinstance(v, torch.Tensor) else v.dtype,
        )
        for k, v in sample_inputs.items()
    ]


def debug_trace_model(model, inputs):
    """Trace model execution to identify problematic operations"""
    with torch.no_grad():

        def hook_fn(module, input, output):
            logger.info(
                f"Module: {module.__class__.__name__}, Input shapes: {[i.shape if isinstance(i, torch.Tensor) else type(i) for i in input]}, Output shapes: {output.shape if isinstance(output, torch.Tensor) else type(output)}"
            )

        hooks = []
        for name, module in model.named_modules():
            hooks.append(module.register_forward_hook(hook_fn))

        try:
            model(*inputs)
        finally:
            for hook in hooks:
                hook.remove()


def find_tensor_split_ops(torchscript_module):
    """Find tensor_split operations in a traced model."""
    # Get the graph
    graph = torchscript_module.graph

    # logger.info the graph for inspection
    logger.info("Model graph:")
    logger.info(graph)

    # Look for tensor_split operations
    tensor_split_nodes = []
    for node in graph.nodes():
        if "tensor_split" in str(node):
            tensor_split_nodes.append(node)

    logger.info(f"Found {len(tensor_split_nodes)} tensor_split operations:")
    for i, node in enumerate(tensor_split_nodes):
        logger.info(f"Node {i+1}:")
        logger.info(f"  Kind: {node.kind()}")
        logger.info(f"  Inputs: {[input.debugName() for input in node.inputs()]}")
        logger.info(f"  Outputs: {[output.debugName() for output in node.outputs()]}")
        logger.info(f"  Source location: {node.sourceRange()}")
        logger.info()

    return tensor_split_nodes


def convert_to_coreml(
    submodule_name: Optional[str],
    torchscript_module: torch.nn.Module,
    sample_inputs: dict,
    output_names: List[str],
    output_dir: str = None,
    output_path: str = None,
    compute_unit: str = None,
    precision: str = None,
    check_output_correctness: bool = False,
) -> str:
    """
    Convert a PyTorch model to CoreML format.

    Args:
        submodule_name: Name of the submodule to convert
        torchscript_module: The PyTorch model to convert
        sample_inputs: Dictionary mapping input names to their shapes
        output_names: List of output names
        output_dir: Directory to save output model
        output_path: Path to save the CoreML model
        compute_unit: Compute unit to use (ALL, CPU_ONLY, etc.)
        precision: Precision to use (FLOAT32, FLOAT16, etc.)
        check_output_correctness: Whether to check the output correctness
    Returns:
        Path to the saved CoreML model
    """

    if output_path is None:
        output_path = _get_out_path(output_dir, submodule_name)

    compute_unit = ct.ComputeUnit[compute_unit]

    if os.path.exists(output_path):
        logger.info(f"Model already exists at {output_path}. Skipping conversion.")
        logger.info(f"Loading model from {output_path}")
        start = time.time()
        # Note: Note that each model load will trigger a model compilation which takes up to a few minutes.
        # The Swifty CLI we provide uses precompiled Core ML models (.mlmodelc) which incurs compilation only
        # upon first load and mitigates the load time in subsequent runs.
        coreml_model = ct.models.MLModel(output_path, compute_units=compute_unit)
        logger.info(f"Loading {output_path} took {time.time() - start:.1f} seconds")

        coreml_model.compute_unit = compute_unit
    else:
        logger.info(f"Converting {submodule_name} to CoreML...")
        coreml_model = ct.convert(
            torchscript_module,
            convert_to="mlprogram",
            minimum_deployment_target=ct.target.iOS16,
            inputs=_get_coreml_inputs(sample_inputs),
            outputs=[
                ct.TensorType(name=name, dtype=np.float32) for name in output_names
            ],
            compute_units=compute_unit,
            compute_precision=precision,
            skip_model_load=not check_output_correctness,
        )

        # Save the model
        output_path = Path(output_path)
        os.makedirs(output_path.parent, exist_ok=True)

        try:
            coreml_model.save(str(output_path))
            logger.info(f"CoreML model saved to {output_path}")
        except Exception as e:
            logger.error(f"Failed to save model: {e}")
            raise

    # Parity check PyTorch vs CoreML
    if check_output_correctness:
        try:
            baseline_out = torchscript_module(**sample_inputs).numpy()
            coreml_out = list(
                coreml_model.predict(
                    {k: v.numpy() for k, v in sample_inputs.items()}
                ).values()
            )[0]
            np.testing.assert_allclose(
                baseline_out,
                coreml_out,
                rtol=1e-2, # 1e-3 may raise error
                atol=1e-2, # 1e-3 may raise error
                err_msg=f"assert allclose {submodule_name} baseline PyTorch to baseline CoreML failed",
            )
        except Exception as e:
            logger.error(f"Failed to check output correctness: {e}")

    del torchscript_module
    gc.collect()

    return coreml_model, output_path

def convert_all(
    model: MDTVAgent,
    input_shapes: dict,
    output_dir: str,
    compute_unit: str = "ALL",
    check_output_correctness: bool = False,
) -> str:
    """
    Convert all models (language goal, visual goal, voltron, gc denoiser) to CoreML.
    examples/mdt/mdt/models/mdtv_agent.py
    """
    logger.info("Converting all models (language goal, visual goal, voltron, gc denoiser) to CoreML...")

    # Define output path
    output_path = _get_out_path(output_dir, "all")

    if os.path.exists(output_path):
        logger.info(f"Model already exists at {output_path}. Skipping conversion.")
        return

    # Define model from the MDTVAgent
    model.language_goal.eval() # LangCLIP
    # model.visual_goal.eval() # DefaultVisionClip, in calvin evaluation, it is not used.
    model.img_encoder.eval() # Voltron
    model.perceiver.eval() # PerceiverResampler
    model.model.eval() # GCDenoiser

    # Define input shapes
    # Define the token sequence length
    sequence_length = input_shapes.get("sequence_length")  # Default CLIP token length
    bs = input_shapes.get("bs")
    from mdt.models.networks.clip import _tokenizer

    # sot_token = _tokenizer.encoder["<|startoftext|>"]
    eot_token = _tokenizer.encoder["<|endoftext|>"]
    agent_inputs = {
        # goal
        # "lang_text": 'take the blue block and rotate it to the right',
        "lang_tokens": torch.randint(
            0, eot_token + 1, (bs, sequence_length)
        ),  # (batch_size, sequence_length)
        # observation
        # rgb_obs
        "rgb_static": torch.randn(bs, 1, 3, 224, 224),
        "rgb_gripper": torch.randn(bs, 1, 3, 84, 84),
        # robot_obs
        # "robot_obs": torch.randn(bs, 1, 8),
        # depth_obs # empty
        # robot_obs_raw
        # "robot_obs_raw": torch.randn(15)
    }
    sigmas = model.get_noise_schedule(model.num_sampling_steps, "exponential")
    action_dim = input_shapes.get("action_dim")

    agent_inputs_spec = {k: (v.shape, v.dtype) for k, v in agent_inputs.items()}
    logger.info(f"Agent sample inputs spec: {agent_inputs_spec}")

    # Define output names
    output_names = ["pred_action_seq"]

    class AgentModelWrapper(torch.nn.Module):
        def __init__(self, model, sigmas, action_dim):
            super().__init__()
            self.language_goal = model.language_goal
            self.img_encoder = model.img_encoder
            self.perceiver = model.perceiver
            self.denoiser = model.model

            self.noise_scheduler = model.noise_scheduler
            self.num_sampling_steps = model.num_sampling_steps # 10 for ddim
            self.act_window_size = model.act_window_size
            self.sigmas = sigmas
            self.sigma_fn = lambda t: t.neg().exp()
            self.t_fn = lambda sigma: sigma.log().neg()
            self.sigma_min = model.sigma_min
            self.sigma_max = model.sigma_max
            self.action_dim = action_dim
        def compute_voltron_embedding(self, rgb_static, rgb_gripper):
            rgb_static = einops.rearrange(rgb_static, 'b t c h w -> (b t) c h w')
            rgb_gripper = einops.rearrange(rgb_gripper, 'b t c h w -> (b t) c h w')
            static_tokens = self.img_encoder(rgb_static)
            gripper_tokens = self.img_encoder(rgb_gripper)
            token_seq = torch.cat([static_tokens, gripper_tokens], dim=1).unsqueeze(1)
            state_images = self.perceiver(token_seq)
            return state_images

        def process_sigma_embeddings(self, sigma):
            sigmas = sigma.log() / 4
            sigmas = einops.rearrange(sigmas, "b -> b 1")
            emb_t = self.denoiser.inner_model.sigma_emb(sigmas)
            if len(emb_t.shape) == 2:
                emb_t = einops.rearrange(emb_t, "b d -> b 1 d")
            return emb_t

        def preprocess_goals(self, goals, states_length):
            if len(goals.shape) == 2:
                goals = einops.rearrange(goals, "b d -> b 1 d")
            if (
                goals.shape[1] == states_length
                and self.denoiser.inner_model.goal_seq_len == 1
            ):
                goals = goals[:, 0, :]
                goals = einops.rearrange(goals, "b d -> b 1 d")
            if goals.shape[-1] == 2 * self.denoiser.inner_model.obs_dim:
                goals = goals[:, :, : self.denoiser.inner_model.obs_dim]
            return goals

        def concatenate_inputs(self, emb_t, goal_embed, state_embed, proprio_embed):
            input_seq_components = [state_embed]
            if self.denoiser.inner_model.goal_conditioned:
                input_seq_components.insert(0, goal_embed)
            if proprio_embed is not None:
                input_seq_components.append(proprio_embed)
            else:
                if not self.denoiser.inner_model.goal_conditioned:
                    input_seq_components.append(
                        self.denoiser.inner_model.drop(goal_embed)
                    )
            if not self.denoiser.inner_model.use_ada_conditioning:
                input_seq_components.insert(0, emb_t)
            input_seq = torch.cat(input_seq_components, dim=1)
            return input_seq

        def forward_enc_only(self, state, goals, sigma):
            goals = self.preprocess_goals(goals, state.size(1))
            state_embed = self.denoiser.inner_model.tok_emb(state)
            goal_embed = self.denoiser.inner_model.lang_emb(goals)
            input_seq = self.concatenate_inputs(None, goal_embed, state_embed, None)
            context = self.denoiser.inner_model.encoder(input_seq)
            self.denoiser.inner_model.latent_encoder_emb = context
            return context

        def forward_dec_only(self, context, action, sigma):
            emb_t = self.process_sigma_embeddings(sigma)
            action_embed = self.denoiser.inner_model.action_emb(action)
            action_x = self.denoiser.inner_model.drop(action_embed)
            if self.denoiser.inner_model.use_ada_conditioning:
                x = self.denoiser.inner_model.decoder(action_x, emb_t, context)
            else:
                x = self.denoiser.inner_model.decoder(action_x, context)
            pred_action = self.denoiser.inner_model.action_pred(x)
            return pred_action

        def denoise_actions(self, latent_goal, state_images):
            # ddim
            action = torch.randn((bs, self.act_window_size, self.action_dim))
            s_in = torch.ones([action.shape[0]])
            for i in range(self.sigmas.size(0) - 1):
                logger.info(f"ddim step {i}")
                sigma = self.sigmas[i] * s_in
                c_skip, c_out, c_in = [
                    x[(...,) + (None,) * (action.ndim - x.ndim)]
                    for x in self.denoiser.get_scalings(sigma)
                ]
                # customized inner_model
                action_in = action * c_in
                # forward_enc_only
                context = self.forward_enc_only(state_images, latent_goal, sigma)
                # forward_dec_only
                pred_action = self.forward_dec_only(context, action_in, sigma)
                denoised = pred_action * c_out + action * c_skip
                # post process
                t, t_next = self.t_fn(self.sigmas[i]), self.t_fn(self.sigmas[i + 1])
                h = t_next - t
                action = (self.sigma_fn(t_next) / self.sigma_fn(t)) * action - (
                    -h
                ).expm1() * denoised
            return action
        def forward(self, lang_tokens, rgb_static, rgb_gripper):
            # get latent_goal
            # tokenzie func in LangCLIP, check if it should be unwraped
            latent_goal = self.language_goal.clip_rn50.encode_text(lang_tokens).unsqueeze(1).to(torch.float32) # (bs, 1, 512)
            # compute voltron embedding
            state_images = self.compute_voltron_embedding(rgb_static, rgb_gripper) # (bs, 3, 384)
            # get action sequence
            action_seq = self.denoise_actions(latent_goal, state_images)
            return action_seq

    # test the agent model
    logger.info("Checking model dimensions...")
    test_agent = AgentModelWrapper(model, sigmas, action_dim).eval()
    with torch.no_grad():
        test_out = test_agent(
            agent_inputs["lang_tokens"],
            agent_inputs["rgb_static"],
            agent_inputs["rgb_gripper"],
        )
        logger.info(f"Test output shape: {test_out.shape}")
    logger.info("Checking model dimensions done.")
    # test the denoiser done.

    reference_agent_model = AgentModelWrapper(model, sigmas, action_dim).eval()
    logger.info(f"JIT tracing reference agent model...")
    traced_reference_agent_model = torch.jit.trace(
        reference_agent_model,
        (agent_inputs["lang_tokens"], agent_inputs["rgb_static"], agent_inputs["rgb_gripper"])
    )
    logger.info(f"JIT tracing reference agent model done")
    
    del model
    gc.collect()
    
    return convert_to_coreml(
        submodule_name="all",
        torchscript_module=traced_reference_agent_model,
        sample_inputs=agent_inputs,
        output_names=output_names,
        output_dir=output_dir,
        compute_unit=compute_unit,
        check_output_correctness=check_output_correctness,
    )

def convert_language_goal(
    model: MDTVAgent,
    input_shapes: dict,
    output_dir: str,
    compute_unit: str = "ALL",
    check_output_correctness: bool = False,
) -> str:
    """
    Convert the language goal component of the MDTVAgent to CoreML.

    Args:
        model: The MDTVAgent model containing the language goal processor
        input_shapes: Dictionary of input shapes
        output_dir: Directory to save output model
        compute_unit: Compute unit to use (ALL, CPU_ONLY, etc.)
        check_output_correctness: Whether to check the output correctness

    Returns:
        Path to the saved CoreML model
    """
    logger.info("Converting language goal model to CoreML...")

    # Define output path
    output_path = _get_out_path(output_dir, "LanguageGoal")

    if os.path.exists(output_path):
        logger.info(
            f"`LanguageGoal` already exists at {output_path}, skipping conversion."
        )
        return

    # Extract language goal component from the MDTVAgent
    language_model = model.language_goal
    language_model = language_model.to(dtype=torch.float32)
    language_model.eval()

    # Define the token sequence length
    sequence_length = input_shapes.get("sequence_length")  # Default CLIP token length
    bs = 1
    from mdt.models.networks.clip import _tokenizer

    # sot_token = _tokenizer.encoder["<|startoftext|>"]
    eot_token = _tokenizer.encoder["<|endoftext|>"]
    # Define input shapes for language model
    lang_inputs = {
        "lang_tokens": torch.randint(
            0, eot_token + 1, (bs, sequence_length)
        )  # (batch_size, sequence_length)
    }
    lang_inputs_spec = {k: (v.shape, v.dtype) for k, v in lang_inputs.items()}
    logger.info(f"Sample inputs spec: {lang_inputs_spec}")

    class ModelWrapper(torch.nn.Module):
        def __init__(self, text_encoder):
            super().__init__()
            self.text_encoder = text_encoder

        def forward(self, lang_tokens):
            output = self.text_encoder.clip_rn50.encode_text(lang_tokens)
            return torch.unsqueeze(output, 1)

    reference_language_model = ModelWrapper(language_model).eval()
    logger.info(f"JIT tracing reference language model...")
    traced_reference_language_model = torch.jit.trace(
        reference_language_model, (lang_inputs["lang_tokens"].to(torch.long),)
    )
    logger.info(f"JIT tracing reference language model done")

    del language_model
    gc.collect()

    # Convert to CoreML
    return convert_to_coreml(
        submodule_name="LanguageGoal",
        torchscript_module=traced_reference_language_model,
        sample_inputs=lang_inputs,
        output_names=["latent_goal"],
        output_dir=output_dir,
        compute_unit=compute_unit,
        check_output_correctness=check_output_correctness,
    )


def convert_visual_goal(
    model: MDTVAgent,
    input_shapes: dict,
    output_dir: str,
    compute_unit: str = "ALL",
    check_output_correctness: bool = False,
) -> str:
    """
    Convert the visual goal component of the MDTVAgent to CoreML.

    Args:
        model: The MDTVAgent model containing the visual goal processor
        input_shapes: Dictionary of input shapes
        output_dir: Directory to save output model
        compute_unit: Compute unit to use (ALL, CPU_ONLY, etc.)
        check_output_correctness: Whether to check the output correctness

    Returns:
        Path to the saved CoreML model
    """
    logger.info("Converting visual goal model to CoreML...")

    # Define output path
    output_path = _get_out_path(output_dir, "VisualGoal")

    if os.path.exists(output_path):
        logger.info(
            f"`VisualGoal` already exists at {output_path}, skipping conversion."
        )
        return

    # Extract visual goal component from the MDTVAgent
    visual_model = model.visual_goal  # DefaultVisionClip
    visual_model = visual_model.to(torch.float32)
    visual_model.eval()

    # Get image dimensions from args
    image_size = input_shapes.get("image_size", 224)
    bs = 1

    # Define input shapes for visual model
    visual_inputs = {
        "image": torch.randn(
            bs, 3, image_size, image_size
        )  # (batch, channels, height, width)
    }

    visual_inputs_spec = {k: (v.shape, v.dtype) for k, v in visual_inputs.items()}
    logger.info(f"Sample inputs spec: {visual_inputs_spec}")

    class ModelWrapper(torch.nn.Module):
        def __init__(self, visual_model):
            super().__init__()
            self.visual_model = visual_model

        def forward(self, image):
            return self.visual_model.clip_model.encode_image(image)

    reference_visual_model = ModelWrapper(visual_model).eval()
    logger.info(f"JIT tracing reference visual model...")
    traced_reference_visual_model = torch.jit.trace(
        reference_visual_model, (visual_inputs["image"].to(torch.float32),)
    )
    logger.info(f"JIT tracing reference visual model done")

    del visual_model
    gc.collect()

    # Convert to CoreML
    return convert_to_coreml(
        submodule_name="VisualGoal",
        torchscript_module=traced_reference_visual_model,
        sample_inputs=visual_inputs,
        output_names=["latent_goal"],
        output_dir=output_dir,
        compute_unit=compute_unit,
        check_output_correctness=check_output_correctness,
    )


def convert_voltron(
    model: MDTVAgent,
    input_shapes: dict,
    output_dir: str,
    compute_unit: str = "ALL",
    check_output_correctness: bool = False,
) -> str:
    """
    Convert the Voltron component of the MDTVAgent to CoreML.

    Args:
        model: The MDTVAgent model containing the Voltron component
        input_shapes: Dictionary of input shapes
        output_dir: Directory to save output model
        compute_unit: Compute unit to use (ALL, CPU_ONLY, etc.)
        check_output_correctness: Whether to check the output correctness
    Returns:
        Path to the saved CoreML model
    """
    logger.info("Converting Voltron model to CoreML...")

    # Define output path
    output_path = _get_out_path(output_dir, "Voltron")

    if os.path.exists(output_path):
        logger.info(f"`Voltron` already exists at {output_path}, skipping conversion.")
        return

    # Extract Voltron component from the MDTVAgent
    voltron_model = model.img_encoder.to(dtype=torch.float32)
    voltron_model.eval()
    perceiver_resampler = model.perceiver.to(dtype=torch.float32)
    perceiver_resampler.eval()

    # Define input shapes for Voltron model
    # The exact shapes depend on the implementation but typically include:
    image_size = input_shapes.get("image_size", 224)
    obs_seq_len = input_shapes.get("obs_seq_len", 1)
    bs = 1

    # preprocess the inputs
    preprocess = voltron_model.preprocess
    lang, lang_mask = [torch.zeros(bs, 20, dtype=int) for _ in range(2)]
    lang[:, 0], lang_mask[:, 0] = voltron_model.vcond.tokenizer.cls_token_id, 1

    voltron_inputs = {
        "rgb_static": torch.randn(
            bs, 3, image_size, image_size
        ),  # (batch, channels, height, width)
        "rgb_gripper": torch.randn(
            bs, 3, image_size, image_size
        ),  # (batch, channels, height, width)
    }

    voltron_inputs_spec = {k: (v.shape, v.dtype) for k, v in voltron_inputs.items()}
    logger.info(f"Voltron sample inputs spec: {voltron_inputs_spec}")

    # perceiver_resampler_inputs = {
    #     "token_seq": torch.randn(bs, 1, 392, 384)  # (batch, num_tokens, channels)
    # }

    # perceiver_resampler_inputs_spec = {
    #     k: (v.shape, v.dtype) for k, v in perceiver_resampler_inputs.items()
    # }
    # logger.info(
    #     f"PerceiverResampler sample inputs spec: {perceiver_resampler_inputs_spec}"
    # )

    class VoltronModelWrapper(torch.nn.Module):
        def __init__(self, voltron_model, perceiver_resampler, lang, lang_mask):
            super().__init__()
            self.vcond = voltron_model.vcond
            self.perceiver_resampler = perceiver_resampler

            self.lang = lang
            self.lang_mask = lang_mask

        def vcond_forward(self, x):
            return self.vcond.encode(x, self.lang, self.lang_mask)

        def forward(self, rgb_static, rgb_gripper):
            # rgb_static = einops.rearrange(rgb_static, "b t c h w -> (b t) c h w")
            # rgb_gripper = einops.rearrange(rgb_gripper, "b t c h w -> (b t) c h w")
            static_tokens = self.vcond_forward(
                rgb_static
            )  # equal to self.vcond(rgb_static, mode='visual')
            gripper_tokens = self.vcond_forward(
                rgb_gripper
            )  # equal to self.vcond(rgb_gripper, mode='visual')
            token_seq = torch.cat([static_tokens, gripper_tokens], dim=1).unsqueeze(1)
            state_images = self.perceiver_resampler(token_seq)
            return state_images

    # class PerceiverResamplerModelWrapper(torch.nn.Module):
    #     def __init__(self, perceiver_resampler):
    #         super().__init__()
    #         self.perceiver_resampler = perceiver_resampler

    #     def forward(self, token_seq):
    #         return self.perceiver_resampler(token_seq)

    reference_voltron_model = VoltronModelWrapper(
        voltron_model, perceiver_resampler, lang, lang_mask
    ).eval()
    # reference_perceiver_resampler_model = PerceiverResamplerModelWrapper(
    #     perceiver_resampler
    # ).eval()

    logger.info(f"JIT tracing reference voltron model...")
    traced_reference_voltron_model = torch.jit.trace(
        reference_voltron_model,
        (
            voltron_inputs["rgb_static"].to(torch.float32),
            voltron_inputs["rgb_gripper"].to(torch.float32),
        ),
    )
    logger.info(f"JIT tracing reference voltron model done")

    # logger.info(f"JIT tracing reference perceiver resampler model...")
    # traced_reference_perceiver_resampler_model = torch.jit.trace(
    #     reference_perceiver_resampler_model,
    #     (perceiver_resampler_inputs["token_seq"].to(torch.float32)),
    # )
    # logger.info(f"JIT tracing reference perceiver resampler model done")

    del voltron_model
    del perceiver_resampler
    gc.collect()

    # Convert to CoreML
    return convert_to_coreml(
        submodule_name="Voltron",
        torchscript_module=traced_reference_voltron_model,
        sample_inputs=voltron_inputs,
        output_names=["state_images"],
        output_dir=output_dir,
        compute_unit=compute_unit,
        check_output_correctness=check_output_correctness,
    )
    # return convert_to_coreml(
    #     submodule_name="PerceiverResampler",
    #     torchscript_module=traced_reference_perceiver_resampler_model,
    #     sample_inputs=perceiver_resampler_inputs,
    #     output_names=["state_images"],
    #     output_dir=output_dir,
    #     compute_unit=compute_unit,
    #     check_output_correctness=check_output_correctness,
    # )


def convert_gcdenoiser(
    model: MDTVAgent,
    input_shapes: dict,
    output_dir: str,
    compute_unit: str = "ALL",
    check_output_correctness: bool = False,
) -> str:
    """
    Convert the GC Denoiser component of the MDTVAgent to CoreML.

    Args:
        model: The MDTVAgent model containing the GC Denoiser
        input_shapes: Dictionary of input shapes
        output_dir: Directory to save output model
        compute_unit: Compute unit to use (ALL, CPU_ONLY, etc.)
        check_output_correctness: Whether to check the output correctness

    Returns:
        Path to the saved CoreML model
    """
    logger.info("Converting GC Denoiser model to CoreML...")

    # Define output path
    output_path = _get_out_path(output_dir, "GCDenoiser")

    if os.path.exists(output_path):
        logger.info(
            f"`GCDenoiser` already exists at {output_path}, skipping conversion."
        )
        return

    # Extract GC Denoiser from the MDTVAgent
    gc_denoiser = model.model
    gc_denoiser.to(torch.float32)
    gc_denoiser.eval()

    # Define input shapes for GC Denoiser
    latent_dim = input_shapes.get("latent_dim")
    perceiver_dim = input_shapes.get("perceiver_dim")
    num_voltron_tokens = input_shapes.get("num_token_voltron")
    num_sampling_steps = input_shapes.get("num_sampling_steps")
    act_window_size = input_shapes.get("act_window_size")
    action_dim = input_shapes.get("action_dim")
    bs = 1

    denoiser_inputs = {
        "state_images": torch.randn(bs, num_voltron_tokens, perceiver_dim),
        "action": torch.randn(bs, act_window_size, action_dim),
        "latent_goal": torch.randn(bs, 1, latent_dim),
    }

    sigmas = model.get_noise_schedule(num_sampling_steps, "exponential")

    class DenoiserModelWrapper(torch.nn.Module):
        def __init__(self, denoiser, sigmas=None, modality="lang"):
            super().__init__()
            self.denoiser = denoiser
            self.sigmas = sigmas
            self.sigma_fn = lambda t: t.neg().exp()
            self.t_fn = lambda sigma: sigma.log().neg()
            self.modality = modality

        def process_sigma_embeddings(self, sigma):
            sigmas = sigma.log() / 4
            sigmas = einops.rearrange(sigmas, "b -> b 1")
            emb_t = self.denoiser.inner_model.sigma_emb(sigmas)
            if len(emb_t.shape) == 2:
                emb_t = einops.rearrange(emb_t, "b d -> b 1 d")
            return emb_t

        def preprocess_goals(self, goals, states_length):
            if len(goals.shape) == 2:
                goals = einops.rearrange(goals, "b d -> b 1 d")
            if (
                goals.shape[1] == states_length
                and self.denoiser.inner_model.goal_seq_len == 1
            ):
                goals = goals[:, 0, :]
                goals = einops.rearrange(goals, "b d -> b 1 d")
            if goals.shape[-1] == 2 * self.denoiser.inner_model.obs_dim:
                goals = goals[:, :, : self.denoiser.inner_model.obs_dim]
            return goals

        def concatenate_inputs(self, emb_t, goal_embed, state_embed, proprio_embed):
            input_seq_components = [state_embed]
            if self.denoiser.inner_model.goal_conditioned:
                input_seq_components.insert(0, goal_embed)
            if proprio_embed is not None:
                input_seq_components.append(proprio_embed)
            else:
                if not self.denoiser.inner_model.goal_conditioned:
                    input_seq_components.append(
                        self.denoiser.inner_model.drop(goal_embed)
                    )
            if not self.denoiser.inner_model.use_ada_conditioning:
                input_seq_components.insert(0, emb_t)
            input_seq = torch.cat(input_seq_components, dim=1)
            return input_seq

        def forward_enc_only(self, state, goals, sigma):
            # logger.info(f"State shape: {state.shape}")
            # logger.info(f"Goals shape: {goals.shape}")

            goals = self.preprocess_goals(goals, state.size(1))
            # logger.info(f"Preprocessed goals shape: {goals.shape}")

            state_embed = self.denoiser.inner_model.tok_emb(state)
            # logger.info(f"State embedding shape: {state_embed.shape}")

            goal_embed = (
                self.denoiser.inner_model.lang_emb(goals)
                if self.modality == "lang"
                else self.denoiser.inner_model.goal_emb(goals)
            )
            # logger.info(f"Goal embedding shape: {goal_embed.shape}")

            input_seq = self.concatenate_inputs(None, goal_embed, state_embed, None)
            # logger.info(f"Input sequence shape: {input_seq.shape}")

            context = self.denoiser.inner_model.encoder(input_seq)
            # logger.info(f"Context shape: {context.shape}")
            self.denoiser.inner_model.latent_encoder_emb = context
            return context

        def forward_dec_only(self, context, action, sigma):
            emb_t = self.process_sigma_embeddings(sigma)
            action_embed = self.denoiser.inner_model.action_emb(action)
            action_x = self.denoiser.inner_model.drop(action_embed)
            if (
                self.denoiser.inner_model.use_ada_conditioning
            ):  # should be True for mdtv
                x = self.denoiser.inner_model.decoder(action_x, emb_t, context)
            else:
                x = self.denoiser.inner_model.decoder(action_x, context)
            pred_action = self.denoiser.inner_model.action_pred(x)
            return pred_action

        def forward(self, state_images, action, latent_goal):
            # ddim
            s_in = torch.ones([action.shape[0]])
            for i in range(self.sigmas.size(0) - 1):
                logger.info(f"ddim step {i}")
                sigma = self.sigmas[i] * s_in
                c_skip, c_out, c_in = [
                    x[(...,) + (None,) * (action.ndim - x.ndim)]
                    for x in self.denoiser.get_scalings(sigma)
                ]
                # customized inner_model
                action_in = action * c_in
                # forward_enc_only
                context = self.forward_enc_only(state_images, latent_goal, sigma)
                # forward_dec_only
                pred_action = self.forward_dec_only(context, action_in, sigma)
                denoised = pred_action * c_out + action * c_skip
                # post process
                t, t_next = self.t_fn(self.sigmas[i]), self.t_fn(self.sigmas[i + 1])
                h = t_next - t
                action = (self.sigma_fn(t_next) / self.sigma_fn(t)) * action - (
                    -h
                ).expm1() * denoised
            return action

    reference_denoiser = DenoiserModelWrapper(gc_denoiser, sigmas).eval()

    # test the denoiser
    logger.info("Checking model dimensions...")
    test_denoiser = DenoiserModelWrapper(gc_denoiser, sigmas).eval()
    with torch.no_grad():
        test_out = test_denoiser(
            denoiser_inputs["state_images"],
            denoiser_inputs["action"],
            denoiser_inputs["latent_goal"],
        )
        logger.info(f"Test output shape: {test_out.shape}")
    logger.info("Checking model dimensions done.")
    # test the denoiser done.

    logger.info(f"JIT tracing reference denoiser...")
    traced_reference_denoiser = torch.jit.trace(
        reference_denoiser,
        (
            denoiser_inputs["state_images"],
            denoiser_inputs["action"],
            denoiser_inputs["latent_goal"],
        ),
        # strict=False,
        # check_trace=True,
    )
    logger.info(f"JIT tracing reference denoiser done")

    # # print and save the graph
    # logger.info("Traced model graph:")
    # logger.info(traced_reference_denoiser.graph)
    # with open(os.path.join(output_dir, "traced_model_graph.txt"), "w") as f:
    #     f.write(str(traced_reference_denoiser.graph))

    del gc_denoiser
    gc.collect()

    # Convert to CoreML
    return convert_to_coreml(
        submodule_name="GCDenoiser",
        torchscript_module=traced_reference_denoiser,
        sample_inputs=denoiser_inputs,
        output_names=["pred_actions"],
        output_dir=output_dir,
        compute_unit=compute_unit,
        check_output_correctness=check_output_correctness,
    )

# python coreml/torch2coreml.py
# python coreml/torch2coreml.py --config-name=convert_prune_e4_d3
# python coreml/torch2coreml.py --config-name=convert_prune_e4_d2
# python coreml/torch2coreml.py --config-name=convert_prune_e4_d1
# python coreml/torch2coreml.py --config-name=convert_prune_e3_d3
# python coreml/torch2coreml.py --config-name=convert_prune_e3_d2
# python coreml/torch2coreml.py --config-name=convert_prune_e3_d1
# python coreml/torch2coreml.py --config-name=convert_prune_e2_d3
# python coreml/torch2coreml.py --config-name=convert_prune_e2_d2
# python coreml/torch2coreml.py --config-name=convert_prune_e2_d1
# python coreml/torch2coreml.py --config-name=convert_prune_e1_d3
# python coreml/torch2coreml.py --config-name=convert_prune_e1_d2
# python coreml/torch2coreml.py --config-name=convert_prune_e1_d1
@hydra.main(config_path=".", config_name="convert")
def main(cfg: DictConfig):
    """Main function for converting MDTVAgent components to CoreML models."""
    # Print version info
    logger.info(f"Using PyTorch {torch.__version__}, CoreMLTools {ct.__version__}")

    # Make output dir an absolute path
    output_dir = os.path.abspath(cfg.output_dir)
    if cfg.clean_output_dir:
        if os.path.exists(output_dir):
            if (input(f"Output directory {output_dir} already exists. Continue? (Y/n)") or "Y") == "Y":
                shutil.rmtree(output_dir)
                logger.info(f"Output directory {output_dir} cleaned.")
            else:
                logger.info(f"Output directory {output_dir} not cleaned.")
                exit()

    # Load the model using hydra config
    model = load_pretrained_model(cfg,
                                  use_ema_weights=cfg.get("use_ema_weights", True),
                                  strict_loading=cfg.get("strict", True))

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    logger.info(f"Output directory: {output_dir}")

    # Get input shapes from config
    input_shapes = {
        "bs": 1,
        "image_size": cfg.get("image_size", 224),
        "obs_seq_len": cfg.get("obs_seq_len", 1),
        "sequence_length": cfg.get("text_token_sequence_length", 77),
        "embedding_dim": 768,  # Default for CLIP/Voltron
        "latent_dim": 512,  # Default dimension for latent goal
        "num_token_voltron": 3,  # Default number of tokens in Voltron with perceiver resampler
        "perceiver_dim": 384,
        "act_window_size": 10,
        "action_dim": 7,
        "num_sampling_steps": 10,
    }

    # Convert requested components
    converted_models = {}

    if cfg.convert_all:
        logger.info("Converting all models (language goal, visual goal, voltron, gc denoiser) to CoreML...")
        converted_models["all"] = convert_all(
            model=model,
            input_shapes=input_shapes,
            output_dir=output_dir,
            compute_unit=cfg.get("compute_unit", "ALL"),
            check_output_correctness=cfg.get("check_output_correctness", False),
        )
        logger.info(
            f"All models converted and saved to {converted_models['all']}"
        )

    if cfg.convert_language_goal:
        logger.info("Converting Language Goal model to CoreML...")
        converted_models["language_goal"] = convert_language_goal(
            model=model,
            input_shapes=input_shapes,
            output_dir=output_dir,
            compute_unit=cfg.get("compute_unit", "ALL"),
            check_output_correctness=cfg.get("check_output_correctness", False),
        )
        logger.info(
            f"Language Goal model converted and saved to {converted_models['language_goal']}"
        )

    if cfg.convert_visual_goal:
        logger.info("Converting Visual Goal model to CoreML...")
        converted_models["visual_goal"] = convert_visual_goal(
            model=model,
            input_shapes=input_shapes,
            output_dir=output_dir,
            compute_unit=cfg.get("compute_unit", "ALL"),
            check_output_correctness=cfg.get("check_output_correctness", False),
        )
        logger.info(
            f"Visual Goal model converted and saved to {converted_models['visual_goal']}"
        )

    if cfg.convert_voltron:
        logger.info("Converting Voltron model to CoreML...")
        converted_models["voltron"] = convert_voltron(
            model=model,
            input_shapes=input_shapes,
            output_dir=output_dir,
            compute_unit=cfg.get("compute_unit", "ALL"),
            check_output_correctness=cfg.get("check_output_correctness", False),
        )
        logger.info(
            f"Voltron model converted and saved to {converted_models['voltron']}"
        )

    if cfg.convert_gcdenoiser:
        logger.info("Converting GC Denoiser model to CoreML...")
        converted_models["gc_denoiser"] = convert_gcdenoiser(
            model=model,
            input_shapes=input_shapes,
            output_dir=output_dir,
            compute_unit=cfg.get("compute_unit", "ALL"),
            check_output_correctness=cfg.get("check_output_correctness", False),
        )
        logger.info(
            f"GC Denoiser model converted and saved to {converted_models['gc_denoiser']}"
        )

    # Bundle resources if requested
    if cfg.get("bundle_resources", False) and converted_models:
        logger.info("Bundling resources for Swift CLI...")
        swift_resources_dir = os.path.join(output_dir, "MDTResources")
        os.makedirs(swift_resources_dir, exist_ok=True)

        # Copy all converted models to the resources directory
        for component, model_path in converted_models.items():
            # Create symbolic links or copy files
            target_dir = os.path.join(swift_resources_dir, os.path.basename(model_path))
            if os.path.exists(target_dir):
                shutil.rmtree(target_dir)
            os.symlink(os.path.abspath(model_path), target_dir)

        logger.info(f"Resources bundled at {swift_resources_dir}")

    # Summary
    if not converted_models:
        logger.warning(
            "No components were converted. Please specify at least one component to convert."
        )
    else:
        logger.info(
            f"Successfully converted {len(converted_models)} components: {', '.join(converted_models.keys())}"
        )


if __name__ == "__main__":
    # Example usage:
    # python -m coreml.torch2coreml --convert-voltron --convert-language-goal --convert-visual-goal --convert-gcdenoiser --output-dir mdt_mlpackage train_folder=pretrained_models/mdt/CALVIN\ D/mdtv_1_d
    main()
