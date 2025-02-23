import torch
import time
from collections import defaultdict
from tabulate import tabulate
import logging
import pandas as pd
import types

from typing import Optional, Dict, Any 
from diffusers.models.resnet import ResnetBlock2D, TemporalResnetBlock, SpatioTemporalResBlock
from diffusers.models.attention import BasicTransformerBlock, TemporalBasicTransformerBlock
from diffusers.models.transformers.transformer_temporal import TransformerSpatioTemporalModel, TransformerTemporalModelOutput

logger = logging.getLogger(__name__)


def pad_dict(data):
    max_length = max(len(lst) for lst in data.values())
    return {key: lst + [None]*(max_length-len(lst)) for key, lst in data.items()}


def write_to_excel(data: defaultdict, filename: str):
    data = dict(data)
    if isinstance(list(data.values())[0], list):
        data = pad_dict(data)
        df = pd.DataFrame(data)
    else:
        df = pd.Series(data)
    df.to_csv(filename)

# Convert the dictionary to a list of lists for tabulate
def print_table(data: defaultdict):
    data = dict(data)
    # round_size = len(data[list(data.keys())[0]])
    # data['Round'] = list(range(1, round_size + 1))
    # table_data = [data[key] for key in data]
    # Transpose the data
    # transposed_data = list(map(list, zip(*table_data)))
    # Print the table
    logger.info(tabulate(data.items(), headers=data.keys()))


# Hooks
def print_module_name_hook(name, module, input):
    logger.info(f"Executing hook for module: {name}")
# module.register_forward_pre_hook(print_module_name_hook)

# Dictionary to store execution times
# sync time with torch.cuda.synchronize()
def start_time_setup_hook(module, input):
    torch.cuda.current_stream().synchronize()
    setattr(module, '_start_time', time.time())
# module.register_forward_pre_hook(start_time_setup_hook)

execution_times = defaultdict(list)
def execute_time_record_hook(name, module, input, output):
    start_time = getattr(module, '_start_time', None)
    if start_time is not None:
        torch.cuda.current_stream().synchronize()
        execution_time = time.time() - start_time
        logger.info(f"module {name}({module.__class__.__name__}): {1e3 * execution_time:.3f} ms")
        execution_times[name].append(1e3 * execution_time)
# module.register_forward_hook(execute_time_record_hook)

parameters_table = defaultdict(float)
# print number of parameters in the model
def count_parameters_hook(name, module, input):
    logger.info(f"module {name}({module.__class__.__name__}): {sum(p.numel() for p in module.parameters() if p.requires_grad) / 1e6:.3f} M")
    parameters_table[name] = sum(p.numel() for p in module.parameters() if p.requires_grad) / 1e6
# module.register_forward_pre_hook(partial(count_parameters_hook, name))

def skip_hook(name, module, input):
    print(f'skip layer {name}')
    return input
# module.register_forward_pre_hook(partial(conditional_skip_hook, #name))


# def stochastic_depth_hook(death_rate, module, input, output):
#     if not module.training or death_rate == 0.:
#         return output

#     survival_prob = 1. - death_rate
#     batch_size = input.shape[0]
#     binary_mask = torch.rand((batch_size, [1] * len(input.shape[1:])), device=output.device) < survival_prob
#     scale_factor = 1 / survival_prob  # Scaling to maintain expected output magnitude

#     if isinstance(module, ResnetBlock3D) or isinstance(module, VanillaTemporalModule):
#         output = (output - input[0]) * binary_mask * scale_factor + input[0]
#     elif isinstance(module, Transformer3DModel):
#         output = output.sample  # output is Transformer3DModelOutput
#         output = (output - input[0]) * binary_mask * scale_factor + input[0]
#         output = Transformer3DModelOutput(sample=output)
#     else:
#         raise NotImplementedError
#     return output


def debug_hook(name, module, input, output):
    print(f"Module: {name} ({module.__class__.__name__})")
    print(f"Inputs: {type(input)}, len(inputs)={len(input)}")
    print(f"Output: {type(output)}, len(inputs)={len(output)}")
    return output  # or manipulate inputs/output as required


# Replace the forward function of the module to support blockwise FVD evaluation
# Set the variable replace_with_identity to True before the forward function
# module.replace_with_identity = True

# replace forward func
# from diffusers.models.resnet import ResnetBlock2D, TemporalResnetBlock, SpatioTemporalResBlock
# from diffusers.models.attention import BasicTransformerBlock, TemporalBasicTransformerBlock
# from diffusers.models.transformers import TransformerSpatioTemporalModel

forward_SpatioTemporalResBlock_time = 0
forward_TransformerSpatioTemporalModel_time = 0
forward_ResnetBlock2D_time = 0
forward_TemporalResnetBlock_time = 0
forward_BasicTransformerBlock_time = 0
forward_TemporalBasicTransformerBlock_time = 0

def forward_SpatioTemporalResBlock(
    self,
    hidden_states: torch.Tensor,
    temb: Optional[torch.Tensor] = None,
    image_only_indicator: Optional[torch.Tensor] = None,
    debug=False,
):
    assert self.replace_with_identity, "self.replace_with_identity should be True"
    if debug:
        global forward_SpatioTemporalResBlock_time
        print(f"forward_SpatioTemporalResBlock_time: {forward_SpatioTemporalResBlock_time}")
        forward_SpatioTemporalResBlock_time += 1
    if self.spatial_res_block.conv_shortcut is not None:
        if debug:
            print('forward_SpatioTemporalResBlock: conv_shortcut')
        hidden_states = self.spatial_res_block.conv_shortcut(hidden_states) # TODO: in_channel and out_channel is not same!!!
    return hidden_states

def forward_TransformerSpatioTemporalModel(
    self,
    hidden_states: torch.Tensor,
    encoder_hidden_states: Optional[torch.Tensor] = None,
    image_only_indicator: Optional[torch.Tensor] = None,
    return_dict: bool = True,
    debug=False,
):
    assert self.replace_with_identity, "self.replace_with_identity should be True"
    if debug:
        global forward_TransformerSpatioTemporalModel_time
        print(f"forward_TransformerSpatioTemporalModel_time: {forward_TransformerSpatioTemporalModel_time}")
        forward_TransformerSpatioTemporalModel_time += 1
    return TransformerTemporalModelOutput(sample=hidden_states)

def forward_ResnetBlock2D(self, input_tensor: torch.Tensor, temb: torch.Tensor, debug=False, *args, **kwargs) -> torch.Tensor:
    assert self.replace_with_identity, "self.replace_with_identity should be True"
    if debug:
        global forward_ResnetBlock2D_time
        print(f"forward_ResnetBlock2D_time: {forward_ResnetBlock2D_time}")
        forward_ResnetBlock2D_time += 1
    output_tensor = input_tensor
    if self.conv_shortcut is not None:
        if debug:
            print('forward_ResnetBlock2D: conv_shortcut')
        output_tensor = self.conv_shortcut(input_tensor) # TODO: in_channel and out_channel is not same!!!

    return output_tensor

def forward_TemporalResnetBlock(self, input_tensor: torch.Tensor, temb: torch.Tensor, debug=False) -> torch.Tensor:
    assert self.replace_with_identity, "self.replace_with_identity should be True"
    if debug:
        global forward_TemporalResnetBlock_time
        print(f"forward_TemporalResnetBlock_time: {forward_TemporalResnetBlock_time}")
        forward_TemporalResnetBlock_time += 1
    output_tensor = input_tensor
    if self.conv_shortcut is not None:
        if debug:
            print('forward_TemporalResnetBlock: conv_shortcut')
        output_tensor = self.conv_shortcut(input_tensor)
    return output_tensor

def forward_BasicTransformerBlock(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    encoder_hidden_states: Optional[torch.Tensor] = None,
    encoder_attention_mask: Optional[torch.Tensor] = None,
    timestep: Optional[torch.LongTensor] = None,
    cross_attention_kwargs: Dict[str, Any] = None,
    class_labels: Optional[torch.LongTensor] = None,
    added_cond_kwargs: Optional[Dict[str, torch.Tensor]] = None,
    debug=False,
) -> torch.Tensor:
    assert self.replace_with_identity, "self.replace_with_identity should be True"
    if debug:
        global forward_BasicTransformerBlock_time
        print(f"forward_BasicTransformerBlock_time: {forward_BasicTransformerBlock_time}")
        forward_BasicTransformerBlock_time += 1
    if hidden_states.ndim == 4:
        hidden_states = hidden_states.squeeze(1)
    output_tensor = hidden_states
    return output_tensor

def forward_TemporalBasicTransformerBlock(
    self,
    hidden_states: torch.Tensor,
    num_frames: int,
    encoder_hidden_states: Optional[torch.Tensor] = None,
    debug=False,
) -> torch.Tensor:
    assert self.replace_with_identity, "self.replace_with_identity should be True"
    if debug:
        global forward_TemporalBasicTransformerBlock_time
        print(f"forward_TemporalBasicTransformerBlock_time: {forward_TemporalBasicTransformerBlock_time}")
        forward_TemporalBasicTransformerBlock_time += 1
    return hidden_states

def replace_forward_func(module, debug=False):
    module.__setattr__('replace_with_identity', True)
    origin_forward_func = module.forward

    def forward_wrapper(forward_func):
        def wrapped_forward(*args, **kwargs):
            if debug:
                print(f"Debugging {module.__class__.__name__} forward pass")
            return forward_func(debug=debug, *args, **kwargs)
        return wrapped_forward

    if isinstance(module, ResnetBlock2D):
        module.forward = forward_wrapper(types.MethodType(forward_ResnetBlock2D, module))
    elif isinstance(module, TemporalResnetBlock):
        module.forward = forward_wrapper(types.MethodType(forward_TemporalResnetBlock, module))
    elif isinstance(module, BasicTransformerBlock):
        module.forward = forward_wrapper(types.MethodType(forward_BasicTransformerBlock, module))
    elif isinstance(module, TemporalBasicTransformerBlock):
        module.forward = forward_wrapper(types.MethodType(forward_TemporalBasicTransformerBlock, module))
    elif isinstance(module, TransformerSpatioTemporalModel):
        module.forward = forward_wrapper(types.MethodType(forward_TransformerSpatioTemporalModel, module))
    elif isinstance(module, SpatioTemporalResBlock):
        module.forward = forward_wrapper(types.MethodType(forward_SpatioTemporalResBlock, module))
    return module, origin_forward_func

def restore_forward_func(module, old_forward_func):
    del module.replace_with_identity
    module.forward = old_forward_func
    return module

if __name__ == "__main__":
    pass
    # # TODO: Implement stochastic depth via register_forward_hook
    # if death_mode != 'none':
    #     assert isinstance(unet, UNet3DConditionModelStochasticDepth), "Stochastic depth is only supported for UNet3DConditionModelStochasticDepth"
    #     # register hooks for elastic depth training
    #     logger.info("registering forward hooks for unet")
    #     # death rates
    #     if death_mode == 'uniform':
    #         modules_death_rate = {k: death_rate for k in module_dict.keys()}
    #     elif death_mode == 'linear':
    #         num_modules = len(module_dict)
    #         death_rates = torch.linspace(0, death_rate, num_modules // 2)
    #         if num_modules % 2 == 0:
    #             death_rates = torch.cat([death_rates, death_rates.flip(0)])
    #         else:
    #             death_rates = torch.cat([death_rates, death_rates[-1:], death_rates.flip(0)])
    #         assert len(death_rates) == num_modules, "death rates length mismatch"
    #         modules_death_rate = {}
    #         for i, k in enumerate(module_keys):
    #             modules_death_rate.update({k: death_rates[i]})
    #     else:
    #         raise ValueError(f"Unknown death mode: {death_mode}")

    #     # block names
    #     named_modules = list(unet.named_modules())
    #     names = [n[0] for n in named_modules]
    #     other_names = ['conv_in', 'time_proj', 'time_embedding', 'conv_norm_out', 'conv_act', 'conv_out']
    #     updownsampler_names = [n for n in names if n.endswith('samplers.0')]
    #     attentions_names = [n for n in names if n.endswith('attentions.0') or n.endswith('attentions.1') or n.endswith('attentions.2')]
    #     resnets_names = [n for n in names if n.endswith('resnets.0') or n.endswith('resnets.1') or n.endswith('resnets.2')]
    #     motion_modules_names = [n for n in names if n.endswith('motion_modules.0') or n.endswith('motion_modules.1') or n.endswith('motion_modules.2')]
    #     register_names = attentions_names + resnets_names + motion_modules_names # + updownsampler_names + other_names
    #     logger.info(f'registering hooks for {len(register_names)} modules')
    #     for name, module in unet.named_modules():
    #         if name not in register_names:
    #             continue
    #         logger.info(f"registering hooks for {name}")
    #         # module.register_forward_pre_hook(partial(count_parameters_hook, name))
    #         # module.register_forward_pre_hook(start_time_setup_hook)
    #         # module.register_forward_hook(partial(execute_time_record_hook, name))
    #         # module.register_forward_hook(partial(debug_hook, name))
    #         # module.register_forward_pre_hook(partial(conditional_skip_hook, name))
    #         # if args.death_mode != 'none':
    #         #     module.register_forward_hook(partial(stochastic_depth_hook, modules_death_rate[name]))
