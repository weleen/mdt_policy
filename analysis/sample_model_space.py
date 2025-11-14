import argparse
import itertools
import csv
from dataclasses import dataclass
from typing import List, Tuple

ENCODER_BLOCKS = 4
DECODER_BLOCKS = 4

# Base parameter counts derived from prior analysis.
BASE_PARAMS = 2_234_887
ENC_BASE = 591_744
DEC_BASE = 2_070_528

MLP_PARAMS = {
    0.4: 471_552,
    0.6: 708_096,
    0.8: 943_872,
}

HEAD_OPTIONS = {
    0.4: 0.4,
    0.6: 0.6,
    0.8: 0.8,
}


@dataclass(frozen=True)
class BlockConfig:
    active: bool
    mlp_ratio: float
    attn_head_ratio: float

    @classmethod
    def off(cls) -> "BlockConfig":
        return cls(False, 0.0, 0.0)

    @classmethod
    def on(cls, ratio: float) -> "BlockConfig":
        return cls(True, ratio, HEAD_OPTIONS[ratio])


def block_state_options() -> List[BlockConfig]:
    options = [BlockConfig.off()]
    for ratio in (0.4, 0.6, 0.8):
        options.append(BlockConfig.on(ratio))
    return options


BLOCK_STATES = block_state_options()


def block_params(is_encoder: bool, block: BlockConfig) -> int:
    if not block.active:
        return 0
    base = ENC_BASE if is_encoder else DEC_BASE
    return base + MLP_PARAMS[block.mlp_ratio]


def enumerate_param_space() -> List[Tuple[int, List[BlockConfig]]]:
    combos: List[Tuple[int, List[BlockConfig]]] = []
    for enc_states in itertools.product(BLOCK_STATES, repeat=ENCODER_BLOCKS):
        enc_params = sum(block_params(True, cfg) for cfg in enc_states)
        for dec_states in itertools.product(BLOCK_STATES, repeat=DECODER_BLOCKS):
            dec_params = sum(block_params(False, cfg) for cfg in dec_states)
            total = BASE_PARAMS + enc_params + dec_params
            combos.append((total, list(enc_states + dec_states)))
    return combos


def uniform_samples(combos: List[Tuple[int, List[BlockConfig]]], num_samples: int) -> List[Tuple[int, List[BlockConfig]]]:
    combos = sorted(combos, key=lambda item: item[0])
    if num_samples >= len(combos):
        return combos
    if num_samples <= 1:
        return [combos[0]]
    indices = [round(i * (len(combos) - 1) / (num_samples - 1)) for i in range(num_samples)]
    return [combos[idx] for idx in indices]


def serialize_samples(samples: List[Tuple[int, List[BlockConfig]]]) -> List[dict]:
    rows = []
    for idx, (params, blocks) in enumerate(samples):
        row = {"id": idx + 1, "parameters": params}
        for block_idx, cfg in enumerate(blocks):
            if block_idx < ENCODER_BLOCKS:
                prefix = f"enc{block_idx + 1}"
            else:
                prefix = f"dec{block_idx - ENCODER_BLOCKS + 1}"

            row[f"{prefix}_z"] = int(cfg.active)
            row[f"{prefix}_r"] = cfg.mlp_ratio if cfg.active else 0.0
            row[f"{prefix}_h"] = cfg.attn_head_ratio if cfg.active else 0.0

        rows.append(row)
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sample MDT model configurations uniformly by parameter count.")
    parser.add_argument("--num-samples", type=int, default=200, help="Number of uniform samples to generate.")
    parser.add_argument(
        "--output",
        type=str,
        default="analysis/model_samples_uniform_200.csv",
        help="Path to write the sampled configurations as CSV.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    combos = enumerate_param_space()
    samples = uniform_samples(combos, args.num_samples)
    serialized = serialize_samples(samples)
    fieldnames = list(serialized[0].keys()) if serialized else []
    with open(args.output, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(serialized)


if __name__ == "__main__":
    main()
