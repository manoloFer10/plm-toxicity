import os
import torch

from utils.models.protgpt2 import ProtGPT2
from utils.extract_activations import get_direction_ablation_input_pre_hook, add_hooks


# make CUDA errors synchronous for clearer debugging
os.environ.setdefault("CUDA_LAUNCH_BLOCKING", "1")


def check_for_bad_values(layer_idx: int):
    """Hook that reports NaN/inf tensors."""

    def hook(_, __, output):
        tensor = output[0] if isinstance(output, tuple) else output
        if not torch.isfinite(tensor).all():
            print(f"\u26a0\ufe0f  non-finite tensor after layer {layer_idx}")
            print("    min:", torch.nanmin(tensor).item(), "max:", torch.nanmax(tensor).item())
            raise RuntimeError(f"non-finite activation at layer {layer_idx}")

    return hook


def main() -> None:
    """Iterate through all saved directions and report failures."""

    # load model once
    model = ProtGPT2("nferruz/ProtGPT2")

    # load directions (shape: [positions, layers, d_model])
    mean_diffs = torch.load("activations/mean_diffs.pt").squeeze(0)

    for layer_idx, direction in enumerate(mean_diffs):
        print(f"\nTesting direction from layer {layer_idx}")
        direction = direction.to(model.device)

        # add ablation hook on every block using this direction
        pre_hooks = [
            (block, get_direction_ablation_input_pre_hook(direction.clone()))
            for block in model.model_block_modules
        ]

        # add diagnostic hooks to catch NaN/inf values
        debug_hooks = [
            (block, check_for_bad_values(i))
            for i, block in enumerate(model.model_block_modules)
        ]

        try:
            with add_hooks(pre_hooks, debug_hooks):
                # generate a few tokens; adjust prompts if needed
                model.generate_de_novo(["M"], batch_size=1, max_new_tokens=10)
        except Exception as err:
            print(f"    -> direction {layer_idx} triggered: {err}")


if __name__ == "__main__":
    main()