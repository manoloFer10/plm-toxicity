import torch
import functools
import contextlib
import functools
from tqdm import tqdm
from typing import List, Callable, Tuple
import os

@contextlib.contextmanager
def add_hooks(
    module_forward_pre_hooks: List[Tuple[torch.nn.Module, Callable]],
    module_forward_hooks: List[Tuple[torch.nn.Module, Callable]],
    **kwargs
):
    """
    Context manager for temporarily adding forward hooks to a model.

    Parameters
    ----------
    module_forward_pre_hooks
        A list of pairs: (module, fnc) The function will be registered as a
            forward pre hook on the module
    module_forward_hooks
        A list of pairs: (module, fnc) The function will be registered as a
            forward hook on the module
    """
    try:
        handles = []
        for module, hook in module_forward_pre_hooks:
            partial_hook = functools.partial(hook, **kwargs)
            handles.append(module.register_forward_pre_hook(partial_hook))
        for module, hook in module_forward_hooks:
            partial_hook = functools.partial(hook, **kwargs)
            handles.append(module.register_forward_hook(partial_hook))
        yield
    finally:
        for h in handles:
            h.remove()


def get_activations(model, tokenizer, instructions, block_modules: List[torch.nn.Module], store_means=True, batch_size=8, positions=[-1]):

    def mean_hook(layer, cache, n_samples, positions):
        def _hook(module, inputs):
            act = inputs[0][:, positions, :].to(cache)          # [B, P, d]
            cache[:, layer] += act.sum(0) / n_samples # mean update
        return _hook


    def sample_hook(layer, cache, positions, batch_start):
        def _hook(module, inputs):
            act = inputs[0][:, positions, :].to(cache)             # [B, P, d]
            B = act.size(0)
            cache[batch_start:batch_start+B, :, layer, :] = act
        return _hook

    torch.cuda.empty_cache()

    n_positions = len(positions)
    n_layers = model.config.num_hidden_layers
    n_samples = len(instructions)
    d_model = model.config.hidden_size
    device= model.device

    if store_means:
        # we store the mean activations in high-precision to avoid numerical issues
        activations = torch.zeros((n_positions, n_layers, d_model), dtype=torch.float64, device='cpu')
    else:
        # to avoid mem issues we fallback to .32 precision
        activations = torch.zeros((n_samples, n_positions, n_layers, d_model), dtype=torch.float32, device='cpu')

    for batch_start in tqdm(range(0, n_samples, batch_size), total = int(n_samples/batch_size)):
        inputs = tokenizer(
            instructions[batch_start:batch_start+batch_size],
            return_tensors="pt",
            padding=True
            )

        if store_means:
            fwd_pre_hooks = [
                (block_modules[l], mean_hook(l, activations, n_samples, positions))
                for l in range(n_layers)
            ]
        else:
            fwd_pre_hooks = [
                (block_modules[l], sample_hook(l, activations, positions, batch_start))
                for l in range(n_layers)
            ]
        with torch.inference_mode():
          with add_hooks(fwd_pre_hooks, []):
              model(
                  input_ids=inputs.input_ids.to(model.device),
                  attention_mask=inputs.attention_mask.to(model.device),
              )


    # of size (n_positions, n_layers, d_model) if store_means = False
    # of size (n_samples, n_positions, n_layers, d_model) if store_means = True
    return activations

def get_activations_for_datasets(model, tokenizer, tox_seqs, non_tox_seqs, block_modules: List[torch.nn.Module], batch_size=32, positions=[-1]):

    activations_tox = get_activations(model, tokenizer, tox_seqs, block_modules, batch_size=batch_size, positions=positions, store_means=False)
    activations_non_tox = get_activations(model, tokenizer, non_tox_seqs, block_modules, batch_size=batch_size, positions=positions, store_means=False)

    torch.save(activations_tox, f"preliminary_exps/acts/tox_acts.pt")
    torch.save(activations_non_tox, f"preliminary_exps/acts/non_tox_acts.pt")

    return activations_tox, activations_non_tox


def get_mean_diff(model, tokenizer, tox_seqs, non_tox_seqs, block_modules: List[torch.nn.Module], batch_size=32, positions=[-1]):
    activations_tox = get_activations(model, tokenizer, tox_seqs, block_modules, batch_size=batch_size, positions=positions, store_means=True)
    activations_non_tox = get_activations(model, tokenizer, non_tox_seqs, block_modules, batch_size=batch_size, positions=positions, store_means=True)

    mean_diff = activations_tox - activations_non_tox

    return mean_diff


def generate_directions(model, tokenizer, block_modules, tox_seqs, non_tox_seqs, artifact_dir):
    if not os.path.exists(artifact_dir):
        os.makedirs(artifact_dir)

    mean_diffs = get_mean_diff(model, tokenizer, tox_seqs, non_tox_seqs, block_modules)

    assert not mean_diffs.isnan().any()

    torch.save(mean_diffs, f"{artifact_dir}/mean_diffs.pt")

    return mean_diffs