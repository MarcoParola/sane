from typing import NamedTuple
from collections import defaultdict
import jax.numpy as jnp
from jax import random
from scipy.optimize import linear_sum_assignment
from src.models.cnn.cnn import CNN
import torch
from pathlib import Path
import numpy as np


rngmix = lambda rng, x: random.fold_in(rng, hash(x))

class PermutationSpec(NamedTuple):
    perm_to_axes: dict
    axes_to_perm: dict


def permutation_spec_from_axes_to_perm(axes_to_perm: dict) -> PermutationSpec:
    perm_to_axes = defaultdict(list)
    for wk, axis_perms in axes_to_perm.items():
        for axis, perm in enumerate(axis_perms):
            if perm is not None:
                perm_to_axes[perm].append((wk, axis))
    return PermutationSpec(perm_to_axes=dict(perm_to_axes), axes_to_perm=axes_to_perm)


def cnn_permutation_spec():
    conv = lambda name, p_in, p_out: {f"{name}.weight": (p_out, p_in), f"{name}.bias": (p_out,)}
    dense = lambda name, p_in, p_out: {f"{name}.weight": (p_out, p_in), f"{name}.bias": (p_out,)}

    return permutation_spec_from_axes_to_perm({
        **conv("module_list.0", None, "P0"),   # Conv1: 3 -> 8
        **conv("module_list.3", "P0", "P1"),   # Conv2: 8 -> 6
        **conv("module_list.6", "P1", "P2"),   # Conv3: 6 -> 4
        **dense("module_list.9", None, "P3"),  # FC1: 36 -> 20
        **dense("module_list.11", "P3", None), # FC2: 20 -> 10
    })


def get_permuted_param(ps: PermutationSpec, perm, k: str, params, except_axis=None):
    """Get parameter `k` from `params`, with the permutations applied."""
    w = params[k]
    for axis, p in enumerate(ps.axes_to_perm[k]):
        # Skip the axis we're trying to permute.
        if axis == except_axis:
            continue

        # None indicates that there is no permutation relevant to that axis.
        if p is not None:
            w = jnp.take(w, perm[p], axis=axis)

    return w


def apply_permutation(ps: PermutationSpec, perm, params):
    """Apply a `perm` to `params`."""
    return {k: get_permuted_param(ps, perm, k, params) for k in params.keys()}


def weight_matching(rng,
                    ps: PermutationSpec,
                    params_a,
                    params_b,
                    max_iter=100,
                    init_perm=None,
                    silent=False):
    """Find a permutation of `params_b` to make them match `params_a`."""
    perm_sizes = {p: params_a[axes[0][0]].shape[axes[0][1]] for p, axes in ps.perm_to_axes.items()}

    perm = {p: jnp.arange(n) for p, n in perm_sizes.items()} if init_perm is None else init_perm
    perm_names = list(perm.keys())

    for iteration in range(max_iter):
        progress = False
        for p_ix in random.permutation(rngmix(rng, iteration), len(perm_names)):
            p = perm_names[p_ix]
            n = perm_sizes[p]
            A = jnp.zeros((n, n))
            for wk, axis in ps.perm_to_axes[p]:
                w_a = params_a[wk]
                w_b = get_permuted_param(ps, perm, wk, params_b, except_axis=axis)
                w_a = jnp.moveaxis(w_a, axis, 0).reshape((n, -1))
                w_b = jnp.moveaxis(w_b, axis, 0).reshape((n, -1))
                A += w_a @ w_b.T

            ri, ci = linear_sum_assignment(A, maximize=True)
            assert (ri == jnp.arange(len(ri))).all()

            oldL = jnp.vdot(A, jnp.eye(n)[perm[p]])
            newL = jnp.vdot(A, jnp.eye(n)[ci, :])
            if not silent: print(f"{iteration}/{p}: {newL - oldL}")
            progress = progress or newL > oldL + 1e-12

            perm[p] = jnp.array(ci)

        if not progress:
            break

    return perm


def permute_model_zoo(zoo):
    # Load model zoo checkpoints
    zoo_path = Path(zoo)
    model_zoo = []
    counter = 0
    for folder in zoo_path.iterdir():
        if folder.is_dir():
            current_checkpoint_path = folder / "checkpoint_000050/checkpoints"
            if current_checkpoint_path.exists():
                model = CNN()
                checkpoint = torch.load(current_checkpoint_path, weights_only=False)
                model.load_state_dict(checkpoint)
                model_zoo.append(model)
                counter = counter+1
                print(f"\rLoaded {counter} model(s)", end='', flush=True)

    print("Permuting the CNN model zoo...")
    
    # Take params of the referment model (the first one)
    params_list = [
        {k: v.detach().cpu().numpy() for k,v in model.state_dict().items()}
        for model in model_zoo
    ]
    params_a = params_list[0]
    aligned_params_list = [params_a]
    
    # Align all other model
    ps = cnn_permutation_spec()
    rng = random.PRNGKey(42)    
    counter = 0
    for params_b in params_list[1:]:
        perm = weight_matching(rng, ps, params_a, params_b)
        params_b_aligned = apply_permutation(ps, perm, params_b)
        counter = counter +1
        print(f"\rAligned {counter} model(s)\n\n", end='', flush=True)
        aligned_params_list.append(params_b_aligned)

    # Convert back to PyTorch state_dicts
    aligned_state_dicts = [
        {k: torch.from_numpy(np.array(jnp.asarray(v))).float() for k, v in params.items()}
        for params in aligned_params_list
    ]

    return aligned_state_dicts


if __name__ == "__main__":
    cnn_zoo_path = "checkpoints/tune_zoo_cifar10_uniform_small"
    aligned_models = permute_model_zoo(cnn_zoo_path)
    print(f"All {len(aligned_models)} models are aligned")