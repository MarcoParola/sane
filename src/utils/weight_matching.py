import torch
from scipy.optimize import linear_sum_assignment
from typing import NamedTuple, Dict
from collections import defaultdict
from pathlib import Path
from src.models.cnn.cnn import CNN


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
    conv = lambda name, p_in, p_out: {f"{name}.weight": (p_out, p_in, None, None), f"{name}.bias": (p_out,)}
    dense = lambda name, p_in, p_out: {f"{name}.weight": (p_out, p_in), f"{name}.bias": (p_out,)}

    return permutation_spec_from_axes_to_perm({
        **conv("module_list.0", None, "P0"),
        **conv("module_list.3", "P0", "P1"),
        **conv("module_list.6", "P1", None),
        **dense("module_list.9", None, "P3"),
        **dense("module_list.11", "P3", None),
    })


def get_permuted_param(ps: PermutationSpec, perm: Dict[str, torch.Tensor], k: str, params: Dict[str, torch.Tensor], except_axis=None):
    """Get parameter `k` from `params`, with the permutations applied."""
    w = params[k]
    for axis, p in enumerate(ps.axes_to_perm[k]):
        # Skip the axis we're trying to permute.
        if axis == except_axis:
            continue
        # None indicates that there is no permutation relevant to that axis.
        if p is not None:
            w = w.index_select(axis, perm[p])
    return w


def apply_permutation(ps: PermutationSpec, perm: Dict[str, torch.Tensor], params: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Apply a `perm` to `params`."""
    return {k: get_permuted_param(ps, perm, k, params) for k in params.keys()}


def weight_matching(ps: PermutationSpec, params_a: Dict[str, torch.Tensor], params_b: Dict[str, torch.Tensor], max_iter: int = 100, init_perm=None, device='cpu'):
    """Find permutation of 'params_b' that best aligns to 'params_a'."""
    perm_sizes = {
        p: params_a[axes[0][0]].shape[axes[0][1]]
        for p, axes in ps.perm_to_axes.items()
    }

    perm = {p: torch.arange(n, device=device) for p, n in perm_sizes.items()} if init_perm is None else init_perm
    perm_names = list(perm.keys())

    for iteration in range(max_iter):
        progress = False
        for p_name in perm_names:
            n = perm_sizes[p_name]
            A = torch.zeros((n, n), device=device)

            for wk, axis in ps.perm_to_axes[p_name]:
                w_a = params_a[wk].to(device)
                w_b = get_permuted_param(ps, perm, wk, params_b, except_axis=axis).to(device)

                w_a = w_a.moveaxis(axis, 0).reshape(n, -1)
                w_b = w_b.moveaxis(axis, 0).reshape(n, -1)

                A += w_a @ w_b.T

            A_np = A.cpu().numpy()
            ri, ci = linear_sum_assignment(A_np, maximize=True)
            ci_torch = torch.tensor(ci, device=device, dtype=torch.long)

            oldL = torch.sum(A[torch.arange(n, device=device), perm[p_name]])
            newL = torch.sum(A[torch.arange(n, device=device), ci_torch])

            print(f"\r{iteration}/{p_name}: {newL-oldL}", end='', flush=True)

            if newL > oldL + 1e-12:
                progress = True
                perm[p_name] = ci_torch

        if not progress:
            break

    return perm


def permute_model_zoo(zoo_path: str, device='cpu'):
    # Load model zoo checkpoints
    zoo_path = Path(zoo_path)
    model_zoo = []
    counter = 0
    for folder in zoo_path.iterdir():
        if folder.is_dir():
            current_checkpoint_path = folder / "checkpoint_000050/checkpoints"
            if current_checkpoint_path.exists():
                model = CNN()
                checkpoint = torch.load(current_checkpoint_path, weights_only=False)
                model.load_state_dict(checkpoint)
                counter = counter+1
                print(f"\rLoaded {counter} model(s)", end='', flush=True)
                model_zoo.append(model)
    print()

    # Take params of the referement model (the first one)
    params_list = [{k: v.detach().clone().to(device) for k, v in m.state_dict().items()} for m in model_zoo]
    params_a = params_list[0]
    aligned_params_list = [params_a]

    ps = cnn_permutation_spec()

    for params_b in params_list[1:]:
        perm = weight_matching(ps, params_a, params_b, device=device)
        aligned = apply_permutation(ps, perm, params_b)
        aligned_params_list.append(aligned)
    
    print(f"\nAligned {len(aligned_params_list)} model(s)", end='', flush=True)

    aligned_state_dicts = [
        {k: v.cpu() for k, v in params.items()}
        for params in aligned_params_list
    ]

    return aligned_state_dicts



if __name__ == "__main__":

    from test_classifier import test_classifier
    from src.datasets.utils import load_dataset
    from src.models.utils import load_model

    cnn_zoo_path = "checkpoints/tune_zoo_cifar10_uniform_small"
    aligned_state_dicts = permute_model_zoo(cnn_zoo_path)

    model_name = "cnn"
    dataset_name="cifar10"
    num_classes = 10
    batch_size=32
    device = "cuda"

    train, val, test, remapping = load_dataset(dataset_name, "data", model_name, 32)
    model = load_model(model_name, dataset_name)
    model.to(device)

    for i in range(870, 880):
        checkpoint_post = aligned_state_dicts[i]
        model.load_state_dict(checkpoint_post)
        model.eval()
        test_classifier(model, test, num_classes, batch_size, device, remapping)