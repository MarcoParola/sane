import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Optional
import numpy as np

def compute_fc_hessian(activations: torch.Tensor) -> torch.Tensor:
    B, d = activations.shape
    a = activations.unsqueeze(-1)
    ones = torch.ones(B, 1, 1, device=activations.device, dtype=activations.dtype)
    v = torch.cat([a, ones], dim=1)
    outer = v @ v.transpose(1, 2)           
    H = outer.mean(dim=0)                    
    return H


def compute_conv_hessian(activations: torch.Tensor,
                         kernel_size: int,
                         stride: int,
                         add_bias: bool = True,
                         stride_factor: int = 1) -> torch.Tensor:
    B, C, H, W = activations.shape
    k = kernel_size
    s = stride * stride_factor

    unfold = nn.Unfold(kernel_size=k, stride=s, padding=(k // 2))
    patches = unfold(activations)
    patches = patches.permute(0, 2, 1).contiguous()
    BL, D = patches.shape[0] * patches.shape[1], patches.shape[2]
    patches_flat = patches.view(-1, D)

    if add_bias:
        ones = torch.ones(patches_flat.size(0), 1, device=patches_flat.device, dtype=patches_flat.dtype)
        v = torch.cat([patches_flat, ones], dim=1)
    else:
        v = patches_flat

    v2 = v.unsqueeze(2) @ v.unsqueeze(1)
    H = v2.mean(dim=0)
    return H


def register_forward_hook_capture(module: nn.Module, capture_dict: dict, name: str):
    def hook(mod, inp, out):
        capture_dict[name] = inp[0].detach()
    handle = module.register_forward_hook(hook)
    return handle


def _get_module_by_name(model: nn.Module, target_name: str) -> Optional[nn.Module]:
    for name, m in model.named_modules():
        if name == target_name or name.endswith(target_name):
            return m
    return None


def generate_hessian(model: nn.Module,
                     trainloader: DataLoader,
                     module_name: str,
                     layer_type: str,
                     n_batch_used: int = 50000,
                     device: str = 'cuda',
                     stride_factor: int = 3) -> torch.Tensor:
    """
    Generate empirical Hessian (averaged over n_batch_used batches) for the specified layer.
    model: PyTorch model (eval mode recommended)
    trainloader: DataLoader yielding (inputs, targets)
    module_name: name of the module to capture (string used in named_modules)
    layer_type: 'F' for FC, 'C' for conv, 'R' for conv-without-bias/res
    n_batch_used: number of batches to use
    device: 'cuda' or 'cpu'
    stride_factor: factor to multiply convolution stride (to reduce patch count)
    Returns: Hessian tensor on device
    """
    model.eval()
    capture = {}
    module = None
    for name, m in model.named_modules():
        if name == module_name or name.endswith(module_name):
            module = m
    if module is None:
            raise ValueError(f"Module {module_name} not found in model.named_modules()")

    handle = register_forward_hook_capture(module, capture, module_name)

    hessian_accum = None
    used = 0
    dev = torch.device(device if torch.cuda.is_available() and device == 'cuda' else 'cpu')
    model.to(dev)
    kernel_size = getattr(module, 'kernel_size', None)
    stride = getattr(module, 'stride', 1)
    if isinstance(kernel_size, tuple):
        k = kernel_size[0]
    elif isinstance(kernel_size, int):
        k = kernel_size
    else:
        k = None

    for batch_idx, (inputs, _) in enumerate(trainloader):
        if batch_idx >= n_batch_used:
            break
        inputs = inputs.to(dev)
        _ = model(inputs)

        if module_name not in capture:
            raise RuntimeError("Forward hook did not capture layer input. Ensure module_name is correct.")
        layer_input = capture[module_name].to(dev)

        if layer_type == 'F':
            H_batch = compute_fc_hessian(layer_input)
        elif layer_type == 'C':
            if k is None:
                w = getattr(module, 'weight', None)
                if (w is not None) and (w.dim() >= 3):
                    k = w.shape[2]
                    stride = module.stride[0] if isinstance(module.stride, tuple) else module.stride
                else:
                    raise ValueError("Could not infer kernel size for conv layer; specify module with .kernel_size")
            add_bias = (layer_type == 'C')
            H_batch = compute_conv_hessian(layer_input, kernel_size=k, stride=stride,
                                           add_bias=add_bias, stride_factor=stride_factor)
        else:
            raise ValueError("layer_type must be 'F' or 'C'")

        if hessian_accum is None:
            hessian_accum = H_batch.detach().cpu()
        else:
            hessian_accum += H_batch.detach().cpu()
        used += 1

    handle.remove()
    if used == 0:
        raise RuntimeError("No batches used to compute Hessian")
    H = (hessian_accum / used).to(dev)
    return H


def generate_hessian_inv_woodbury(model: nn.Module,
                                  trainloader: DataLoader,
                                  module_name: str,
                                  layer_type: str,
                                  n_batch_used: int = 50000,
                                  device: str = 'cuda',
                                  stride_factor: int = 3,
                                  init_diag: float = 1e6) -> torch.Tensor:
    """
    Compute inverse Hessian via iterative Woodbury (Sherman-Morrison) updates.
    Mirrors original TF logic: initial hessian_inverse = init_diag * I, then for each sample (or patch)
    do rank-1 update:
        h_inv <- h_inv - (h_inv * w^T * w * h_inv) / (dataset_size + w * h_inv * w^T)
    Returns: hessian_inverse tensor (d, d) on device.
    layer_type: 'F' -> FC (append bias), 'C' -> conv (append bias), 'R' -> res (no bias)
    """
    model.eval()
    capture = {}
    module = _get_module_by_name(model, module_name)
    if module is None:
        raise ValueError(f"Module {module_name} not found in model.named_modules()")
    handle = register_forward_hook_capture(module, capture, module_name)

    dev = torch.device(device if torch.cuda.is_available() and device == 'cuda' else 'cpu')
    model.to(dev)
    hessian_inv = None
    dataset_size = 0
    used_batches = 0

    kernel_size = getattr(module, 'kernel_size', None)
    if isinstance(kernel_size, tuple):
        k = kernel_size[0]
    elif isinstance(kernel_size, int):
        k = kernel_size
    else:
        k = None

    for batch_idx, (inputs, _) in enumerate(trainloader):
        if batch_idx >= n_batch_used:
            break
        inputs = inputs.to(dev)
        _ = model(inputs)
        layer_input = capture[module_name].to(dev)

        if layer_type == 'F':
            B = layer_input.shape[0]
            dataset_size = n_batch_used * (trainloader.batch_size if trainloader.batch_size is not None else B)
            d_in = layer_input.shape[1]
            if hessian_inv is None:
                hessian_inv = torch.eye(d_in + 1, device=dev, dtype=layer_input.dtype) * init_diag
            for i in range(B):
                xi = layer_input[i]
                wb = torch.cat([xi, torch.tensor([1.0], device=dev, dtype=xi.dtype)])
                if wb.dim() == 2 and wb.shape[0] == 1:
                    wb = wb.squeeze(0)
                h_inv_v = hessian_inv.matmul(wb)
                denom = dataset_size + (wb @ h_inv_v)
                numerator = torch.ger(h_inv_v, h_inv_v)
                hessian_inv = hessian_inv - numerator / denom
        elif layer_type == 'C':
            if k is None:
                w = getattr(module, 'weight', None)
                if (w is not None) and (w.dim() >= 3):
                    k = w.shape[2]
                else:
                    raise ValueError("Could not infer kernel size for conv layer; specify module with .kernel_size")
            add_bias = (layer_type == 'C')
            unfold = nn.Unfold(kernel_size=k, stride=module.stride[0] * stride_factor, padding=(k // 2))
            patches = unfold(layer_input)
            patches = patches.permute(0, 2, 1).contiguous()
            B, L, D = patches.shape
            dataset_size = n_batch_used * B * L
            if hessian_inv is None:
                dim = D + (1 if add_bias else 0)
                hessian_inv = torch.eye(dim, device=dev, dtype=patches.dtype) * init_diag
            for b in range(B):
                for l in range(L):
                    patch = patches[b, l]
                    if add_bias:
                        wb = torch.cat([patch, torch.tensor([1.0], device=dev, dtype=patch.dtype)])
                    else:
                        wb = patch
                    if wb.dim() == 2 and wb.shape[0] == 1:
                        wb = wb.squeeze(0)
                    h_inv_v = hessian_inv.matmul(wb)
                    denom = dataset_size + (wb @ h_inv_v)
                    numerator = torch.ger(h_inv_v, h_inv_v)
                    hessian_inv = hessian_inv - numerator / denom
        else:
            raise ValueError("layer_type must be 'F' or 'C'")

        used_batches += 1

    handle.remove()
    return hessian_inv


def unfold_kernel(kernel):
	"""
	In pytorch format, kernel is stored as [out_channel, in_channel, height, width]
	Unfold kernel into a 2-dimension weights: [height * width * in_channel, out_channel]
	:param kernel: numpy ndarray
	:return:
	"""
	k_shape = kernel.shape
	weight = np.zeros([k_shape[1] * k_shape[2] * k_shape[3], k_shape[0]])
	for i in range(k_shape[0]):
		weight[:, i] = np.reshape(kernel[i, :, :, :], [-1])

	return weight


def fold_weights(weights, kernel_shape):
	"""
	In pytorch format, kernel is stored as [out_channel, in_channel, width, height]
	Fold weights into a 4-dimensional tensor as [out_channel, in_channel, width, height]
	:param weights:
	:param kernel_shape:
	:return:
	"""
	kernel = np.zeros(shape=kernel_shape)
	for i in range(kernel_shape[0]):
		kernel[i,:,:,:] = weights[:, i].reshape([kernel_shape[1], kernel_shape[2], kernel_shape[3]])

	return kernel