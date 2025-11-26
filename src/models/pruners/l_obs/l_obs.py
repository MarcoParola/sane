import torch 
from src.datasets.utils import load_dataset
from src.models.pruners.l_obs.utils import *
import numpy as np
from numpy.linalg import inv, pinv
from pathlib import Path

def prune_weights(store_location, checkpoint, dataset_name, data_dir, model_name, img_size):
    current_zoo_model_path = Path(*store_location.parts[3:])
    current_path = f"{model_name}_{dataset_name}" / current_zoo_model_path

    # GENERATE HESSIAN INV
    hessian_inverse_root = Path("src/models/pruners/l_obs/hessians") / current_path / "hessian_inv"
    hessian_inverse_root.mkdir(parents=True, exist_ok=True)
    
    layer_name_list = [
        'module_list.0',
        'module_list.3',
        'module_list.6',
        'module_list.9',
        'module_list.11'
    ]
    
    train, *_ = load_dataset(dataset_name, data_dir, model_name, img_size)
    hessian_loader = torch.utils.data.DataLoader(train, batch_size = 2, shuffle=True)

    for layer_name in layer_name_list:
        # skip the hessian computation if already saved
        hessian_file = hessian_inverse_root / f"{layer_name}.npy"
        if hessian_file.exists():
            print(f"Hessian inverse for {layer_name} already exists. Skipping computation.")
            continue

        if layer_name in ['module_list.0', 'module_list.3', 'module_list.6']:
            hessian = generate_hessian(checkpoint, hessian_loader, layer_name, 'C')
            # Inverse Hessian
            try:
                hessian_inv = inv(hessian.cpu().numpy())
            except Exception as err:
                print(err)
                hessian_inv = pinv(hessian.cpu().numpy())
        elif layer_name in ['module_list.9', 'module_list.11']:
            hessian_inv = generate_hessian_inv_woodbury(checkpoint, hessian_loader, layer_name, 'F')
        
        # safe conversion for both Tensor and NumPy array
        if isinstance(hessian_inv, torch.Tensor):
            arr = hessian_inv.detach().cpu().numpy()
        else:
            arr = hessian_inv
        np.save(hessian_inverse_root / f"{layer_name}.npy", arr)


    # PRUNE WEIGHTS
    state_dict = checkpoint.state_dict()
    target_sparsities = [0.3,0.5,0.8]

    for sparsity in target_sparsities:
        cr_save_dir = Path(f"checkpoints/sparsified_zoos/{model_name}_{dataset_name}_l_obs_sparsity_{sparsity}") / current_zoo_model_path
        cr_save_dir.mkdir(parents=True, exist_ok=True)
        pruned_checkpoint_file = cr_save_dir / "checkpoints"

        # Skip pruning if checkpoint already exists
        if pruned_checkpoint_file.exists():
            print(f"Pruned checkpoint for sparsity {sparsity} already exists. Skipping...")
            continue

        pruned_state_dict = {}

        for layer_name in layer_name_list:
            if layer_name in ['module_list.0', 'module_list.3', 'module_list.6']:
                layer_type = 'C'
                kernel = state_dict[f'{layer_name}.weight'].cpu().numpy()
                kernel_shape = kernel.shape
                weight = unfold_kernel(kernel)
                bias   = state_dict[f'{layer_name}.bias'].cpu().numpy()
                wb = np.concatenate([weight, bias.reshape(1, -1)], axis = 0)
            elif layer_name in ['module_list.9', 'module_list.11']:
                layer_type = 'F'
                weight = state_dict[f'{layer_name}.weight'].cpu().numpy()
                bias   = state_dict[f'{layer_name}.bias'].cpu().numpy()
                wb = np.hstack([weight, bias.reshape(-1, 1)]).transpose()
        
            # Load Hessian inverse
            hessian_inv = np.load(f'{hessian_inverse_root}/{layer_name}.npy')

            # Sensitivity ranking
            diag = np.diag(hessian_inv).reshape(-1, 1)
            L = (wb**2) / (diag + 1e-5)
            sen_rank = np.argsort(L.ravel())

            # determine number of weights to prune
            l1, l2 = wb.shape
            n_prune = int(l1*l2*sparsity)            
            mask = np.ones(wb.shape)
                    
            # prune weights
            for prune_idx in sen_rank[:n_prune]:
                prune_row_idx = prune_idx // l2
                prune_col_idx = prune_idx % l2
                delta_W = - wb[prune_row_idx, prune_col_idx] / (hessian_inv[prune_row_idx, prune_row_idx] + 1e-5) * hessian_inv[:, prune_row_idx]
                wb[:, prune_col_idx] += delta_W
                mask[prune_row_idx, prune_col_idx] = 0

            # save pruned weights into state_dict
            wb_masked = np.multiply(wb, mask)

            if layer_type == 'F':
                w = wb_masked[0:-1, :].transpose()
                b = wb_masked[-1, :].transpose()
                pruned_state_dict[f"{layer_name}.weight"] = torch.from_numpy(w)
                pruned_state_dict[f"{layer_name}.bias"] = torch.from_numpy(b)
            elif layer_type == 'C':
                kernel_pruned = fold_weights(wb_masked[0:-1, :], kernel_shape)
                bias_pruned = wb_masked[-1, :]
                pruned_state_dict[f"{layer_name}.weight"] = torch.from_numpy(kernel_pruned)
                pruned_state_dict[f"{layer_name}.bias"] = torch.from_numpy(bias_pruned)

        torch.save(pruned_state_dict, f"{cr_save_dir}/checkpoints")