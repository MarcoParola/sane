
import torch
from torch.utils.data import DataLoader
import copy
from src.datasets.utils import load_dataset
from torch.utils.data import DataLoader
from src.models.pruners.chita.chita import CHITA
from src.models.pruners.chita.utils import get_pvec

def update_weights(model, params, w_flat):
    idx = 0
    state_dict = model.state_dict()
    
    for p in params:
        tensor = state_dict[p]
        numel  = tensor.numel()
        new_tensor = w_flat[idx:idx+numel].view_as(tensor)
        state_dict[p].copy_(new_tensor)
        idx += numel
    
    model.load_state_dict(state_dict)


def apply_mask(model, params, mask_flat):
    idx = 0
    state_dict = model.state_dict()

    for p in params:
        tensor = state_dict[p]
        numel = tensor.numel()
        mask_tensor = mask_flat[idx:idx+numel].view_as(tensor)
        state_dict[p].mul_(mask_tensor)
        idx += numel

    model.load_state_dict(state_dict)


def run_chita(model, sparsity_level, dataset_name, data_dir, model_name, img_size):    
    train_dataset, *_   = load_dataset(dataset_name, data_dir, model_name, img_size)
    prun_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=2, pin_memory=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model.to(device)
    model.eval()

    modules_to_prune = []
    for name, layer in model.named_modules():
        if isinstance(layer, torch.nn.Conv2d) or isinstance(layer, torch.nn.Linear):
            modules_to_prune.append(name+'.weight')


    model_pruned = copy.deepcopy(model)
    mask = torch.ones_like(get_pvec(model_pruned, modules_to_prune)).to(device).bool()

    pruner = CHITA(model_pruned, modules_to_prune, prun_dataloader, device)

    w_pruned, mask = pruner.prune(mask, sparsity_level)

    update_weights(model_pruned, modules_to_prune, torch.from_numpy(w_pruned).to(device))
    apply_mask(model_pruned, modules_to_prune, mask.to(device))

    return model_pruned