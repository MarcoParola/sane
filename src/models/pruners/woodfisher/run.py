from src.datasets.utils import load_dataset
import torch
import torch.nn as nn
import numpy as np
from src.models.pruners.woodfisher.woodfisher import WoodburryFisherPruner

def run_woodfisher(model, dataset_name, data_dir, model_name, img_size, sparsity_lvl, device):
    model.eval()

    if dataset_name == "cifar10":
        num_samples = 1000
    else: 
        num_samples = 400

    data_train, *_  = load_dataset(dataset_name, data_dir, model_name, img_size)

    subset_inds = np.random.choice(len(data_train), num_samples, replace=False)

    PRUNABLE_MODULES = (
        nn.Conv2d, nn.Linear
    )

    for name, module in model.named_modules():
        if isinstance(module, PRUNABLE_MODULES):
            module.weight_mask = torch.ones_like(module.weight, dtype=torch.float32, device=module.weight.device)
    
    pruner = WoodburryFisherPruner(model, num_samples) 
    pruner.prune(data_train, subset_inds, sparsity_lvl, device)