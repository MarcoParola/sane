import torch
from pathlib import Path
import hydra
from src.models.utils import load_model
import os
from src.models.pruners.l_obs.prune_small_cnn import prune_small_cnn
from src.models.pruners.l_obs.prune_large_cnn import prune_large_cnn
from src.models.pruners.woodfisher.run import run_woodfisher

def prune_model(model, pruner_type, sparsity_level, store_location, dataset_name, data_dir, model_name, img_size, device):
    if pruner_type == "magnitude":
        import torch.nn.utils.prune as prune
        for name, module in model.named_modules():
            if isinstance(module, torch.nn.Conv2d) or isinstance(module, torch.nn.Linear):
                prune.l1_unstructured(module, name='weight', amount=sparsity_level)
                prune.remove(module, 'weight')
        torch.save(model.state_dict(), store_location / "checkpoints")
    elif pruner_type == "l_obs":
        if model_name == "small_cnn":
            prune_small_cnn(store_location, model, dataset_name, data_dir, model_name, img_size)
        elif model_name == "large_cnn":
            prune_large_cnn(store_location, model, dataset_name, data_dir, model_name, img_size)
    elif pruner_type == "woodfisher":
        run_woodfisher(model, dataset_name, data_dir, model_name, img_size, sparsity_level, device)
        torch.save(model.state_dict(), store_location / "checkpoints")

@hydra.main(version_base=None, config_path="../config", config_name="config")
def main(cfg):
    model_name = cfg.model.name
    dataset_name = cfg.dataset.name
    device = cfg.training.device
    data_dir = cfg.data_dir
    img_size = cfg[dataset_name].img_size

    zoo_name = cfg.model.name + "_" + cfg.dataset.name
    zoo_path = Path(cfg[zoo_name].zoo_path)

    model = load_model(model_name, dataset_name)
    model.to(device)

    pruner = cfg.sparsification.pruner

    sparsified_zoo_path = Path("checkpoints/sparsified_zoos/" + zoo_name + f"_{pruner}_sparsity_{cfg.sparsification.sparsity_level}") 
    os.makedirs(sparsified_zoo_path, exist_ok=True)

    if pruner == "magnitude":
        print(f"Sparsifying {zoo_name} models with {pruner} pruner at sparsity level {cfg.sparsification.sparsity_level}")
    elif pruner == "l_obs":
        print(f"Sparsifying {zoo_name} models with {pruner} pruner at sparsity levels 0.3, 0.5 and 0.8 ")
        
    counter = 0
    for folder in zoo_path.iterdir():
        if folder.is_dir():
            current_checkpoint_path = folder / "checkpoint_000050/checkpoints"
            if current_checkpoint_path.exists():
                checkpoint = torch.load(current_checkpoint_path, weights_only=False)
                model.load_state_dict(checkpoint)
                store_location = Path(sparsified_zoo_path / folder.name / "checkpoint_000050")
                os.makedirs(store_location, exist_ok=True)
                prune_model(model, pruner, cfg.sparsification.sparsity_level, store_location, dataset_name, data_dir, model_name, img_size, device)
                counter = counter+1
                print(f"\rSparsified {counter} model(s)", end='', flush=True)
    print()    


if __name__ == "__main__":
    main()