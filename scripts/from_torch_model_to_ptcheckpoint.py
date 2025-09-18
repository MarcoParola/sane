import os
import torch
import hydra
from src.models.utils import load_model


@hydra.main(version_base=None, config_path="../config", config_name="config")
def main(cfg):
    # Load model and dataset configurations
    model_name = cfg.model.name
    dataset_name = cfg.dataset.name

    # Load model
    model = load_model(model_name, dataset_name)

    # Save model checkpoint
    checkpoint_dir = "../checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, f"{model_name}_{dataset_name}.pt")
    torch.save(model.state_dict(), checkpoint_path)
    print(f"Model checkpoint {model_name}_{dataset_name} successfully saved")


if __name__ == "__main__":
    main()