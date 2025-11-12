import hydra
import torch
from pathlib import Path
from src.datasets.utils import load_dataset
from src.models.utils import load_model
from src.utils.metrics import SparsificationMetrics 
from test_classifier import test_classifier


@hydra.main(version_base=None, config_path="config", config_name="config")
def main(cfg):
    model_name = cfg.model.name
    dataset_name = cfg.dataset.name
    num_classes = cfg[dataset_name].num_classes
    batch_size = cfg.training.batch_size
    device = cfg.training.device
    data_dir = cfg.data_dir

    train, val, test, remapping = load_dataset(dataset_name, data_dir, model_name)
    model = load_model(model_name, dataset_name)
    model.to(device)
    print(f"Loaded model {model_name} for dataset {dataset_name}.")

    zoo_name = model_name + "_" + dataset_name
    original_zoo_path = Path(cfg[zoo_name].zoo_path)
    sparsified_zoo_path = Path("checkpoints/sparsified_zoos/" + zoo_name + f"_{cfg.sparsification.pruner}_sparsity_{cfg.sparsification.sparsity_level}") 

    compression_rates = []
    accuracy_retentions = []
    start_index = 140
    end_index = start_index + 20
    original_zoo_dirs = sorted(p for p in original_zoo_path.iterdir() if p.is_dir())

    print(f"Evaluating sparsified models obtained by {cfg.sparsification.pruner} at sparsity level {cfg.sparsification.sparsity_level}.")

    for idx, model_dir in enumerate(original_zoo_dirs[start_index:end_index], start=start_index):

        print(f"Evaluating checkpoint {idx}...")

        original_checkpoint_path = model_dir / "checkpoint_000050" / "checkpoints"
        sparsified_checkpoint_path = sparsified_zoo_path /  f"checkpoint_{idx}.pt"

        original_checkpoint = torch.load(original_checkpoint_path, weights_only=False)
        sparsified_checkpoints = torch.load(sparsified_checkpoint_path, weights_only=False)


        print("Classification Metrics:")
        print("\tOriginal Model:")
        model.load_state_dict(original_checkpoint)
        model.eval()
        original_metrics = test_classifier(model, test, num_classes, batch_size, device, remapping)
        original_accuracy = original_metrics.accuracy()

        print("\tSparsified Model:")
        model.load_state_dict(sparsified_checkpoints)
        model.eval()
        sparsified_metrics = test_classifier(model, test, num_classes, batch_size, device, remapping)
        sparsified_accuracy = sparsified_metrics.accuracy()

        print("Sparsification Metrics:")
        metrics = SparsificationMetrics(original_checkpoint, sparsified_checkpoints, original_accuracy, sparsified_accuracy, device)
        print(metrics)

        compression_rates.append(metrics.compression_ratio())
        accuracy_retentions.append(metrics.accuracy_retention())

    avg_compression = sum(compression_rates) / len(compression_rates)
    avg_accuracy_retention = sum(accuracy_retentions) / len(accuracy_retentions)
    print(f"Average Compression Rate: {avg_compression:.2f}x")
    print(f"Average Accuracy Retention: {avg_accuracy_retention*100:.2f}%")


if __name__ == "__main__":
    main()