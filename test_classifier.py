from src.datasets.utils import load_dataset
from src.models.utils import load_model
import torch
from src.utils.metrics import ClassificationMetrics
import hydra


def test_classifier(model, testset, num_classes, batch_size, device):
    metrics = ClassificationMetrics(num_classes=num_classes, device=device)

    test_loader = torch.utils.data.DataLoader(testset, batch_size, shuffle=False)
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        metrics(outputs, labels)

    print(metrics)
    return metrics


@hydra.main(version_base=None, config_path="config", config_name="config")
def main(cfg):
    # Load model and dataset configurations
    model_name = cfg.model.name
    dataset_name = cfg.dataset.name
    num_classes = cfg[dataset_name].num_classes
    if (model_name == "vit" and dataset_name == "cifar10"):
        img_size = 224
    else:
        img_size = cfg[dataset_name].img_size
    batch_size = 32 if model_name == "vit" else cfg.training.batch_size
    device = cfg.training.device
    data_dir = cfg.data_dir

    # Load dataset
    train, val, test = load_dataset(dataset_name, data_dir, img_size)

    # Load model
    model = load_model(model_name, dataset_name)
    model.to(device)

    print(f"\nModel: {model_name} \nDataset: {dataset_name} \nNum Classes: {num_classes} \nImage Size: {img_size} \nBatch Size: {batch_size}")


if __name__ == "__main__":
    main()