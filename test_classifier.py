from src.datasets.utils import load_dataset
from src.models.utils import load_model
import torch
from src.utils.metrics import ClassificationMetrics
import hydra


def test_classifier(model, testset, num_classes, batch_size, device, remapping=None):
    metrics = ClassificationMetrics(num_classes=num_classes, device=device)

    test_loader = torch.utils.data.DataLoader(testset, batch_size, shuffle=False)
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)

        if remapping is not None:
            labels = torch.tensor(
                [remapping[int(label)] for label in labels],
                device=device
            )

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

    print(f"\nModel: {model_name} \nDataset: {dataset_name} \nNum Classes: {num_classes} \nImage Size: {img_size} \nBatch Size: {batch_size}")

    # Load model architecture
    model = load_model(model_name, dataset_name)
    model.to(device)
    
    # Load the state dictionary
    #checkpoint = torch.load(f"checkpoints/{model_name}_{dataset_name}.pt", weights_only=False)
    checkpoint = torch.load(f"checkpoints/tiny-imagenet_resnet18_kaiming_uniform_subset/NN_tune_trainable_dbca4_00115_115_seed=116_2022-08-25_12-19-06/checkpoint_000060/checkpoints", weights_only=False)
        
    model.load_state_dict(checkpoint)
    model.eval()
    
    if dataset_name=="tinyimagenet":
        # wnids.txt mapping
        with open("data/tiny-imagenet-200/wnids.txt") as f:
            trained_classes = [line.strip() for line in f]

        # Current mapping (list in alphabetical order)
        current_classes = test.classes

        # Build mapping: current_classes -> wnids.txt
        remapping = {current_classes.index(c): trained_classes.index(c) for c in current_classes}

        test_classifier(model, test, num_classes, batch_size, device, remapping)
    else:
        test_classifier(model, test, num_classes, batch_size, device)


if __name__ == "__main__":
    main()