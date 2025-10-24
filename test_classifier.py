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
    train, val, test, remapping = load_dataset(dataset_name, data_dir, model_name, img_size)

    print(f"\nModel: {model_name} \nDataset: {dataset_name}")

    # Load model architecture
    model = load_model(model_name, dataset_name)
    model.to(device)
    
    # Load the state dictionary
    #checkpoint = torch.load(f"checkpoints/{model_name}_{dataset_name}_injected.pt", weights_only=False)
    #checkpoint = torch.load("checkpoints/tiny-imagenet_resnet18_kaiming_uniform_subset/NN_tune_trainable_dbca4_00050_50_seed=51_2022-08-23_21-28-42/checkpoint_000060/checkpoints", weights_only=False)
    #checkpoint = torch.load("checkpoints/injections/single_model_trained/resnet18_tinyimagenet/injected_100.pt", weights_only=False) 
    #checkpoint = torch.load("checkpoints/tune_zoo_cifar10_uniform_small/NN_tune_trainable_86fd7_00000_0_seed=1_2021-09-25_11-41-33/checkpoint_000050/checkpoints", weights_only=False)
    checkpoint = torch.load("checkpoints/tune_zoo_cifar10_uniform_small/NN_tune_trainable_86fd7_00986_986_seed=987_2021-09-27_00-21-18/checkpoint_000050/checkpoints", weights_only=False)

    model.load_state_dict(checkpoint)
    if dataset_name == "tinyimagenet" or model_name == "cnn":
        model.eval()
    
    test_classifier(model, test, num_classes, batch_size, device, remapping)


if __name__ == "__main__":
    main()