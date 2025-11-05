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
    # SVHN
    #checkpoint = torch.load("checkpoints/tune_zoo_svhn_uniform/NN_tune_trainable_97ebe_00000_0_seed=1_2021-07-26_17-33-32/checkpoint_000050/checkpoints", weights_only=False)
    # MNIST
    #checkpoint = torch.load("checkpoints/tune_zoo_mnist_uniform/NN_tune_trainable_c0371_00000_0_seed=1_2021-07-01_16-59-53/checkpoint_000050/checkpoints", weights_only=False)
    # STL10 small
    #checkpoint = torch.load("checkpoints/tune_zoo_stl_small_uniform/NN_tune_trainable_3e946_00092_92_seed=93_2021-09-27_07-56-25/checkpoint_000050/checkpoints", weights_only=False)
    # STL10 large
    # checkpoint = torch.load("checkpoints/tune_zoo_stl_uniform_large/NN_tune_trainable_01314_00000_0_seed=1_2021-09-26_21-02-07/checkpoint_000050/checkpoints", weights_only=False)
    # CIFAR10 large
    checkpoint = torch.load("checkpoints/tune_zoo_cifar10_uniform_large/NN_tune_trainable_da045_00000_0_seed=1_2021-09-25_11-43-53/checkpoint_000050/checkpoints", weights_only=False)

    model.load_state_dict(checkpoint)
    model.eval()
    
    test_classifier(model, test, num_classes, batch_size, device, remapping)


if __name__ == "__main__":
    main()