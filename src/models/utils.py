import timm
import detectors
import torch
from src.models.resnet.resnet import ResNet18
from src.models.cnn.cnn import CNN

def load_model(model_name: str, dataset_name: str):
    """
    Load a model from the Hugging Face model hub.
    
    Args:
        model_name (str): The name of the model to load.
        dataset_name (str): The name of the dataset to use.

    Returns:
        model: The loaded model.
    """
    model = None
    if model_name == 'resnet18':
        if dataset_name == 'cifar10':
            model = timm.create_model("resnet18_cifar10", pretrained=True)
        elif dataset_name == 'cifar100':
            model = timm.create_model("resnet18_cifar100", pretrained=True)
        elif dataset_name == 'tinyimagenet':
            model = ResNet18(out_dim=200)
    elif model_name == 'vit':
        if dataset_name == 'cifar10':
            model = timm.create_model("vit_base_patch16_224_in21k_ft_cifar10", pretrained=True)
        elif dataset_name == 'cifar100':
            model = timm.create_model("vit_base_patch16_224_in21k_ft_cifar100", pretrained = True)
    elif model_name == 'cnn':
        if dataset_name == 'cifar10':
            model = CNN()

    return model

if __name__ == "__main__":
    # Example usage
    model_name = 'resnet18'
    dataset_name = 'cifar10'
    model = load_model(model_name, dataset_name)

    img = torch.randn(1, 3, 224, 224)  # Example input
    preds = model(img)
    print(preds)

    dataset_name = 'cifar100'
    model = load_model(model_name, dataset_name)
    img = torch.randn(1, 3, 224, 224)  # Example input
    preds = model(img)
    print(preds)