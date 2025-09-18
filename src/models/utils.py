import timm
import detectors
import torch
from huggingface_hub import hf_hub_download


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
    elif model_name == 'vit':
        if dataset_name == 'cifar10':
            checkpoint = hf_hub_download(repo_id="nateraw/vit-base-patch16-224-cifar10", filename="pytorch_model.bin", repo_type="model")
            model = timm.create_model("vit_base_patch16_224", num_classes=10, pretrained=False)
            # Load the state dict with strict=False to handle key mismatches
            state_dict = torch.load(checkpoint, map_location='cpu', weights_only=False)
            model.load_state_dict(state_dict, strict=False)
        elif dataset_name == 'cifar100':
            checkpoint = hf_hub_download(repo_id="Ahmed9275/Vit-Cifar100", filename="pytorch_model.bin", repo_type="model")
            model = timm.create_model("vit_base_patch16_224", num_classes=100, pretrained=False)
            # Load the state dict with strict=False to handle key mismatches
            state_dict = torch.load(checkpoint, map_location='cpu', weights_only=False)
            model.load_state_dict(state_dict, strict=False)

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