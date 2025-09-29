import timm
import detectors
import torch
import torch.nn as nn

from torchvision.models.resnet import ResNet, BasicBlock
class ResNet18(ResNet):
    def __init__(
        self,
        channels_in=3,
        out_dim=10,
        init_type="kaiming_uniform",
    ):
        # call init from parent class
        super().__init__(block=BasicBlock, layers=[2, 2, 2, 2], num_classes=out_dim)
        # adpat first layer to fit dimensions
        self.conv1 = nn.Conv2d(
            channels_in,
            64,
            kernel_size=(3, 3),
            stride=(1, 1),
            padding=(1, 1),
            bias=False,
        )
        self.maxpool = nn.Identity()

        if init_type is not None:
            self.initialize_weights(init_type)

    def initialize_weights(self, init_type):
        """
        applies initialization method on all layers in the network
        """
        for m in self.modules():
            m = self.init_single(init_type, m)

    def init_single(self, init_type, m):
        """
        applies initialization method on module object
        """
        if type(m) == nn.Linear or type(m) == nn.Conv2d:
            if init_type == "xavier_uniform":
                torch.nn.init.xavier_uniform_(m.weight)
            if init_type == "xavier_normal":
                torch.nn.init.xavier_normal_(m.weight)
            if init_type == "uniform":
                torch.nn.init.uniform_(m.weight)
            if init_type == "normal":
                torch.nn.init.normal_(m.weight)
            if init_type == "kaiming_normal":
                torch.nn.init.kaiming_normal_(m.weight)
            if init_type == "kaiming_uniform":
                torch.nn.init.kaiming_uniform_(m.weight)
            # set bias to some small non-zero value
            try:
                m.bias.data.fill_(0.01)
            except Exception as e:
                pass
        return m


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
            model = ResNet18(out_dim=200, dropout=0.0)
    elif model_name == 'vit':
        if dataset_name == 'cifar10':
            model = timm.create_model("vit_base_patch16_224_in21k_ft_cifar10", pretrained=True)
        elif dataset_name == 'cifar100':
            model = timm.create_model("vit_base_patch16_224_in21k_ft_cifar100", pretrained = True)

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