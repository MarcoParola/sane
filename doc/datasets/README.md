# Datasets

## Supported datasets:
- **Cifar10**
- **Cifar100**
- **Tiny-ImageNet**

Cifar10 and Cifar100 are automatically downloaded and saved in `./data` 
through the Pytorch library `torchvision.datasets`, while Tiny-Imagenet needs to be manually downloaded from [Kaggle](https://www.kaggle.com/datasets/akash2sharma/tiny-imagenet) and decompressed in `./data`.

## Supported model zoos (and dataset weights):
- **Tiny-Imagenet Resnet18**

    Download link: https://zenodo.org/records/7023278/files/tiny-imagenet_resnet18_subset.zip?download=1

    The archive needs to be downloaded from the link above and decompressed in `./checkpoints`.