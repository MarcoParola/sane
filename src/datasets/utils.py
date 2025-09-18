import torch
import torchvision
import numpy as np


class ImgDataset(torch.utils.data.Dataset):
    def __init__(self, orig_dataset, transform=None):
        self.orig_dataset = orig_dataset
        self.transform = transform

    def __len__(self):
        return self.orig_dataset.__len__()

    def __getitem__(self, idx):
        img, lbl = self.orig_dataset.__getitem__(idx)
        img = torch.clamp(img, 0, 1)  # clamp the image to be between 0 and 1
        # clip image to be three channels
        if img.shape[0] == 1:
            img = img.repeat(3, 1, 1)
        
        if self.transform:
            img = self.transform(img)

        return img, lbl


def load_dataset(dataset, data_dir, img_size=None, val_split=0.15, test_split=0.15):
    train, val, test = None, None, None

    torch.manual_seed(42)
    np.random.seed(42)

    # Create transform with optional resize
    transform_list = [torchvision.transforms.ToTensor()]
    if img_size is not None:
        transform_list.append(torchvision.transforms.Resize((img_size, img_size)))
    transform = torchvision.transforms.Compose(transform_list)

    # CIFAR-10
    if dataset == 'cifar10':
        train = torchvision.datasets.CIFAR10(data_dir, train=True, download=True, transform=transform)
        test = torchvision.datasets.CIFAR10(data_dir, train=False, download=True, transform=transform)

        split = int(len(train) * val_split)
        train, val = torch.utils.data.random_split(train, [len(train) - split, split])

    # CIFAR-100
    elif dataset == 'cifar100':
        train = torchvision.datasets.CIFAR100(data_dir, train=True, download=True, transform=transform)
        test = torchvision.datasets.CIFAR100(data_dir, train=False, download=True, transform=transform)

        split = int(len(train) * val_split)
        train, val = torch.utils.data.random_split(train, [len(train) - split, split])

    else:
        raise ValueError(f'Unknown dataset: {dataset}')

    train = ImgDataset(train)
    val = ImgDataset(val)
    test = ImgDataset(test)

    return train, val, test


if __name__ == "__main__":
    # Example usage
    dataset_name = 'cifar10'
    data_dir = './data'
    train, val, test = load_dataset(dataset_name, data_dir, img_size=32)
    print(f'{dataset_name} Train size: {len(train)}, Val size: {len(val)}, Test size: {len(test)}')

    dataset_name = 'cifar100'
    train, val, test = load_dataset(dataset_name, data_dir, img_size=32)
    print(f'{dataset_name} Train size: {len(train)}, Val size: {len(val)}, Test size: {len(test)}')