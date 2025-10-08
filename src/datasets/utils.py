import torch
import torchvision
import numpy as np
import detectors
from torchvision import transforms
from torchvision.datasets import ImageFolder


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

    # TINY IMAGENET
    elif dataset == 'tinyimagenet':
        train = detectors.create_dataset("tiny_imagenet", split="train")
        test = detectors.create_dataset("tiny_imagenet", split="val")

        train_transforms_list = [torchvision.transforms.ToTensor()]
        if img_size is not None:
            train_transforms_list.append(torchvision.transforms.Resize((img_size, img_size)))

        test_transforms_list = [torchvision.transforms.ToTensor()]
        if img_size is not None:
            test_transforms_list.append(torchvision.transforms.Resize((img_size, img_size)))

        train_transforms_list.extend([
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(20),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

        test_transforms_list.extend([
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

        train_transforms = torchvision.transforms.Compose(train_transforms_list)
        test_transforms = torchvision.transforms.Compose(test_transforms_list)

        train = ImageFolder("data/tiny-imagenet-200/train", transform=train_transforms)
        test = ImageFolder("data/tiny-imagenet-200/val", transform=test_transforms)
        split = int(len(train) * val_split)
        train, val = torch.utils.data.random_split(train, [len(train) - split, split])
        
        # Build mapping: current_classes -> wnids.txt
        with open("data/tiny-imagenet-200/wnids.txt") as f:
            trained_classes = [line.strip() for line in f]

        current_classes = test.classes
        remapping = {current_classes.index(c): trained_classes.index(c) for c in current_classes}

        return train, val, test, remapping

    else:
        raise ValueError(f'Unknown dataset: {dataset}')

    train = ImgDataset(train)
    val = ImgDataset(val)
    test = ImgDataset(test)

    return train, val, test, None


if __name__ == "__main__":
    # Example usage
    dataset_name = 'tinyimagenet'
    data_dir = './data'
    train, val, test = load_dataset(dataset_name, data_dir)
    print(f'{dataset_name} Train size: {len(train)}, Val size: {len(val)}, Test size: {len(test)}')

    img, lbl = train.__getitem__(0)
    print(f'Image shape: {img.shape}, Label: {lbl}', 'max:', img.max(), 'min:', img.min())

    from matplotlib import pyplot as plt
    for i in range(5):
        img, lbl = train[i]
        plt.imshow(img.permute(1, 2, 0).numpy())
        plt.title(f'Label: {lbl}')
        plt.show()

    '''
    dataset_name = 'cifar100'
    train, val, test = load_dataset(dataset_name, data_dir, img_size=32)
    print(f'{dataset_name} Train size: {len(train)}, Val size: {len(val)}, Test size: {len(test)}')
    '''