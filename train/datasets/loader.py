from typing import Dict, Tuple
import os

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.datasets import Caltech256, CIFAR10, CIFAR100, Flowers102


class CalTech256Split(Dataset):
    def __init__(self, split_file, root_dir, transform=None, download=False):
        self.dataset = Caltech256(
            root=root_dir, transform=transform, download=download)
        self.transform = transform
        self.data = []
        self.class_to_idx = {}
        self.num_classes = 256

        with open(split_file, 'r') as f:
            for line in f:
                img_path, class_id, class_name = line.strip().split()
                self.data.append((img_path, int(class_id)))
                if class_name not in self.class_to_idx:
                    self.class_to_idx[class_name] = int(class_id)

        self.classes = list(self.class_to_idx.keys())

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path, class_id = self.data[idx]
        full_img_path = os.path.join(self.dataset.root, img_path)
        image = Image.open(full_img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image, class_id


class CIFAR10Split(Dataset):
    def __init__(self, root_dir, transform=None, download=False, train=True):
        self.dataset = CIFAR10(
            root=root_dir, transform=transform, download=download,train=train)
        self.transform = transform
        self.num_classes = 10

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image, class_id = self.dataset[idx]
        return image, class_id


class Flowers102CIFAR10Split(Dataset):
    def __init__(self, root_dir, transform=None, download=False, split="train"):
        self.dataset = Flowers102(
            root=root_dir, transform=transform, download=download,split=split)
        self.transform = transform
        self.num_classes = 10

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image, class_id = self.dataset[idx]
        return image, class_id


class CIFAR100Split(Dataset):  # New class for CIFAR-100
    def __init__(self, root_dir, transform=None, download=False, train=True):
        self.dataset = CIFAR100(
            root=root_dir, transform=transform, download=download,train=train)
        self.num_classes = 100

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image, class_id = self.dataset[idx]

        return image, class_id


train_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.ConvertImageDtype(torch.uint8),
    transforms.RandAugment(),
    transforms.ConvertImageDtype(torch.float32),
    transforms.Resize((224, 224)),
    transforms.Normalize([0.5538, 0.5341, 0.5063], [0.2364, 0.2356, 0.2381])
])
val_transform = transforms.Compose([

    transforms.ToTensor(),
    transforms.Resize((224, 224)),
    transforms.Normalize([.5538, 0.5341, 0.5063], [0.2364, 0.2356, 0.2381])

])
cifar_train_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.ConvertImageDtype(torch.uint8),
    transforms.RandAugment(),
    transforms.ConvertImageDtype(torch.float32),
    transforms.Resize((224, 224)),
    transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2470, 0.2435, 0.2616])
])
cifar_val_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((224, 224)),
    transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2470, 0.2435, 0.2616])

])


current_dir = os.path.dirname(os.path.abspath(__file__))
caltech_train_lst_path = os.path.join(current_dir, "caltech256_train_lst.txt")
caltech_val_lst_path = os.path.join(current_dir, "caltech256_val_lst.txt")
caltech_256_train = CalTech256Split(caltech_train_lst_path, root_dir=current_dir,
                                    transform=train_transform, download=True)
caltech_256_val = CalTech256Split(caltech_val_lst_path, root_dir=current_dir,
                                  transform=val_transform)
cifar10_train = CIFAR10Split(
    root_dir=current_dir, transform=cifar_train_transform, download=True, train=True)
cifar10_val = CIFAR10Split(root_dir=current_dir, transform=cifar_val_transform, train=False)
cifar100_train = CIFAR100Split(
    root_dir=current_dir, transform=cifar_train_transform, download=True)
cifar100_val = CIFAR100Split(root_dir=current_dir, transform=cifar_val_transform)

DATASETS: Dict[str, Tuple[Dataset, Dataset]] = {
    "caltech256": (caltech_256_train, caltech_256_val),
    "cifar10": (cifar10_train, cifar10_val),
    "cifar100": (cifar100_train, cifar100_val)
}
