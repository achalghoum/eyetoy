import os

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import datasets, transforms


class CalTech256Split(Dataset):
    def __init__(self, split_file, root_dir, transform=None, download=False):
        self.dataset = datasets.Caltech256(root=root_dir, transform=transform, download=download)
        self.transform = transform
        self.data = []
        self.class_to_idx = {}

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

train_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.ConvertImageDtype(torch.uint8),
    transforms.AutoAugment(),
    transforms.ConvertImageDtype(torch.float32),
    transforms.Resize((128, 128)),
    transforms.Normalize([0.5538, 0.5341, 0.5063], [0.2364, 0.2356, 0.2381])
])
val_transform = transforms.Compose([

    transforms.ToTensor(),
    transforms.Resize((128, 128)),
    transforms.Normalize([0.5538, 0.5341, 0.5063], [0.2364, 0.2356, 0.2381])

])

current_dir = os.path.dirname(os.path.abspath(__file__))
caltech_train_lst_path = os.path.join(current_dir, "caltech256_train_lst.txt")
caltech_val_lst_path = os.path.join(current_dir, "caltech256_val_lst.txt")
cifar_dir = os.path.join(current_dir, "cifar_10")
caltech_256_train = CalTech256Split(caltech_train_lst_path, root_dir=current_dir,
                                    transform=train_transform, download=True)
caltech_256_val = CalTech256Split(caltech_val_lst_path, root_dir=current_dir,
                                  transform=val_transform)
