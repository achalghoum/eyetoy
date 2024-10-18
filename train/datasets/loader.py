import os
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import datasets, transforms


class CalTech256Split(Dataset):
    def __init__(self, split_file, root_dir, transform=None):
        self.root_dir = root_dir
        self.data = []
        self.transform = transform
        self.class_to_idx = {}
        
        import random

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
        full_img_path = os.path.join(self.root_dir, img_path)
        image = Image.open(full_img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        return image, class_id
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

current_dir = os.path.dirname(os.path.abspath(__file__))
train_lst_path = os.path.join(current_dir, "256_ObjectCategories", "train_lst.txt")
val_lst_path = os.path.join(current_dir, "256_ObjectCategories", "val_lst.txt")

caltech_256_train = CalTech256Split(train_lst_path, root_dir=current_dir, transform=transform)
caltech_256_val = CalTech256Split(val_lst_path, root_dir=current_dir)