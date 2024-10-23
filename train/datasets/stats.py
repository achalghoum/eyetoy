from torch.utils.data import DataLoader
import torch

from loader import caltech_256_train

# Assuming `dataset` is your Caltech-256 dataset
loader = DataLoader(caltech_256_train, batch_size=32, shuffle=False, num_workers=2)

# Initialize tensors to store the sums
mean = torch.zeros(3)
std = torch.zeros(3)
nb_samples = 0

# Loop through the dataset to accumulate the mean and standard deviation
for images, _ in loader:
    batch_samples = images.size(0)  # current batch size
    images = images.view(batch_samples, images.size(1), -1)  # flatten the image pixels
    mean += images.mean(2).sum(0)
    std += images.std(2).sum(0)
    nb_samples += batch_samples

mean /= nb_samples
std /= nb_samples

print(f"Mean: {mean}")
print(f"Std: {std}")
