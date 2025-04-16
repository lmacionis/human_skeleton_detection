from torchvision import datasets, transforms
import torch
from check_data import path


N_CHANNELS = 3

dataset = datasets.ImageFolder(path, transform=transforms.ToTensor())
full_loader = torch.utils.data.DataLoader(dataset, shuffle=True)

mean = torch.zeros(N_CHANNELS)
std = torch.zeros(N_CHANNELS)

print("==> Computing mean and std")
for inputs, _labels in full_loader:
    for i in range(N_CHANNELS):
        mean[i] += inputs[:, i, :, :].mean()
        std[i] += inputs[:, i, :, :].std()

mean.div_(len(dataset))
std.div_(len(dataset))
print(mean, std)

"""
mean > tensor([0.6981, 0.6837, 0.6703])
std > tensor([0.3245, 0.3306, 0.3320])
"""
