from torchvision import datasets, transforms
import torch


path = "./datasets/human_action_recognition"
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
mean > tensor([0.5726, 0.5376, 0.5073])
std > tensor([0.2456, 0.2437, 0.2470])
"""
