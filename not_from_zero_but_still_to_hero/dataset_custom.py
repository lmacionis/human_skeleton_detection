from torch.utils.data import DataLoader, Dataset
import os
from check_data import *
from torchvision.io import read_image

class CustomDataset(Dataset):
    def __init__(self, img_dir, transform=None, target_transform=None):
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform
        self.img_labels = os.listdir(img_dir)     # list of image names

    def __len__(self):
        return len(self.img_labels)     # 635

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels[idx])
        image = read_image(img_path)
        label = self.img_labels[idx]

        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        
        return image, label
