from torchvision import transforms
from check_data import path, img_name_list_test, path_train_csv, path_train, path_test, path_test_csv
from torch.utils.data import DataLoader, random_split, Dataset
import pandas as pd
import os, torch, cv2
from torchvision.io import read_image
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
import numpy as np


class HumanPoseDataset(Dataset):
    def __init__(self, img_dir, annotations_file, transform=None, target_transform=None):
        self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        image = read_image(img_path)
        label = self.img_labels.iloc[idx, 1]
        
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        
        return image, label


transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((300, 300)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5726, 0.5376, 0.5073], std=[0.2456, 0.2437, 0.2470])
])


val_size = int(0.6 * len(img_name_list_test))
test_size = len(img_name_list_test) - val_size
print(val_size + test_size, len(img_name_list_test))

train_dataset = HumanPoseDataset(path_train, path_train_csv, transform=transform)
test_val_dataset = HumanPoseDataset(path_test, path_test_csv, transform=transform)
val_dataset, test_dataset = random_split(test_val_dataset, [val_size, test_size])


train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=64, shuffle=True)

# Display image and label.
train_features, train_labels = next(iter(train_dataloader))
print(f"Feature batch shape: {train_features.size()}")
print(f"Labels batch shape: {train_labels}")
img = train_features[0].squeeze()
label = train_labels[0]
plt.text(7, 15, label, bbox={'facecolor': 'white', 'pad': 5})
img = np.mean(img.numpy(), axis=0)
plt.imshow(img, cmap="grey")
plt.show()
