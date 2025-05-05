from torch.utils.data import DataLoader, Dataset
import torch
import os
from check_data import *
from torchvision.io import read_image

class CustomDataset(Dataset):
    def __init__(self, img_dir, coord_dir, transform=None, target_transform=None):
        self.img_dir = img_dir
        self.coord_dir = coord_dir
        self.transform = transform
        self.target_transform = target_transform
        
        # we need sorted beacause, os.listdir reades directory according to 
        # os filesystem implementation, it is not the same as you see it in dir tree.
        self.img_coord = sorted(os.listdir(coord_dir))    # list of coordinate file names
        self.img_labels = sorted(os.listdir(img_dir))     # list of image names

    def __len__(self):
        return len(self.img_labels)     # 635

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels[idx])
        coord_path = os.path.join(self.coord_dir, self.img_coord[idx])
        image = read_image(img_path)
        # label = self.img_labels[idx]

        with open(coord_path) as f:
            coord = f.readline()
            f.close()
        
        if len(coord) == 0:
            coord = "0 " * 8

        # Converts string of coordinates into tensor of coordinates
        coord = coord.strip().split()
        coord = [float(x) for x in coord]
        coord = torch.tensor(coord, dtype=torch.float32)

        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            coord = self.target_transform(coord)
        
        return image, coord


# dataset = CustomDataset(path_test, path_test_coord)
# sample_image, sample_coord = dataset[0]
# print("Image tensor:", sample_image, type(sample_image))
# print("Coordinate:", sample_coord, type(sample_coord))
# nepamirsti sukti ir koordinaciu
# koordinate transformuoti i img, su 0 ir 1
