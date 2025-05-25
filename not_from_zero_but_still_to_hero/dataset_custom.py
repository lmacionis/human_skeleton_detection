from torch.utils.data import DataLoader, Dataset
import torch
import os
from check_data import *
from torchvision.io import read_image
from PIL import Image, ImageDraw
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F


class CustomDataset(Dataset):
    def __init__(self, img_dir, coord_dir, transform=None):
        self.img_dir = img_dir
        self.coord_dir = coord_dir
        self.transform = transform
        
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

        with open(coord_path) as f:
            coord = f.readline()
            f.close()

        # Converts string of coordinates into tensor of coordinates
        coord = coord.strip().split()
        coord = [float(x) for x in coord]
        if len(coord) != 8:
            coord = [0.0] * 8
        
        x, y = coord[1], coord[2]
        # Scaling coordinates in order to create clasification map since yolov5 from roboflow gave normalised coordinates
        x_pixel = int(x * 20)
        y_pixel = int(y * 20)
        # print((x_pixel, y_pixel))

        classification_map = np.zeros((20, 20), dtype=np.float32)     # Ground Truth Mask
        for i in range(x_pixel-5, x_pixel+5):
            for j in range(y_pixel-5, y_pixel+5):
                classification_map[j, i] = 1
        # classification_map[x_pixel, y_pixel] = 1
        classification_map = torch.tensor(classification_map, dtype=torch.float32)

        """Leaving as exmple for future"""
        # For more clear view, that we could see clasification not as a sinle point, cause it is to small on the image and cant be seen with naked eye.
        # Gaussian distribution, for beter data visualisation in statistics...
        # Also creates a distribution around that 1 pixel of smaller values like 0.7, 0.5 and  similar.
        # sigma = 2
        # y_grid, x_grid = torch.meshgrid(
        #     torch.arange(320, dtype=torch.float32),
        #     torch.arange(320, dtype=torch.float32),
        #     indexing='ij'
        # )
        # gaussian = torch.exp(-((x_grid - x_pixel)**2 + (y_grid - y_pixel)**2) / (2 * sigma**2))     # formula of gaussian distribution, or standart normal distribution
        # classification_map = torch.clamp(gaussian, 0, 1)

        if self.transform:
            image = self.transform(image)
        
        return image, classification_map
        

if __name__=="__main__":
    dataset = CustomDataset(path_test, path_test_coord)
    sample_image, classification_map = dataset[60]
    print(sample_image.shape)
    print(classification_map.shape)
    plt.figure()
    sample_image = sample_image.permute(1, 2, 0)    # Rearaging tensor
    plt.imshow(sample_image)
    plt.imshow(classification_map, alpha=0.5*(classification_map>0))
    plt.show()
