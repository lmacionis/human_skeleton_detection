# Import Packages
import pandas as pd
import matplotlib.pyplot  as plt
from pathlib import Path
import imagesize
from check_data import path_train, path_test, img_name_list_train, img_name_list_test

img_name_list = img_name_list_train
img_dir = path_train
# img_name_list = img_name_list_test
# img_dir = path_test


# Get list of all image pathes
all_img_path_list = []

def img_path_list(path):
    for img_name in img_name_list:
        all_img_path_list.append(path + "/" + img_name)

    return all_img_path_list


# Check resoliution of images
img_resolution_dict = {}
for item in img_path_list(img_dir):
    img_resolution_dict[str(item)] = imagesize.get(item)

print(img_resolution_dict)


# Create data frame in order to check how all images are scattered by size.
img_resolution_dict_df = pd.DataFrame.from_dict([img_resolution_dict]).T.reset_index().set_axis(['FileName', 'Size'], axis='columns')
img_resolution_dict_df[["Width", "Height"]] = pd.DataFrame(img_resolution_dict_df["Size"].tolist(), index=img_resolution_dict_df.index)

# Showing data in a Scatter Plot
fig = plt.figure(figsize=(4, 4))
ax = fig.add_subplot()
points = ax.scatter(img_resolution_dict_df.Width, img_resolution_dict_df.Height, color='blue', alpha=0.5, label=f'Number of Images: {len(img_resolution_dict_df)}')
ax.set_title("Image Resolution")
ax.set_xlabel("Width", size=14)
ax.set_ylabel("Height", size=14)
ax.legend()
plt.show()
plt.close()