import os
from PIL import Image
import matplotlib.pyplot as plt


path_train = "./not_from_zero_but_still_to_hero/datasets/Head.v2i.yolov5pytorch/train/images"
path_test = "./not_from_zero_but_still_to_hero/datasets/Head.v2i.yolov5pytorch/test/images"
path_valid = "./not_from_zero_but_still_to_hero/datasets/Head.v2i.yolov5pytorch/valid/images"
path = "./not_from_zero_but_still_to_hero/datasets/Head.v2i.yolov5pytorch"


def img_check(path):
    img_list = []
    for img_name in os.listdir(path):
        img_list.append(img_name)

    return img_list # len(img_list)

img_name_list_train = img_check(path_train)
img_name_list_test = img_check(path_test)
img_name_list_valid = img_check(path_valid)
# print("Images for training: ", len(img_name_list_train))
# print("Images for testing: ", len(img_name_list_test))
# print("Images for validation: ", len(img_name_list_valid))

"""
Images for training:  635
Images for testing:  90
Images for validation:  182
"""
