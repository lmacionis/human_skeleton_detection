import os
from PIL import Image
import matplotlib.pyplot as plt


path_train = "./datasets/human_action_recognition/train"
path_test = "./datasets/human_action_recognition/test"
path = "./datasets/human_action_recognition"

def img_check(path):
    img_list = []
    for img_name in os.listdir(path):
        img_list.append(img_name)

    return img_list # len(img_list)

img_name_list_train = img_check(path_train)
img_name_list_test = img_check(path_test)
# print("Images for training: ", len(img_name_list_train))
# print("Images for testing: ", len(img_name_list_test))
"""
Images for training:  12601
Images for testing:  5410
"""
