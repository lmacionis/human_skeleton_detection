import os
from PIL import Image
import matplotlib.pyplot as plt


path_train = "./from_zero_to_hero/datasets/human_action_recognition/train"
path_test = "./from_zero_to_hero/datasets/human_action_recognition/test"
path = "./from_zero_to_hero/datasets/human_action_recognition"
path_train_csv = "./from_zero_to_hero/datasets/human_action_recognition/Training_set.csv"
path_test_csv = "./from_zero_to_hero/datasets/human_action_recognition/Testing_set.csv"

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
Images for testing:  5400
"""
