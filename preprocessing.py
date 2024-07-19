#!/usr/bin/python3

import numpy as np

from PIL import Image
from skimage.measure import label
from sklearn.cluster import KMeans
from matplotlib import pyplot as plt
from skimage.morphology import binary_dilation, disk

from utils import hpp
from color import RGB2gray

def show_img(img):
    plt.axis('off')
    plt.imshow(img)
    plt.show()

    return

def preprocess(img_path):
    img = Image.open(img_path)
    img_arr = np.asarray(img)

    # Step-1
    gray = RGB2gray(img_arr)
    gray_reshaped = gray.reshape(gray.size, 1)
    show_img(gray)

    # Step-2
    cluster = KMeans(n_clusters=2, random_state=88)
    gray_predict = cluster.fit_predict(gray_reshaped)
    gray_predict = gray_predict.reshape(gray.shape)
    show_img(gray_predict)

    # Step-3
    micro_letter_free = hpp(gray_predict)
    plt.imshow(micro_letter_free)
    plt.show()

    # Step-4
    remove_noise = (label(micro_letter_free, background=0) > 25)
    show_img(remove_noise)
    # Step-5
    # reconstructed = binary_dilation(remove_noise, disk(1))
    # remove_noise = reconstructed

    # Step-6
    mask_3d = np.array([remove_noise, remove_noise, remove_noise]).transpose(1, 2, 0)
    preprocessed_img = np.where(mask_3d, img_arr, 255)
    show_img(preprocessed_img)

    return preprocessed_img

if __name__ == '__main__':
    name = 'Cheque 120616.tif'
    preprocess(name)
