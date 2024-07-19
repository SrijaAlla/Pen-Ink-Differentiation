#!/usr/bin/python3

import cv2
import numpy as np
from scipy.stats import skew, kurtosis

def hpp(bin_img):
    max_fg_pix = np.max(bin_img.sum(axis=1))
    mask = (bin_img.sum(axis=1) >= 2*max_fg_pix//3)

    bin_img[mask] = [0] * bin_img.shape[1]

    return bin_img

def distance_value(word1_path, word2_path):
    H = [np.array([]), np.array([])]

    for i, image_path in enumerate([word1_path, word2_path]):
        img = cv2.imread(image_path, 1)
        chs = cv2.split(img)

        for c in chs:
            hist, _ = np.histogram(c, bins=range(257), normed=True)
            H[i] = np.concatenate([H[i], hist])

    D = H[0] - H[1]
    D12 = np.sum(np.abs(D))

    return D12

def compute_features(word1, word2):
    feature_vector = []

    for img in [word1, word2]:
        chs = cv2.split(img)

        for c in chs:
            feature_vector.append(np.mean(c))
            feature_vector.append(np.var(c))
            feature_vector.append(skew(c.flatten()))
            feature_vector.append(kurtosis(c.flatten()))
            feature_vector.append(np.mean(np.abs(c - np.mean(c))))

    return feature_vector