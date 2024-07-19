#!/usr/bin/python3

#Function to obtain grayscale image
def RGB2gray(img_arr):
    gray = 0.299 * img_arr[:, :, 0] + 0.587 * img_arr[:, :, 1] + 0.114 * img_arr[:, :, 2]

    return gray.astype('int64')
