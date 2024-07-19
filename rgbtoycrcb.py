import numpy as np
import cv2

img = cv2.imread('img4.jpg')

def RGB2YCrCb(image):
    ycrcbimg = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)

    return ycrcbimg

cv2.imshow('YCrCb Image', RGB2YCrCb(img))

cv2.waitKey(10000)
cv2.destroyAllWindows()

