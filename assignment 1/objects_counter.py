import cv2
# compressed zip
# python file for counting and another for originalimage
import numpy as np
import matplotlib.pyplot as plt
from skimage.feature import blob_log

im = cv2.imread("OriginalImage.tif", cv2.IMREAD_GRAYSCALE)
# binarize the image
im_bw, th = cv2.threshold(im, 100, 255, cv2.THRESH_BINARY)


r_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (20, 20))
rice = cv2.morphologyEx(th, cv2.MORPH_OPEN, kernel=r_kernel)
# cv2.imshow("rice", rice)


rg_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
grain_rice = cv2.morphologyEx(th, cv2.MORPH_OPEN, kernel=rg_kernel)

grain = grain_rice - rice
f_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10, 10))
grain = cv2.morphologyEx(grain, cv2.MORPH_OPEN, kernel=f_kernel)
# cv2.imshow("grain", grain)


spaghetti = th - grain_rice

f_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10, 10))
spaghetti = cv2.morphologyEx(spaghetti, cv2.MORPH_OPEN, kernel=f_kernel)
# cv2.imshow("spaghetti", spaghetti)

rice_blobs = blob_log(rice, max_sigma=100, min_sigma=15,
                      num_sigma=3, threshold=0.3, overlap=0.4)
grain_blobs = blob_log(grain, max_sigma=100, min_sigma=10,
                       num_sigma=3, threshold=0.3, overlap=0.4)

contours, hierarchy = cv2.findContours(
    spaghetti, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)


rice_gray = cv2.bitwise_and(im, im, mask=rice)
grain_gray = cv2.bitwise_and(im, im, mask=grain)
spaghetti_gray = cv2.bitwise_and(im, im, mask=spaghetti)

cv2.imshow('rice_gray', rice_gray)
cv2.imshow('grain_gray', grain_gray)
cv2.imshow('spaghetti_gray', spaghetti_gray)

print(f'{len(rice_blobs)} rice')
print(f'{len(grain_blobs)} grains')
print(f'{len(contours)} spaghetti')


cv2.waitKey(0)
cv2.destroyAllWindows()


#your job now is to find the exact size of the kernels, so they have to be relative to the input image size (not hard coded)
