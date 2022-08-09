from pickletools import uint8
from random import gauss
import cv2
import numpy as np

def build_laplacian_pyramid(img, levels):
    gaussian = img.copy()
    gaussian = gaussian.astype(np.float32)
    gp_imgs = [gaussian]
    for i in range(levels):
        gaussian = cv2.pyrDown(gaussian)
        gp_imgs.append(gaussian)
        
    laplacian_pyramid = [gp_imgs[levels - 1]]
    for i in range(levels - 1, 0, -1):
        size = (gp_imgs[i-1].shape[1], gp_imgs[i-1].shape[0])
        GE = cv2.pyrUp(gp_imgs[i], dstsize=size)
        L = cv2.subtract(gp_imgs[i-1],GE)
        laplacian_pyramid.append(L)

    return laplacian_pyramid


def transfer_lighting(end_pyramid, n_layers):
    ls_ = end_pyramid[0]
    for i in range(1, n_layers):
        size = (end_pyramid[i].shape[1], end_pyramid[i].shape[0])
        ls_ = cv2.pyrUp(ls_, dstsize=size)
        ls_ = cv2.add(ls_, end_pyramid[i])
        ls_[ls_ < 0] = 0
        ls_[ls_ > 255] = 255
    
    ls_ = ls_.astype(np.uint8)
    return ls_


def get_mean_and_std(mat):
    mat_mean, mat_std = cv2.meanStdDev(mat)
    mat_mean = np.hstack(np.around(mat_mean,2))
    mat_std = np.hstack(np.around(mat_std,2))
    return mat_mean, mat_std


def apply_color_transfer(source_img, target_img):
    source_img = cv2.cvtColor(source_img, cv2.COLOR_RGB2LAB)
    target_img = cv2.cvtColor(target_img, cv2.COLOR_RGB2LAB)

    source_mean, source_std = get_mean_and_std(source_img)
    target_mean, target_std = get_mean_and_std(target_img)

    height, width, channel = source_img.shape
    for i in range(0,height):
        for j in range(0,width):
            for k in range(0,channel):
                x = source_img[i,j,k]
                x = ((x - source_mean[k]) * (target_std[k] / source_std[k])) + target_mean[k]
                # round or +0.5
                x = round(x)
                # boundary check
                x = 0 if x < 0 else x
                x = 255 if x > 255 else x
                source_img[i,j,k] = x
    
    colored_image = cv2.cvtColor(source_img, cv2.COLOR_LAB2RGB)
    return colored_image