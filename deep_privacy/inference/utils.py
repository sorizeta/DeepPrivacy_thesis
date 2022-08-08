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