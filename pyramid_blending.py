# pyramid blending

import cv2
import numpy as np

PYRAMID_LEVEL = 6

def generate_gaussian_pyramid(img, n_layers):
    gaussian = img.copy()
    gp_imgs = [gaussian]
    for i in range(n_layers):
        gaussian = cv2.pyrDown(gaussian)
        gp_imgs.append(gaussian)
    return gp_imgs


def generate_laplacian_pyramid(gaussian_pyramid, n_layers):
    laplacian_pyramid = [gaussian_pyramid[n_layers - 1]]
    for i in range(n_layers - 1, 0, -1):
        GE = cv2.pyrUp(gaussian_pyramid[i])
        L = cv2.subtract(gaussian_pyramid[i-1],GE)
        laplacian_pyramid.append(L)
    return laplacian_pyramid


def reconstruct_laplacian_pyramid(laplacian_modified, n_layers):
    ls_ = laplacian_modified[0]
    for i in range(1, n_layers):
        ls_ = cv2.pyrUp(ls_)
        ls_ = cv2.add(ls_, laplacian_modified[i])
    return ls_

light_path = "C:\\Users\\sofia\\imgHQ05074\\imgHQ05074_00.png"
start_path = "C:\\Users\\sofia\\imgHQ05074\\imgHQ05074_03.png"

start_im = cv2.imread(start_path)
light_im = cv2.imread(light_path)

start_im_norm = out = cv2.normalize(start_im.astype('float'), None, 0.0, 1.0, cv2.NORM_MINMAX)
light_im_norm = out = cv2.normalize(light_im.astype('float'), None, 0.0, 1.0, cv2.NORM_MINMAX)


gaussian_sp = generate_gaussian_pyramid(start_im_norm, PYRAMID_LEVEL)
gaussian_lp = generate_gaussian_pyramid(light_im_norm, PYRAMID_LEVEL)
laplacian_sp = generate_laplacian_pyramid(gaussian_sp, PYRAMID_LEVEL)
laplacian_lp =  generate_laplacian_pyramid(gaussian_lp, PYRAMID_LEVEL)

#laplacian_sp[1] = laplacian_lp[1]
laplacian_sp[0] = laplacian_lp[0]

sp_recon = reconstruct_laplacian_pyramid(laplacian_sp, PYRAMID_LEVEL)

im_to_show = np.hstack((start_im_norm, sp_recon))
# now reconstructing our image using pyrUp and stating pyramid levels

cv2.namedWindow("image", cv2.WINDOW_NORMAL)
cv2.imshow('image', im_to_show)
cv2.waitKey(0) 
cv2.destroyAllWindows() 
