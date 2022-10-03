import glob
import cv2
import numpy as np
import argparse
import os


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
        L = cv2.subtract(gp_imgs[i-1], GE)
        laplacian_pyramid.append(L)

    return laplacian_pyramid


def expand_pyramid(end_pyramid, n_layers):
    ls_ = end_pyramid[0]
    for i in range(1, n_layers):
        size = (end_pyramid[i].shape[1], end_pyramid[i].shape[0])
        ls_ = cv2.pyrUp(ls_, dstsize=size)
        ls_ = cv2.add(ls_, end_pyramid[i])
        ls_[ls_ < 0] = 0
        ls_[ls_ > 255] = 255

    ls_ = ls_.astype(np.uint8)
    return ls_


def transfer_lighting(source_image, target_image):
    s_ycrcb = cv2.cvtColor(source_image, cv2.COLOR_RGB2YCrCb)
    t_ycrcb = cv2.cvtColor(target_image, cv2.COLOR_RGB2YCrCb)
    
    s_y, _, _ = cv2.split(s_ycrcb)
    t_y, t_cr, t_cb = cv2.split(t_ycrcb)
    
    source_pyramid = build_laplacian_pyramid(s_y, 4)
    target_pyramid = build_laplacian_pyramid(t_y, 4)
    
    target_pyramid[0] = source_pyramid[0]
    
    final_y = expand_pyramid(target_pyramid, 4)
    
    im = cv2.merge((final_y, t_cr, t_cb))
    im = cv2.cvtColor(im, cv2.COLOR_YCrCb2RGB)
    
    return im



def apply_color_transfer(target_img, ref_img):
    target_img = cv2.cvtColor(target_img, cv2.COLOR_RGB2LAB)
    ref_img = cv2.cvtColor(ref_img, cv2.COLOR_RGB2LAB)

    source_mean, source_std = cv2.meanStdDev(target_img)
    ref_mean, ref_std = cv2.meanStdDev(ref_img)

    _, _, channel = target_img.shape

    for k in range(0, channel):
        ch = target_img[:, :, k]
        ch = np.add(
            np.multiply(
                np.add(
                    ch,
                    -source_mean[:, :, k]
                ),
                np.divide(
                    ref_std[:, :, k],
                    source_std[:, :, k]
                )
            ),
            ref_mean[:, :, k])
        x = np.round(ch)
        x = np.clip(ch, 0, 255)
        target_img[:, :, k] = x

    colored_image = cv2.cvtColor(target_img, cv2.COLOR_LAB2RGB)
    return colored_image


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Apply color transfer and light direction transfer")
    parser.add_argument('src_folder', type=str, help="Original image folder")
    parser.add_argument('dest_folder', type=str,
                        help="Anonymised image folder")
    args = parser.parse_args()

    src_folder = os.path.abspath(args.src_folder)
    dest_folder = os.path.abspath(args.dest_folder)

    allowed_file_ext = ['.jpg', '.png', '.jpeg']
    source_files = []
    
    for type in allowed_file_ext:
        source_files.append = glob.glob(src_folder + '/**/*' + type)
    
    for image in source_files:
        filename = image.split('/')[-1]
        anonymised_image = dest_folder + filename
        if os.path.exists(anonymised_image):
            
            source_image = cv2.imread(image)
            anon_image = cv2.imread(anonymised_image)
            
            anon_image_light = transfer_lighting(source_image, anon_image)
            final_image = apply_color_transfer(anon_image_light, source_image)
            
            cv2.imwrite(dest_folder + filename.split('.')[0] + '_corrected' + filename.split('.')[1], final_image)
            
        