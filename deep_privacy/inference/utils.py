import cv2

def build_laplacian_pyramid(img, levels):
    gaussian = img.copy()
    gp_imgs = [gaussian]
    
    for i in range(levels):
        gaussian = cv2.pyrDown(gaussian)
        gp_imgs.append(gaussian)

    laplacian_pyramid = [gp_imgs[levels - 1]]
    
    for i in range(levels - 1, 0, -1):
        GE = cv2.pyrUp(gp_imgs[i])
        L = cv2.subtract(gp_imgs[i-1],GE)
        laplacian_pyramid.append(L)
    return laplacian_pyramid


def transfer_lighting(start_pyramid, end_pyramid, n_layers):
    end_pyramid[0] = start_pyramid[0]
    ls_ = end_pyramid[0]
    for i in range(1, n_layers):
        ls_ = cv2.pyrUp(ls_)
        ls_ = cv2.add(ls_, end_pyramid[i])
    return ls_