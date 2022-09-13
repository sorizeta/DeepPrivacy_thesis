import cv2
import argparse
import numpy as np
import colour

s_name = ''
t_name = ''

s_image = cv2.imread(s_name).astype(np.float32) / 255
t_image = cv2.imread(t_name).astype(np.float32) / 255

lab_s_image = cv2.cvtColor(s_image, cv2.COLOR_BGR2LAB)
lab_t_image = cv2.cvtColor(t_image, cv2.COLOR_BGR2LAB)

L_s, A_s, B_s = cv2.split(lab_s_image)
L_t, A_t, B_t = cv2.split(lab_t_image)

delta_E = colour.deltaE(lab_s_image, lab_t_image)
delta_L = np.mean(np.add(-L_t, L_s))
delta_C = np.mean(
    np.add(
        -np.sqrt(np.power(A_t, 2) + np.power(B_t, 2)),
        np.sqrt(np.power(A_s, 2) + np.power(B_s, 2))
    )
)

print('Delta L: {}, Delta C: {}, Delta E: {}'.format(delta_L, delta_C, delta_E))