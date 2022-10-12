import pandas as pd
import numpy as np
import os
import csv
import itertools
import cv2

def evaluate_normalised_mean_error(lnd_gt, lnd_pred):
    # normalised by the distance between landmarks
    norm_dist = np.sum(np.square(lnd_gt[45] - lnd_gt[55]))
    img_error = 0
    for i in range(68):
        pair_error = np.sum(np.square(lnd_gt[i] - lnd_pred[i])) / norm_dist
        img_error += pair_error
    return norm_dist, img_error / 68

def print_landmark_image(filename, landmark_set, out_name):
    im = cv2.imread(filename)
    basename = os.path.basename(filename)
    lnd = landmark_set[basename]
    for i in range(68):
        cv2.circle(im, (lnd[i, 0], lnd[i, 1]), 3, (255, 0, 0), -1)
    cv2.imwrite(out_name, im)

def read_landmarks_from_pts(filename):
    pts_filename = os.path.basename(filename).split('.')[0] + '.pts'
    pts_path = os.path.join('C:\\', 'Users', 'sofia', 'Desktop', 'tesi', 'lfpw', pts_filename)
    data = []
    with open(pts_path) as f:
        for line in itertools.islice(f, 3, 71):
            dataline=line.rstrip()
            dataline=np.fromstring(dataline, sep=' ', dtype=float)
            data.append(dataline)
    data= np.asarray(data)
    assert data.shape == (68, 2)
    return data


def save_normalised_mean_error(landmarks_dict, filename):
    for key, value in landmarks_dict.items():
        gt_landmarks = read_landmarks_from_pts(key)
        normalisation, error = evaluate_normalised_mean_error(gt_landmarks, value)
        with open(filename, 'a') as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow([os.path.basename(key), normalisation, error])
            

def read_landmarks_from_csv(filename):
    landmark_df = pd.read_csv(filename, names = ["filename", "landmarks"])
    landmark_df = landmark_df.sort_values(by="filename")
    landmark_names = landmark_df["filename"].values
    landmark_str = landmark_df["landmarks"].values
    landmarks = dict()
    for idx, s in enumerate(landmark_str):
        s = s.replace('\n', '')
        s = s.replace('[', '')
        s = s.replace(']', '')
        lnd = np.fromstring(s, sep=' ', dtype=int)
        lnd = np.array_split(lnd, 68)
        lnd = np.asarray(lnd)
        if lnd.shape[1] > 0:
            assert lnd.shape == (68, 2)
            landmarks[os.path.basename(landmark_names[idx])] = lnd
    
    return landmarks
        
dlib_landmarks = read_landmarks_from_csv('C:\\Users\\sofia\\Desktop\\tesi\\prova_mse_landmarks\\dlib\\features_dlib.csv')
pyl_landmarks = read_landmarks_from_csv("C:\\Users\\sofia\\Desktop\\tesi\\prova_mse_landmarks\\PyLandmark\\features_pylandmark.csv")

print_landmark_image("C:\\Users\\sofia\\Desktop\\tesi\\lfpw\\image_0003.png", dlib_landmarks, "dlib_out.png")
print_landmark_image("C:\\Users\\sofia\\Desktop\\tesi\\lfpw\\image_0003.png", pyl_landmarks, "pyl_out.png")

save_normalised_mean_error(dlib_landmarks, "dlib_lnd.csv")
save_normalised_mean_error(pyl_landmarks, "pyl_lnd.csv")    


