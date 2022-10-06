import pandas as pd
import numpy as np

dlib_landmark_df = pd.read_csv('C:\\Users\\sofia\\Desktop\\tesi\\prova_mse_landmarks\\dlib\\features_dlib.csv', names = ["filename", "landmarks"])
dlib_landmark_str = dlib_landmark_df["landmarks"].values
dlib_landmarks = []
for s in dlib_landmark_str:
    s = s.replace('\n', '')
    s = s.replace('[', '')
    s = s.replace(']', '')
    d_lnd = np.fromstring(s, sep=' ', dtype=int)
    dlib_landmarks.append(d_lnd)
    
pyl_landmark_df = pd.read_csv("C:\\Users\\sofia\\Desktop\\tesi\\prova_mse_landmarks\\PyLandmark\\features_new_box.csv", names=["filename", "landmarks"])
pyl_landmark_str = dlib_landmark_df["landmarks"].values
pyl_landmarks = []
for s in pyl_landmark_str:
    s = s.replace('\n', '')
    s = s.replace('[', '')
    s = s.replace(']', '')
    p_lnd = np.fromstring(s, sep=' ', dtype=int)
    pyl_landmarks.append(p_lnd)
    