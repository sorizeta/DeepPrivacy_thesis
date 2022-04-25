import pandas as pd
import numpy as np
import argparse
import os
import glob
import sys

def detect_landmarks(landmark_detector, image, bounding_box):
    rect = [int(bounding_box[0]), int(bounding_box[1]), 
            int(bounding_box[2]-bounding_box[0]), int(bounding_box[3]-bounding_box[1])]
    pts = landmark_detector.detect(image, rect, [], 1)
    return pts

def detect_bounding_box(face_detector, image):
    boxes, _ = face_detector.detect_face(image)
    im_boxes = boxes.astype(int)
    return im_boxes

detector_dir = './deep_privacy/face-datasets/'
sys.path.insert(0, detector_dir+'facealign')
sys.path.insert(0, detector_dir+'util')

from MtcnnPycaffe import MtcnnDetector
import PyLandmark as LandmarkDetector

parser = argparse.ArgumentParser(description='find landmarks and bounding boxes')
parser.add_argument('source', metavar='source', type=str, help='Folder containing images to process')
parser.add_argument('destination', metavar='destination', type='str', help='Output folder for bounding boxes and landmark files')

args = parser.parse_args()
source_path = args.source
dest_path = args.destination

face_detector = MtcnnDetector()
landmark_detector = LandmarkDetector.create("./model/")

if os.path.exists(source_path):
    source_path = os.path.abspath(source_path)
    landmarks = []
    bounding_boxes = []
    for f in glob.glob(source_path + '/*.png'):
        bbox = detect_bounding_box(face_detector, f)
        im_landmarks = detect_landmarks(landmark_detector, f, bbox)
        bounding_boxes.append(bbox)
        landmarks.append(im_landmarks)
    with open(dest_path + '/bounding_boxes.npy', 'wb') as f:
        np.save(f, np.asarray(bounding_boxes))
    with open(dest_path + '/landmarks.npy', 'wb') as f:
        np.save(f, np.asarray(landmarks))
    


