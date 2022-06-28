import numpy as np
import argparse
import os
import glob
import sys
import cv2
import concurrent.futures
import csv

def detect_features(image_path):
    im = cv2.imread(image_path)
    boxes, _ = MtcnnDetector().detect_face(im)
    if len(boxes) > 0:
        boxes = boxes.astype(int)
        box = boxes[0]
    else:
        box = []
    
    if len(box) > 0:
        rect = [int(box[0]), int(box[1]), 
                int(box[2]-box[0]), int(box[3]-box[1])]

    
        pts = LandmarkDetector.detect(im, rect, [], 1)
        im_cp = im.copy()
        
        draw_landmarks(im_cp, pts)
        
        draw_rect(im_cp, box)
        cv2.imwrite("/home/ubuntu/lnd_imgs/" + image_path.split('/')[-1], im_cp)
        box = np.asarray(box)
        pts = np.asarray(pts)

    else:

        pts = []

    with open(dest_path + '/features_new_box.csv', 'a') as csv_file:
         writer = csv.writer(csv_file)
         writer.writerow([image_path, box, pts])
    
    
    return box, pts

def draw_landmarks(im, landmarks):
    for i in range(len(landmarks)/2):
        cv2.circle(im,(int(round(landmarks[i*2])),int(round(landmarks[i*2+1]))),1,(0,255,0),2)

def draw_rect(img, bbox):
    cv2.rectangle(img, (int(bbox[0]), int(bbox[1])), (int(bbox[2]),int(bbox[3])) ,(0,0,255), 2)


detector_dir = './face-datasets/'
sys.path.insert(0, detector_dir+'facealign')
sys.path.insert(0, detector_dir+'util')

import PyLandmark as LandmarkDetector
from MtcnnPycaffe import MtcnnDetector

LandmarkDetector.create('./detection/model/')

parser = argparse.ArgumentParser(description='find landmarks and bounding boxes')
parser.add_argument('source', metavar='source', type=str, help='Folder containing images to process')
parser.add_argument('destination', metavar='destination', type=str, help='Output folder for bounding boxes and landmark files')

args = parser.parse_args()
source_path = args.source
dest_path = args.destination


if os.path.exists(source_path):
    source_path = os.path.abspath(source_path)

    with concurrent.futures.ProcessPoolExecutor(max_workers=8) as executor:
            file_list = glob.glob(source_path + '/*.png')
            future_proc = {executor.submit(detect_features, f): f for f in file_list}
