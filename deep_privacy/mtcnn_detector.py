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
        h, w, _ = im.shape
        box = [0, 0, h, w]
    
    # [x0, y0, x1, y1]
    rect = [int(box[0]), int(box[1]), 
            int(box[2]-box[0]), int(box[3]-box[1])]
    
    pts = LandmarkDetector.detect(im, rect, [], 1)
    return f, box, pts

detector_dir = './face-datasets/'
sys.path.insert(0, detector_dir+'facealign')
sys.path.insert(0, detector_dir+'util')

import PyLandmark as LandmarkDetector
from MtcnnPycaffe import MtcnnDetector

parser = argparse.ArgumentParser(description='find landmarks and bounding boxes')
parser.add_argument('source', metavar='source', type=str, help='Folder containing images to process')
parser.add_argument('destination', metavar='destination', type=str, help='Output folder for bounding boxes and landmark files')

args = parser.parse_args()
source_path = args.source
dest_path = args.destination

face_detector = MtcnnDetector()
landmark_detector = LandmarkDetector.create("./detection/model/")

if os.path.exists(source_path):
    source_path = os.path.abspath(source_path)

    with open(dest_path + '/features.csv', 'a') as csv_file:
        with concurrent.futures.ProcessPoolExecutor(max_workers=8) as executor:
            file_list = glob.glob(source_path + '/*.jpg')
            future_proc = {executor.submit(detect_features, f): f for f in file_list}
            for future in concurrent.futures.as_completed(future_proc):
                filename, box, lnd = future.result()
                writer = csv.writer(csv_file)
                writer.writerow([filename, box, lnd])


