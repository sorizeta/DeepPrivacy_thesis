import argparse
import numpy as np
import os
import pandas as pd
import csv

def bb_intersection_over_union(box1, box2):
    area1 = np.zeros((256, 256), dtype=bool)
    area2 = np.zeros((256, 256), dtype=bool)
    
    area1[box1[0] : box1[2], box1[1]: box1[3]] = 1
    area2[box2[0] : box2[2], box2[1]: box2[3]] = 1
    
    intersection = area1 * area2
    union = area1 + area2
    IOU = intersection.sum()/float(union.sum())
	
    return IOU

def calculate_bboxes(filename):
    df_boxes = pd.read_csv(filename, header=0, names=['filename', 'bbox', 'bbox_idx', 'ann'])
    df_boxes = df_boxes.sort_values(by="filename")
    bbox_names = df_boxes["filename"].values
    bbox_str = df_boxes["bbox"].values
    bboxes = dict()
    for idx, s in enumerate(bbox_str):
        s = s.replace('\n', '')
        s = s.replace('[', '')
        s = s.replace(']', '')
        bbox = np.fromstring(s, sep=' ', dtype=int)[:4]
        if bbox is None:
            bbox = np.empty()
        bbox = np.asarray(bbox)

        bboxes[os.path.basename(bbox_names[idx])] = bbox
    
    return bboxes

def calculate_overlap(bounding_boxes, report_name):
    for filename, box in bounding_boxes.items():
        celeba_bbox = np.array([48, 28, 208, 228])
        try:
            if box.shape[0] == 0:
                IOU = 0
            else:
                IOU = bb_intersection_over_union(celeba_bbox, box)
                
            with open(report_name, 'a') as csv_file:
                writer = csv.writer(csv_file)
                writer.writerow([filename, IOU])
                
        except Exception as e:
            print(e)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Calculate IOU between the CelebA-HQ bounding boxes, fixed at the centre of the image, and the detected boxes")
    parser.add_argument('bbox_file', type=str, help="Bounding box file")
    parser.add_argument('results_filename', type=str, help='Name of the export csv file')
    args = parser.parse_args()

    bbox_file = args.bbox_file
    results_filename = args.results_filename

    bounding_boxes = calculate_bboxes(bbox_file)
    calculate_overlap(bounding_boxes, results_filename)
