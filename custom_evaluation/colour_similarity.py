from unittest.util import strclass
import cv2
import argparse
import numpy as np
import colour
import glob
import csv
import os
import concurrent.futures
import pandas as pd

def create_crop(cell):
    box = cell["bbox"].values[0]
    image_bbox= np.fromstring(box[1:-1], sep=" ", dtype=int)
    image_bbox = image_bbox[:4]
    return image_bbox


def retrieve_bounding_box(filename, bbox_file):
    # Here we assume that the bounding box file is a csv file
    bboxes = pd.read_csv(bbox_file, names=["filename", "bbox"], sep=';')
    line = bboxes[bboxes['filename'].str.contains(filename)]
    bbox = create_crop(line)
    return bbox
    
    
def calculate_deltas(s_path, dest_folder, deltas_filename):
    filename = s_path.split('\\')[-1]
    d_path = dest_folder + '\\' + filename
    bbox = retrieve_bounding_box(filename, "all_bounding_boxes.csv")
    try:
        s_image = cv2.imread(s_path).astype(np.float32) / 255
        t_image = cv2.imread(d_path).astype(np.float32) / 255
        
        
        crop_s_image = s_image[bbox[0]:bbox[2], bbox[1]:bbox[3]]
        crop_t_image = t_image[bbox[0]:bbox[2], bbox[1]:bbox[3]]
        
        lab_s_image = cv2.cvtColor(crop_s_image, cv2.COLOR_BGR2LAB)
        lab_t_image = cv2.cvtColor(crop_t_image, cv2.COLOR_BGR2LAB)

        L_s, A_s, B_s = cv2.split(lab_s_image)
        L_t, A_t, B_t = cv2.split(lab_t_image)

        delta_E = np.mean(colour.difference.delta_E_CIE2000(
            lab_s_image, lab_t_image))
        delta_L = np.abs(
            np.mean(
                np.add(-L_t, L_s)
            )
        )
        delta_C = np.abs(
            np.mean(
                np.add(
                    -np.sqrt(np.power(A_t, 2) + np.power(B_t, 2)),
                    np.sqrt(np.power(A_s, 2) + np.power(B_s, 2))
                )
            )
        )
        
        with open(deltas_filename, 'a') as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow([filename, delta_L, delta_C, delta_E])
        
    except Exception as e:
        print(e)
    return

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Calculate deltaL, deltaC and deltaE between pairs of images with the same name")
    parser.add_argument('src_folder', type=str, help="First folder")
    parser.add_argument('dest_folder', type=str, help="Second folder")
    parser.add_argument('file_ext', type=str, help="File extension")
    parser.add_argument('deltas_filename', type=str, help='name of the deltas file')
    args = parser.parse_args()

    src_folder = args.src_folder.rstrip()
    dest_folder = args.dest_folder.rstrip()
    file_ext = args.file_ext
    deltas_filename = args.deltas_filename

    if os.path.exists(src_folder) and os.path.exists(dest_folder):
        src_folder = os.path.abspath(src_folder)
        dest_folder = os.path.abspath(dest_folder)

        with concurrent.futures.ProcessPoolExecutor(max_workers=8) as executor:
            file_list = glob.glob(src_folder + '/*.' + file_ext.lower())
            future_proc = {executor.submit(
                calculate_deltas, f, dest_folder, deltas_filename): f for f in file_list}
