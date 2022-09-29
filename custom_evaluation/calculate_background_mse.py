import argparse
import cv2
import numpy as np
import os
import concurrent.futures
import glob
import pandas as pd
import csv
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

def calculate_bg_mse(s_path, dest_folder, report_name):
    filename = s_path.split('\\')[-1]
    d_path = dest_folder + '\\' + filename
    bbox = retrieve_bounding_box(filename, "all_bounding_boxes.csv")
    try:
        s_image = cv2.imread(s_path).astype(np.float32) / 255
        t_image = cv2.imread(d_path).astype(np.float32) / 255
        
        s_image[bbox[0]:bbox[2], bbox[1]:bbox[3]] = 0
        t_image[bbox[0]:bbox[2], bbox[1]:bbox[3]] = 0
        
        mse = np.square(np.subtract(s_image,t_image)).mean()
        
        with open(report_name, 'a') as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow([filename, mse])
    
    except Exception as e:
        print(e)
        
    return

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Calculate mse between the background of source and dest images")
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
                calculate_bg_mse, f, dest_folder, deltas_filename): f for f in file_list}
