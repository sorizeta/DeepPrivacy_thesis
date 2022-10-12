import argparse
import os
import glob
from deep_privacy.detection.detection_api import *
from deep_privacy.detection.utils import *
import cv2
import csv

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Apply color transfer and light direction transfer")
    parser.add_argument('src_folder', type=str, help="Original image folder")
    args = parser.parse_args()
    src_folder = os.path.abspath(args.src_folder)

    allowed_file_ext = ['.jpg', '.png', '.jpeg']
    source_files = []

    for type in allowed_file_ext:
        search_path = os.path.join(src_folder, '*' + type)
        source_files = source_files + glob.glob(search_path)
    
    images = []

    for image in source_files:
        im = cv2.imread(image)
        images.append(im)

    face_detector = RCNNDetector(keypoint_threshold=0.2,
                                 rcnn_batch_size=16,
                                 densepose_threshold=.3,
                                 simple_expand=True,
                                 align_faces=False,
                                 resize_background=True,
                                 generator_imsize = 256,
                                 face_detector_cfg=dict(
                                    name="RetinaNetResNet50",
                                    confidence_threshold=.3,
                                    nms_iou_threshold=.3,
                                    max_resolution=1080,
                                    fp16_inference=True,
                                    clip_boxes=True
                                 )
    )

    image_annotations = face_detector.get_detections(images)
    bboxes = []
    for idx, ann in enumerate(image_annotations):
        bbox_idx, _ = match_bbox_keypoint(ann.bbox_XYXY, keypoints=ann.keypoints)
        with open('/home/ubuntu/face_detection_test/matches_deep.csv', 'a') as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow([source_files[idx], ann.bbox_XYXY, bbox_idx, ann.keypoints])
