from .mask_util import generate_mask
from .custom import CustomDataset
from .build import DATASET_REGISTRY
import numpy as np
import pandas as pd
import os

def parse_arrays(row, col_name):
    lnd = row[col_name][1:-1]
    image_lnd = np.fromstring(lnd, sep=" ", dtype=int)
    return image_lnd

@DATASET_REGISTRY.register_module
class CelebAHQThesis(CustomDataset):

    def __init__(self, *args, is_train, **kwargs):
        super().__init__(*args, **kwargs)
        self.is_train = is_train
        self.load_landmarks()
        self.load_bounding_box()

    def _load_impaths(self):
        image_dir = self.dirpath
        image_paths_tmp = list(image_dir.glob("*.png"))
        excluded_paths = pd.read_csv("/home/ubuntu/cancel_files.csv", sep=",", names=['filename'])
        excluded_paths = excluded_paths["filename"].to_list()
        image_paths = [x for x in image_paths_tmp if x not in excluded_paths]
        image_paths.sort(key=lambda x: int(x.stem))
        return image_paths

    def get_mask(self, idx):
        mask = np.ones((self.imsize, self.imsize), dtype=np.bool)
        bounding_box = self.bounding_boxes[idx]
        x0, y0, x1, y1 = bounding_box
        mask[y0:y1, x0:x1] = 0

        return mask

    def load_bounding_box(self):
        # I think I'll add an option here
        # An if is better
        filepath = "/home/ubuntu/parsed_boxes_256.csv"
        assert os.path.isfile(filepath), \
            f"Did not find bounding boxes at: {filepath}"

        image_paths = [str(i).split("/")[-1] for i in self.image_paths]

        file_boxes = pd.read_csv(filepath, header=None, names=["filename", "bounding_boxes"])
        file_boxes = file_boxes[file_boxes["filename"].isin(image_paths)]
        file_boxes["parsed_boxes"] = file_boxes.apply(parse_arrays, args=("bounding_boxes", ), axis=1)
        bbox = np.stack(file_boxes["parsed_boxes"].values)
        self.bounding_boxes = bbox[:len(self)]
        assert len(self.bounding_boxes) == len(self)


    def load_landmarks(self):
        filepath = "/home/ubuntu/parsed_landmarks_256.csv"
        assert os.path.isfile(filepath), \
            f"Did not find landmarks at: {filepath}"
       
        image_paths = [str(i).split("/")[-1] for i in self.image_paths]
        landmarks_file = pd.read_csv(filepath, header=None, names=["filename", "landmarks"])
        landmarks_file = landmarks_file[landmarks_file["filename"].isin(image_paths)]
        landmarks_file['parsed_landmarks'] = landmarks_file.apply(parse_arrays, args=("landmarks", ), axis=1)
        landmarks = np.stack(landmarks_file["parsed_landmarks"].values)
        landmarks = np.reshape(landmarks, (-1, 68, 2))
        landmarks = landmarks[:, 28:, :]
        landmarks = landmarks.astype(np.float32)
        self.landmarks = landmarks[:len(self)]
        assert len(self.landmarks) == len(self),\
            f"Number of images: {len(self)}, landmarks: {len(landmarks)}"

    # A cosa serve questa cosa?
    # Potrebbe dovermi servire
    def get_item(self, index):
        batch = super().get_item(index)
        landmark = self.landmarks[index]
        batch["landmarks"] = landmark
        return batch
