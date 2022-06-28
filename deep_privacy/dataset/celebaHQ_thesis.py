from .mask_util import generate_mask
from .custom import CustomDataset
from .build import DATASET_REGISTRY
import numpy as np
import pandas as pd
import os
from pathlib import Path

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
        image_df = pd.read_csv(self.dirpath, header=None, names=['filename'], sep=';')
        image_list = image_df['filename'].to_list()
        image_paths = [Path(x) for x in image_list]
        '''
        image_paths_tmp = list(image_dir.glob("*.png"))
        excluded_paths = pd.read_csv("/home/ubuntu/cancel_files.csv", sep=",", names=['filename'])
        excluded_paths = excluded_paths["filename"].to_list()
        image_paths = [x for x in image_paths_tmp if x not in excluded_paths]
        image_paths.sort(key=lambda x: int(x.stem))
        print('Print:', len(image_paths))
        '''
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
        filepath = self.dirpath.joinpath(self.boxespath)
        assert filepath.is_file(), \
            f"Did not find landmarks at: {filepath}"
        boxes = np.load(filepath, allow_pickle=True)
        boxes = np.vstack(boxes)
        boxes = boxes.astype(np.int)
        self.bounding_boxes = boxes


    def load_landmarks(self):
        filepath = self.dirpath.joinpath(self.landmarkspath)
        assert filepath.is_file(), \
            f"Did not find landmarks at: {filepath}"
        landmarks = np.load(filepath, allow_pickle=True)
        landmarks = np.vstack(landmarks)
        landmarks = np.reshape(landmarks, (-1, 45, 2))
        landmarks = landmarks.astype(np.float32)
        self.landmarks = landmarks

    def get_item(self, index):
        batch = super().get_item(index)
        landmark = self.landmarks[index]
        batch["landmarks"] = landmark
        return batch
