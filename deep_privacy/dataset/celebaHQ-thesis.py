from .mask_util import generate_mask
from .custom import CustomDataset
from .build import DATASET_REGISTRY
import numpy as np


@DATASET_REGISTRY.register_module
class CelebAHQDataset(CustomDataset):

    def __init__(self, *args, is_train, **kwargs):
        super().__init__(*args, **kwargs)
        self.is_train = is_train
        self.load_landmarks()
        self.load_bounding_box()

    def _load_impaths(self):
        image_dir = self.dirpath
        image_paths = list(image_dir.glob("*.jpg"))
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
        filepath = self.dirpath.joinpath("bounding_boxes.npy")
        assert filepath.is_file(), \
            f"Did not find bounding boxes at: {filepath}"
        bbox = np.load(filepath)
        self.bounding_boxes = bbox[:len(self)]
        assert len(self.bounding_boxes) == len(self)

    ### Trovata trovata yeeeeee
    ### Caricamento landmark

    def load_landmarks(self):
        filepath = self.dirpath.joinpath("landmarks.npy")
        assert filepath.is_file(), \
            f"Did not find landmarks at: {filepath}"
        landmarks = np.load(filepath).reshape(-1, 7, 2)
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