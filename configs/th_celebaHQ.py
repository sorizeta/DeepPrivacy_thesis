import os

_base_config_ = "defaults.py"

dataset_type = "CelebAHQThesis"
data_train = dict(
    dataset=dict(
        type=dataset_type,
        dirpath="/home/ubuntu/labels_new/train_files_new.csv",
        landmarkspath = "/home/ubuntu/labels_new/landmarks_train_new.npy",
        boxespath = "/home/ubuntu/labels_new/boxes_train_new.npy",
        percentage=1.0,
        is_train=True
    ),
    transforms=[dict(type="FlattenLandmark")],
)
data_val = dict(
    dataset=dict(
        type=dataset_type,
        dirpath="/home/ubuntu/labels_new/test_files_new.csv",
        landmarkspath = "/home/ubuntu/labels_new/landmarks_test_new.npy",
        boxespath = "/home/ubuntu/labels_new/boxes_test_new.npy",
        percentage=.2,
        is_train=False
    ),
    transforms=[dict(type="FlattenLandmark")],
)
