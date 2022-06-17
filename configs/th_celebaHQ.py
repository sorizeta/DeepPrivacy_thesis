import os

_base_config_ = "defaults.py"

dataset_type = "CelebAHQThesis"
data_train = dict(
    dataset=dict(
        type=dataset_type,
        dirpath="/home/ubuntu/labels/train_files.csv",
        landmarkspath = "/home/ubuntu/labels/landmarks_train.npy",
        boxespath = "/home/ubuntu/labels/boxes_train.npy",
        percentage=1.0,
        is_train=True
    ),
    transforms=[dict(type="FlattenLandmark")],
)
data_val = dict(
    dataset=dict(
        type=dataset_type,
        dirpath="/home/ubuntu/labels/test_files.csv",
        landmarkspath = "/home/ubuntu/labels/landmarks_test.npy",
        boxespath = "/home/ubuntu/labels/boxes_test.npy",
        percentage=.2,
        is_train=False
    ),
    transforms=[dict(type="FlattenLandmark")],
)
