import os

_base_config_ = "defaults.py"

models = dict(
    pose_size=0
)

dataset_type = "CelebAHQThesis"
data_train = dict(
    dataset=dict(
        type=dataset_type,
        dirpath="/datasets/celeba/img_celeba",
        percentage=1.0,
        is_train=True
    ),
    transforms=[
        dict(type="RandomFlip", flip_ratio=0.5),
    ],
)
data_val = dict(
    dataset=dict(
        type=dataset_type,
        dirpath=os.path.join("/home/ubuntu/test_img_celeba/"),
        percentage=.2,
        is_train=False
    ),
    transforms=[
    ],
)
