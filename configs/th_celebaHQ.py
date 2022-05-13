import os

_base_config_ = "defaults.py"

dataset_type = "CelebAHQThesis"
data_train = dict(
    dataset=dict(
        type=dataset_type,
        dirpath="/datasets/celeba/celeba-hq/256x256/",
        percentage=1.0,
        is_train=True
    ),
    transforms=[],
)
data_val = dict(
    dataset=dict(
        type=dataset_type,
        dirpath=os.path.join("/home/ubuntu/celeba-test-256/"),
        percentage=.2,
        is_train=False
    ),
    transforms=[],
)
