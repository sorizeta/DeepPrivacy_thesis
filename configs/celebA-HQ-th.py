import os

_base_config_ = "defaults.py"

#model_url = "outputs/celebA-HQ-th/checkpoints/step_200000.ckpt"


dataset_type = "CelebAHQThesis"
data_train = dict(
    dataset=dict(
        type=dataset_type,
        dirpath="/datasets/celeba/celeba-hq/256x256/",
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
        dirpath=os.path.join("/home/ubuntu/celeba-test-256/"),
        percentage=.2,
        is_train=False
    ),
    transforms=[
    ],
)
