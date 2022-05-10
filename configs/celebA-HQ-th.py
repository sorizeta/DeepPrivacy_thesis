import os

_base_config_ = "defaults.py"

#model_url = "outputs/celebA-HQ-th/checkpoints/step_200000.ckpt"

models = dict(
    min_imsize=256,
    max_imsize=256,
    pose_size=136
)

landmarks = ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12",
             "13", "14", "15", "16", "17", "18", "19", "20", "21", "22", "23",
             "24", "25", "26", "27", "28", "29", "30", "31", "32", "33", "34",
             "35", "36", "37", "38", "39", "40", "41", "42", "43", "44", "45",
             "46", "47", "48", "49", "50", "51", "52", "53", "54", "55", "56",
             "57", "58", "59", "60", "61", "62", "63", "64", "65", "66", "67", "68"]

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
