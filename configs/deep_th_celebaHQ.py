_base_config_ = "th_celebaHQ.py"
model_size = 512

models = dict(
    conv_size={
        4: model_size//2,
        8: model_size//2,
        16: model_size//2,
        32: model_size//2,
        64: model_size//4,
        128: model_size//8,
        256: model_size//16,
        512: model_size//32
    },
    generator=dict(
        residual=True,
        scalar_pose_input=True
    ),
    discriminator=dict(
        residual=True
    )
)

trainer = dict(
    max_images_to_train=20e6,
    progressive=dict(
        enabled=False,
    ),
)
