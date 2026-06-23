_base_ = [
    '../_base_/models/upernet_convnext.py',
    '../_base_/datasets/spalling.py',
    '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_40k.py'
]

custom_imports = dict(
    imports=['mmpretrain.models', 'spalling_seg.datasets'], allow_failed_imports=False)


crop_size = (512, 512)
data_preprocessor = dict(size=crop_size)
checkpoint_file = 'checkpoints/convnext-large_3rdparty_in21k_20220301-e6e0ea0a.pth'  # local

# ConvNeXt-L + UPerNet, num_classes=2 for binary spalling
model = dict(
    data_preprocessor=data_preprocessor,
    backbone=dict(
        type='mmpretrain.ConvNeXt',
        arch='large',
        out_indices=[0, 1, 2, 3],
        drop_path_rate=0.4,
        layer_scale_init_value=1.0,
        gap_before_final_norm=False,
        init_cfg=dict(
            type='Pretrained', checkpoint=checkpoint_file,
            prefix='backbone.')),
    decode_head=dict(
        in_channels=[192, 384, 768, 1536],
        num_classes=2,
    ),
    auxiliary_head=dict(in_channels=768, num_classes=2),
    test_cfg=dict(mode='slide', crop_size=crop_size, stride=(341, 341)),
)

optim_wrapper = dict(
    _delete_=True,
    type='AmpOptimWrapper',
    optimizer=dict(
        type='AdamW', lr=0.0001, betas=(0.9, 0.999), weight_decay=0.05),
    paramwise_cfg={
        'decay_rate': 0.9,
        'decay_type': 'stage_wise',
        'num_layers': 12
    },
    constructor='LearningRateDecayOptimizerConstructor',
    loss_scale='dynamic')

param_scheduler = [
    dict(type='LinearLR', start_factor=1e-6, by_epoch=False, begin=0, end=1500),
    dict(type='PolyLR', power=1.0, begin=1500, end=40000, eta_min=0.0, by_epoch=False)
]

train_dataloader = dict(batch_size=2, num_workers=2)
val_dataloader = dict(batch_size=1, num_workers=4)
test_dataloader = val_dataloader

default_hooks = dict(
    checkpoint=dict(type='CheckpointHook', by_epoch=False, interval=4000,
                    save_best='mIoU', max_keep_ckpts=1))
