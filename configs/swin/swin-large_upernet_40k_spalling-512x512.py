_base_ = [
    '../_base_/models/upernet_swin.py',
    '../_base_/datasets/spalling.py',
    '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_40k.py',
]
crop_size = (512, 512)
data_preprocessor = dict(size=crop_size)
checkpoint_file = 'checkpoints/swin_large_patch4_window12_384_22k_20220412-6580f57d.pth'  # local

# Swin-L backbone (in22k-384 pretrained) + UPerNet, num_classes=2 binary spalling
model = dict(
    data_preprocessor=data_preprocessor,
    backbone=dict(
        init_cfg=dict(type='Pretrained', checkpoint=checkpoint_file),
        pretrain_img_size=384,
        embed_dims=192,
        depths=[2, 2, 18, 2],
        num_heads=[6, 12, 24, 48],
        window_size=12,
        use_abs_pos_embed=False,
        drop_path_rate=0.3,
        patch_norm=True),
    decode_head=dict(in_channels=[192, 384, 768, 1536], num_classes=2),
    auxiliary_head=dict(in_channels=768, num_classes=2))

# AdamW + AMP (16GB GPU). No weight decay for pos embed / norm.
optim_wrapper = dict(
    _delete_=True,
    type='AmpOptimWrapper',
    loss_scale='dynamic',
    optimizer=dict(
        type='AdamW', lr=0.00006, betas=(0.9, 0.999), weight_decay=0.01),
    paramwise_cfg=dict(
        custom_keys={
            'absolute_pos_embed': dict(decay_mult=0.),
            'relative_position_bias_table': dict(decay_mult=0.),
            'norm': dict(decay_mult=0.)
        }))

param_scheduler = [
    dict(type='LinearLR', start_factor=1e-6, by_epoch=False, begin=0, end=1500),
    dict(type='PolyLR', eta_min=0.0, power=1.0, begin=1500, end=40000,
         by_epoch=False)
]

train_dataloader = dict(batch_size=2, num_workers=2)
val_dataloader = dict(batch_size=1, num_workers=4)
test_dataloader = val_dataloader

default_hooks = dict(
    checkpoint=dict(type='CheckpointHook', by_epoch=False, interval=4000,
                    save_best='mIoU', max_keep_ckpts=1))
