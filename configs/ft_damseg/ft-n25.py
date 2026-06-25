# Data-efficiency fine-tune: start from LODO(no-DamSeg) model, fine-tune on 25 DamSeg imgs.
# Eval on fixed 100-img DamSeg test. Short schedule, low LR.
_base_ = ['../convnext/convnext-large_upernet_40k_spalling-512x512.py']

data_root = 'data/ft_damseg/n25'
train_dataloader = dict(dataset=dict(data_root=data_root))
val_dataloader = dict(dataset=dict(data_root=data_root))
test_dataloader = dict(dataset=dict(data_root=data_root))

load_from = 'work_dirs/lodo_convnext-large_40k_damseg/best_mIoU_iter_32000.pth'

train_cfg = dict(type='IterBasedTrainLoop', max_iters=2000, val_interval=2000)
param_scheduler = [
    dict(type='LinearLR', start_factor=1e-6, by_epoch=False, begin=0, end=100),
    dict(type='PolyLR', power=1.0, begin=100, end=2000, eta_min=0.0, by_epoch=False),
]
optim_wrapper = dict(optimizer=dict(lr=2e-5))
default_hooks = dict(
    checkpoint=dict(type='CheckpointHook', by_epoch=False, interval=2000, max_keep_ckpts=1))

# 단일 GPU 공유 환경: cuDNN autotune 스파이크 억제 (OOM 방지)
env_cfg = dict(cudnn_benchmark=False)
