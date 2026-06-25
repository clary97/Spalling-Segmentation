# 누수 제거(clean) baseline: CUBIT을 씬 단위로 분할한 통합 데이터셋으로 재학습.
# data: tools/build_spalling_dataset.py --cubit-scene-split --dst data/spalling_clean
_base_ = ['../convnext/convnext-large_upernet_40k_spalling-512x512.py']

data_root = 'data/spalling_clean'
train_dataloader = dict(dataset=dict(data_root=data_root))
val_dataloader = dict(dataset=dict(data_root=data_root))
test_dataloader = dict(dataset=dict(data_root=data_root))
