# LODO fold: held-out = hrcds. Train on the other two datasets, test on all of hrcds.
# Leakage-free + cross-domain generalization probe. data built by tools/build_lodo_folds.py
_base_ = ['../convnext/convnext-large_upernet_40k_spalling-512x512.py']

data_root = 'data/spalling_lodo_hrcds'
train_dataloader = dict(dataset=dict(data_root=data_root))
val_dataloader = dict(dataset=dict(data_root=data_root))
test_dataloader = dict(dataset=dict(data_root=data_root))
