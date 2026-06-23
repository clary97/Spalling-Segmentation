#!/usr/bin/env bash
# Leave-one-dataset-out: train ConvNeXt-L on 2 datasets, evaluate on the held-out one.
# 3 folds, sequential, 2-GPU DDP, 40k iters each. Run from repo root.
set -e
cd /home/ldh/minkyung/spalling-segmentation
source /home/ldh/anaconda3/etc/profile.d/conda.sh
conda activate internimage
export CUDA_VISIBLE_DEVICES=0,1

mkdir -p results/lodo
CKPT_CFG=configs/convnext/convnext-large_upernet_40k_spalling-512x512.py

for H in damseg hrcds cubit; do
  CFG=configs/lodo/convnext-large_upernet_40k_lodo-${H}.py
  WD=work_dirs/lodo_convnext-large_40k_${H}
  echo "================ [$(date +%H:%M:%S)] FOLD held-out=${H} : TRAIN ================"
  bash tools/dist_train.sh "$CFG" 2 --work-dir "$WD"

  BEST=$(ls -t "$WD"/best_mIoU_iter_*.pth 2>/dev/null | head -1)
  [ -z "$BEST" ] && BEST=$(ls -t "$WD"/iter_*.pth 2>/dev/null | head -1)
  echo "================ [$(date +%H:%M:%S)] FOLD held-out=${H} : EVAL ($BEST) ========"
  CUDA_VISIBLE_DEVICES=0 python tools/eval_per_dataset.py \
    --config "$CFG" --checkpoint "$BEST" \
    --img-dir data/spalling_lodo_${H}/test/images \
    --gt-dir  data/spalling_lodo_${H}/test/masks \
    | tee results/lodo/metrics_${H}.txt
done
echo "================ [$(date +%H:%M:%S)] ALL LODO FOLDS DONE ================"
