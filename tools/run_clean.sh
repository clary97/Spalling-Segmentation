#!/usr/bin/env bash
# 누수 제거 baseline: scene-split 통합 데이터셋으로 ConvNeXt-L 재학습 (40k, 2-GPU) + 평가.
set -e
cd /home/ldh/minkyung/spalling-segmentation
source /home/ldh/anaconda3/etc/profile.d/conda.sh
conda activate internimage
export CUDA_VISIBLE_DEVICES=0,1

CFG=configs/clean/convnext-large_upernet_40k_clean.py
WD=work_dirs/clean_convnext-large_40k
mkdir -p results/clean

echo "================ [$(date +%H:%M:%S)] CLEAN baseline : TRAIN ================"
bash tools/dist_train.sh "$CFG" 2 --work-dir "$WD"

BEST=$(ls -t "$WD"/best_mIoU_iter_*.pth 2>/dev/null | head -1)
[ -z "$BEST" ] && BEST=$(ls -t "$WD"/iter_*.pth 2>/dev/null | head -1)
echo "================ [$(date +%H:%M:%S)] CLEAN baseline : EVAL ($BEST) ================"
CUDA_VISIBLE_DEVICES=0 python tools/eval_per_dataset.py \
  --config "$CFG" --checkpoint "$BEST" \
  --img-dir data/spalling_clean/test/images --gt-dir data/spalling_clean/test/masks \
  | tee results/clean/metrics_clean.txt
echo "================ [$(date +%H:%M:%S)] CLEAN baseline DONE ================"
