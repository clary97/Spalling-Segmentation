#!/usr/bin/env bash
# Data-efficiency curve: fine-tune LODO(no-DamSeg) model on N DamSeg imgs, eval on fixed test.
# N=0 = base checkpoint, no fine-tune. 2-GPU DDP for training.
set -e
cd /home/ldh/minkyung/spalling-segmentation
source /home/ldh/anaconda3/etc/profile.d/conda.sh
conda activate internimage
# 단일 GPU (GPU 0). corrosion 학습과 공유하므로 2-GPU DDP 미사용 (OOM 방지)
export CUDA_VISIBLE_DEVICES=0

BASE=work_dirs/lodo_convnext-large_40k_damseg/best_mIoU_iter_32000.pth
BASECFG=configs/lodo/convnext-large_upernet_40k_lodo-damseg.py
OUT=results/ft_curve

# N=0: evaluate base checkpoint (no fine-tune) on the fixed test
echo "================ [$(date +%H:%M:%S)] N=0 (no fine-tune) EVAL ================"
CUDA_VISIBLE_DEVICES=0 python tools/eval_per_dataset.py \
  --config "$BASECFG" --checkpoint "$BASE" \
  --img-dir data/ft_damseg/n0/test/images --gt-dir data/ft_damseg/n0/test/masks \
  | tee "$OUT/metrics_n0.txt"

for N in 10 25 50 100 200; do
  CFG=configs/ft_damseg/ft-n${N}.py
  WD=work_dirs/ft_damseg_n${N}
  echo "================ [$(date +%H:%M:%S)] N=${N} : FINE-TUNE (single GPU) ================"
  python tools/train.py "$CFG" --work-dir "$WD"
  CKPT="$WD/iter_2000.pth"
  echo "================ [$(date +%H:%M:%S)] N=${N} : EVAL ($CKPT) ================"
  CUDA_VISIBLE_DEVICES=0 python tools/eval_per_dataset.py \
    --config "$CFG" --checkpoint "$CKPT" \
    --img-dir data/ft_damseg/n${N}/test/images --gt-dir data/ft_damseg/n${N}/test/masks \
    | tee "$OUT/metrics_n${N}.txt"
done
echo "================ [$(date +%H:%M:%S)] FT CURVE DONE ================"
