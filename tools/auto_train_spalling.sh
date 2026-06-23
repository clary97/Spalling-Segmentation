#!/bin/bash
# Sequential 2-GPU DDP training of the 3 spalling models via mim (installed mmsegmentation).
# After each run, prune periodic iter_*.pth (keep best_mIoU_*) to save disk.
# 실행: repo 루트에서 (conda env 활성화 + `pip install -e .` 선행). GPU 수는 --gpus 로 조정.
set -u
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1}
NGPUS=$(echo "$CUDA_VISIBLE_DEVICES" | tr ',' '\n' | grep -c .)

CONFIGS=(
  "configs/mask2former/mask2former_swin-b_40k_spalling-512x512.py"
  "configs/swin/swin-large_upernet_40k_spalling-512x512.py"
  "configs/convnext/convnext-large_upernet_40k_spalling-512x512.py"
)

for CFG in "${CONFIGS[@]}"; do
  NAME=$(basename "$CFG" .py)
  WD="work_dirs/$NAME"
  echo "==================================================================="
  echo "[$(date '+%F %T')] START  $NAME (gpus=$NGPUS)"
  echo "==================================================================="
  bash tools/dist_train.sh "$CFG" "$NGPUS" --work-dir "$WD"
  RC=$?
  if [ $RC -ne 0 ]; then
    echo "[$(date '+%F %T')] FAILED $NAME (rc=$RC) — 다음 모델로 진행"
    continue
  fi
  # 디스크 절약: best 외 periodic 체크포인트 제거
  if ls "$WD"/best_mIoU_*.pth >/dev/null 2>&1; then
    rm -f "$WD"/iter_*.pth
    echo "[$(date '+%F %T')] pruned periodic ckpts, kept best for $NAME"
  fi
  echo "[$(date '+%F %T')] DONE   $NAME"
done
echo "[$(date '+%F %T')] ===== ALL SPALLING TRAININGS COMPLETE ====="
