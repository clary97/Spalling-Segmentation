#!/usr/bin/env bash
# 3순위(정성 실패분석) + 4순위(임계값 sweep). 단일 GPU(0). 재학습과 공유.
set -e
cd /home/ldh/minkyung/spalling-segmentation
source /home/ldh/anaconda3/etc/profile.d/conda.sh
conda activate internimage
export CUDA_VISIBLE_DEVICES=0
mkdir -p results/failure results/threshold

CFG=configs/convnext/convnext-large_upernet_40k_spalling-512x512.py

echo "######## 3순위: LODO cross-domain 실패 분석 ########"
declare -A CK=(
  [damseg]=work_dirs/lodo_convnext-large_40k_damseg/best_mIoU_iter_32000.pth
  [hrcds]=work_dirs/lodo_convnext-large_40k_hrcds/best_mIoU_iter_40000.pth
  [cubit]=work_dirs/lodo_convnext-large_40k_cubit/best_mIoU_iter_20000.pth
)
for H in damseg hrcds cubit; do
  python tools/failure_analysis.py \
    --config "$CFG" --checkpoint "${CK[$H]}" \
    --img-dir data/spalling_lodo_${H}/test/images \
    --gt-dir  data/spalling_lodo_${H}/test/masks \
    --out results/failure/${H} --topk 8 \
    | tee results/failure/summary_${H}.txt
done

echo "######## 4순위: 임계값 sweep (baseline, negatives 포함 test) ########"
python tools/threshold_sweep.py \
  --config "$CFG" \
  --checkpoint /home/ldh/minkyung/mmsegmentation/work_dirs/convnext-large_upernet_40k_spalling-512x512/best_mIoU_iter_36000.pth \
  --img-dir data/spalling_with_neg/test/images \
  --gt-dir  data/spalling_with_neg/test/masks \
  | tee results/threshold/sweep_baseline.txt
echo "######## 3,4순위 DONE ########"
