# Spalling Segmentation (Binary)

콘크리트 구조물 **박락(spalling)** 이진 분할. 3개 공개 데이터셋(CUBIT_Seg, DamSeg, HRCDS)에서
spalling 라벨만 추출·통합하여 binary(배경 0 / spalling 1) 세그멘테이션 모델 3종을 학습·평가했다.

> [mmsegmentation](https://github.com/open-mmlab/mmsegmentation) 1.x를 **패키지 의존성**(`pip install`)으로 사용한다.
> mmseg 소스를 수정하지 않고, 커스텀 모듈(`SpallingDataset`)은 `spalling_seg/` 패키지에 두고
> 각 config의 `custom_imports`로 등록한다. mmseg의 base config(default_runtime/schedule/upernet 등)는
> 자립성을 위해 `configs/_base_/`에 vendoring(mmseg 1.2.2 기준)했다.

## 결과 (test, 225장)

| Model | spalling IoU | F1 | Precision | Recall | FPR | mIoU |
|---|---|---|---|---|---|---|
| Mask2Former + Swin-B | 92.04 | 95.86 | 94.75 | 96.98 | 1.55 | 94.82 |
| UPerNet + Swin-L | 91.59 | 95.61 | 94.93 | 96.30 | 1.48 | 94.53 |
| **UPerNet + ConvNeXt-L** | **92.42** | **96.06** | 95.79 | 96.33 | **1.22** | **95.08** |

검증셋(val, 229장) 최고 mIoU: Swin-B 93.09 / Swin-L 93.14 / ConvNeXt-L 93.50.
세 모델 모두 spalling IoU ~92%로 우수하며 **ConvNeXt-L이 근소 우위**(IoU 최고, FPR 최저).
per-class 수치는 [`results/`](results/)의 `metrics_*.txt`, 샘플 예측은 [`results/samples/`](results/samples/) (좌→우: 원본 / GT / 예측).

## 구조

```
spalling_seg/            # 커스텀 mmseg 모듈 (pip install -e . 로 설치)
  datasets/spalling.py   #   SpallingDataset (background, spalling)
configs/
  _base_/datasets/spalling.py                       # 데이터셋 설정 (SpallingDataset)
  _base_/{default_runtime,schedules/schedule_40k,models/upernet_*}.py  # mmseg vendored base
  mask2former/mask2former_swin-b_40k_spalling-512x512.py   # custom_imports 는 각 leaf 에
  swin/swin-large_upernet_40k_spalling-512x512.py
  convnext/convnext-large_upernet_40k_spalling-512x512.py
tools/
  build_spalling_dataset.py   # 3개 데이터셋 -> 통합 이진 데이터셋
  auto_train_spalling.sh      # 3개 모델 순차 학습 (mim)
  inference_spalling.py       # test 평가 + metrics.txt (FPR 포함)
results/                 # metrics_*.txt + samples/
```

## 설치

```bash
# 1) PyTorch (CUDA에 맞게 먼저)
# 2) OpenMMLab 스택 + 이 패키지
pip install -r requirements.txt
pip install -e .          # spalling_seg 를 import 가능하게 (custom_imports 용)
```

## 데이터셋

`/mnt/nas_200`의 3개 데이터셋에서 spalling만 추출 (검증된 인코딩):

| 데이터셋 | 마스크 형식 | spalling 값 | 비고 |
|---|---|---|---|
| CUBIT_Seg/spalling512 | RGB | `(128,0,0)` | spalling 전용 폴더 |
| DamSeg (Easy/Medium/Hard) | RGB | `(0,0,255)` 파랑 = category_id 1 | crack=빨강. 형태학+육안 확인 |
| HRCDS | index PNG | index `2` | 1=crack,3=corrosion,4=exposed rebar |

```bash
python tools/build_spalling_dataset.py        # 기본: spalling 포함 이미지만
# python tools/build_spalling_dataset.py --include-negatives   # 전체 이미지(나머지→배경)
```
출력: `data/spalling/{train,val,test}/{images(.jpg symlink),masks(.png 0/1)}`.
분할: HRCDS 자체 split, CUBIT·DamSeg 80/10/10(seed=42).
**spalling-only 결과**: train 2092 / val 229 / test 225 (CUBIT 1160 + DamSeg 366 + HRCDS 1020; spalling 없는 1314장 제외).

## 학습 / 평가

```bash
# 학습 (3개 순차, 2-GPU DDP)
CUDA_VISIBLE_DEVICES=0,1 bash tools/auto_train_spalling.sh
# 또는 개별 (tools/train.py, dist_train.sh 는 mmseg 1.2.2 에서 vendoring):
CUDA_VISIBLE_DEVICES=0,1 bash tools/dist_train.sh \
  configs/convnext/convnext-large_upernet_40k_spalling-512x512.py 2
# (clean pip 환경이면 `mim train mmsegmentation <cfg> --gpus 2 --launcher pytorch` 도 가능)

# test 평가 -> results_eval/metrics.txt
python tools/inference_spalling.py \
  --config configs/convnext/convnext-large_upernet_40k_spalling-512x512.py \
  --checkpoint <best_mIoU_*.pth> \
  --input data/spalling/test/images --gt-dir data/spalling/test/masks \
  --compute-metrics --output results_eval
```

공통: 512×512 crop, batch 2/GPU, 40k iter, AMP, num_classes=2, `SpallingDataset`.
백본 pretrained: swin_base/large_patch4_window12_384_22k, convnext-large_3rdparty_in21k (mmseg/mmpretrain model zoo).

## 환경 / 메모리

- 2× NVIDIA RTX A4000 (16GB). 학습 메모리(bs2/512/AMP): Swin-B ~4.9GB, ConvNeXt-L ~5.8GB, Swin-L ~6.5GB (1st-iter cuDNN 스파이크 12~14GB) — 모두 16GB 적합.
- 스택: torch + mmengine 0.10.7 / mmcv 2.1.0 / mmsegmentation 1.2.2 / mmdet 3.3.0 / mmpretrain 1.2.0.

## 체크포인트 위치 (이 저장소 미포함, 2.2GB)

학습 환경(`/home/ldh/minkyung/mmsegmentation`)의 work_dirs:
```
work_dirs/mask2former_swin-b_40k_spalling-512x512/best_mIoU_iter_36000.pth
work_dirs/swin-large_upernet_40k_spalling-512x512/best_mIoU_iter_40000.pth
work_dirs/convnext-large_upernet_40k_spalling-512x512/best_mIoU_iter_36000.pth
```
