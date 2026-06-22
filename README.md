# Spalling Segmentation (Binary)

콘크리트 구조물 **박락(spalling)** 이진 분할. 3개 공개 데이터셋(CUBIT_Seg, DamSeg, HRCDS)에서
spalling 라벨만 추출·통합하여 binary(배경 0 / spalling 1) 세그멘테이션 모델 3종을 학습·평가했다.

> 이 저장소는 [open-mmlab/mmsegmentation](https://github.com/open-mmlab/mmsegmentation) 1.x 기반이며,
> config/스크립트를 mmsegmentation 체크아웃에 **드롭인**하여 재현한다 (아래 *재현* 참고).

## 결과 (test, 225장)

| Model | spalling IoU | F1 | Precision | Recall | FPR | mIoU |
|---|---|---|---|---|---|---|
| Mask2Former + Swin-B | 92.04 | 95.86 | 94.75 | 96.98 | 1.55 | 94.82 |
| UPerNet + Swin-L | 91.59 | 95.61 | 94.93 | 96.30 | 1.48 | 94.53 |
| **UPerNet + ConvNeXt-L** | **92.42** | **96.06** | 95.79 | 96.33 | **1.22** | **95.08** |

검증셋(val, 229장) 최고 mIoU: Swin-B 93.09 / Swin-L 93.14 / ConvNeXt-L 93.50.
세 모델 모두 spalling IoU ~92%로 우수하며 **ConvNeXt-L이 근소 우위**(IoU 최고, FPR 최저).

원본 per-class 수치는 [`results/`](results/)의 `metrics_*.txt` 참고. 샘플 예측은 [`results/samples/`](results/samples/) (좌→우: 원본 / GT / 예측).

## 데이터셋

`/mnt/nas_200`의 3개 데이터셋에서 spalling만 추출 (검증된 인코딩):

| 데이터셋 | 마스크 형식 | spalling 값 | 비고 |
|---|---|---|---|
| CUBIT_Seg/spalling512 | RGB | `(128,0,0)` | spalling 전용 폴더 |
| DamSeg (Easy/Medium/Hard) | RGB | `(0,0,255)` 파랑 = category_id 1 | crack=빨강. 형태학+육안 확인 |
| HRCDS | index PNG | index `2` | 1=crack,3=corrosion,4=exposed rebar |

빌드: [`tools/build_spalling_dataset.py`](tools/build_spalling_dataset.py)
- 기본: spalling 포함 이미지만. `--include-negatives`: 전체 이미지(나머지→배경).
- 분할: HRCDS는 자체 train/val/test, CUBIT·DamSeg는 80/10/10(seed=42).
- 출력 `data/spalling/{train,val,test}/{images(.jpg symlink),masks(.png 0/1)}`.

**spalling-only 결과**: train 2092 / val 229 / test 225 (CUBIT 1160 + DamSeg 366 + HRCDS 1020; spalling 없는 1314장 제외).

## 모델 / config

| Model | config | pretrained 백본 |
|---|---|---|
| Mask2Former + Swin-B | [configs/mask2former/...](configs/mask2former/mask2former_swin-b_40k_spalling-512x512.py) | swin_base_patch4_window12_384_22k |
| UPerNet + Swin-L | [configs/swin/...](configs/swin/swin-large_upernet_40k_spalling-512x512.py) | swin_large_patch4_window12_384_22k |
| UPerNet + ConvNeXt-L | [configs/convnext/...](configs/convnext/convnext-large_upernet_40k_spalling-512x512.py) | convnext-large_3rdparty_in21k |

공통: 512×512 crop, batch 2/GPU, 40k iter, AMP, num_classes=2, `SpallingDataset`.

## 환경 / 메모리

- 2× NVIDIA RTX A4000 (16GB). 학습 메모리(bs2/512/AMP): Swin-B ~4.9GB, ConvNeXt-L ~5.8GB, Swin-L ~6.5GB (1st-iter cuDNN 스파이크 12~14GB) — 모두 16GB에 적합.
- conda env `internimage` (PyTorch + mmcv 2.x + mmsegmentation 1.x + mmdet + mmpretrain).
- ConvNeXt 백본은 `mmpretrain` 필요.

## 재현

```bash
# 1) mmsegmentation 1.x 체크아웃에 이 repo의 파일을 복사
cp -r mmseg/datasets/spalling.py        <MMSEG>/mmseg/datasets/
cp -r configs/_base_/datasets/spalling.py <MMSEG>/configs/_base_/datasets/
cp -r configs/*/                          <MMSEG>/configs/   # 모델 config
cp tools/*                                <MMSEG>/tools/
# 2) mmseg/datasets/__init__.py 에 SpallingDataset 등록
#    from .spalling import SpallingDataset  +  __all__ 에 추가
# 3) 데이터 빌드
python tools/build_spalling_dataset.py
# 4) 학습 (2-GPU 순차)
bash tools/auto_train_spalling.sh
# 5) 평가 (test) → metrics.txt
python tools/inference_spalling.py --config <CFG> --checkpoint <CKPT> \
  --input data/spalling/test/images --gt-dir data/spalling/test/masks \
  --compute-metrics --output results_eval
```

## 체크포인트 위치 (이 저장소에는 미포함, 2.2GB)

학습 환경(`/home/ldh/minkyung/mmsegmentation`)의 work_dirs:
```
work_dirs/mask2former_swin-b_40k_spalling-512x512/best_mIoU_iter_36000.pth
work_dirs/swin-large_upernet_40k_spalling-512x512/best_mIoU_iter_40000.pth
work_dirs/convnext-large_upernet_40k_spalling-512x512/best_mIoU_iter_36000.pth
```
