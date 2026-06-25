# Spalling Segmentation (Binary)

콘크리트 구조물 **박락(spalling)** 이진 분할. 3개 공개 데이터셋(CUBIT_Seg, DamSeg, HRCDS)에서
spalling 라벨만 추출·통합하여 binary(배경 0 / spalling 1) 세그멘테이션 베이스라인을 구축했다.
모델 3종을 비교해 **UPerNet+ConvNeXt-L** 를 선정하고, 데이터 누수를 바로잡은 정직한 수치(통합 IoU 90.02)와
일반화·데이터효율 진단을 마쳤다. 향후 **실증 데이터로 파인튜닝**해 도메인 확장 예정.

> [mmsegmentation](https://github.com/open-mmlab/mmsegmentation) 1.x를 **패키지 의존성**(`pip install`)으로 사용한다.
> mmseg 소스를 수정하지 않고, 커스텀 모듈(`SpallingDataset`)은 `spalling_seg/` 패키지에 두고
> 각 config의 `custom_imports`로 등록한다. mmseg의 base config(default_runtime/schedule/upernet 등)는
> 자립성을 위해 `configs/_base_/`에 vendoring(mmseg 1.2.2 기준)했다.

## 베이스라인 모델

초기에 3종(Mask2Former+Swin-B, UPerNet+Swin-L, UPerNet+ConvNeXt-L)을 비교해
**UPerNet + ConvNeXt-Large** 를 베이스라인으로 선정했다(IoU 최고·헛검출 최저).

| 구성 | 내용 |
|---|---|
| 아키텍처 | UPerNet (UPerHead + FCN 보조헤드) + **ConvNeXt-Large** 백본, `EncoderDecoder` |
| 백본 사전학습 | ImageNet-21k (`convnext-large_3rdparty_in21k`) |
| 클래스 | 2 (배경 0 / 박락 1), CrossEntropyLoss (main 1.0 + aux 0.4) |
| 총 파라미터 | 약 235M |
| 입력 / 추론 | 512×512 학습 crop / slide-window 추론(crop 512, stride 341) |
| 증강 | RandomResize(0.5–2.0×) · RandomCrop · RandomFlip(0.5) · PhotoMetricDistortion |
| 옵티마이저 | AdamW (lr 1e-4, wd 0.05, layer-wise LR decay 0.9), AMP 혼합정밀 |
| 스케줄 / 반복 | LinearLR 워밍업(0–1.5k) → PolyLR, **40,000 iter** |
| 배치 / 하드웨어 | 2/GPU × 2 (RTX A4000 16GB) = 유효 4 |

## 결과 (clean test, 222장)

**누수 제거(leakage-free) 통합 baseline** — CUBIT을 원본 사진(씬) 단위로 분할해 재학습한
정직한 수치다. 시작 체크포인트: `work_dirs/clean_convnext-large_40k/best_mIoU_iter_28000.pth`.

| 데이터셋 | n | spalling IoU | F1 | Precision | Recall |
|---|---|---|---|---|---|
| CUBIT | 113 | 82.89 | 90.65 | 90.80 | 90.49 |
| DamSeg | 35 | 82.75 | 90.56 | 94.64 | 86.82 |
| HRCDS | 74 | 95.75 | 97.83 | 97.09 | 98.58 |
| **통합(ALL)** | **222** | **90.02** | **94.75** | 94.51 | 94.99 |

> **데이터 누수 보정**: CUBIT_Seg/spalling512는 큰 원본 사진을 512 타일로 자른 데이터(1160 타일 ←
> 59장 원본)인데, 초기 분할이 타일을 통째로 섞어 나눠 **같은 사진의 타일이 train·test에 동시에**
> 들어가 있었다(CUBIT test 타일 100%가 학습 사진과 겹침). 모델이 "본 표면을 기억"해 점수가
> 부풀려졌다. 씬 단위 분할로 바로잡자 CUBIT 88.25→82.89, 통합 **92.42→90.02**(−2.4%p)로 정직해졌다.
> (옛 누수 포함 수치: Mask2Former+Swin-B 92.04 / UPerNet+Swin-L 91.59 / ConvNeXt-L 92.42 IoU — 참고용)

**일반화 진단 & 실증 데이터 계획**: 학습에 없던 새 도메인에선 IoU가 35~63%로 떨어지나(leave-one-dataset-out),
**라벨 25~50장 파인튜닝이면 ~85% 회복**됨을 검증했다. 누수·헛검출·도메인갭·데이터효율·임계값
전체 진단은 [`results/BASELINE_DIAGNOSIS.md`](results/BASELINE_DIAGNOSIS.md) 참조. 샘플 예측은
[`results/samples/`](results/samples/) (좌→우: 원본 / GT / 예측).

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
  build_spalling_dataset.py   # 3개 데이터셋 -> 통합 이진 데이터셋 (--cubit-scene-split: 누수 제거)
  auto_train_spalling.sh      # 3개 모델 순차 학습 (mim)
  inference_spalling.py       # test 평가 + metrics.txt (FPR 포함)
  run_clean.sh                # 누수 제거 데이터로 재학습 + 평가 (정직한 baseline)
  eval_per_dataset.py         # 데이터셋별 IoU 분해
  build_lodo_folds.py / run_lodo*.sh   # leave-one-dataset-out (도메인갭 진단)
  build_ft_curve.py / run_ft_curve.sh  # 데이터효율 곡선 (파인튜닝 리허설)
  failure_analysis.py         # cross-domain 실패 정성 분석
  threshold_sweep.py          # 운영 임계값(precision/recall/헛검출) sweep
results/                 # metrics_*.txt, samples/, BASELINE_DIAGNOSIS.md, lodo/ ft_curve/ ...
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
python tools/build_spalling_dataset.py        # 기본: spalling 포함 이미지만 (옛 split, CUBIT 타일 누수 있음)
python tools/build_spalling_dataset.py --cubit-scene-split --dst data/spalling_clean  # 누수 제거(권장)
# --include-negatives : 박락 없는 이미지도 포함(나머지→배경). 헛검출 평가용
```
출력: `<dst>/{train,val,test}/{images(.jpg symlink),masks(.png 0/1)}`.
분할: HRCDS 자체 split, DamSeg 80/10/10(seed=42), CUBIT은 `--cubit-scene-split` 시 **원본 사진 단위** 80/10/10.
- 옛 spalling-only: train 2092 / val 229 / test 225 (CUBIT 타일 누수 — 위 결과표 주석 참고)
- **누수 제거 clean**: train 2098 / val 226 / test 222 (CUBIT 113 + DamSeg 35 + HRCDS 74)

## 학습 / 평가

```bash
# 권장: 누수 제거 baseline 재학습 + 평가 (2-GPU DDP, ~4h)
python tools/build_spalling_dataset.py --cubit-scene-split --dst data/spalling_clean
CUDA_VISIBLE_DEVICES=0,1 bash tools/run_clean.sh

# 개별 학습 (tools/train.py, dist_train.sh 는 mmseg 1.2.2 에서 vendoring):
CUDA_VISIBLE_DEVICES=0,1 bash tools/dist_train.sh \
  configs/clean/convnext-large_upernet_40k_clean.py 2

# test 평가 (데이터셋별 IoU 분해)
CUDA_VISIBLE_DEVICES=0 python tools/eval_per_dataset.py \
  --config configs/clean/convnext-large_upernet_40k_clean.py \
  --checkpoint work_dirs/clean_convnext-large_40k/best_mIoU_iter_*.pth \
  --img-dir data/spalling_clean/test/images --gt-dir data/spalling_clean/test/masks
```

공통: 512×512 crop, batch 2/GPU, 40k iter, AMP, num_classes=2, `SpallingDataset`.
백본 pretrained: swin_base/large_patch4_window12_384_22k, convnext-large_3rdparty_in21k (mmseg/mmpretrain model zoo).

## 환경 / 메모리

- 2× NVIDIA RTX A4000 (16GB). 학습 메모리(bs2/512/AMP): Swin-B ~4.9GB, ConvNeXt-L ~5.8GB, Swin-L ~6.5GB (1st-iter cuDNN 스파이크 12~14GB) — 모두 16GB 적합.
- 스택: torch + mmengine 0.10.7 / mmcv 2.1.0 / mmsegmentation 1.2.2 / mmdet 3.3.0 / mmpretrain 1.2.0.

## 체크포인트 위치 (이 저장소 미포함, 용량 큼)

```
# 권장 baseline (누수 제거, 실증 파인튜닝 시작점):
work_dirs/clean_convnext-large_40k/best_mIoU_iter_28000.pth

# 진단용 (leave-one-dataset-out, 도메인별로 한 데이터셋 제외 학습):
work_dirs/lodo_convnext-large_40k_{damseg,hrcds,cubit}/best_mIoU_iter_*.pth

# 옛 누수 포함 baseline (참고용, 학습 환경 /home/ldh/minkyung/mmsegmentation):
work_dirs/convnext-large_upernet_40k_spalling-512x512/best_mIoU_iter_36000.pth
```
