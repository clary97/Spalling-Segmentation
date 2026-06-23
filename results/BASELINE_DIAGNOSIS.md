# Baseline 진단 (실증 데이터 투입 전)

> 목적: 실증(현장) 데이터가 들어오기 전, 현재 공개 데이터셋 baseline의 **신뢰성**을 점검하고
> 실증 비교의 기준점을 확정한다. 결론부터: 보고된 92% IoU는 **① 데이터 누수로 과대평가**되어
> 있고 **② 헛검출(false positive)이 미측정**이라, "실증 예상 성능"으로 쓰면 위험하다.
> 비교 기준점으로는 유효하되, 아래 함정을 보정한 정직한 수치와 함께 봐야 한다.

기준 모델: **UPerNet + ConvNeXt-L** (`best_mIoU_iter_36000.pth`), test 225장(spalling-only).

---

## 1. Negative(박락 없는) 이미지 헛검출 — 기존 미측정

기존 데이터셋은 **박락 픽셀이 있는 이미지만** 사용(`build_spalling_dataset.py` 기본 모드,
1314장 제외). 따라서 박락 없는 현장 이미지에서의 오경보율이 측정된 적이 없다.
negative 141장(DamSeg/HRCDS; CUBIT은 spalling 전용이라 negative 없음)을 추가해 재평가.

- **픽셀 단위**: spalling IoU 91.48, FPR 0.83% ← 배경 픽셀 급증으로 **희석된 값**, 배포 신뢰도로 부적합
- **이미지 단위 (배포 관점 오경보율, negative 141장 중)**:

| 헛검출 기준 | 해당 이미지 | 비율 |
|---|---|---|
| 픽셀 1개라도 | 34/141 | **24.1%** |
| 면적 ≥0.1% | 25/141 | 17.7% |
| 면적 ≥1% | 7/141 | 5.0% |
| 면적 ≥5% | 2/141 | 1.4% |

데이터셋별(≥1% 기준): **HRCDS 15.4% (4/26)**, DamSeg 2.6% (3/115).
최악 사례: HRCDS 9.15% / 8.64% 면적을 박락으로 오인.

→ 박락 없는 이미지 4장 중 1장에서 무언가를 검출. HRCDS류 영상에선 무시 못 할 수준.
재현: `tools/eval_negatives.py` (데이터: `tools/build_spalling_dataset.py --include-negatives --dst data/spalling_with_neg`)

---

## 2. CUBIT 데이터 누수 — 확정, 심각

CUBIT(`spalling512`)은 512 **타일** 데이터다. 1160 타일이 **단 59개 원본 씬**에서 잘렸고
(씬당 평균 ~20장), 빌드 스크립트가 1160 타일을 통째로 섞어 80/10/10 분할한다.

- 씬의 **95%(56/59)** 가 train/val/test에 걸쳐 분포
- **CUBIT test 타일의 100%(116/116)** 가 같은 원본 씬의 다른 타일을 train에도 가짐
- CUBIT은 test set의 **52%(116/225)** → **test의 절반이 학습 중 본 표면의 인접 크롭**

→ 보고된 CUBIT IoU는 거의 무의미하고, 통합 92% IoU도 상당히 부풀려진 값.
DamSeg(640×640 개별 사진, 번호 단위)·HRCDS(자체 split)는 누수 없음 — CUBIT만의 문제.
재현: `tools/build_lodo_folds.py`의 prefix 그룹핑 / 씬 단위 집계 로직.

---

## 3. 데이터셋별 분해 — 재학습 없이 얻은 "정직한 신호"

현재(누수된) 체크포인트의 test 데이터셋별 spalling IoU:

| 데이터셋 | n | IoU | F1 | Precision | Recall | 비고 |
|---|---|---|---|---|---|---|
| CUBIT | 116 | 88.25 | 93.76 | 93.46 | 94.05 | 🔴 누수 (실제론 더 낮음) |
| DamSeg | 35 | 82.78 | 90.58 | 93.80 | 87.57 | ✅ 깨끗 |
| HRCDS | 74 | 95.95 | 97.93 | 97.50 | 98.36 | ✅ 깨끗 |
| **ALL** | 225 | **92.42** | 96.06 | 95.79 | 96.33 | |

- 통합 92.42는 **HRCDS(95.95, 쉬운 영상)** 가 끌어올린 값
- 가장 정직한 "어려운 영상" 신호는 **DamSeg 82.78** (통합치보다 ~10%p 낮음)
- CUBIT 88.25는 누수 inflated → de-leaked 시 더 낮아지고, test 절반을 차지하므로 통합 IoU도 하락

→ **진짜 baseline은 ~92%가 아니라 80%대 중후반**으로 봐야 함. 실증(새 도메인)에선 추가 하락 예상.
재현: `tools/eval_per_dataset.py`

---

## 4. Leave-One-Dataset-Out (LODO) — 진행 중

데이터셋 2개로 학습 → 나머지 1개 전체로 평가. **누수 자동 제거 + 도메인 갭** 동시 측정.
- 모델: ConvNeXt-L, 40k iter, 3 fold(held-out = damseg / hrcds / cubit)
- 데이터: `tools/build_lodo_folds.py` (prefix 필터, 소스 재처리 없음)
- 실행: `tools/run_lodo.sh` (2-GPU DDP, 순차)
- 결과: `results/lodo/metrics_{damseg,hrcds,cubit}.txt` (완료 후 본 문서에 표 추가 예정)

| held-out | n(test) | IoU | F1 | 비고 |
|---|---|---|---|---|
| DamSeg | 366 | _(진행 중)_ | | 도메인 갭 |
| HRCDS | 1020 | _(진행 중)_ | | 도메인 갭 |
| CUBIT | 1160 | _(진행 중)_ | | 누수 제거된 진짜 CUBIT 성능 |

---

## 실증 전 권고

IoU를 더 짜내는 것보다 **평가의 정직성**을 확보하는 게 우선:

1. **CUBIT을 씬 단위로 재분할**(같은 원본 씬은 한 split에만) 후 재학습 → 누수 없는 in-distribution baseline 확정
2. **negative 포함 test를 표준 평가셋으로 고정** → 이미지 단위 오경보율을 상시 추적
3. LODO 수치를 "실증에서 떨어질 폭"의 사전 추정치로 문서화
4. 실증 데이터는 이 baseline을 **fine-tuning 시작점 + 비교 기준**으로만 사용
