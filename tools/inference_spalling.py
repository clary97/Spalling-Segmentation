"""
UPerNet+ConvNeXt로 학습한 efflorescence 모델 인퍼런스.

단일 이미지, 폴더, val/test 세트 평가 모두 지원.

Usage:
    # 1) 단일 이미지
    python tools/inference_dacl_efflorescence.py \
        --config configs/convnext/convnext-base_upernet_40k_dacl-efflorescence-512x512.py \
        --checkpoint work_dirs/upernet_convnext-base_dacl_efflorescence/iter_40000.pth \
        --input /path/to/image.jpg \
        --output results/

    # 2) 폴더 내 모든 이미지
    python tools/inference_dacl_efflorescence.py \
        --config configs/convnext/convnext-base_upernet_40k_dacl-efflorescence-512x512.py \
        --checkpoint work_dirs/upernet_convnext-base_dacl_efflorescence/iter_40000.pth \
        --input /path/to/folder \
        --output results/

    # 3) GT 마스크와 비교 평가 (메트릭 계산)
    python tools/inference_dacl_efflorescence.py \
        --config configs/convnext/convnext-base_upernet_40k_dacl-efflorescence-512x512.py \
        --checkpoint work_dirs/upernet_convnext-base_dacl_efflorescence/iter_40000.pth \
        --input /path/to/images \
        --gt-dir /path/to/masks \
        --output results/ \
        --compute-metrics
"""

import argparse
import os
import os.path as osp

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from mmengine.model import revert_sync_batchnorm
from prettytable import PrettyTable

from mmseg.apis import init_model, inference_model

# 클래스 색상 (binary: background, efflorescence)
PALETTE = np.array([
    [0, 0, 0],        # background
    [255, 0, 0],      # efflorescence (red)
], dtype=np.uint8)

CLASS_NAMES = ('background', 'Efflorescence')


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True, help='Config file path')
    parser.add_argument('--checkpoint', required=True, help='Checkpoint path')
    parser.add_argument('--input', required=True,
                        help='Input image or folder path')
    parser.add_argument('--output', default='inference_results',
                        help='Output directory')
    parser.add_argument('--gt-dir', default=None,
                        help='Optional: GT mask directory (for metrics)')
    parser.add_argument('--compute-metrics', action='store_true',
                        help='Compute IoU/F1/TPR/FPR (requires --gt-dir)')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--opacity', type=float, default=0.5,
                        help='Overlay opacity (0~1)')
    parser.add_argument('--save-binary-mask', action='store_true',
                        help='Also save raw binary masks')
    return parser.parse_args()


def collect_images(input_path):
    """단일 이미지 또는 폴더에서 이미지 목록 수집."""
    if osp.isfile(input_path):
        return [input_path]
    elif osp.isdir(input_path):
        images = []
        for f in sorted(os.listdir(input_path)):
            if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                images.append(osp.join(input_path, f))
        return images
    else:
        raise FileNotFoundError(f'Input not found: {input_path}')


def visualize(img_bgr, pred, opacity=0.5):
    """이미지 위에 예측 mask 시각화."""
    h, w = img_bgr.shape[:2]
    color_mask = PALETTE[pred]  # (H, W, 3) RGB
    color_mask_bgr = cv2.cvtColor(color_mask, cv2.COLOR_RGB2BGR)

    # alpha blending only where pred != 0
    mask_3 = (pred > 0)[:, :, None]
    overlay = img_bgr.copy()
    overlay = np.where(
        mask_3,
        (img_bgr.astype(np.float32) * (1 - opacity) +
         color_mask_bgr.astype(np.float32) * opacity).astype(np.uint8),
        img_bgr,
    )
    return overlay


def make_side_by_side(img, gt, pred, opacity=0.5):
    """[원본] [GT(있으면)] [예측] 가로로 붙임."""
    overlay_pred = visualize(img, pred, opacity)
    images = [img, overlay_pred]
    labels = ['Original', 'Prediction']

    if gt is not None:
        overlay_gt = visualize(img, gt, opacity)
        images.insert(1, overlay_gt)
        labels.insert(1, 'GT')

    # text 표시
    font = cv2.FONT_HERSHEY_SIMPLEX
    for img_, label in zip(images, labels):
        cv2.putText(img_, label, (10, 30), font, 0.8, (255, 255, 255), 2)

    return np.concatenate(images, axis=1)


def compute_metrics_single(pred, gt):
    """Binary segmentation per-pixel metrics."""
    tp = int(((pred == 1) & (gt == 1)).sum())
    fp = int(((pred == 1) & (gt == 0)).sum())
    fn = int(((pred == 0) & (gt == 1)).sum())
    tn = int(((pred == 0) & (gt == 0)).sum())
    return tp, fp, fn, tn


def main():
    args = parse_args()

    # 모델 로드
    print(f'Loading model from {args.checkpoint}...')
    model = init_model(args.config, args.checkpoint, device=args.device)
    if args.device == 'cpu':
        model = revert_sync_batchnorm(model)

    # 이미지 수집
    images = collect_images(args.input)
    print(f'Found {len(images)} images.')

    # 출력 폴더 준비
    os.makedirs(args.output, exist_ok=True)
    mask_dir = osp.join(args.output, 'masks')
    if args.save_binary_mask:
        os.makedirs(mask_dir, exist_ok=True)

    # 메트릭 누적용
    total_tp = total_fp = total_fn = total_tn = 0
    valid_with_gt = 0

    for i, img_path in enumerate(images):
        if (i + 1) % 20 == 0 or (i + 1) == len(images):
            print(f'  [{i+1}/{len(images)}] {osp.basename(img_path)}')

        # 추론
        result = inference_model(model, img_path)
        pred = result.pred_sem_seg.data[0].cpu().numpy().astype(np.uint8)

        # 원본 이미지
        img_bgr = cv2.imread(img_path)
        if img_bgr.shape[:2] != pred.shape:
            img_bgr = cv2.resize(img_bgr,
                                  (pred.shape[1], pred.shape[0]))

        # GT 로드 (옵션)
        gt = None
        if args.gt_dir is not None:
            stem = osp.splitext(osp.basename(img_path))[0]
            gt_path = osp.join(args.gt_dir, stem + '.png')
            if osp.exists(gt_path):
                gt = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
                # 0/255 → 0/1 정규화
                if gt.max() > 1:
                    gt = (gt > 0).astype(np.uint8)
                if gt.shape != pred.shape:
                    gt = cv2.resize(gt, (pred.shape[1], pred.shape[0]),
                                    interpolation=cv2.INTER_NEAREST)
                if args.compute_metrics:
                    tp, fp, fn, tn = compute_metrics_single(pred, gt)
                    total_tp += tp
                    total_fp += fp
                    total_fn += fn
                    total_tn += tn
                    valid_with_gt += 1

        # 시각화 저장
        out_vis = make_side_by_side(img_bgr.copy(), gt, pred, args.opacity)
        stem = osp.splitext(osp.basename(img_path))[0]
        cv2.imwrite(osp.join(args.output, f'{stem}_compare.jpg'), out_vis)

        # binary mask 저장 (옵션)
        if args.save_binary_mask:
            cv2.imwrite(osp.join(mask_dir, f'{stem}.png'),
                        (pred * 255).astype(np.uint8))

    print(f'\n✓ Saved {len(images)} comparison images to: {args.output}')

    # 전체 메트릭 출력
    if args.compute_metrics and valid_with_gt > 0:
        print('\n' + '=' * 60)
        print(f'Metrics on {valid_with_gt} images (with GT)')
        print('=' * 60)

        # per-class
        precision_d = total_tp / max(total_tp + total_fp, 1)
        recall_d = total_tp / max(total_tp + total_fn, 1)
        iou_d = total_tp / max(total_tp + total_fp + total_fn, 1)
        f1_d = 2 * precision_d * recall_d / max(precision_d + recall_d, 1e-10)
        fpr_d = total_fp / max(total_fp + total_tn, 1)

        precision_b = total_tn / max(total_tn + total_fn, 1)
        recall_b = total_tn / max(total_tn + total_fp, 1)
        iou_b = total_tn / max(total_tn + total_fp + total_fn, 1)
        f1_b = 2 * precision_b * recall_b / max(precision_b + recall_b, 1e-10)

        table = PrettyTable()
        table.field_names = ['Class', 'IoU', 'F1', 'Precision', 'Recall', 'FPR']
        table.add_row(['background',
                       f'{iou_b*100:.2f}', f'{f1_b*100:.2f}',
                       f'{precision_b*100:.2f}', f'{recall_b*100:.2f}', '-'])
        table.add_row(['Efflorescence',
                       f'{iou_d*100:.2f}', f'{f1_d*100:.2f}',
                       f'{precision_d*100:.2f}', f'{recall_d*100:.2f}',
                       f'{fpr_d*100:.2f}'])
        print(table)

        mean_iou = (iou_b + iou_d) / 2
        mean_f1 = (f1_b + f1_d) / 2
        print(f'\nmIoU: {mean_iou*100:.2f}')
        print(f'mF1:  {mean_f1*100:.2f}')

        # 결과 저장
        metrics_file = osp.join(args.output, 'metrics.txt')
        with open(metrics_file, 'w') as fout:
            fout.write(f'Config:     {args.config}\n')
            fout.write(f'Checkpoint: {args.checkpoint}\n')
            fout.write(f'Images:     {valid_with_gt}\n\n')
            fout.write(str(table) + '\n')
            fout.write(f'\nmIoU: {mean_iou*100:.2f}\n')
            fout.write(f'mF1:  {mean_f1*100:.2f}\n')
        print(f'\nMetrics saved to: {metrics_file}')


if __name__ == '__main__':
    main()
