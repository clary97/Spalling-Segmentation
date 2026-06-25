"""Operating-point (decision-threshold) sweep for binary spalling.

The model outputs 2-class logits; spalling probability = softmax[...,1]. Default prediction
uses argmax (== threshold 0.5). This sweeps the threshold to expose the precision/recall and
false-alarm tradeoff, so a deployment operating point can be chosen (e.g. fewer false alarms
on spalling-free images at the cost of some recall).

Reports, per threshold:
  - pixel IoU / precision / recall / F1 (over all images)
  - image-level false-alarm rate on truly-negative images (>=1% predicted spalling area)
Run on the negatives-included test set to make the false-alarm column meaningful.
"""
import argparse, glob, os.path as osp
import numpy as np
import torch, torch.nn.functional as F
from PIL import Image
from mmseg.apis import init_model, inference_model

THRESHOLDS = [0.3, 0.5, 0.7, 0.9, 0.95, 0.99]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    ap.add_argument('--checkpoint', required=True)
    ap.add_argument('--img-dir', required=True)
    ap.add_argument('--gt-dir', required=True)
    ap.add_argument('--device', default='cuda:0')
    args = ap.parse_args()

    model = init_model(args.config, args.checkpoint, device=args.device)
    imgs = sorted(glob.glob(osp.join(args.img_dir, '*.jpg')))

    # per-threshold pixel confusion + per-threshold negative-image false alarms
    T = len(THRESHOLDS)
    tp = np.zeros(T); fp = np.zeros(T); fn = np.zeros(T)
    neg_total = 0
    neg_falarm = np.zeros(T)  # negative images flagged (>=1% area) at each threshold

    for ip in imgs:
        stem = osp.splitext(osp.basename(ip))[0]
        gt = np.array(Image.open(osp.join(args.gt_dir, stem + '.png'))) == 1
        res = inference_model(model, ip)
        logits = res.seg_logits.data  # (2,H,W) tensor on device
        prob1 = F.softmax(logits, dim=0)[1].cpu().numpy()  # spalling prob
        if prob1.shape != gt.shape:
            gt = np.array(Image.fromarray(gt.astype(np.uint8)).resize(
                (prob1.shape[1], prob1.shape[0]), Image.NEAREST)).astype(bool)
        is_neg = not gt.any()
        if is_neg:
            neg_total += 1
        for i, thr in enumerate(THRESHOLDS):
            pred = prob1 >= thr
            tp[i] += (pred & gt).sum(); fp[i] += (pred & ~gt).sum(); fn[i] += (~pred & gt).sum()
            if is_neg and pred.mean() >= 0.01:
                neg_falarm[i] += 1

    print(f'\nThreshold sweep on {len(imgs)} imgs (negatives={neg_total})')
    print('+-------+-------+-----------+--------+-------+----------------------+')
    print('| thr   |  IoU  | Precision | Recall |  F1   | neg img false-alarm  |')
    print('+-------+-------+-----------+--------+-------+----------------------+')
    for i, thr in enumerate(THRESHOLDS):
        iou = tp[i] / (tp[i] + fp[i] + fn[i])
        prec = tp[i] / (tp[i] + fp[i]) if (tp[i] + fp[i]) else 0
        rec = tp[i] / (tp[i] + fn[i]) if (tp[i] + fn[i]) else 0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0
        fa = (100 * neg_falarm[i] / neg_total) if neg_total else float('nan')
        print(f'| {thr:<5.2f} | {iou*100:5.2f} |   {prec*100:5.2f}   | {rec*100:5.2f}  | '
              f'{f1*100:5.2f} | {int(neg_falarm[i]):3d}/{neg_total} = {fa:5.1f}%      |')
    print('+-------+-------+-----------+--------+-------+----------------------+')
    print('(thr 0.5 == default argmax prediction)')


if __name__ == '__main__':
    main()
