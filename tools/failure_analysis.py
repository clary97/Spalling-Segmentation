"""Qualitative failure analysis for a LODO (cross-domain) model.

For a model + held-out test set, compute per-image spalling IoU, rank, and:
  - print worst-K and best-K with FP/FN area fractions (over/under-detection signal)
  - save worst-K side-by-side panels (original | GT-overlay | pred-overlay) for eyeballing

Tells us WHAT kinds of cross-domain images break, to prioritize field-data collection.
"""
import argparse, glob, os, os.path as osp
import numpy as np
from PIL import Image
from mmseg.apis import init_model, inference_model


def overlay(img, mask, color):
    out = img.copy().astype(np.float32)
    m = mask.astype(bool)
    for c in range(3):
        out[..., c][m] = 0.5 * out[..., c][m] + 0.5 * color[c]
    return out.astype(np.uint8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    ap.add_argument('--checkpoint', required=True)
    ap.add_argument('--img-dir', required=True)
    ap.add_argument('--gt-dir', required=True)
    ap.add_argument('--out', required=True, help='dir to save worst-K panels')
    ap.add_argument('--topk', type=int, default=8)
    ap.add_argument('--device', default='cuda:0')
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    model = init_model(args.config, args.checkpoint, device=args.device)
    imgs = sorted(glob.glob(osp.join(args.img_dir, '*.jpg')))

    rows = []  # (iou, stem, fp_frac, fn_frac, gt_frac)
    for ip in imgs:
        stem = osp.splitext(osp.basename(ip))[0]
        gt = np.array(Image.open(osp.join(args.gt_dir, stem + '.png'))) == 1
        res = inference_model(model, ip)
        pred = res.pred_sem_seg.data.squeeze().cpu().numpy() == 1
        tp = (pred & gt).sum(); fp = (pred & ~gt).sum(); fn = (~pred & gt).sum()
        iou = tp / (tp + fp + fn) if (tp + fp + fn) else 1.0
        npx = gt.size
        rows.append((float(iou), stem, ip, fp / npx, fn / npx, gt.mean()))
    rows.sort(key=lambda r: r[0])

    ious = np.array([r[0] for r in rows])
    print(f'\n=== {osp.basename(args.out)}: {len(rows)} imgs | '
          f'mean per-img IoU {ious.mean()*100:.1f} | '
          f'median {np.median(ious)*100:.1f} | '
          f'<0.5 IoU: {(ious<0.5).sum()} imgs ({100*(ious<0.5).mean():.0f}%) ===')
    print('worst {}: (IoU | FPfrac | FNfrac | GTfrac | name)'.format(args.topk))
    for iou, stem, ip, fpf, fnf, gtf in rows[:args.topk]:
        flag = 'OVER-detect' if fpf > fnf else 'UNDER-detect'
        print(f'  {iou*100:5.1f} | {fpf*100:5.2f} | {fnf*100:5.2f} | {gtf*100:5.2f} | '
              f'{flag:12s} | {stem}')

    # save worst-K panels
    for rank, (iou, stem, ip, fpf, fnf, gtf) in enumerate(rows[:args.topk]):
        img = np.array(Image.open(ip).convert('RGB'))
        gt = np.array(Image.open(osp.join(args.gt_dir, stem + '.png'))) == 1
        res = inference_model(model, ip)
        pred = res.pred_sem_seg.data.squeeze().cpu().numpy() == 1
        if gt.shape != img.shape[:2]:
            gt = np.array(Image.fromarray(gt.astype(np.uint8)).resize(
                (img.shape[1], img.shape[0]), Image.NEAREST)).astype(bool)
        if pred.shape != img.shape[:2]:
            pred = np.array(Image.fromarray(pred.astype(np.uint8)).resize(
                (img.shape[1], img.shape[0]), Image.NEAREST)).astype(bool)
        panel = np.concatenate(
            [img, overlay(img, gt, (0, 255, 0)), overlay(img, pred, (255, 0, 0))], axis=1)
        Image.fromarray(panel).save(
            osp.join(args.out, f'worst{rank:02d}_iou{int(iou*100):02d}_{stem}.jpg'))
    print(f'saved {min(args.topk, len(rows))} panels -> {args.out} '
          f'(left=orig, mid=GT green, right=pred red)')


if __name__ == '__main__':
    main()
