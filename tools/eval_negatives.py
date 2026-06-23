"""Image-level false-positive analysis on truly-negative (no-spalling) test images.

Pixel-level FPR dilutes real-world false alarms because negatives add huge
amounts of true-negative background. For a deployment baseline we want:
  - how many negative images trigger ANY spalling prediction
  - how large those false predictions are (area fraction)
  - an image-level false-alarm rate at a few area thresholds
"""
import argparse, glob, os.path as osp
import numpy as np
from PIL import Image
import torch
from mmseg.apis import init_model, inference_model


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
    neg, pos = [], []
    for ip in imgs:
        stem = osp.splitext(osp.basename(ip))[0]
        gt = np.array(Image.open(osp.join(args.gt_dir, stem + '.png')))
        (pos if gt.any() else neg).append(ip)
    print(f'total={len(imgs)}  negatives(no spalling)={len(neg)}  positives={len(pos)}')

    # per-image false-positive area fraction on NEGATIVE images
    fracs, per_ds = [], {}
    for ip in neg:
        res = inference_model(model, ip)
        pred = res.pred_sem_seg.data.squeeze().cpu().numpy()
        frac = float((pred == 1).mean())
        fracs.append((osp.basename(ip), frac))
        ds = osp.basename(ip).split('_')[0]
        per_ds.setdefault(ds, []).append(frac)
    fracs.sort(key=lambda x: -x[1])
    arr = np.array([f for _, f in fracs])

    print('\n=== NEGATIVE-image false positives (n={}) ==='.format(len(neg)))
    for thr, name in [(0.0, '>0 px'), (0.001, '>=0.1% area'),
                      (0.01, '>=1% area'), (0.05, '>=5% area')]:
        k = int((arr > thr).sum()) if thr == 0.0 else int((arr >= thr).sum())
        print(f'  images flagged {name:>12}: {k:3d} / {len(neg)}  '
              f'= {100*k/len(neg):5.1f}%')
    print(f'  mean FP area fraction: {arr.mean()*100:.3f}%   '
          f'max: {arr.max()*100:.2f}%')
    print('\n  per-dataset image-level false-alarm rate (>=1% area):')
    for ds, fs in sorted(per_ds.items()):
        fa = sum(1 for f in fs if f >= 0.01)
        print(f'    {ds:8s}: {fa:3d}/{len(fs):3d} = {100*fa/len(fs):5.1f}%')
    print('\n  worst 10 negative images (FP area fraction):')
    for name, f in fracs[:10]:
        print(f'    {f*100:6.2f}%  {name}')


if __name__ == '__main__':
    main()
