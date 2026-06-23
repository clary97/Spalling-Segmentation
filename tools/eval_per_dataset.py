"""Per-dataset spalling metrics on the (positives) test set.

HRCDS (native split) and DamSeg (per-image random split) are leakage-free,
so their numbers are honest in-distribution performance. CUBIT is tile-leaked
(100% of test tiles share a source scene with training tiles), so its number
is inflated. The CUBIT-vs-HRCDS gap shows the size of the leakage bubble.
"""
import argparse, glob, os.path as osp
import numpy as np
from PIL import Image
from mmseg.apis import init_model, inference_model


def stats_for(tp, fp, fn):
    iou = tp / (tp + fp + fn) if (tp + fp + fn) else float('nan')
    prec = tp / (tp + fp) if (tp + fp) else float('nan')
    rec = tp / (tp + fn) if (tp + fn) else float('nan')
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else float('nan')
    return iou, f1, prec, rec


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

    # accumulate pixel confusion per dataset prefix + overall
    acc = {}  # ds -> [tp, fp, fn, n]
    for ip in imgs:
        stem = osp.splitext(osp.basename(ip))[0]
        ds = stem.split('_')[0]
        gt = np.array(Image.open(osp.join(args.gt_dir, stem + '.png'))) == 1
        res = inference_model(model, ip)
        pred = res.pred_sem_seg.data.squeeze().cpu().numpy() == 1
        tp = int((pred & gt).sum()); fp = int((pred & ~gt).sum()); fn = int((~pred & gt).sum())
        for key in (ds, '__all__'):
            a = acc.setdefault(key, [0, 0, 0, 0])
            a[0] += tp; a[1] += fp; a[2] += fn; a[3] += 1

    print(f'\nPer-dataset spalling metrics (test positives, n={len(imgs)})')
    print('+----------+-----+-------+-------+-----------+--------+')
    print('| dataset  |  n  |  IoU  |   F1  | Precision | Recall |')
    print('+----------+-----+-------+-------+-----------+--------+')
    order = [k for k in ('cubit', 'damseg', 'hrcds') if k in acc] + ['__all__']
    for k in order:
        tp, fp, fn, n = acc[k]
        iou, f1, prec, rec = stats_for(tp, fp, fn)
        name = 'ALL' if k == '__all__' else k
        tag = '  <- LEAKED' if k == 'cubit' else ('  <- clean' if k in ('hrcds', 'damseg') else '')
        print(f'| {name:8s} | {n:3d} | {iou*100:5.2f} | {f1*100:5.2f} |   {prec*100:5.2f}   | '
              f'{rec*100:5.2f} |{tag}')
    print('+----------+-----+-------+-------+-----------+--------+')


if __name__ == '__main__':
    main()
