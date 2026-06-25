"""Build a unified binary *spalling* segmentation dataset from CUBIT_Seg, DamSeg, HRCDS.

spalling -> 1, everything else -> 0.

Per-dataset spalling encoding (verified):
  - CUBIT_Seg/spalling512 : RGB mask, spalling = (128,0,0)
  - DamSeg                : RGB mask, spalling = (0,0,255) blue  (category_id 1)
  - HRCDS                 : index mask, spalling = index 2

Output layout (mmseg-friendly):
  data/spalling/{train,val,test}/images/<stem>.jpg   (symlink to source, any codec)
  data/spalling/{train,val,test}/masks/<stem>.png    (uint8, values {0,1})

Splits: HRCDS keeps its native train/val/test; CUBIT & DamSeg are split 80/10/10
(deterministic, seeded). Image stems are sanitized + dataset-prefixed to avoid
collisions / spaces.

Modes:
  default            : keep only images containing >=1 spalling pixel (spalling-only)
  --include-negatives: keep ALL images (non-spalling -> all background)
"""
import argparse, os, os.path as osp, glob, random, re
import numpy as np
from PIL import Image

SRC = '/mnt/nas_200'
DST = 'data/spalling'
SEED = 42


def sanitize(s):
    return re.sub(r'[^A-Za-z0-9]+', '_', s).strip('_')


def ensure_dirs():
    for sp in ('train', 'val', 'test'):
        for sub in ('images', 'masks'):
            os.makedirs(osp.join(DST, sp, sub), exist_ok=True)


def write_pair(split, stem, img_src, spalling_bool, include_neg, stats):
    """Symlink image + write binary mask if it qualifies. Returns True if written."""
    has = bool(spalling_bool.any())
    if not has and not include_neg:
        stats['skipped_no_spalling'] += 1
        return False
    img_link = osp.join(DST, split, 'images', stem + '.jpg')
    msk_path = osp.join(DST, split, 'masks', stem + '.png')
    if osp.islink(img_link) or osp.exists(img_link):
        os.remove(img_link)
    os.symlink(img_src, img_link)
    Image.fromarray(spalling_bool.astype(np.uint8)).save(msk_path)  # values {0,1}
    stats['written'] += 1
    stats['with_spalling'] += int(has)
    return True


def split_of(idx, n, ratios=(0.8, 0.1, 0.1)):
    # deterministic split by shuffled index
    return None  # unused; we precompute below


# ---------------- CUBIT ----------------
def proc_cubit(include_neg, stats, scene_split=False):
    rgb_dir = f'{SRC}/CUBIT_Seg/spalling512/outputs_RGB'
    msk_dir = f'{SRC}/CUBIT_Seg/spalling512/outputs_Mask'
    # use flat top-level files (canonical set)
    masks = sorted(glob.glob(osp.join(msk_dir, '*.png')))
    items = []
    for m in masks:
        stem = osp.splitext(osp.basename(m))[0]
        img = osp.join(rgb_dir, stem + '.jpg')
        if osp.exists(img):
            items.append((stem, img, m))

    if scene_split:
        # CUBIT tiles are 512 crops of ~59 source scenes (e.g. "1_10_38" -> scene "1_10").
        # Split by SCENE so all tiles of one source photo land in the same split (no leakage).
        scenes = {}
        for it in items:
            scene = '_'.join(it[0].split('_')[:2])
            scenes.setdefault(scene, []).append(it)
        keys = sorted(scenes)
        rng = random.Random(SEED)
        rng.shuffle(keys)
        nsc = len(keys)
        ntr, nva = int(nsc * 0.8), int(nsc * 0.1)
        assign = {}
        for i, k in enumerate(keys):
            assign[k] = 'train' if i < ntr else ('val' if i < ntr + nva else 'test')
        ordered = [(assign['_'.join(it[0].split('_')[:2])], it) for it in items]
    else:
        rng = random.Random(SEED)
        rng.shuffle(items)
        n = len(items)
        ntr, nva = int(n * 0.8), int(n * 0.1)
        ordered = [('train' if i < ntr else ('val' if i < ntr + nva else 'test'), it)
                   for i, it in enumerate(items)]

    for split, (stem, img, m) in ordered:
        arr = np.array(Image.open(m).convert('RGB'))
        sp = (arr[..., 0] == 128) & (arr[..., 1] == 0) & (arr[..., 2] == 0)
        write_pair(split, 'cubit_' + sanitize(stem), img, sp, include_neg, stats)


# ---------------- DamSeg ----------------
def proc_damseg(include_neg, stats):
    items = []
    for level in ('Easy', 'Medium', 'Hard'):
        for m in sorted(glob.glob(f'{SRC}/DamSeg/{level}/Labels/Mask/*_mask.png')):
            stem = osp.basename(m)[:-len('_mask.png')]
            img = f'{SRC}/DamSeg/{level}/Images/{stem}.jpg'
            if osp.exists(img):
                items.append((f'{level}_{stem}', img, m))
    rng = random.Random(SEED)
    rng.shuffle(items)
    n = len(items)
    ntr, nva = int(n * 0.8), int(n * 0.1)
    for i, (stem, img, m) in enumerate(items):
        split = 'train' if i < ntr else ('val' if i < ntr + nva else 'test')
        arr = np.array(Image.open(m).convert('RGB'))
        sp = (arr[..., 0] == 0) & (arr[..., 1] == 0) & (arr[..., 2] == 255)
        write_pair(split, 'damseg_' + sanitize(stem), img, sp, include_neg, stats)


# ---------------- HRCDS ----------------
def proc_hrcds(include_neg, stats):
    for split in ('train', 'val', 'test'):
        for img in sorted(glob.glob(f'{SRC}/HRCDS/{split}_image/*.jpg')):
            stem = osp.splitext(osp.basename(img))[0]
            m = f'{SRC}/HRCDS/{split}_mask/{stem}_mask.png'
            if not osp.exists(m):
                continue
            arr = np.array(Image.open(m))
            sp = (arr == 2)
            write_pair(split, 'hrcds_' + sanitize(stem), img, sp, include_neg, stats)


def main():
    global DST
    ap = argparse.ArgumentParser()
    ap.add_argument('--include-negatives', action='store_true',
                    help='keep all images (non-spalling -> background). Default: spalling-only.')
    ap.add_argument('--dst', default=DST,
                    help='output root (default: data/spalling). Use a separate dir to avoid '
                         'overwriting the baseline dataset.')
    ap.add_argument('--cubit-scene-split', action='store_true',
                    help='split CUBIT by source scene (no tile leakage across train/val/test).')
    args = ap.parse_args()
    DST = args.dst
    ensure_dirs()
    stats = dict(written=0, with_spalling=0, skipped_no_spalling=0)
    proc_cubit(args.include_negatives, stats, scene_split=args.cubit_scene_split)
    proc_damseg(args.include_negatives, stats)
    proc_hrcds(args.include_negatives, stats)

    print('=== build done ===')
    print('mode:', 'INCLUDE-NEGATIVES' if args.include_negatives else 'SPALLING-ONLY')
    print(stats)
    for sp in ('train', 'val', 'test'):
        ni = len(os.listdir(osp.join(DST, sp, 'images')))
        nm = len(os.listdir(osp.join(DST, sp, 'masks')))
        print(f'  {sp}: images={ni} masks={nm}')


if __name__ == '__main__':
    main()
