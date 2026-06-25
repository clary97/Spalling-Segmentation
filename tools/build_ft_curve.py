"""Build a data-efficiency (fine-tuning rehearsal) experiment for DamSeg.

Simulates the upcoming field-data fine-tuning: the LODO model trained WITHOUT DamSeg
is fine-tuned on increasing amounts of DamSeg, then evaluated on a FIXED held-out
DamSeg test set. Answers "how many field labels do we need?".

Deterministic split of all 366 DamSeg images:
  - fixed TEST = 100 images (never used for fine-tuning; same across all N)
  - POOL       = remaining 266 images (fine-tune source)

Output:
  data/ft_damseg/test/{images,masks}          # fixed 100
  data/ft_damseg/n{N}/{train,val,test}/...     # train=N from pool; val=test=fixed 100
  for N in 10,25,50,100,200
(N=0 = no fine-tune; just evaluate the base checkpoint on the fixed test.)
"""
import os, os.path as osp, glob, random

SRC = 'data/spalling'
DST = 'data/ft_damseg'
SEED = 42
N_TEST = 100
SIZES = [10, 25, 50, 100, 200]


def link(src, dst):
    if osp.islink(dst) or osp.exists(dst):
        os.remove(dst)
    os.symlink(osp.abspath(src), dst)


def all_damseg():
    items = []
    for sp in ('train', 'val', 'test'):
        for img in glob.glob(osp.join(SRC, sp, 'images', 'damseg_*.jpg')):
            stem = osp.splitext(osp.basename(img))[0]
            msk = osp.join(SRC, sp, 'masks', stem + '.png')
            if osp.exists(msk):
                items.append((stem, img, msk))
    return sorted(items)


def write_set(root, split, items):
    for sub in ('images', 'masks'):
        os.makedirs(osp.join(root, split, sub), exist_ok=True)
    for stem, img, msk in items:
        link(img, osp.join(root, split, 'images', stem + '.jpg'))
        link(msk, osp.join(root, split, 'masks', stem + '.png'))


def main():
    items = all_damseg()
    rng = random.Random(SEED)
    rng.shuffle(items)
    test, pool = items[:N_TEST], items[N_TEST:]
    print(f'total damseg={len(items)}  fixed_test={len(test)}  pool={len(pool)}')

    write_set(DST, 'test', test)          # shared fixed test
    for N in SIZES:
        root = osp.join(DST, f'n{N}')
        train = pool[:N]
        write_set(root, 'train', train)
        write_set(root, 'val', test)      # monitor only; final eval uses fixed test
        write_set(root, 'test', test)
        print(f'  n{N}: train={len(train)} val/test={len(test)}')
    # also expose the fixed test as n0 (for base-ckpt eval convenience)
    write_set(osp.join(DST, 'n0'), 'test', test)
    print('done.')


if __name__ == '__main__':
    main()
