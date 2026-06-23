"""Build leave-one-dataset-out (LODO) folds from the existing data/spalling split
by symlink-filtering on dataset prefix (cubit_/damseg_/hrcds_). No source reprocessing.

For held-out dataset H:
  train/ = the OTHER two datasets' train images+masks
  val/   = the OTHER two datasets' val   images+masks  (in-domain model selection)
  test/  = ALL of H (train+val+test)                   (cross-domain evaluation)

Output: data/spalling_lodo_<H>/{train,val,test}/{images,masks}
This measures cross-domain generalization AND is leakage-free (H never seen in training).
"""
import os, os.path as osp, glob

SRC = 'data/spalling'
DATASETS = ['cubit', 'damseg', 'hrcds']


def link(src_path, dst_path):
    if osp.islink(dst_path) or osp.exists(dst_path):
        os.remove(dst_path)
    os.symlink(osp.abspath(src_path), dst_path)


def main():
    for heldout in DATASETS:
        train_on = [d for d in DATASETS if d != heldout]
        root = f'data/spalling_lodo_{heldout}'
        for sp in ('train', 'val', 'test'):
            for sub in ('images', 'masks'):
                os.makedirs(osp.join(root, sp, sub), exist_ok=True)

        counts = {}
        # train/val: the other two datasets, keep their native train/val
        for out_split, src_splits in (('train', ['train']), ('val', ['val'])):
            n = 0
            for src_split in src_splits:
                for pref in train_on:
                    for img in glob.glob(osp.join(SRC, src_split, 'images', f'{pref}_*.jpg')):
                        stem = osp.splitext(osp.basename(img))[0]
                        msk = osp.join(SRC, src_split, 'masks', stem + '.png')
                        link(img, osp.join(root, out_split, 'images', stem + '.jpg'))
                        link(msk, osp.join(root, out_split, 'masks', stem + '.png'))
                        n += 1
            counts[out_split] = n
        # test: ALL of held-out dataset (train+val+test)
        n = 0
        for src_split in ('train', 'val', 'test'):
            for img in glob.glob(osp.join(SRC, src_split, 'images', f'{heldout}_*.jpg')):
                stem = osp.splitext(osp.basename(img))[0]
                msk = osp.join(SRC, src_split, 'masks', stem + '.png')
                link(img, osp.join(root, 'test', 'images', stem + '.jpg'))
                link(msk, osp.join(root, 'test', 'masks', stem + '.png'))
                n += 1
        counts['test'] = n
        print(f'{root}: train={counts["train"]} val={counts["val"]} '
              f'test({heldout}-all)={counts["test"]}')


if __name__ == '__main__':
    main()
