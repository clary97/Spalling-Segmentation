from mmseg.registry import DATASETS
from mmseg.datasets import BaseSegDataset


@DATASETS.register_module()
class SpallingDataset(BaseSegDataset):
    """Unified binary *spalling* dataset (CUBIT_Seg + DamSeg + HRCDS).

    2 classes:
        0: background
        1: spalling
    """
    METAINFO = dict(
        classes=('background', 'spalling'),
        palette=[[0, 0, 0], [255, 0, 0]])

    def __init__(self,
                 img_suffix='.jpg',
                 seg_map_suffix='.png',
                 reduce_zero_label=False,
                 **kwargs) -> None:
        super().__init__(
            img_suffix=img_suffix,
            seg_map_suffix=seg_map_suffix,
            reduce_zero_label=reduce_zero_label,
            **kwargs)
