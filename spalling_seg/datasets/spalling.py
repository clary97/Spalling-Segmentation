from mmseg.registry import DATASETS
from mmseg.datasets import BaseSegDataset


# force=True: 동일 이름이 이미 등록돼 있어도(예: mmseg 소스에 in-tree 복사본이
# 있는 개발 포크 환경) 안전하게 override. 깨끗한 pip-install 환경에선 override 대상이
# 없으므로 일반 등록과 동일하게 동작한다.
@DATASETS.register_module(force=True)
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
