from typing import List, Tuple

import numpy as np

from text_renderer.utils.bbox import BBox

from .layout import Layout
from ..utils import FontText


class SameLineLayout(Layout):
    """
    Draw multiple texts horizontally at random intervals.

    TODO: add vertical offset
    """

    def __init__(self, h_spacing: Tuple[float, float] = (0, 1)):
        """
        Args:
            h_spacing (float, float): Horizontal spacing between each bbox. scale * average height of text
        """
        self.h_spacing = h_spacing

    def apply(self, text_bboxes: List[BBox], img_bboxes: List[BBox],) -> List[BBox]:
        avg_height = sum([it.height for it in img_bboxes]) / len(img_bboxes)

        for i in range(0, len(img_bboxes) - 1):
            h_spacing_scale = np.random.uniform(*self.h_spacing)
            h_spacing = int(avg_height * h_spacing_scale)
            img_bboxes[i].right += h_spacing

        merged_bbox = BBox.from_bboxes(img_bboxes)

        img_bboxes[0].offset_(img_bboxes[0].left_cnt, merged_bbox.left_cnt)

        for i in range(1, len(img_bboxes)):
            img_bboxes[i].offset_(img_bboxes[i].left_cnt, img_bboxes[i - 1].right_cnt)

        return img_bboxes
