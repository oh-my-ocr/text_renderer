import random
from typing import List

from text_renderer.utils.bbox import BBox
from text_renderer.utils.font_text import FontText
from text_renderer.utils.utils import prob

from .layout import Layout


class ExtraTextLineLayout(Layout):
    """
    Draw extra text line above/under main text. The extra text line will not appear in label
    main text: first corpus
    extra text: second corpus
    """

    def __init__(self, bottom_prob: float = 0.5):
        """

        Parameters
        ----------
        bottom_prob : float
                   Probability of draw extra text under main text line
        """
        assert 0 <= bottom_prob <= 1
        self.bottom_prob = bottom_prob

    def apply(
        self,
        text_bboxes: List[BBox],
        img_bboxes: List[BBox],
    ) -> List[BBox]:
        assert (
            len(text_bboxes) == 2
        ), "ExtraTextLineLayout only support input two text bboxes"

        main_text_mask_bbox, extra_text_mask_bbox = (
            img_bboxes[0],
            img_bboxes[1],
        )

        if prob(1 - self.bottom_prob):
            # above extra line
            main_offset = int(main_text_mask_bbox.height * random.uniform(0.3, 0.4))
            main_text_mask_bbox.offset_(
                main_text_mask_bbox.left_top,
                (
                    main_text_mask_bbox.left_top[0],
                    main_text_mask_bbox.left_top[1] + main_offset,
                ),
            )
            extra_offset = int(random.uniform(0, main_offset // 2))
            extra_text_mask_bbox.offset_(
                extra_text_mask_bbox.left_bottom,
                (
                    main_text_mask_bbox.left_top[0],
                    main_text_mask_bbox.left_top[1] - extra_offset,
                ),
            )
        else:
            # bottom extra line
            extra_text_mask_bbox.offset_(
                extra_text_mask_bbox.left_top, main_text_mask_bbox.left_bottom
            )
            extra_text_mask_bbox.bottom -= int(
                extra_text_mask_bbox.height * random.uniform(0.65, 0.9)
            )

        if extra_text_mask_bbox.width > main_text_mask_bbox.width:
            extra_text_mask_bbox.right -= (
                extra_text_mask_bbox.width - main_text_mask_bbox.width
            )

        return img_bboxes

    def merge_texts(self, font_texts: List[FontText]) -> str:
        return font_texts[0].text
