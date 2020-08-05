from typing import Tuple

import numpy as np

from text_renderer.utils.bbox import BBox
from text_renderer.utils.draw_utils import transparent_img
from text_renderer.utils.types import PILImage
from text_renderer.utils.utils import random_xy_offset

from .base_effect import Effect


class Padding(Effect):
    def __init__(
        self, p=0.5, w_ratio=(0.0, 0.05), h_ratio=(0.0, 0.3),
    ):
        """

        Parameters
        ----------
        p : float
            Probability of apply this effect
        w_ratio : tuple
            [min_ratio, max_ratio)
        h_ratio : tuple
            [min_ratio, max_ratio)
        """

        super().__init__(p)
        self.w_ratio = w_ratio
        self.h_ratio = h_ratio

    def apply(self, img: PILImage, text_bbox: BBox) -> Tuple[PILImage, BBox]:
        w_ratio = np.random.uniform(*self.w_ratio)
        h_ratio = np.random.uniform(*self.h_ratio)
        new_w = int(img.width + img.width * w_ratio)
        new_h = int(img.height + img.height * h_ratio)

        new_img = transparent_img((new_w, new_h))
        xy = random_xy_offset(img.size, (new_w, new_h))
        new_img.paste(img, xy)

        new_bbox = text_bbox.move_origin(xy)
        return new_img, new_bbox
