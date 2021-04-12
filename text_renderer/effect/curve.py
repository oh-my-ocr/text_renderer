from typing import Tuple

import cv2
import numpy as np
from PIL import Image

from text_renderer.utils.bbox import BBox
from text_renderer.utils.types import PILImage
from .base_effect import Effect


class Curve(Effect):
    def __init__(
        self,
        p=0.5,
        period: float = 180,
        amplitude: Tuple[float, float] = (1, 5),
    ):
        """

        Parameters
        ----------
        p : float
            Probability of apply this effect
        period : float
            in degree
        amplitude : tuple
        """

        super().__init__(p)
        assert amplitude[0] < amplitude[1]
        self.period = period
        self.amplitude = amplitude

    def apply(self, img: PILImage, text_bbox: BBox) -> Tuple[PILImage, BBox]:
        max_val = np.random.uniform(*self.amplitude)

        word_img = np.array(img)
        h, w = word_img.shape[:2]

        img_x = np.zeros((h, w), np.float32)
        img_y = np.zeros((h, w), np.float32)

        xmin = text_bbox.left
        xmax = text_bbox.right
        ymin = text_bbox.top
        ymax = text_bbox.bottom

        remap_y_min = ymin
        remap_y_max = ymax

        for y in range(h):
            for x in range(w):
                remaped_y = y + self._remap_y(x, max_val)

                if y == ymin:
                    if remaped_y < remap_y_min:
                        remap_y_min = remaped_y

                if y == ymax:
                    if remaped_y > remap_y_max:
                        remap_y_max = remaped_y

                img_y[y, x] = remaped_y
                img_x[y, x] = x

        dst = cv2.remap(word_img, img_x, img_y, cv2.INTER_CUBIC)
        bbox = BBox(left=xmin, top=remap_y_min, right=xmax, bottom=remap_y_max)
        bbox = bbox.offset((bbox.left, bbox.top), (0, 0))
        return Image.fromarray(dst), bbox

    def _remap_y(self, x, max_val):
        return int(max_val * np.math.sin(2 * 3.14 * x / self.period))
