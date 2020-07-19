import random
from typing import Tuple

from text_renderer.utils.bbox import BBox
from text_renderer.utils.types import PILImage

from .base_effect import Effect


class DropoutVertical(Effect):
    def __init__(self, p=0.5, num_line=50):
        """

        Parameters
        ----------
        p : float
            Probability of apply this effect
        num_line : int
            Number of vertical dropout lines
        """
        super().__init__(p)
        self.num_line = num_line

    def apply(self, img: PILImage, text_bbox: BBox) -> Tuple[PILImage, BBox]:
        pim = img.load()

        for _ in range(self.num_line):
            col = random.randint(1, img.width - 1)
            for row in range(img.height):
                self.rand_pick(pim, col, row)

        return img, text_bbox
