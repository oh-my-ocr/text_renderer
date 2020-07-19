import random
from typing import Tuple

from text_renderer.utils.bbox import BBox
from text_renderer.utils.types import PILImage
from .base_effect import Effect


class DropoutHorizontal(Effect):
    def __init__(self, p=0.5, num_line=5):
        """

        Parameters
        ----------
        p : float
            Probability of apply this effect
        num_line : int
            Number of horizontal dropout lines

        """
        super().__init__(p)
        self.num_line = num_line

    def apply(self, img: PILImage, text_bbox: BBox) -> Tuple[PILImage, BBox]:
        pim = img.load()

        for _ in range(self.num_line):
            row = random.randint(1, img.height - 1)
            for col in range(img.width):
                self.rand_pick(pim, col, row)
        return img, text_bbox
