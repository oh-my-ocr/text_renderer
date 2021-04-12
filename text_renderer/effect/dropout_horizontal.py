import random
from typing import Tuple

from text_renderer.utils.bbox import BBox
from text_renderer.utils.types import PILImage
from .base_effect import Effect


class DropoutHorizontal(Effect):
    def __init__(self, p=0.5, num_line=3, thickness: int = 3):
        """

        Parameters
        ----------
        p : float
            Probability of apply this effect
        num_line : int
            Number of horizontal dropout lines
        thickness: int
        """
        super().__init__(p)
        self.num_line = num_line
        self.thickness = thickness

    def apply(self, img: PILImage, text_bbox: BBox) -> Tuple[PILImage, BBox]:
        pim = img.load()

        for _ in range(self.num_line):
            row = random.randint(1, img.height - self.thickness - 1)
            for i in range(self.thickness):
                for col in range(img.width):
                    self.fix_pick(pim, col, row + i, (0, 20))

        return img, text_bbox
