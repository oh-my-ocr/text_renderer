import random
from typing import Tuple

from text_renderer.utils.bbox import BBox
from text_renderer.utils.types import PILImage

from .base_effect import Effect


class DropoutVertical(Effect):
    def __init__(self, p=0.5, num_line=8, thickness: int = 3):
        """

        Parameters
        ----------
        p : float
            Probability of apply this effect
        num_line : int
            Number of vertical dropout lines
        thickness: int
        """
        super().__init__(p)
        self.num_line = num_line
        self.thickness = thickness

    def apply(self, img: PILImage, text_bbox: BBox) -> Tuple[PILImage, BBox]:
        # Check if there's enough space for the dropout effect
        if img.width <= self.thickness + 1:
            # Not enough space for dropout, return original image
            return img, text_bbox
            
        pim = img.load()

        for _ in range(self.num_line):
            col = random.randint(1, img.width - self.thickness - 1)
            for i in range(self.thickness):
                for row in range(img.height):
                    self.fix_pick(pim, col + i, row, (0, 20))

        return img, text_bbox
