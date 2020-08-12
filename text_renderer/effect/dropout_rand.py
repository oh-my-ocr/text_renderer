import random
from typing import Tuple

import numpy as np

from text_renderer.utils.bbox import BBox
from text_renderer.utils.types import PILImage

from .base_effect import Effect


class DropoutRand(Effect):
    def __init__(self, p=0.5, dropout_p=(0.2, 0.4)):
        """

        Parameters
        ----------
        p : float
            Probability of apply this effect
        dropout_p : Tuple[float, float]
            The percentage range of pixels to be discarded

        """
        super().__init__(p)
        self.dropout_p = dropout_p

    def apply(self, img: PILImage, text_bbox: BBox) -> Tuple[PILImage, BBox]:
        pim = img.load()

        alpha_channel = np.array(img).astype(np.uint8)[:, :, 3]
        nonzero_idxes = np.argwhere(alpha_channel != 0)

        nonzero_count = nonzero_idxes.shape[0]
        random_dropout_count = random.randint(
            int(nonzero_count * self.dropout_p[0]),
            int(nonzero_count * self.dropout_p[1]),
        )
        shuffled = np.random.permutation(nonzero_count)
        shuffled = shuffled[:random_dropout_count]

        for i in shuffled:
            y, x = nonzero_idxes[i]
            col = int(x)
            row = int(y)
            self.rand_pick(pim, col, row)

        return img, text_bbox
