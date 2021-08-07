from typing import Tuple

from PIL import Image
from imgaug.augmenters import Augmenter
import numpy as np

from text_renderer.utils.bbox import BBox
from text_renderer.utils.types import PILImage

from .base_effect import Effect


class ImgAugEffect(Effect):
    """
    Apply imgaug(https://github.com/aleju/imgaug) Augmenter on image.
    """
    def __init__(self, p=1.0, aug: Augmenter = None):
        super().__init__(p)
        self.aug = aug

    def apply(self, img: PILImage, text_bbox: BBox) -> Tuple[PILImage, BBox]:
        if self.aug is None:
            return img, text_bbox

        word_img = np.array(img)
        # TODO: test self.aug.augment_bounding_boxes()
        return Image.fromarray(self.aug.augment_image(word_img)), text_bbox
