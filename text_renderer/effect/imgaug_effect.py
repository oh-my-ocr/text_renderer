from typing import Tuple

from PIL import Image
from imgaug.augmenters import Augmenter
import imgaug.augmenters as iaa
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


class Emboss(ImgAugEffect):
    def __init__(self, p=1.0, alpha=(0, 9, 1.0), strength=(1.5, 1.6)):
        """

        Parameters
        ----------
        p:
        alpha:
            see imgaug `doc`_
        strength:
            see imgaug `doc`_


        .. _doc: https://imgaug.readthedocs.io/en/latest/source/api_augmenters_convolutional.html#imgaug.augmenters.convolutional.Emboss
        """
        super().__init__(p, iaa.Emboss(alpha=alpha, strength=strength))


class MotionBlur(ImgAugEffect):
    def __init__(self, p=1.0, k=(3, 7), angle=(0, 360), direction=(-1.0, 1.0)):
        """

        Parameters
        ----------
        p
        k:
            see imgaug `doc`_
        angle
            see imgaug `doc`_
        direction
            see imgaug `doc`_


        .. _doc: https://imgaug.readthedocs.io/en/latest/source/api_augmenters_blur.html#imgaug.augmenters.blur.MotionBlur
        """

        super().__init__(p, iaa.MotionBlur(k, angle, direction))
