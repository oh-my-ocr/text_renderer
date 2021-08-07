import random
from abc import abstractmethod
from typing import List, Union, Tuple

from PIL import PyAccess

from text_renderer.effect.selector import Selector
from text_renderer.utils.bbox import BBox
from text_renderer.utils.types import PILImage
from text_renderer.utils.utils import prob


class Effect:
    """
    Apply different augmentations to image.

    E.g. add noise, add dropout, add padding...
    """

    def __init__(self, p=0.5):
        """

        Parameters
        ----------
        p : float
            Probability of apply this effect

        """
        self.p = p

    def __call__(self, img, text_bbox):
        if prob(self.p):
            return self.apply(img, text_bbox)
        return img, text_bbox

    @abstractmethod
    def apply(self, img: PILImage, text_bbox: BBox):
        """

        Parameters
        ----------
        img : PILImage
            Image to apply effect
        text_bbox : BBox
            bbox of text on input Image

        Returns
        -------
        PILImage:
            Image changed
        BBox:
            Text bbox on image after apply effect.
            Some effects (such as Padding) may modify the relative position of the text in the image.

        """
        pass

    @staticmethod
    def rand_pick(pim, col, row):
        """
        Randomly reset pixel value at [col, row]

        new_pixel_value = random.randint(0, pixel_value)

        Parameters
        ----------
        pim : PyAccess
            Get from pil_img.load()
        col : int
        row : int
        """

        pim[col, row] = (
            random.randint(0, pim[col, row][0]),
            random.randint(0, pim[col, row][1]),
            random.randint(0, pim[col, row][2]),
            random.randint(0, pim[col, row][3]),
        )

    @staticmethod
    def fix_pick(pim, col, row, value_range: Tuple[int, int]):
        value = random.randint(*value_range)
        pim[col, row] = (value, value, value, value)


class NoEffects:
    """
    Placeholder when you don't want to apply effects for multi corpus
    """

    def apply_effects(self, img: PILImage, bbox: BBox) -> Tuple[PILImage, BBox]:
        return img, bbox


class Effects:
    """
    Apply multiple effects
    """

    def __init__(self, effects: Union[Effect, List[Effect], Selector, List[Selector]]):
        """

        Parameters
        ----------
        effects : Effect or List[Effect]
        """
        if not isinstance(effects, list):
            effects = [effects]
        self.effects = effects

    def apply_effects(self, img: PILImage, bbox: BBox) -> Tuple[PILImage, BBox]:
        """

        Args:
            img:
            bbox: bbox of text on input Image

        Returns:

        """
        for e in self.effects:
            img, bbox = e(img, bbox)
        return img, bbox
