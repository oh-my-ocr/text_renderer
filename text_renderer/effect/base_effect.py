"""
Base effect classes for text rendering augmentation.

This module provides the base classes and utilities for implementing
image augmentation effects in text rendering operations.
"""

import random
from abc import abstractmethod
from typing import List, Tuple, Union

from text_renderer.effect.selector import Selector
from text_renderer.utils.bbox import BBox
from text_renderer.utils.types import PILImage
from text_renderer.utils.utils import prob


class Effect:
    """
    Base class for applying different augmentations to images.

    This abstract base class defines the interface for all image effects
    used in text rendering. Effects can modify images and potentially
    update the text bounding box coordinates.

    Examples of effects include: adding noise, dropout, padding, etc.

    Args:
        p (float): Probability of applying this effect (default: 0.5)
    """

    def __init__(self, p: float = 0.5):
        self.p = p

    def __call__(self, img: PILImage, text_bbox: BBox) -> Tuple[PILImage, BBox]:
        """
        Apply the effect with the configured probability.

        Args:
            img (PILImage): Input image to apply effect to
            text_bbox (BBox): Bounding box of text in the image

        Returns:
            Tuple[PILImage, BBox]: Modified image and updated bounding box
        """
        if prob(self.p):
            # Create a copy to ensure the image is writable
            img = img.copy()
            return self.apply(img, text_bbox)
        return img, text_bbox

    @abstractmethod
    def apply(self, img: PILImage, text_bbox: BBox) -> Tuple[PILImage, BBox]:
        """
        Apply the effect to the image.

        This method must be implemented by all subclasses to define
        the specific augmentation behavior.

        Args:
            img (PILImage): Image to apply effect to
            text_bbox (BBox): Bounding box of text in the image

        Returns:
            Tuple[PILImage, BBox]: Modified image and updated bounding box.
                Some effects (such as Padding) may modify the relative
                position of the text in the image.
        """
        pass

    @staticmethod
    def rand_pick(pim, col: int, row: int):
        """
        Randomly reset pixel value at [col, row].

        This utility method randomly reduces the pixel values at the
        specified position. The new pixel value is a random integer
        between 0 and the original pixel value.

        Args:
            pim: Pixel access object from pil_img.load()
            col (int): Column coordinate
            row (int): Row coordinate
        """
        pim[col, row] = (
            random.randint(0, pim[col, row][0]),
            random.randint(0, pim[col, row][1]),
            random.randint(0, pim[col, row][2]),
            random.randint(0, pim[col, row][3]),
        )

    @staticmethod
    def fix_pick(pim, col: int, row: int, value_range: Tuple[int, int]):
        """
        Set pixel value to a random value within the specified range.

        Args:
            pim: Pixel access object from pil_img.load()
            col (int): Column coordinate
            row (int): Row coordinate
            value_range (Tuple[int, int]): Range for random value selection
        """
        value = random.randint(*value_range)
        pim[col, row] = (value, value, value, value)


class NoEffects:
    """
    Placeholder class when no effects are desired for multi-corpus scenarios.

    This class provides a no-op implementation that simply returns the
    input image and bounding box unchanged.
    """

    def apply_effects(self, img: PILImage, bbox: BBox) -> Tuple[PILImage, BBox]:
        """
        Return the input image and bounding box unchanged.

        Args:
            img (PILImage): Input image
            bbox (BBox): Input bounding box

        Returns:
            Tuple[PILImage, BBox]: Unchanged image and bounding box
        """
        return img, bbox


class Effects:
    """
    Apply multiple effects in sequence.

    This class manages the application of multiple effects to an image.
    It can handle individual effects, lists of effects, or effect selectors.

    Args:
        effects (Union[Effect, List[Effect], Selector, List[Selector]]):
            Effect(s) to apply. Can be a single effect, list of effects,
            selector, or list of selectors.
    """

    def __init__(self, effects: Union[Effect, List[Effect], Selector, List[Selector]]):
        """
        Initialize the Effects container.

        Args:
            effects (Union[Effect, List[Effect], Selector, List[Selector]]):
                Effect(s) to apply. Can be a single effect, list of effects,
                selector, or list of selectors.
        """
        if not isinstance(effects, list):
            effects = [effects]
        self.effects = effects

    def apply_effects(self, img: PILImage, bbox: BBox) -> Tuple[PILImage, BBox]:
        """
        Apply all configured effects to the image.

        Args:
            img (PILImage): Input image to apply effects to
            bbox (BBox): Bounding box of text in the image

        Returns:
            Tuple[PILImage, BBox]: Image with all effects applied and updated bounding box
        """
        # Create a copy to ensure the image is writable
        img = img.copy()
        for e in self.effects:
            img, bbox = e(img, bbox)
        return img, bbox
