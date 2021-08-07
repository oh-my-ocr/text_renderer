import typing
from typing import Tuple, List

if typing.TYPE_CHECKING:
    from text_renderer.effect import Effect

from text_renderer.utils.bbox import BBox
from text_renderer.utils.types import PILImage
from text_renderer.utils.utils import random_choice


class Selector:
    """
    Selects a random Effect from given list
    """

    def __init__(self, effects: List["Effect"]):
        """

        Parameters
        ----------
        effects : :obj:`list` of :obj:`Effect`
        """
        self.effects = effects


class OneOf(Selector):
    def __call__(self, img: PILImage, text_bbox: BBox) -> Tuple[PILImage, BBox]:
        effect = random_choice(self.effects)
        return effect(img, text_bbox)
