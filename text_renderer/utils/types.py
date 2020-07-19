from typing import Tuple
from PIL.Image import Image

# RGBA
FontColor = Tuple[int, int, int, int]
PILImage = Image
Point = Tuple[int, int]


def is_list(obj) -> bool:
    return isinstance(obj, list)
