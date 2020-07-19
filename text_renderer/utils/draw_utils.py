from typing import Tuple

from PIL import ImageDraw, Image
from PIL.Image import Image as PILImage

from text_renderer.utils.font_text import FontText


def transparent_img(size: Tuple[int, int]) -> PILImage:
    """

    Args:
        size: (width, height)

    Returns:

    """
    return Image.new("RGBA", (size[0], size[1]), (255, 255, 255, 0))


def draw_text_on_bg(
    font_text: FontText,
    text_color: Tuple[int, int, int, int] = (0, 0, 0, 255),
    bg: PILImage = None,
    xy: Tuple[int, int] = None,
) -> PILImage:
    """

    Args:
        font_text:
        text_color: RGBA。default is black
        bg: default will draw on a transparent RGBA image
        xy:

    Returns:

    """
    text_width, text_height = font_text.size

    if bg is None:
        text_mask = transparent_img((text_width, text_height))
    else:
        text_mask = bg.copy()
    draw = ImageDraw.Draw(text_mask)

    if xy is None:
        xy = font_text.xy

    # TODO: figure out anchor
    draw.text(
        xy, font_text.text, font=font_text.font, fill=text_color, anchor=None,
    )

    return text_mask
