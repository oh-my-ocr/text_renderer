from typing import Tuple, Union

from PIL import ImageDraw, Image
from PIL.Image import Image as PILImage
import numpy as np

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
    char_spacing: Union[float, Tuple[float, float]] = -1,
) -> PILImage:
    """

    Parameters
    ----------
    font_text : FontText
    text_color : RGBA
        Default is black
    char_spacing : Union[float, Tuple[float, float]]
        Draw character with spacing. If tuple, random choice between [min, max)
        Set -1 to disable

    Returns
    -------
        PILImage:
            RGBA Pillow image with text on a transparent image
    -------

    """
    if char_spacing == -1:
        if font_text.horizontal:
            return _draw_text_on_bg(font_text, text_color)
        else:
            char_spacing = 0

    chars_size = []
    widths = []
    heights = []

    for c in font_text.text:
        size = font_text.font.getsize(c)
        chars_size.append(size)
        widths.append(size[0])
        heights.append(size[1])

    if font_text.horizontal:
        width = sum(widths)
        height = max(heights)
    else:
        width = max(widths)
        height = sum(heights)

    char_spacings = []

    cs_height = font_text.size[1]
    for i in range(len(font_text.text)):
        if isinstance(char_spacing, list) or isinstance(char_spacing, tuple):
            s = np.random.uniform(*char_spacing)
            char_spacings.append(int(s * cs_height))
        else:
            char_spacings.append(int(char_spacing * cs_height))

    if font_text.horizontal:
        width += sum(char_spacings[:-1])
    else:
        height += sum(char_spacings[:-1])

    text_mask = transparent_img((width, height))
    draw = ImageDraw.Draw(text_mask)

    c_x = 0
    c_y = 0

    if font_text.horizontal:
        y_offset = font_text.offset[1]
        for i, c in enumerate(font_text.text):
            draw.text((c_x, c_y - y_offset), c, fill=text_color, font=font_text.font)
            c_x += chars_size[i][0] + char_spacings[i]
    else:
        x_offset = font_text.offset[0]
        for i, c in enumerate(font_text.text):
            draw.text((c_x - x_offset, c_y), c, fill=text_color, font=font_text.font)
            c_y += chars_size[i][1] + char_spacings[i]
        text_mask = text_mask.rotate(90, expand=True)

    return text_mask


def _draw_text_on_bg(
    font_text: FontText,
    text_color: Tuple[int, int, int, int] = (0, 0, 0, 255),
) -> PILImage:
    """
    Draw text

    Parameters
    ----------
    font_text : FontText
    text_color : RGBA
        Default is black

    Returns
    -------
        PILImage:
            RGBA Pillow image with text on a transparent image
    """
    text_width, text_height = font_text.size
    text_mask = transparent_img((text_width, text_height))
    draw = ImageDraw.Draw(text_mask)

    xy = font_text.xy

    # TODO: figure out anchor
    draw.text(
        xy,
        font_text.text,
        font=font_text.font,
        fill=text_color,
        anchor=None,
    )

    return text_mask
