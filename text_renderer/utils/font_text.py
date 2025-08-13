"""
Font text utilities for text rendering.

This module provides the FontText class which encapsulates font and text information
for rendering operations, including support for both horizontal and vertical text.
"""

from dataclasses import dataclass
from typing import Tuple

from PIL.ImageFont import FreeTypeFont


@dataclass
class FontText:
    """
    Encapsulates font and text information for rendering operations.

    This class holds the font object, text content, font path, and orientation
    information needed for text rendering. It provides utility methods for
    calculating text dimensions and positioning.

    Args:
        font (FreeTypeFont): PIL font object for text rendering
        text (str): Text content to be rendered
        font_path (str): Path to the font file
        horizontal (bool): True for horizontal text, False for vertical text
    """

    font: FreeTypeFont
    text: str
    font_path: str
    horizontal: bool = True

    @property
    def xy(self) -> Tuple[int, int]:
        """
        Get the x, y offset for text positioning.

        This property calculates the offset needed to position text correctly
        by using the font's bounding box information.

        Returns:
            Tuple[int, int]: (x_offset, y_offset) for text positioning
        """
        # Use getbbox() instead of deprecated getoffset()
        bbox = self.font.getbbox(self.text)
        if bbox[2] > bbox[0] and bbox[3] > bbox[1]:  # Valid bbox
            left, top, right, bottom = bbox
        else:
            # Fallback for empty or invalid bbox
            left, top, right, bottom = 0, 0, 0, self.font.size
        return 0 - left, 0 - top

    @property
    def offset(self) -> Tuple[int, int]:
        """
        Get the text offset from the font's bounding box.

        This property returns the left and top offset values from the font's
        bounding box, which can be used for precise text positioning.

        Returns:
            Tuple[int, int]: (left_offset, top_offset) from font bbox
        """
        # Use getbbox() instead of deprecated getoffset()
        bbox = self.font.getbbox(self.text)
        if bbox[2] > bbox[0] and bbox[3] > bbox[1]:  # Valid bbox
            return (bbox[0], bbox[1])  # Return left, top as offset
        else:
            # Fallback for empty or invalid bbox
            return (0, 0)

    @property
    def size(self) -> Tuple[int, int]:
        """
        Get the text size without offset.

        This property calculates the actual size of the text content,
        handling both horizontal and vertical text orientations.

        For horizontal text: returns (width, height)
        For vertical text: returns (height, width) where height is the sum
        of individual character heights and width is the maximum character width.

        Returns:
            Tuple[int, int]: (width, height) for horizontal text or (height, width) for vertical text
        """
        if self.horizontal:
            bbox = self.font.getbbox(self.text)
            if bbox[2] > bbox[0] and bbox[3] > bbox[1]:  # Valid bbox
                size = (bbox[2] - bbox[0], bbox[3] - bbox[1])
            else:
                # Fallback for empty or invalid bbox
                size = (0, self.font.size)
            # Use bbox directly instead of getoffset
            width = size[0]
            height = size[1]
            return width, height
        else:
            widths = []
            heights = []
            for c in self.text:
                bbox = self.font.getbbox(c)
                if bbox[2] > bbox[0] and bbox[3] > bbox[1]:  # Valid bbox
                    char_size = (bbox[2] - bbox[0], bbox[3] - bbox[1])
                else:
                    # Fallback for empty or invalid bbox
                    char_size = (0, self.font.size)
                widths.append(char_size[0])
                heights.append(char_size[1])
            width = max(widths)
            height = sum(heights)
            return height, width
