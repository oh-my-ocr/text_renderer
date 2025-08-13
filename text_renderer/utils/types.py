"""
Type definitions and utility functions for the text_renderer package.

This module provides common type aliases and utility functions used throughout
the text_renderer package for type checking and validation.
"""

from typing import Tuple

from PIL.Image import Image

# RGBA color tuple for font rendering
FontColor = Tuple[int, int, int, int]

# PIL Image type alias for better type hints
PILImage = Image

# 2D point tuple (x, y)
Point = Tuple[int, int]


def is_list(obj) -> bool:
    """
    Check if an object is a list.

    Args:
        obj: Object to check

    Returns:
        bool: True if obj is a list, False otherwise
    """
    return isinstance(obj, list)
