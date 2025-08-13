"""
Utility functions for text rendering operations.

This module provides various utility functions for text rendering including
probability calculations, random selection, drawing utilities, and file operations.
"""

import random
from typing import Any, List, Set, Tuple

import cv2
import numpy as np
from loguru import logger

from text_renderer.utils.errors import PanicError

SPACE_CHAR = " "


def prob(percent: float) -> bool:
    """
    Return True with the given probability.

    Args:
        percent (float): Probability between 0 and 1 (e.g., 0.1 means 10% chance)

    Returns:
        bool: True with the specified probability, False otherwise

    Raises:
        AssertionError: If percent is not between 0 and 1
    """
    assert 0 <= percent <= 1
    if random.uniform(0, 1) <= percent:
        return True
    return False


def random_choice(items: List[Any], size: int = 1) -> Any:
    """
    Randomly select items from a list.

    This function is optimized for performance compared to numpy.random.choice
    for simple random selection operations.

    Args:
        items (List[Any]): List of items to choose from
        size (int): Number of items to select (default: 1)

    Returns:
        Any: Selected item(s). If size=1, returns a single item.
             If size>1, returns a list of items.
    """
    # np.random.choice is very slow
    out = []
    for _ in range(size):
        i = np.random.randint(0, len(items))
        out.append(items[i])
    if size == 1:
        return out[0]
    return out


def draw_box(
    img: np.ndarray, pnts: np.ndarray, color: Tuple[int, int, int]
) -> np.ndarray:
    """
    Draw a rectangular box on an image.

    Args:
        img (np.ndarray): Input image (will be converted to BGR if grayscale)
        pnts (np.ndarray): Four corner points in order: left-top, right-top, right-bottom, left-bottom
        color (Tuple[int, int, int]): BGR color tuple for the box

    Returns:
        np.ndarray: Image with the box drawn on it
    """
    if isinstance(pnts, np.ndarray):
        pnts = pnts.astype(np.int32)

    if len(img.shape) > 2:
        dst = img
    else:
        dst = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    thickness = 1
    linetype = cv2.LINE_AA
    cv2.line(
        dst,
        (pnts[0][0], pnts[0][1]),
        (pnts[1][0], pnts[1][1]),
        color=color,
        thickness=thickness,
        lineType=linetype,
    )
    cv2.line(
        dst,
        (pnts[1][0], pnts[1][1]),
        (pnts[2][0], pnts[2][1]),
        color=color,
        thickness=thickness,
        lineType=linetype,
    )
    cv2.line(
        dst,
        (pnts[2][0], pnts[2][1]),
        (pnts[3][0], pnts[3][1]),
        color=color,
        thickness=thickness,
        lineType=linetype,
    )
    cv2.line(
        dst,
        (pnts[3][0], pnts[3][1]),
        (pnts[0][0], pnts[0][1]),
        color=color,
        thickness=thickness,
        lineType=linetype,
    )
    return dst


def draw_bbox(
    img: np.ndarray, bbox: Tuple[int, int, int, int], color: Tuple[int, int, int]
) -> np.ndarray:
    """
    Draw a bounding box on an image.

    Args:
        img (np.ndarray): Input image
        bbox (Tuple[int, int, int, int]): Bounding box as (x, y, width, height)
        color (Tuple[int, int, int]): BGR color tuple for the box

    Returns:
        np.ndarray: Image with the bounding box drawn on it
    """
    pnts = [
        [bbox[0], bbox[1]],
        [bbox[0] + bbox[2], bbox[1]],
        [bbox[0] + bbox[2], bbox[1] + bbox[3]],
        [bbox[0], bbox[1] + bbox[3]],
    ]
    return draw_box(img, pnts, color)


def random_xy_offset(
    small_size: Tuple[int, int], big_size: Tuple[int, int]
) -> Tuple[int, int]:
    """
    Get random left-top point for putting a small rectangle in a large rectangle.

    This function calculates a random position within the larger rectangle
    where the smaller rectangle can be placed without extending beyond the bounds.

    Args:
        small_size (Tuple[int, int]): Size of the small rectangle (width, height)
        big_size (Tuple[int, int]): Size of the large rectangle (width, height)

    Returns:
        Tuple[int, int]: Random (x, y) offset for positioning the small rectangle
    """
    small_rect_width, small_rect_height = small_size
    big_rect_width, big_rect_height = big_size

    y_max_offset = 0
    if big_rect_height > small_rect_height:
        y_max_offset = big_rect_height - small_rect_height

    x_max_offset = 0
    if big_rect_width > small_rect_width:
        x_max_offset = big_rect_width - small_rect_width

    y_offset = 0
    if y_max_offset != 0:
        y_offset = random.randint(0, y_max_offset)

    x_offset = 0
    if x_max_offset != 0:
        x_offset = random.randint(0, x_max_offset)

    return x_offset, y_offset


def size_to_pnts(size: Tuple[int, int]) -> np.ndarray:
    """
    Convert image size to four corner points.

    Args:
        size (Tuple[int, int]): Image size as (width, height)

    Returns:
        np.ndarray: Array of shape (4, 2) containing corner points in order:
                   left-top, right-top, right-bottom, left-bottom
    """
    width = size[0]
    height = size[1]
    return np.array([[0, 0], [width, 0], [width, height], [0, height]])


def load_chars_file(chars_file, log: bool = False) -> Set[str]:
    """
    Load characters from a file where each line contains one character.

    This function reads a text file and extracts individual characters,
    handling special cases like space characters and empty lines.

    Args:
        chars_file: Path to the character file
        log (bool): Whether to print log messages during loading

    Returns:
        Set[str]: Set of unique characters from the file

    Raises:
        PanicError: If the file doesn't exist or contains invalid lines
    """
    assumed_space = False
    with open(str(chars_file), "r", encoding="utf-8") as f:
        lines = f.readlines()
        _lines = []
        for i, line in enumerate(lines):
            line_striped = line.strip()
            if len(line_striped) > 1:
                raise PanicError(
                    f"Line {i} in {chars_file} is invalid, make sure one char one line"
                )

            if len(line_striped) == 0 and SPACE_CHAR in line:
                if assumed_space is True:
                    raise PanicError(f"Find two space in {chars_file}")

                if log:
                    logger.info(f"Find space in line {i} when load {chars_file}")
                assumed_space = True
                _lines.append(SPACE_CHAR)
                continue

            _lines.append(line_striped)

        lines = _lines
        chars = set("".join(lines))
    if log:
        logger.info(f"load {len(chars)} chars from: {chars_file}")
    return chars
