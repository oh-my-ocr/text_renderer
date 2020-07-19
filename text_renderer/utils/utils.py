import glob
import os
import random
import cv2
import numpy as np
import hashlib
from PIL import Image


def prob(percent):
    """
    percent: 0 ~ 1, e.g: 如果 percent=0.1，有 10% 的可能性
    """
    assert 0 <= percent <= 1
    if random.uniform(0, 1) <= percent:
        return True
    return False


def draw_box(img, pnts, color):
    """
    :param img: gray image, will be convert to BGR image
    :param pnts: left-top, right-top, right-bottom, left-bottom
    :param color:
    :return:
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


def draw_bbox(img, bbox, color):
    pnts = [
        [bbox[0], bbox[1]],
        [bbox[0] + bbox[2], bbox[1]],
        [bbox[0] + bbox[2], bbox[1] + bbox[3]],
        [bbox[0], bbox[1] + bbox[3]],
    ]
    return draw_box(img, pnts, color)


def random_xy_offset(small_size, big_size):
    """
    Get random left-top point for putting a small rect in a large rect.
    Args:
        small_size: (width, height)
        big_size: (width, height)

    Returns:

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


def size_to_pnts(size) -> np.ndarray:
    """
    获得图片 size 的四个角点 (4,2)
    """
    width = size[0]
    height = size[1]
    return np.array([[0, 0], [width, 0], [width, height], [0, height]])
