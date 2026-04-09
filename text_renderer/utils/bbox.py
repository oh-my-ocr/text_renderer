"""
Bounding box utilities for text rendering.

This module provides the BBox class for representing and manipulating
rectangular bounding boxes used in text rendering operations.
"""

import copy
import math
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np

from text_renderer.utils.types import Point


@dataclass
class BBox:
    """
    Represents a rectangular bounding box with integer coordinates.

    This class provides utilities for working with bounding boxes including
    property accessors, geometric operations, and coordinate transformations.

    Args:
        left (int): Left edge x-coordinate
        top (int): Top edge y-coordinate
        right (int): Right edge x-coordinate
        bottom (int): Bottom edge y-coordinate
    """

    left: int
    top: int
    right: int
    bottom: int

    @property
    def cx(self) -> int:
        """
        Get the center x-coordinate of the bounding box.

        Returns:
            int: Center x-coordinate
        """
        return int((self.left + self.right) // 2)

    @property
    def cy(self) -> int:
        """
        Get the center y-coordinate of the bounding box.

        Returns:
            int: Center y-coordinate
        """
        return int((self.top + self.bottom) // 2)

    @property
    def cnt(self) -> Tuple[int, int]:
        """
        Get the center point of the bounding box.

        Returns:
            Tuple[int, int]: (center_x, center_y)
        """
        return self.cx, self.cy

    @property
    def left_cnt(self) -> Tuple[int, int]:
        """
        Get the left center point of the bounding box.

        Returns:
            Tuple[int, int]: (left, center_y)
        """
        return self.left, self.cy

    @property
    def top_cnt(self) -> Tuple[int, int]:
        """
        Get the top center point of the bounding box.

        Returns:
            Tuple[int, int]: (center_x, top)
        """
        return self.cx, self.top

    @property
    def right_cnt(self) -> Tuple[int, int]:
        """
        Get the right center point of the bounding box.

        Returns:
            Tuple[int, int]: (right, center_y)
        """
        return self.right, self.cy

    @property
    def bottom_cnt(self) -> Tuple[int, int]:
        """
        Get the bottom center point of the bounding box.

        Returns:
            Tuple[int, int]: (center_x, bottom)
        """
        return self.cx, self.bottom

    @property
    def left_top(self) -> Tuple[int, int]:
        """
        Get the left-top corner point of the bounding box.

        Returns:
            Tuple[int, int]: (left, top)
        """
        return self.left, self.top

    @property
    def left_bottom(self) -> Tuple[int, int]:
        """
        Get the left-bottom corner point of the bounding box.

        Returns:
            Tuple[int, int]: (left, bottom)
        """
        return self.left, self.bottom

    @property
    def right_top(self) -> Tuple[int, int]:
        """
        Get the right-top corner point of the bounding box.

        Returns:
            Tuple[int, int]: (right, top)
        """
        return self.right, self.top

    @property
    def right_bottom(self) -> Tuple[int, int]:
        """
        Get the right-bottom corner point of the bounding box.

        Returns:
            Tuple[int, int]: (right, bottom)
        """
        return self.right, self.bottom

    @property
    def height(self) -> int:
        """
        Get the height of the bounding box.

        Returns:
            int: Height (bottom - top)
        """
        return self.bottom - self.top

    @property
    def width(self) -> int:
        """
        Get the width of the bounding box.

        Returns:
            int: Width (right - left)
        """
        return self.right - self.left

    @property
    def size(self) -> Tuple[int, int]:
        """
        Get the size of the bounding box.

        Returns:
            Tuple[int, int]: (width, height)
        """
        return self.width, self.height

    @staticmethod
    def from_bboxes(boxes: List["BBox"]) -> "BBox":
        """
        Create a bounding box that encompasses all input bounding boxes.

        Args:
            boxes (List[BBox]): List of bounding boxes to combine

        Returns:
            BBox: A new bounding box that contains all input boxes
        """
        left = max(min([it.left for it in boxes]), 0)
        top = max(min([it.top for it in boxes]), 0)
        right = max([it.right for it in boxes])
        bottom = max([it.bottom for it in boxes])
        return BBox(left, top, right, bottom)

    @staticmethod
    def from_size(size: Tuple[int, int]) -> "BBox":
        """
        Create a bounding box from size with origin at (0, 0).

        Args:
            size (Tuple[int, int]): (width, height) of the bounding box

        Returns:
            BBox: A new bounding box with given size at origin
        """
        return BBox(0, 0, size[0], size[1])

    def pnts(self) -> np.ndarray:
        """
        Get the four corner points of the bounding box as a numpy array.

        Returns:
            np.ndarray: Array of shape (4, 2) containing corner points in order:
                       left-top, right-top, right-bottom, left-bottom
        """
        return np.array(
            [
                [self.left, self.top],
                [self.right, self.top],
                [self.right, self.bottom],
                [self.left, self.bottom],
            ]
        )

    def copy(self) -> "BBox":
        """
        Create a copy of this bounding box.

        Returns:
            BBox: A new bounding box with the same coordinates
        """
        return BBox(self.left, self.top, self.right, self.bottom)

    def offset(self, anchor: Point, move_to: Point) -> "BBox":
        """
        Create a new bounding box by moving this one so that anchor overlaps with move_to.

        This method creates a new bounding box without modifying the original.

        Args:
            anchor (Point): Point on this bounding box to use as anchor
            move_to (Point): Target position for the anchor point

        Returns:
            BBox: A new bounding box with updated position
        """
        bbox = copy.deepcopy(self)
        bbox.offset_(anchor, move_to)
        return bbox

    def offset_(self, anchor: Point, move_to: Point):
        """
        Move this bounding box so that anchor overlaps with move_to.

        This method modifies the current bounding box in-place.

        Args:
            anchor (Point): Point on this bounding box to use as anchor
            move_to (Point): Target position for the anchor point
        """
        offset_x = move_to[0] - anchor[0]
        offset_y = move_to[1] - anchor[1]

        self.left += offset_x
        self.right += offset_x
        self.top += offset_y
        self.bottom += offset_y

    def move_origin(self, pnt: Point) -> "BBox":
        """
        Create a new bounding box by moving the origin to the specified point.

        This method creates a new bounding box without modifying the original.

        Args:
            pnt (Point): New origin point

        Returns:
            BBox: A new bounding box with moved origin
        """
        bbox = self.copy()
        bbox.move_origin_(pnt)
        return bbox

    def move_origin_(self, pnt: Point):
        """
        Move the origin of this bounding box to the specified point.

        This method modifies the current bounding box in-place.

        Args:
            pnt (Point): New origin point
        """
        self.offset_((0, 0), pnt)


@dataclass
class CharBBox:
    """
    Represents a single character bounding box.

    Args:
        char (str): Character associated with this bounding box
        left (int): Left edge x-coordinate
        top (int): Top edge y-coordinate
        right (int): Right edge x-coordinate
        bottom (int): Bottom edge y-coordinate
    """

    char: str
    left: int
    top: int
    right: int
    bottom: int

    def to_dict(self) -> dict:
        """
        Convert the character bounding box into a serializable dictionary.

        Returns:
            dict: {"char": <char>, "bbox": [left, top, right, bottom]}
        """
        return {"char": self.char, "bbox": [self.left, self.top, self.right, self.bottom]}

    def offset(self, dx: int, dy: int) -> "CharBBox":
        """
        Return a new character bounding box translated by the given offset.

        Args:
            dx (int): Horizontal offset
            dy (int): Vertical offset

        Returns:
            CharBBox: Offset bounding box
        """
        return CharBBox(
            self.char,
            self.left + dx,
            self.top + dy,
            self.right + dx,
            self.bottom + dy,
        )

    def scale(self, sx: float, sy: float) -> "CharBBox":
        """
        Return a scaled copy of the character bounding box.

        Args:
            sx (float): Horizontal scale factor
            sy (float): Vertical scale factor

        Returns:
            CharBBox: Scaled bounding box
        """
        # Use floor/ceil so the scaled box encloses the scaled glyph
        # when sx/sy are non-integer; plain int() truncates toward zero
        # and under-bounds negative left/top or fractional right/bottom.
        return CharBBox(
            self.char,
            int(math.floor(self.left * sx)),
            int(math.floor(self.top * sy)),
            int(math.ceil(self.right * sx)),
            int(math.ceil(self.bottom * sy)),
        )

    def pnts(self) -> np.ndarray:
        """
        Get the four corner points of the bounding box as a float32 array.

        Returns:
            np.ndarray: Array of shape (4, 2)
        """
        return np.array(
            [
                [self.left, self.top],
                [self.right, self.top],
                [self.right, self.bottom],
                [self.left, self.bottom],
            ],
            dtype=np.float32,
        )

    @staticmethod
    def from_pnts(char: str, pnts: np.ndarray) -> "CharBBox":
        """
        Create an axis-aligned character bounding box from corner points.

        Args:
            char (str): Character associated with the bounding box
            pnts (np.ndarray): Corner points with shape (4, 2)

        Returns:
            CharBBox: Bounding box enclosing the provided points
        """
        # Use floor/ceil to ensure the returned bbox encloses non-integer
        # transformed points; plain int() truncates toward zero and would
        # produce a box that excludes points like -0.4 or 20.9.
        return CharBBox(
            char,
            int(math.floor(pnts[:, 0].min())),
            int(math.floor(pnts[:, 1].min())),
            int(math.ceil(pnts[:, 0].max())),
            int(math.ceil(pnts[:, 1].max())),
        )
