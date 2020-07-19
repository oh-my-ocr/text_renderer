import copy
from dataclasses import dataclass
from typing import Tuple, List

import numpy as np

from text_renderer.utils.types import Point


@dataclass
class BBox:
    left: int
    top: int
    right: int
    bottom: int

    @property
    def cx(self) -> int:
        return int((self.left + self.right) // 2)

    @property
    def cy(self) -> int:
        return int((self.top + self.bottom) // 2)

    @property
    def cnt(self) -> Tuple[int, int]:
        return self.cx, self.cy

    @property
    def left_cnt(self) -> Tuple[int, int]:
        return self.left, self.cy

    @property
    def top_cnt(self) -> Tuple[int, int]:
        return self.cx, self.top

    @property
    def right_cnt(self) -> Tuple[int, int]:
        return self.right, self.cy

    @property
    def bottom_cnt(self) -> Tuple[int, int]:
        return self.cx, self.bottom

    @property
    def left_top(self) -> Tuple[int, int]:
        return self.left, self.top

    @property
    def left_bottom(self) -> Tuple[int, int]:
        return self.left, self.bottom

    @property
    def right_top(self) -> Tuple[int, int]:
        return self.right, self.top

    @property
    def right_bottom(self) -> Tuple[int, int]:
        return self.right, self.bottom

    @property
    def height(self) -> int:
        return self.bottom - self.top

    @property
    def width(self) -> int:
        return self.right - self.left

    @property
    def size(self) -> Tuple[int, int]:
        return self.width, self.height

    @staticmethod
    def from_bboxes(boxes: List["BBox"]) -> "BBox":
        left = max(min([it.left for it in boxes]), 0)
        top = max(min([it.top for it in boxes]), 0)
        right = max([it.right for it in boxes])
        bottom = max([it.bottom for it in boxes])
        return BBox(left, top, right, bottom)

    @staticmethod
    def from_size(size: Tuple[int, int]) -> "BBox":
        return BBox(0, 0, size[0], size[1])

    def pnts(self) -> np.ndarray:
        return np.array(
            [
                [self.left, self.top],
                [self.right, self.top],
                [self.right, self.bottom],
                [self.left, self.bottom],
            ]
        )

    def copy(self) -> "BBox":
        return BBox(self.left, self.top, self.right, self.bottom)

    def offset(self, anchor: Point, move_to: Point) -> "BBox":
        """
        Modify current bbox's position, let anchor overlap with move_to
        Args:
            anchor:
            move_to:
        """
        bbox = copy.deepcopy(self)
        bbox.offset_(anchor, move_to)
        return bbox

    def offset_(self, anchor: Point, move_to: Point):
        """
        Modify current bbox's position, let anchor overlap with move_to
        Args:
            anchor:
            move_to:
        """

        offset_x = move_to[0] - anchor[0]
        offset_y = move_to[1] - anchor[1]

        self.left += offset_x
        self.right += offset_x
        self.top += offset_y
        self.bottom += offset_y

    def move_origin(self, pnt: Point) -> "BBox":
        """
        Move origin to point
        """
        bbox = self.copy()
        bbox.move_origin_(pnt)
        return bbox

    def move_origin_(self, pnt: Point):
        """
        Move origin to point
        """
        self.offset_((0, 0), pnt)
