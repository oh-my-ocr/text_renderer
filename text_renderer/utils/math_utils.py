"""
Mathematical utilities for text rendering operations.

This module provides mathematical functions and classes for geometric transformations,
particularly perspective transformations used in text rendering.
"""

import math

#!/usr/env/bin python3
from functools import reduce
from typing import Tuple

import cv2
import numpy as np
from PIL import Image

from text_renderer.config import PerspectiveTransformCfg
from text_renderer.utils import utils


# http://planning.cs.uiuc.edu/node102.html
def get_rotate_matrix(x: float, y: float, z: float) -> np.matrix:
    """
    Generate rotation matrix for 3D rotation in ZYX order.

    This function creates a 4x4 homogeneous transformation matrix for 3D rotation.
    Rotations are applied in ZYX order (first around Z-axis, then Y-axis, then X-axis).
    All rotations are clockwise when viewed from the positive axis direction.

    Args:
        x (float): Rotation angle around X-axis in degrees
        y (float): Rotation angle around Y-axis in degrees
        z (float): Rotation angle around Z-axis in degrees

    Returns:
        np.matrix: 4x4 homogeneous rotation matrix
    """
    x = math.radians(x)
    y = math.radians(y)
    z = math.radians(z)

    c, s = math.cos(y), math.sin(y)
    M_y = np.matrix(
        [
            [c, 0.0, s, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [-s, 0.0, c, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )

    c, s = math.cos(x), math.sin(x)
    M_x = np.matrix(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, c, -s, 0.0],
            [0.0, s, c, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )

    c, s = math.cos(z), math.sin(z)
    M_z = np.matrix(
        [
            [c, -s, 0.0, 0.0],
            [s, c, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )

    return M_x * M_y * M_z


# https://stackoverflow.com/questions/17087446/how-to-calculate-perspective-transform-for-opencv-from-rotation-angles
# https://nbviewer.jupyter.org/github/manisoftwartist/perspectiveproj/blob/master/perspective.ipynb
# http://planning.cs.uiuc.edu/node102.html
class PerspectiveTransform(object):
    """
    Apply perspective transformation to images based on 3D rotation parameters.

    This class implements perspective transformation using 3D rotation angles
    and camera parameters to create realistic perspective effects on text images.

    Args:
        cfg (PerspectiveTransformCfg): Configuration object containing rotation
                                      angles, scale, and field of view parameters
    """

    def __init__(self, cfg: PerspectiveTransformCfg):
        self.x, self.y, self.z = cfg.get_xyz()
        self.scale = cfg.scale
        self.fovy = cfg.fovy

    def get_transformed_size(self, size: Tuple[int, int]) -> Tuple[int, int]:
        """
        Calculate the size of the image after perspective transformation.

        Args:
            size (Tuple[int, int]): Original image size (width, height)

        Returns:
            Tuple[int, int]: Size of the transformed image (width, height)
        """
        width, height = size
        _, _, _, text_box_pnts_transformed = self.gen_warp_matrix(width, height)

        transformed_bbox = cv2.boundingRect(text_box_pnts_transformed)
        bbox_width = transformed_bbox[2]
        bbox_height = transformed_bbox[3]

        return int(bbox_width), int(bbox_height)

    def do_warp_perspective(
        self, pil_img: Image.Image
    ) -> Tuple[Image.Image, np.ndarray]:
        """
        Apply perspective transformation to a PIL image.

        Args:
            pil_img (Image.Image): Input PIL image to transform

        Returns:
            Tuple[Image.Image, np.ndarray]: A tuple containing:
                - Image.Image: Transformed image
                - np.ndarray: Transformed corner points
        """
        text_box_pnts = utils.size_to_pnts(pil_img.size)
        img = np.array(pil_img).astype(np.uint8)

        dst = cv2.warpPerspective(
            img,
            self.M33,
            (self.sl, self.sl),
            flags=cv2.INTER_CUBIC,
            borderValue=(255, 255, 255, 0),
        )
        transformed_pnts = self.transform_pnts(text_box_pnts, self.M33)

        transformed_text_box = cv2.boundingRect(transformed_pnts)
        transformed_text_box = list(transformed_text_box)
        dst = Image.fromarray(dst)
        dst = dst.crop(
            [
                transformed_text_box[0],
                transformed_text_box[1],
                transformed_text_box[0] + transformed_text_box[2],
                transformed_text_box[1] + transformed_text_box[3],
            ]
        )

        transformed_pnts[:, 0] -= transformed_text_box[0]
        transformed_pnts[:, 1] -= transformed_text_box[1]

        return dst, transformed_pnts

    def transform_pnts(self, pnts: np.ndarray, M33: np.ndarray) -> np.ndarray:
        """
        Transform 2D points using a perspective transformation matrix.

        Args:
            pnts (np.ndarray): 2D points in order: left-top, right-top, right-bottom, left-bottom
            M33 (np.ndarray): 3x3 perspective transformation matrix

        Returns:
            np.ndarray: Transformed 2D points with integer coordinates
        """
        pnts = np.asarray(pnts, dtype=np.float32)
        pnts = np.array([pnts])
        dst_pnts = cv2.perspectiveTransform(pnts, M33)[0]
        return np.array(dst_pnts).astype(np.int32)

    def get_warped_pnts(
        self, ptsIn: np.ndarray, ptsOut: np.ndarray, W: int, H: int, sidelength: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Convert 3D points to 2D points for perspective transformation.

        Args:
            ptsIn (np.ndarray): Input 3D points
            ptsOut (np.ndarray): Output 3D points
            W (int): Width of the image
            H (int): Height of the image
            sidelength (float): Side length parameter for scaling

        Returns:
            Tuple[np.ndarray, np.ndarray]: A tuple containing:
                - np.ndarray: Input 2D points
                - np.ndarray: Output 2D points
        """
        ptsIn2D = ptsIn[0, :]
        ptsOut2D = ptsOut[0, :]
        ptsOut2Dlist = []
        ptsIn2Dlist = []

        for i in range(0, 4):
            ptsOut2Dlist.append([ptsOut2D[i, 0], ptsOut2D[i, 1]])
            ptsIn2Dlist.append([ptsIn2D[i, 0], ptsIn2D[i, 1]])

        pin = np.array(ptsIn2Dlist) + [W / 2.0, H / 2.0]
        pout = (np.array(ptsOut2Dlist) + [1.0, 1.0]) * (0.5 * sidelength)
        pin = pin.astype(np.float32)
        pout = pout.astype(np.float32)

        return pin, pout

    def gen_warp_matrix(
        self, width: int, height: int
    ) -> Tuple[np.ndarray, float, np.ndarray, np.ndarray]:
        """
        Generate perspective transformation matrix and parameters.

        This method creates the perspective transformation matrix using 3D rotation
        angles, scale, and field of view parameters. It implements a complete 3D
        to 2D projection pipeline.

        Args:
            width (int): Width of the input image
            height (int): Height of the input image

        Returns:
            Tuple[np.ndarray, float, np.ndarray, np.ndarray]: A tuple containing:
                - np.ndarray: 3x3 perspective transformation matrix
                - float: Side length for output image
                - np.ndarray: Input 2D points
                - np.ndarray: Output 2D points
        """
        x = self.x
        y = self.y
        z = self.z
        scale = self.scale
        fV = self.fovy

        fVhalf = np.deg2rad(fV / 2.0)
        d = np.sqrt(width * width + height * height)
        sideLength = scale * d / np.cos(fVhalf)
        h = d / (2.0 * np.sin(fVhalf))
        n = h - (d / 2.0)
        f = h + (d / 2.0)

        # Translation along Z-axis by -h
        T = np.eye(4, 4)
        T[2, 3] = -h

        # Rotation matrices around x,y,z
        R = get_rotate_matrix(x, y, z)

        # Projection Matrix
        P = np.eye(4, 4)
        P[0, 0] = 1.0 / np.tan(fVhalf)
        P[1, 1] = P[0, 0]
        P[2, 2] = -(f + n) / (f - n)
        P[2, 3] = -(2.0 * f * n) / (f - n)
        P[3, 2] = -1.0
        P[3, 3] = 1.0

        # pythonic matrix multiplication
        M44 = reduce(lambda x, y: np.matmul(x, y), [P, T, R])

        # shape should be 1,4,3 for ptsIn and ptsOut since perspectiveTransform() expects data in this way.
        # In C++, this can be achieved by Mat ptsIn(1,4,CV_64FC3);
        ptsIn = np.array(
            [
                [
                    [-width / 2.0, height / 2.0, 0.0],
                    [width / 2.0, height / 2.0, 0.0],
                    [width / 2.0, -height / 2.0, 0.0],
                    [-width / 2.0, -height / 2.0, 0.0],
                ]
            ]
        )
        ptsOut = cv2.perspectiveTransform(ptsIn, M44)

        ptsInPt2f, ptsOutPt2f = self.get_warped_pnts(
            ptsIn, ptsOut, width, height, sideLength
        )

        # check float32 otherwise OpenCV throws an error
        assert ptsInPt2f.dtype == np.float32
        assert ptsOutPt2f.dtype == np.float32
        M33 = cv2.getPerspectiveTransform(ptsInPt2f, ptsOutPt2f).astype(np.float32)

        self.sl = int(sideLength)
        self.M33 = M33

        return M33, sideLength, ptsInPt2f, ptsOutPt2f
