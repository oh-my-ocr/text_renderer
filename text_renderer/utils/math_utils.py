#!/usr/env/bin python3
from functools import reduce
from typing import Tuple

import numpy as np
import cv2
import math

from PIL import Image

from text_renderer.config import PerspectiveTransformCfg
from text_renderer.utils import utils


# http://planning.cs.uiuc.edu/node102.html
def get_rotate_matrix(x, y, z):
    """
    按照 zyx 的顺序旋转，输入角度单位为 degrees, 均为顺时针旋转
    :param x: X-axis
    :param y: Y-axis
    :param z: Z-axis
    :return:
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
    def __init__(self, cfg: PerspectiveTransformCfg):
        self.x, self.y, self.z = cfg.get_xyz()
        self.scale = cfg.scale
        self.fovy = cfg.fovy

    def get_transformed_size(self, size: Tuple[int, int]) -> Tuple[int, int]:
        """
        Args:
            size: (width, height)

        Returns:
            (width, height)
        """
        width, height = size
        _, _, _, text_box_pnts_transformed = self.gen_warp_matrix(width, height)

        transformed_bbox = cv2.boundingRect(text_box_pnts_transformed)
        bbox_width = transformed_bbox[2]
        bbox_height = transformed_bbox[3]

        return int(bbox_width), int(bbox_height)

    def do_warp_perspective(self, pil_img):
        """

        Args:
            pil_img:

        Returns:

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

    def transform_pnts(self, pnts, M33):
        """
        :param pnts: 2D pnts, left-top, right-top, right-bottom, left-bottom
        :param M33: output from transform_image()
        :return: 2D pnts apply perspective transform
        """
        pnts = np.asarray(pnts, dtype=np.float32)
        pnts = np.array([pnts])
        dst_pnts = cv2.perspectiveTransform(pnts, M33)[0]
        return np.array(dst_pnts).astype(np.int)

    def get_warped_pnts(self, ptsIn, ptsOut, W, H, sidelength):
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

    def gen_warp_matrix(self, width, height):
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
