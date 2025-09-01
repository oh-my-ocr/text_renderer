"""
Dataset utilities for text rendering operations.

This module provides dataset classes for storing and retrieving generated text images
and their corresponding labels. It supports both image file storage and LMDB database storage.
"""

import json
import os
from typing import Dict, Tuple

import cv2
import lmdb
import numpy as np


class Dataset:
    """
    Abstract base class for dataset storage and retrieval.

    This class provides a common interface for storing generated text images
    and their corresponding labels. It supports both image file storage and
    database storage formats.

    Args:
        data_dir (str): Directory path for storing dataset files
        jpg_quality (int): JPEG compression quality (1-100, default: 95)
    """

    def __init__(self, data_dir: str, jpg_quality: int = 95):
        self.data_dir = data_dir
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)
        self.jpg_quality = jpg_quality

    def encode_param(self) -> list:
        """
        Get JPEG encoding parameters for image compression.

        Returns:
            list: OpenCV JPEG encoding parameters
        """
        return [int(cv2.IMWRITE_JPEG_QUALITY), self.jpg_quality]

    def write(self, name: str, image: np.ndarray, label: str):
        """
        Write an image and its label to the dataset.

        Args:
            name (str): Unique identifier for the image
            image (np.ndarray): Image data as numpy array
            label (str): Text label corresponding to the image
        """
        pass

    def read(self, name: str) -> Dict:
        """
        Read an image and its metadata from the dataset.

        Args:
            name (str): Unique identifier for the image

        Returns:
            Dict: Dictionary containing:
                - "image": Image data as numpy array
                - "label": Text label for the image
                - "size": [width, height] of the image
        """
        pass

    def read_count(self) -> int:
        """
        Get the total number of samples in the dataset.

        Returns:
            int: Number of samples in the dataset
        """
        pass

    def write_count(self, count: int):
        """
        Write the total count of samples to the dataset.

        Args:
            count (int): Total number of samples
        """
        pass

    def close(self):
        """
        Close the dataset and release any resources.
        """
        pass

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()


class ImgDataset(Dataset):
    """
    Save generated images as JPEG files with labels and metadata in JSON.

    This dataset implementation stores images as individual JPEG files in an
    'images' subdirectory and maintains a JSON file with labels and metadata.

    JSON file format:

    .. code-block:: text

        {
            "labels": {
                "000000000": "test",
                "000000001": "text2"
            },
            "sizes": {
                "000000000": [width, height],
                "000000001": [width, height]
            },
            "num-samples": 2
        }

    Where:
        - width: Image width in pixels
        - height: Image height in pixels

    Args:
        data_dir (str): Directory path for storing dataset files
    """

    LABEL_NAME = "labels.json"

    def __init__(self, data_dir: str):
        super().__init__(data_dir)
        self._img_dir = os.path.join(data_dir, "images")
        if not os.path.exists(self._img_dir):
            os.makedirs(self._img_dir)
        self._label_path = os.path.join(data_dir, self.LABEL_NAME)

        self._data = {"num-samples": 0, "labels": {}, "sizes": {}}
        if os.path.exists(self._label_path):
            with open(self._label_path, "r", encoding="utf-8") as f:
                self._data = json.load(f)

    def write(self, name: str, image: np.ndarray, label: str):
        """
        Write an image as JPEG file and update the JSON metadata.

        Args:
            name (str): Unique identifier for the image
            image (np.ndarray): Image data as numpy array
            label (str): Text label corresponding to the image
        """
        img_path = os.path.join(self._img_dir, name + ".jpg")
        cv2.imwrite(img_path, image, self.encode_param())
        self._data["labels"][name] = label

        height, width = image.shape[:2]
        self._data["sizes"][name] = (width, height)

    def read(self, name: str) -> Dict:
        img_path = os.path.join(self._img_dir, name + ".jpg")
        image = cv2.imread(img_path)
        label = self._data["labels"][name]
        size = self._data["sizes"][name]
        return {"image": image, "label": label, "size": size}

    def read_size(self, name: str) -> Tuple[int, int]:
        """
        Read only the size information for an image.

        Args:
            name (str): Unique identifier for the image

        Returns:
            Tuple[int, int]: (width, height) of the image
        """
        return self._data["sizes"][name]

    def read_count(self) -> int:
        return self._data.get("num-samples", 0)

    def write_count(self, count: int):
        self._data["num-samples"] = count

    def close(self):
        with open(self._label_path, "w", encoding="utf-8") as f:
            json.dump(self._data, f, indent=2, ensure_ascii=False)


class LmdbDataset(Dataset):
    """
    Save generated images into LMDB database format.

    This dataset implementation stores images in LMDB (Lightning Memory-Mapped Database)
    format, which is compatible with PaddleOCR and provides efficient storage and
    retrieval for large datasets.

    LMDB Keys format:
        - image-{name}: Image raw bytes (JPEG encoded)
        - label-{name}: Text label as string
        - size-{name}: Image dimensions as "width,height" string
        - num-samples: Total number of samples in the dataset

    Args:
        data_dir (str): Directory path for the LMDB database
    """

    def __init__(self, data_dir: str):
        """
        Initialize the LMDB dataset.

        Args:
            data_dir (str): Directory path for the LMDB database
        """
        super().__init__(data_dir)
        self._lmdb_env = lmdb.open(self.data_dir, map_size=1099511627776)  # 1T
        self._lmdb_txn = self._lmdb_env.begin(write=True)

    def write(self, name: str, image: np.ndarray, label: str):
        """
        Write an image and its label to the LMDB database.

        Args:
            name (str): Unique identifier for the image
            image (np.ndarray): Image data as numpy array
            label (str): Text label corresponding to the image
        """
        self._lmdb_txn.put(
            self.image_key(name),
            cv2.imencode(".jpg", image, self.encode_param())[1].tobytes(),
        )
        self._lmdb_txn.put(self.label_key(name), label.encode())

        height, width = image.shape[:2]
        self._lmdb_txn.put(self.size_key(name), f"{width},{height}".encode())

    def read(self, name: str) -> Dict:
        """
        Read an image and its metadata from the LMDB database.

        Args:
            name (str): Unique identifier for the image

        Returns:
            Dict: Dictionary containing:
                - "image": Image data as numpy array
                - "label": Text label for the image
                - "size": [width, height] of the image
        """
        label = self._lmdb_txn.get(self.label_key(name)).decode()
        size_str = self._lmdb_txn.get(self.size_key(name)).decode()
        size = [int(it) for it in size_str.split(",")]

        image_bytes = self._lmdb_txn.get(self.image_key(name))
        image_buf = np.frombuffer(image_bytes, dtype=np.uint8)
        image = cv2.imdecode(image_buf, cv2.IMREAD_UNCHANGED)

        return {"image": image, "label": label, "size": size}

    def read_size(self, name: str) -> Tuple[int, int]:
        """
        Read only the size information for an image from the LMDB database.

        Args:
            name (str): Unique identifier for the image

        Returns:
            Tuple[int, int]: (width, height) of the image
        """
        size_str = self._lmdb_txn.get(self.size_key(name)).decode()
        width, height = map(int, size_str.split(","))

        return width, height

    def read_count(self) -> int:
        """
        Get the total number of samples in the LMDB dataset.

        Returns:
            int: Number of samples in the dataset
        """
        count = self._lmdb_txn.get("num-samples".encode())
        if count is None:
            return 0
        return int(count)

    def write_count(self, count: int):
        """
        Write the total count of samples to the LMDB dataset.

        Args:
            count (int): Total number of samples
        """
        self._lmdb_txn.put("num-samples".encode(), str(count).encode())

    def image_key(self, name: str) -> bytes:
        """
        Generate the LMDB key for image data.

        Args:
            name (str): Image identifier

        Returns:
            bytes: Encoded key for image data
        """
        return f"image-{name}".encode()

    def label_key(self, name: str) -> bytes:
        """
        Generate the LMDB key for label data.

        Args:
            name (str): Image identifier

        Returns:
            bytes: Encoded key for label data
        """
        return f"label-{name}".encode()

    def size_key(self, name: str) -> bytes:
        """
        Generate the LMDB key for size data.

        Args:
            name (str): Image identifier

        Returns:
            bytes: Encoded key for size data
        """
        return f"size-{name}".encode()

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        """
        Context manager exit.

        This method properly closes the LMDB transaction and environment
        to ensure data is committed and resources are released.
        """
        self._lmdb_txn.__exit__(exc_type, exc_value, traceback)
        self._lmdb_env.close()


if __name__ == "__main__":
    # image = cv2.imread("f_004.jpg")
    # label = "test"

    with LmdbDataset("./test/train") as writer:
        # writer.write("test", image, label)
        writer.write_count(1)
        print(writer.read_count())

    # with LmdbDataset("train") as ld:
    #     data = ld.read("test")
    #     cv2.imwrite("test.jpg", data["image"])
