import os
import json
from typing import Dict

import lmdb
import cv2
import numpy as np


class Dataset:
    def __init__(self, data_dir: str, jpg_quality: int = 95):
        self.data_dir = data_dir
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)
        self.jpg_quality = jpg_quality

    def encode_param(self):
        return [int(cv2.IMWRITE_JPEG_QUALITY), self.jpg_quality]

    def write(self, name: str, image: np.ndarray, label: str):
        pass

    def read(self, name) -> Dict:
        """

        Parameters
        ----------
            name : str
                000000001

        Returns
        -------
            dict :

                .. code-block:: bash

                    {
                        "image": ndarray,
                        "label": "label",
                        "size": [int_width, int_height]
                    }
        """
        pass

    def read_count(self) -> int:
        pass

    def write_count(self, count: int):
        pass

    def close(self):
        pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


class ImgDataset(Dataset):
    """
    Save generated image as jpg file, save label and meta in json
    json file format:

    .. code-block:: bash

        {
             "labels": {
                "000000000": "test",
                "000000001": "text2"
             },
             "sizes": {
                "000000000": [width, height],
                "000000001": [width, height],
             }
             "num-samples": 2,
        }
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

    def read_size(self, name: str) -> [int, int]:
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
    Save generated image into lmdb. Compatible with https://github.com/PaddlePaddle/PaddleOCR
    Keys in lmdb:

        - image-000000001: image raw bytes
        - label-000000001: string
        - size-000000001: "width,height"

    """

    def __init__(self, data_dir: str):
        super().__init__(data_dir)
        self._lmdb_env = lmdb.open(self.data_dir, map_size=1099511627776)  # 1T
        self._lmdb_txn = self._lmdb_env.begin(write=True)

    def write(self, name: str, image: np.ndarray, label: str):
        self._lmdb_txn.put(
            self.image_key(name),
            cv2.imencode(".jpg", image, self.encode_param())[1].tobytes(),
        )
        self._lmdb_txn.put(self.label_key(name), label.encode())

        height, width = image.shape[:2]
        self._lmdb_txn.put(self.size_key(name), f"{width},{height}".encode())

    def read(self, name: str) -> Dict:
        label = self._lmdb_txn.get(self.label_key(name)).decode()
        size_str = self._lmdb_txn.get(self.size_key(name)).decode()
        size = [int(it) for it in size_str.split(",")]

        image_bytes = self._lmdb_txn.get(self.image_key(name))
        image_buf = np.frombuffer(image_bytes, dtype=np.uint8)
        image = cv2.imdecode(image_buf, cv2.IMREAD_UNCHANGED)

        return {"image": image, "label": label, "size": size}

    def read_size(self, name: str) -> [int, int]:
        """

        Args:
            name:

        Returns: (width, height)

        """
        size_key = f"size_{name}"

        size = self._lmdb_txn.get(size_key.encode()).decode()
        width = int(size.split[","][0])
        height = int(size.split[","][1])

        return width, height

    def read_count(self) -> int:
        count = self._lmdb_txn.get("num-samples".encode())
        if count is None:
            return 0
        return int(count)

    def write_count(self, count: int):
        self._lmdb_txn.put("num-samples".encode(), str(count).encode())

    def image_key(self, name: str):
        return f"image-{name}".encode()

    def label_key(self, name: str):
        return f"label-{name}".encode()

    def size_key(self, name: str):
        return f"size-{name}".encode()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
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
