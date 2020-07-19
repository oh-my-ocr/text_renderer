import os
import json
from typing import Dict

import lmdb
import cv2
import numpy as np


class Dataset:
    def __init__(self, data_dir: str):
        self.data_dir = data_dir
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)

    def write(self, name: str, image: np.ndarray, label: str):
        pass

    def read(self, name) -> Dict:
        """
        Args:
            name:
        Returns: Dict
            {
                "image": ndarray,
                "label": label
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
    save label and count in json
    """

    LABEL_NAME = "labels.json"

    def __init__(self, data_dir: str):
        super().__init__(data_dir)
        self._img_dir = os.path.join(data_dir, "images")
        if not os.path.exists(self._img_dir):
            os.makedirs(self._img_dir)
        self._label_path = os.path.join(data_dir, self.LABEL_NAME)

        # labels: {'00000000': "test", "00000001": "text2"}
        # count: 2
        self._data = {"count": 0, "labels": {}, "sizes": {}}
        if os.path.exists(self._label_path):
            with open(self._label_path, "r", encoding="utf-8") as f:
                self._data = json.load(f)

    def write(self, name: str, image: np.ndarray, label: str):
        img_path = os.path.join(self._img_dir, name + ".jpg")
        cv2.imwrite(img_path, image)
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
        return self._data.get("count", 0)

    def write_count(self, count: int):
        self._data["count"] = count

    def close(self):
        with open(self._label_path, "w", encoding="utf-8") as f:
            json.dump(self._data, f, indent=2, ensure_ascii=False)


class LmdbDataset(Dataset):
    def __init__(self, data_dir: str):
        super().__init__(data_dir)
        self._lmdb_env = lmdb.open(self.data_dir, map_size=1099511627776)  # 1T
        self._lmdb_txn = self._lmdb_env.begin(write=True)

    def write(self, name: str, image: np.ndarray, label: str):
        image_key = f"image_{name}"
        label_key = f"label_{name}"
        size_key = f"size_{name}"

        self._lmdb_txn.put(image_key.encode(), cv2.imencode(".jpg", image)[1].tobytes())
        self._lmdb_txn.put(label_key.encode(), label.encode())

        height, width = image.shape[:2]
        self._lmdb_txn.put(size_key.encode(), f"{width},{height}".encode())

    def read(self, name: str) -> Dict:
        image_key = f"image_{name}"
        label_key = f"label_{name}"
        size_key = f"label_{name}"

        label = self._lmdb_txn.get(label_key.encode()).decode()
        size_str = self._lmdb_txn.get(size_key.encode()).decode()
        size = [int(it) for it in size_str.split(",")]

        image_bytes = self._lmdb_txn.get(image_key.encode())
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
        count = self._lmdb_txn.get("count".encode())
        if count is None:
            return 0
        return int(count)

    def write_count(self, count: int):
        self._lmdb_txn.put("count".encode(), str(count).encode())

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
