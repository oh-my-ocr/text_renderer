from tempfile import TemporaryDirectory

import numpy as np

from text_renderer.dataset import ImgDataset, LmdbDataset


def test_lmdb():
    height, width = 5, 10
    img = np.random.randint(0, 255, (height, width), dtype=np.uint8)
    idx = 1
    label = "hello"
    with TemporaryDirectory() as d:
        name = f"{idx:09d}"
        with LmdbDataset(d) as dataset:
            dataset.write(name, img, label)
            dataset.write_count(1)

        with LmdbDataset(d) as dataset:
            data = dataset.read(name)
            assert "image" in data
            # when encode, jpg will be compressed
            # print(np.array_equal(data["image"], img))
            assert data["label"] == label
            assert data["size"] == [width, height]
            assert dataset.read_count() == 1


def test_img_dataset():
    height, width = 5, 10
    img = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
    idx = 1
    label = "hello"
    name = f"{idx:09d}"
    with TemporaryDirectory() as d:
        with ImgDataset(d) as dataset:
            dataset.write(name, img, label)
            dataset.write_count(1)

        with ImgDataset(d) as dataset:
            data = dataset.read(name)
            assert "image" in data
            # jpg is lossy, only assert shape rather than pixel equality
            assert data["image"].shape == img.shape
            assert data["label"] == label
            assert tuple(data["size"]) == (width, height)
            assert dataset.read_count() == 1
            assert tuple(dataset.read_size(name)) == (width, height)
