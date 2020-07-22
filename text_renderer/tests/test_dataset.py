from tempfile import TemporaryDirectory
import numpy as np

from text_renderer.dataset import LmdbDataset


def test_lmdb():
    height, width = 5, 10
    img = np.random.randint(0, 255, (height, width), dtype=np.uint8)
    idx = 1
    label = "hello"
    with TemporaryDirectory() as d:
        name = f"{idx:09d}"
        with LmdbDataset(d) as dataset:
            dataset.write(name, img, label)

        with LmdbDataset(d) as dataset:
            data = dataset.read(name)
            assert "image" in data
            # when encode, jpg will be compressed
            # print(np.array_equal(data["image"], img))
            assert data["label"] == label
            assert data["size"] == [width, height]
