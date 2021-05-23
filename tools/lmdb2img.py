import os
import cv2

import fire
from tqdm import tqdm
from text_renderer.dataset import LmdbDataset


def hello(name="World"):
    return "Hello %s!" % name


def lmdb2img(input: str, output: str, num: int = -1):
    labels = []
    if os.path.exists(output):
        print(f"Output exists.")
        return
    else:
        os.makedirs(output)

    with LmdbDataset(input) as db:
        count = db.read_count()

        if num == -1 or num > count:
            convert_count = count
        else:
            convert_count = num

        print(f"Total count: {count}, will convert: {convert_count}")
        for i in tqdm(range(convert_count)):
            num = "{:09d}".format(i)
            ret = db.read(num)
            label = ret["label"]
            image = ret["image"]
            cv2.imwrite(os.path.join(output, num + ".jpg"), image)
            labels.append((num, label))

    with open(os.path.join(output, "label.txt"), "w", encoding="utf-8") as f:
        for num, label in labels:
            f.write(f"{num} {label}\n")


if __name__ == "__main__":
    fire.Fire(lmdb2img)
