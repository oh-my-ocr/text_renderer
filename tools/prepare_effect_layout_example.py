import os
from pathlib import Path
import shutil
import pandas as pd

CURRENT_DIR = Path(os.path.abspath(os.path.dirname(__file__)))

if __name__ == "__main__":
    effect_layout_image_dir = (
        CURRENT_DIR.parent / "example_data" / "effect_layout_image"
    )

    for subdir in effect_layout_image_dir.iterdir():
        if subdir.is_dir():
            img_p = subdir / "images" / "000000000.jpg"
            img_save_path = effect_layout_image_dir / (subdir.name + ".jpg")
            shutil.copy(img_p, img_save_path)
            shutil.rmtree(subdir)

    markdown_data = []
    for img_p in effect_layout_image_dir.glob("*.jpg"):
        markdown_data.append(
            {
                "Name": img_p.stem,
                "Example": f"![{img_p.name}](https://github.com/oh-my-ocr/text_renderer/raw/master/example_data/effect_layout_image/{img_p.name})",
            }
        )
    markdown_data.sort(key=lambda x: x["Name"])
    df = pd.DataFrame(markdown_data)
    markdown_table = df.to_markdown()
    print(markdown_table)
