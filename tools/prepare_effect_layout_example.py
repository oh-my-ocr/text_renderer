import shutil
from pathlib import Path

CURRENT_DIR = Path(__file__).resolve().parent

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

    rows = []
    for img_p in sorted(effect_layout_image_dir.glob("*.jpg"), key=lambda p: p.stem):
        url = f"https://github.com/oh-my-ocr/text_renderer/raw/master/example_data/effect_layout_image/{img_p.name}"
        rows.append(f"| {img_p.stem} | ![{img_p.name}]({url}) |")

    print("| Name | Example |")
    print("|------|---------|")
    print("\n".join(rows))
