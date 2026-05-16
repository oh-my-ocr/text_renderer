from pathlib import Path

import numpy as np

from text_renderer.config import RenderCfg
from text_renderer.corpus import EnumCorpus, EnumCorpusCfg
from text_renderer.render import Render

EXAMPLE_DATA = Path(__file__).resolve().parents[2] / "example_data"


def _make_render(gray: bool = True) -> Render:
    corpus = EnumCorpus(
        EnumCorpusCfg(
            text_paths=[EXAMPLE_DATA / "text" / "enum_text.txt"],
            font_dir=EXAMPLE_DATA / "font",
            font_list_file=EXAMPLE_DATA / "font_list" / "font_list.txt",
            font_size=(30, 31),
        )
    )
    cfg = RenderCfg(bg_dir=EXAMPLE_DATA / "bg", corpus=corpus, gray=gray)
    return Render(cfg)


def test_render_returns_image_and_label():
    render = _make_render()
    img, text = render()

    assert isinstance(img, np.ndarray)
    assert img.ndim in (2, 3)
    assert img.size > 0
    assert img.dtype == np.uint8
    assert isinstance(text, str)
    assert len(text) > 0


def test_render_color_mode_produces_three_channels():
    render = _make_render(gray=False)
    img, _ = render()

    assert img.ndim == 3
    assert img.shape[2] == 3
