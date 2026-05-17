from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from text_renderer.config import RenderCfg
from text_renderer.corpus import EnumCorpus, EnumCorpusCfg
from text_renderer.effect import Effects, Padding
from text_renderer.render import Render
from text_renderer.utils.errors import PanicError

EXAMPLE_DATA = Path(__file__).resolve().parents[2] / "example_data"


def _make_corpus() -> EnumCorpus:
    return EnumCorpus(
        EnumCorpusCfg(
            text_paths=[EXAMPLE_DATA / "text" / "enum_text.txt"],
            font_dir=EXAMPLE_DATA / "font",
            font_list_file=EXAMPLE_DATA / "font_list" / "font_list.txt",
            font_size=(30, 31),
        )
    )


def _make_render(
    gray: bool = True,
    return_bg_and_mask: bool = False,
    height: int = 32,
) -> Render:
    cfg = RenderCfg(
        bg_dir=EXAMPLE_DATA / "bg",
        corpus=_make_corpus(),
        gray=gray,
        return_bg_and_mask=return_bg_and_mask,
        height=height,
    )
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


def test_render_return_bg_and_mask_keeps_triple_width_after_height_normalization():
    render = _make_render(gray=False, return_bg_and_mask=True, height=32)
    size = (100, 37)
    text_img = Image.new("RGB", size, (10, 20, 30))
    bg_img = Image.new("RGB", size, (40, 50, 60))
    text_mask = Image.new("RGBA", size, (255, 255, 255, 255))

    def fake_gen_single_corpus():
        return text_img, "abc", bg_img, text_mask

    render.gen_single_corpus = fake_gen_single_corpus

    img, text = render()

    assert text == "abc"
    assert img.shape[0] == 32
    assert img.shape[1] % 3 == 0


def test_render_rejects_list_corpus_with_scalar_effects():
    cfg = RenderCfg(
        bg_dir=EXAMPLE_DATA / "bg",
        corpus=[_make_corpus(), _make_corpus()],
        corpus_effects=Effects([Padding()]),
    )
    with pytest.raises(PanicError):
        Render(cfg)


def test_render_rejects_corpus_effects_length_mismatch():
    cfg = RenderCfg(
        bg_dir=EXAMPLE_DATA / "bg",
        corpus=[_make_corpus(), _make_corpus()],
        corpus_effects=[Effects([Padding()])],
    )
    with pytest.raises(PanicError):
        Render(cfg)
