# All Effect/Layout example config
# 1. Run effect_layout_example.py, generate images in effect_layout_image
# 2. Push generated images to always_force_push branch
# 3. Update README.md
# 4. Update docs file

import os
from pathlib import Path

from text_renderer.effect import *
from text_renderer.corpus import *
from text_renderer.config import (
    RenderCfg,
    NormPerspectiveTransformCfg,
    GeneratorCfg,
    SimpleTextColorCfg,
)

CURRENT_DIR = Path(os.path.abspath(os.path.dirname(__file__)))
BG_DIR = CURRENT_DIR / "bg"

enum_data = GeneratorCfg(
    num_image=50,
    save_dir=CURRENT_DIR / "effect_layout_image",
    render_cfg=RenderCfg(
        bg_dir=BG_DIR,
        perspective_transform=perspective_transform,
        corpus=EnumCorpus(
            EnumCorpusCfg(
                text_paths=[TEXT_DIR / "enum_text.txt"],
                filter_by_chars=True,
                chars_file=CHAR_DIR / "chn.txt",
                **font_cfg
            ),
        ),
    ),
)

def story_data():
    return GeneratorCfg(
        num_image=10,
        save_dir=CURRENT_DIR / "output",
        render_cfg=RenderCfg(
            bg_dir=CURRENT_DIR / "bg",
            height=32,
            perspective_transform=NormPerspectiveTransformCfg(20, 20, 1.5),
            corpus=WordCorpus(
                WordCorpusCfg(
                    text_paths=[CURRENT_DIR / "corpus" / "eng_text.txt"],
                    font_dir=CURRENT_DIR / "font",
                    font_size=(20, 30),
                    num_word=(2, 3),
                ),
            ),
            corpus_effects=Effects(Line(0.9, thickness=(2, 5))),
            gray=False,
            text_color_cfg=SimpleTextColorCfg(),
        ),
    )


configs = [story_data()]
