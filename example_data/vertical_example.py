# All Effect/Layout example config
# 1. Run effect_layout_example.py, generate images in effect_layout_image
# 2. Update README.md
import inspect
import os
from pathlib import Path

from text_renderer.config import (
    RenderCfg,
    GeneratorCfg,
    FixedTextColorCfg,
)
from text_renderer.corpus import *

CURRENT_DIR = Path(os.path.abspath(os.path.dirname(__file__)))
BG_DIR = CURRENT_DIR / "bg"


OUT_DIR = CURRENT_DIR / "output"
DATA_DIR = CURRENT_DIR
# BG_DIR = DATA_DIR / "bg"
CHAR_DIR = CURRENT_DIR / "char"
FONT_DIR = CURRENT_DIR / "font"
FONT_LIST_DIR = CURRENT_DIR / "font_list"
TEXT_DIR = CURRENT_DIR / "text"

font_cfg = dict(
    font_dir=CURRENT_DIR / "font",
    font_size=(30, 31),
)

small_font_cfg = dict(
    font_dir=CURRENT_DIR / "font",
    font_size=(20, 21),
)


def base_cfg(name: str):
    return GeneratorCfg(
        num_image=5,
        save_dir=CURRENT_DIR / "vertical" / name,
        render_cfg=RenderCfg(
            bg_dir=BG_DIR,
            # corpus=EnumCorpus(
            #     EnumCorpusCfg(
            #         items=["Hello World!"],
            #         text_color_cfg=FixedTextColorCfg(),
            #         **font_cfg,
            #     ),
            # ),

            corpus=CharCorpus(
                CharCorpusCfg(
                    text_paths=[TEXT_DIR / "chn_text.txt", TEXT_DIR / "chn_text.txt"],
                    filter_by_chars=True,
                    chars_file=CHAR_DIR / "chinese_charset_v2.txt",
                    length=(10, 25),
                    **font_cfg
                ),
            ),
        ),
    )


def vertical_text():
    cfg = base_cfg(inspect.currentframe().f_code.co_name)
    cfg.render_cfg.corpus.cfg.horizontal = False
    cfg.render_cfg.corpus.cfg.char_spacing = 0.1
    return cfg


configs = [
    vertical_text()
]
