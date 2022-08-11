import os
from pathlib import Path

from text_renderer.config import (
    RenderCfg,
    NormPerspectiveTransformCfg,
    GeneratorCfg,
)
from text_renderer.corpus import *
from text_renderer.effect import *

CURRENT_DIR = Path(os.path.abspath(os.path.dirname(__file__)))
OUT_DIR = CURRENT_DIR / "output"
DATA_DIR = CURRENT_DIR
BG_DIR = DATA_DIR / "bg"
CHAR_DIR = DATA_DIR / "char"
FONT_DIR = DATA_DIR / "font"
FONT_LIST_DIR = DATA_DIR / "font_list"
TEXT_DIR = DATA_DIR / "text"

font_cfg = dict(
    font_dir=FONT_DIR,
    font_list_file=FONT_LIST_DIR / "chn_font_list.txt",
    font_size=(30, 31),
)

perspective_transform = NormPerspectiveTransformCfg(5, 5, 1.5)

rand_data = GeneratorCfg(
    num_image=1000000,
    save_dir=OUT_DIR / "rand_corpus",
    render_cfg=RenderCfg(
        bg_dir=BG_DIR,
        perspective_transform=perspective_transform,
        corpus=RandCorpus(RandCorpusCfg(chars_file=CHAR_DIR / "idcard.supplement.random.txt", **font_cfg),),
    ),
)
configs = [
    rand_data
]
