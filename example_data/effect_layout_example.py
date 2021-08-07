# All Effect/Layout example config
# 1. Run effect_layout_example.py, generate images in effect_layout_image
# 2. Update README.md
import inspect
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from text_renderer.effect import *
from text_renderer.corpus import *
from text_renderer.config import (
    RenderCfg,
    NormPerspectiveTransformCfg,
    GeneratorCfg,
    SimpleTextColorCfg,
    TextColorCfg,
    FixedTextColorCfg,
    FixedPerspectiveTransformCfg,
)
from text_renderer.effect.curve import Curve
from text_renderer.layout import SameLineLayout, ExtraTextLineLayout

CURRENT_DIR = Path(os.path.abspath(os.path.dirname(__file__)))
BG_DIR = CURRENT_DIR / "bg"

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
        save_dir=CURRENT_DIR / "effect_layout_image" / name,
        render_cfg=RenderCfg(
            bg_dir=BG_DIR,
            corpus=EnumCorpus(
                EnumCorpusCfg(
                    items=["Hello! 你好！"],
                    text_color_cfg=FixedTextColorCfg(),
                    **font_cfg,
                ),
            ),
        ),
    )


def dropout_rand():
    cfg = base_cfg(inspect.currentframe().f_code.co_name)
    cfg.render_cfg.corpus_effects = Effects(DropoutRand(p=1, dropout_p=(0.3, 0.5)))
    return cfg


def dropout_horizontal():
    cfg = base_cfg(inspect.currentframe().f_code.co_name)
    cfg.render_cfg.corpus_effects = Effects(
        DropoutHorizontal(p=1, num_line=2, thickness=3)
    )
    return cfg


def dropout_vertical():
    cfg = base_cfg(inspect.currentframe().f_code.co_name)
    cfg.render_cfg.corpus_effects = Effects(DropoutVertical(p=1, num_line=15))
    return cfg


def line():
    poses = [
        "top",
        "bottom",
        "left",
        "right",
        "top_left",
        "top_right",
        "bottom_left",
        "bottom_right",
        "horizontal_middle",
        "vertical_middle",
    ]
    cfgs = []
    for i, pos in enumerate(poses):
        pos_p = [0] * len(poses)
        pos_p[i] = 1
        cfg = base_cfg(f"{inspect.currentframe().f_code.co_name}_{pos}")
        cfg.render_cfg.corpus_effects = Effects(
            Line(p=1, thickness=(3, 4), line_pos_p=pos_p)
        )
        cfgs.append(cfg)
    return cfgs


def padding():
    cfg = base_cfg(inspect.currentframe().f_code.co_name)
    cfg.render_cfg.corpus_effects = Effects(
        Padding(p=1, w_ratio=[0.2, 0.21], h_ratio=[0.7, 0.71], center=True)
    )
    return cfg


def same_line_layout_different_font_size():
    cfg = base_cfg(inspect.currentframe().f_code.co_name)
    cfg.render_cfg.layout = SameLineLayout(h_spacing=(0.9, 0.91))
    cfg.render_cfg.corpus = [
        EnumCorpus(
            EnumCorpusCfg(
                items=["Hello "],
                text_color_cfg=FixedTextColorCfg(),
                **font_cfg,
            ),
        ),
        EnumCorpus(
            EnumCorpusCfg(
                items=[" World!"],
                text_color_cfg=FixedTextColorCfg(),
                **small_font_cfg,
            ),
        ),
    ]
    return cfg


def extra_text_line_layout():
    cfg = base_cfg(inspect.currentframe().f_code.co_name)
    cfg.render_cfg.layout = ExtraTextLineLayout(bottom_prob=1.0)
    cfg.render_cfg.corpus = [
        EnumCorpus(
            EnumCorpusCfg(
                items=["Hello world"],
                text_color_cfg=FixedTextColorCfg(),
                **font_cfg,
            ),
        ),
        EnumCorpus(
            EnumCorpusCfg(
                items=["THIS IS AN EXTRA TEXT LINE!"],
                text_color_cfg=FixedTextColorCfg(),
                **font_cfg,
            ),
        ),
    ]
    return cfg


def color_image():
    cfg = base_cfg(inspect.currentframe().f_code.co_name)
    cfg.render_cfg.gray = False
    return cfg


def perspective_transform():
    cfg = base_cfg(inspect.currentframe().f_code.co_name)
    cfg.render_cfg.perspective_transform = FixedPerspectiveTransformCfg(30, 30, 1.5)
    return cfg


def compact_char_spacing():
    cfg = base_cfg(inspect.currentframe().f_code.co_name)
    cfg.render_cfg.corpus.cfg.char_spacing = -0.3
    return cfg


def large_char_spacing():
    cfg = base_cfg(inspect.currentframe().f_code.co_name)
    cfg.render_cfg.corpus.cfg.char_spacing = 0.5
    return cfg


def curve():
    cfg = base_cfg(inspect.currentframe().f_code.co_name)
    cfg.render_cfg.corpus_effects = Effects(
        [
            Padding(p=1, w_ratio=[0.2, 0.21], h_ratio=[0.7, 0.71], center=True),
            Curve(p=1, period=180, amplitude=(4, 5)),
        ]
    )
    return cfg


def vertical_text():
    cfg = base_cfg(inspect.currentframe().f_code.co_name)
    cfg.render_cfg.corpus.cfg.horizontal = False
    # cfg.render_cfg.corpus.cfg.char_spacing = 0.1
    return cfg


def bg_and_text_mask():
    cfg = base_cfg(inspect.currentframe().f_code.co_name)
    cfg.render_cfg.perspective_transform = FixedPerspectiveTransformCfg(30, 30, 1.5)
    cfg.render_cfg.return_bg_and_mask = True
    cfg.render_cfg.height = 48
    return cfg


def emboss():
    import imgaug.augmenters as iaa

    cfg = base_cfg(inspect.currentframe().f_code.co_name)
    cfg.render_cfg.height = 48
    cfg.render_cfg.corpus_effects = Effects(
        [
            Padding(p=1, w_ratio=[0.2, 0.21], h_ratio=[0.7, 0.71], center=True),
            ImgAugEffect(aug=iaa.Emboss(alpha=(0.9, 1.0), strength=(1.5, 1.6))),
        ]
    )
    return cfg


configs = [
    # bg_and_text_mask()
    emboss()
    # vertical_text()
    # extra_text_line_layout()
    # char_spacing_compact(),
    # char_spacing_large(),
    # *line(),
    # perspective_transform(),
    # color_image(),
    # dropout_rand(),
    # dropout_horizontal(),
    # dropout_vertical(),
    # padding(),
    # same_line_layout_different_font_size(),
]
