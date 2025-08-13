import inspect
import os
from pathlib import Path

from text_renderer.config import (
    FixedTextColorCfg,
    GeneratorCfg,
    NormPerspectiveTransformCfg,
    RangeTextColorCfg,
    RenderCfg,
)
from text_renderer.corpus import *
from text_renderer.effect import *
from text_renderer.layout.extra_text_line import ExtraTextLineLayout
from text_renderer.layout.same_line import SameLineLayout

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
    font_list_file=FONT_LIST_DIR / "font_list.txt",
    font_size=(30, 31),
)

perspective_transform = NormPerspectiveTransformCfg(20, 20, 1.5)


def get_char_corpus():
    return CharCorpus(
        CharCorpusCfg(
            text_paths=[TEXT_DIR / "chn_text.txt", TEXT_DIR / "eng_text.txt"],
            filter_by_chars=True,
            chars_file=CHAR_DIR / "chn.txt",
            length=(5, 10),
            char_spacing=(-0.3, 1.3),
            **font_cfg
        ),
    )


def base_cfg(
    name: str, corpus, corpus_effects=None, layout_effects=None, layout=None, gray=True
):
    return GeneratorCfg(
        num_image=50,
        save_dir=OUT_DIR / name,
        render_cfg=RenderCfg(
            bg_dir=BG_DIR,
            perspective_transform=perspective_transform,
            gray=gray,
            layout_effects=layout_effects,
            layout=layout,
            corpus=corpus,
            corpus_effects=corpus_effects,
        ),
    )


def chn_data():
    return base_cfg(
        inspect.currentframe().f_code.co_name,
        corpus=get_char_corpus(),
        corpus_effects=Effects(
            [
                Line(0.5, color_cfg=FixedTextColorCfg()),
                OneOf([DropoutRand(), DropoutVertical()]),
            ]
        ),
    )


def enum_data():
    return base_cfg(
        inspect.currentframe().f_code.co_name,
        corpus=EnumCorpus(
            EnumCorpusCfg(
                text_paths=[TEXT_DIR / "enum_text.txt"],
                filter_by_chars=True,
                chars_file=CHAR_DIR / "chn.txt",
                **font_cfg
            ),
        ),
    )


def rand_data():
    return base_cfg(
        inspect.currentframe().f_code.co_name,
        corpus=RandCorpus(
            RandCorpusCfg(chars_file=CHAR_DIR / "chn.txt", **font_cfg),
        ),
    )


def eng_word_data():
    return base_cfg(
        inspect.currentframe().f_code.co_name,
        corpus=WordCorpus(
            WordCorpusCfg(
                text_paths=[TEXT_DIR / "eng_text.txt"],
                filter_by_chars=True,
                chars_file=CHAR_DIR / "eng.txt",
                **font_cfg
            ),
        ),
    )


def same_line_data():
    return base_cfg(
        inspect.currentframe().f_code.co_name,
        layout=SameLineLayout(),
        gray=False,
        corpus=[
            EnumCorpus(
                EnumCorpusCfg(
                    text_paths=[TEXT_DIR / "enum_text.txt"],
                    filter_by_chars=True,
                    chars_file=CHAR_DIR / "chn.txt",
                    **font_cfg
                ),
            ),
            CharCorpus(
                CharCorpusCfg(
                    text_paths=[
                        TEXT_DIR / "chn_text.txt",
                        TEXT_DIR / "eng_text.txt",
                    ],
                    filter_by_chars=True,
                    chars_file=CHAR_DIR / "chn.txt",
                    length=(5, 10),
                    font_dir=font_cfg["font_dir"],
                    font_list_file=font_cfg["font_list_file"],
                    font_size=(30, 35),
                ),
            ),
        ],
        corpus_effects=[Effects([Padding(), DropoutRand()]), NoEffects()],
        layout_effects=Effects(Line(p=1)),
    )


def extra_text_line_data():
    return base_cfg(
        inspect.currentframe().f_code.co_name,
        layout=ExtraTextLineLayout(),
        corpus=[
            CharCorpus(
                CharCorpusCfg(
                    text_paths=[
                        TEXT_DIR / "chn_text.txt",
                        TEXT_DIR / "eng_text.txt",
                    ],
                    filter_by_chars=True,
                    chars_file=CHAR_DIR / "chn.txt",
                    length=(9, 10),
                    font_dir=font_cfg["font_dir"],
                    font_list_file=font_cfg["font_list_file"],
                    font_size=(30, 35),
                ),
            ),
            CharCorpus(
                CharCorpusCfg(
                    text_paths=[
                        TEXT_DIR / "chn_text.txt",
                        TEXT_DIR / "eng_text.txt",
                    ],
                    filter_by_chars=True,
                    chars_file=CHAR_DIR / "chn.txt",
                    length=(9, 10),
                    font_dir=font_cfg["font_dir"],
                    font_list_file=font_cfg["font_list_file"],
                    font_size=(30, 35),
                ),
            ),
        ],
        corpus_effects=[Effects([Padding()]), NoEffects()],
        layout_effects=Effects(Line(p=1)),
    )


def albumentations_emboss_example():
    return base_cfg(
        inspect.currentframe().f_code.co_name,
        corpus=get_char_corpus(),
        corpus_effects=Effects(
            [
                Padding(p=1, w_ratio=[0.2, 0.21], h_ratio=[0.7, 0.71], center=True),
                AlbumentationsEmboss(alpha=(0.9, 1.0), strength=(1.5, 1.6)),
                GaussianBlur(blur_limit=(3, 7)),
            ]
        ),
    )


def range_color_example():
    """
    Example using RangeTextColorCfg with blue and brown color ranges
    """
    # Define color ranges with fractions
    color_ranges = {
        "blue": {
            "fraction": 0.5,
            "l_boundary": [0, 0, 150],
            "h_boundary": [60, 60, 253],
        },
        "brown": {
            "fraction": 0.5,
            "l_boundary": [139, 70, 19],
            "h_boundary": [160, 82, 43],
        },
    }

    return base_cfg(
        inspect.currentframe().f_code.co_name,
        corpus=get_char_corpus(),
        gray=False,  # Set to False to enable color output
        corpus_effects=Effects(
            [
                Line(0.5, color_cfg=RangeTextColorCfg(color_ranges=color_ranges)),
                OneOf([DropoutRand(), DropoutVertical()]),
            ]
        ),
    )


def text_border_example():
    """
    Example using TextBorder effect with different styles
    """
    return base_cfg(
        inspect.currentframe().f_code.co_name,
        corpus=get_char_corpus(),
        gray=False,  # Set to False to enable color output
        corpus_effects=Effects(
            [
                # Solid border with red color
                TextBorder(
                    p=0.7,
                    border_width=(2, 4),
                    border_color_cfg=FixedTextColorCfg(),
                    border_style="solid",
                    blur_radius=0,
                ),
                # Dashed border with blue color
                TextBorder(
                    p=0.3,
                    border_width=(1, 2),
                    border_color_cfg=RangeTextColorCfg(
                        {
                            "blue": {
                                "fraction": 1.0,
                                "l_boundary": [0, 0, 200],
                                "h_boundary": [50, 50, 255],
                            }
                        }
                    ),
                    border_style="dashed",
                    blur_radius=0,
                ),
                # Dotted border with green color
                TextBorder(
                    p=0.2,
                    border_width=(1, 3),
                    border_color_cfg=RangeTextColorCfg(
                        {
                            "green": {
                                "fraction": 1.0,
                                "l_boundary": [0, 150, 0],
                                "h_boundary": [50, 255, 50],
                            }
                        }
                    ),
                    border_style="dotted",
                    blur_radius=1,
                ),
            ]
        ),
    )


def text_border_light_dark_example():
    """
    Example using TextBorder effect with light/dark configuration
    """
    return base_cfg(
        inspect.currentframe().f_code.co_name,
        corpus=get_char_corpus(),
        gray=False,  # Set to False to enable color output
        corpus_effects=Effects(
            [
                # Text border with light/dark configuration
                TextBorder(
                    p=1.0,  # Always apply when enabled
                    border_width=(2, 3),
                    border_style="solid",
                    blur_radius=0,
                    # Configuration following the standard format
                    enable=True,
                    fraction=0.5,
                    light_enable=True,
                    light_fraction=0.5,
                    dark_enable=True,
                    dark_fraction=0.5,
                ),
                # Add some dropout effects
                OneOf([DropoutRand(), DropoutVertical()]),
            ]
        ),
    )


# fmt: off
# The configuration file must have a configs variable
configs = [
    chn_data(),
    enum_data(),
    rand_data(),
    eng_word_data(),
    same_line_data(),
    extra_text_line_data(),
    albumentations_emboss_example(),
    range_color_example(),
    text_border_example(),
    text_border_light_dark_example()
]
# fmt: on
