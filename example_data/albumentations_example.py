import inspect
from pathlib import Path

from text_renderer.config import (
    RenderCfg,
    GeneratorCfg,
    NormPerspectiveTransformCfg,
    FixedTextColorCfg,
    FixedPerspectiveTransformCfg,
)
from text_renderer.corpus import (
    EnumCorpusCfg,
    EnumCorpus,
    CharCorpusCfg,
    CharCorpus,
)
from text_renderer.effect import (
    Effects,
    NoEffects,
    Padding,
    DropoutRand,
    Line,
    AlbumentationsEffect,
    AlbumentationsEmboss,
    AlbumentationsMotionBlur,
    GaussianBlur,
    Noise,
    BrightnessContrast,
    Rotate,
    ShiftScaleRotate,
    ElasticTransform,
    GridDistortion,
    OpticalDistortion,
)
from text_renderer.layout import (
    SameLineLayout,
    ExtraTextLineLayout,
)
import albumentations as A

TEXT_DIR = Path(__file__).parent / "text"
CHAR_DIR = Path(__file__).parent / "char"
FONT_DIR = Path(__file__).parent / "font"

font_cfg = {
    "font_dir": FONT_DIR,
    "font_list_file": Path(__file__).parent / "font_list" / "font_list.txt",
    "font_size": (30, 35),
}


def base_cfg(name, **kwargs):
    return GeneratorCfg(
        num_image=50,
        save_dir=Path(__file__).parent / "output" / name,
        render_cfg=RenderCfg(
            height=32,
            **kwargs,
        ),
    )


def get_char_corpus():
    return [
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
    ]


def chn_data():
    return base_cfg(
        inspect.currentframe().f_code.co_name,
        corpus=get_char_corpus(),
        corpus_effects=Effects([Padding(), DropoutRand()]),
    )


def enum_data():
    return base_cfg(
        inspect.currentframe().f_code.co_name,
        corpus=[
            EnumCorpus(
                EnumCorpusCfg(
                    text_paths=[TEXT_DIR / "enum_text.txt"],
                    filter_by_chars=True,
                    chars_file=CHAR_DIR / "chn.txt",
                    **font_cfg
                ),
            ),
        ],
        corpus_effects=Effects([Padding(), DropoutRand()]),
    )


def rand_data():
    return base_cfg(
        inspect.currentframe().f_code.co_name,
        corpus=get_char_corpus(),
        corpus_effects=Effects([Padding(), DropoutRand()]),
        layout_effects=Effects(Line(p=1)),
    )


def eng_word_data():
    return base_cfg(
        inspect.currentframe().f_code.co_name,
        corpus=[
            CharCorpus(
                CharCorpusCfg(
                    text_paths=[TEXT_DIR / "eng_text.txt"],
                    filter_by_chars=True,
                    chars_file=CHAR_DIR / "eng.txt",
                    length=(5, 10),
                    font_dir=font_cfg["font_dir"],
                    font_list_file=font_cfg["font_list_file"],
                    font_size=(30, 35),
                ),
            ),
        ],
        corpus_effects=Effects([Padding(), DropoutRand()]),
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
            ]
        ),
    )


def albumentations_motion_blur_example():
    return base_cfg(
        inspect.currentframe().f_code.co_name,
        corpus=get_char_corpus(),
        corpus_effects=Effects(
            [
                Padding(p=1, w_ratio=[0.2, 0.21], h_ratio=[0.7, 0.71], center=True),
                AlbumentationsMotionBlur(blur_limit=(3, 7)),
            ]
        ),
    )


def albumentations_gaussian_blur_example():
    return base_cfg(
        inspect.currentframe().f_code.co_name,
        corpus=get_char_corpus(),
        corpus_effects=Effects(
            [
                Padding(p=1, w_ratio=[0.2, 0.21], h_ratio=[0.7, 0.71], center=True),
                GaussianBlur(blur_limit=(3, 7)),
            ]
        ),
    )


def albumentations_noise_example():
    return base_cfg(
        inspect.currentframe().f_code.co_name,
        corpus=get_char_corpus(),
        corpus_effects=Effects(
            [
                Padding(p=1, w_ratio=[0.2, 0.21], h_ratio=[0.7, 0.71], center=True),
                Noise(var_limit=(10.0, 50.0)),
            ]
        ),
    )


def albumentations_brightness_contrast_example():
    return base_cfg(
        inspect.currentframe().f_code.co_name,
        corpus=get_char_corpus(),
        corpus_effects=Effects(
            [
                Padding(p=1, w_ratio=[0.2, 0.21], h_ratio=[0.7, 0.71], center=True),
                BrightnessContrast(brightness_limit=0.2, contrast_limit=0.2),
            ]
        ),
    )


def albumentations_rotate_example():
    return base_cfg(
        inspect.currentframe().f_code.co_name,
        corpus=get_char_corpus(),
        corpus_effects=Effects(
            [
                Padding(p=1, w_ratio=[0.2, 0.21], h_ratio=[0.7, 0.71], center=True),
                Rotate(limit=10),
            ]
        ),
    )


def albumentations_shift_scale_rotate_example():
    return base_cfg(
        inspect.currentframe().f_code.co_name,
        corpus=get_char_corpus(),
        corpus_effects=Effects(
            [
                Padding(p=1, w_ratio=[0.2, 0.21], h_ratio=[0.7, 0.71], center=True),
                ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=10),
            ]
        ),
    )


def albumentations_elastic_transform_example():
    return base_cfg(
        inspect.currentframe().f_code.co_name,
        corpus=get_char_corpus(),
        corpus_effects=Effects(
            [
                Padding(p=1, w_ratio=[0.2, 0.21], h_ratio=[0.7, 0.71], center=True),
                ElasticTransform(alpha=1, sigma=50, alpha_affine=50),
            ]
        ),
    )


def albumentations_grid_distortion_example():
    return base_cfg(
        inspect.currentframe().f_code.co_name,
        corpus=get_char_corpus(),
        corpus_effects=Effects(
            [
                Padding(p=1, w_ratio=[0.2, 0.21], h_ratio=[0.7, 0.71], center=True),
                GridDistortion(num_steps=5, distort_limit=0.3),
            ]
        ),
    )


def albumentations_optical_distortion_example():
    return base_cfg(
        inspect.currentframe().f_code.co_name,
        corpus=get_char_corpus(),
        corpus_effects=Effects(
            [
                Padding(p=1, w_ratio=[0.2, 0.21], h_ratio=[0.7, 0.71], center=True),
                OpticalDistortion(distort_limit=0.05, shift_limit=0.05),
            ]
        ),
    )


def albumentations_custom_transform_example():
    # Example of using a custom Albumentations transform
    custom_transform = A.Compose([
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=1.0),
        A.GaussNoise(var_limit=(10.0, 50.0), p=1.0),
        A.MotionBlur(blur_limit=(3, 7), p=1.0),
    ])
    
    return base_cfg(
        inspect.currentframe().f_code.co_name,
        corpus=get_char_corpus(),
        corpus_effects=Effects(
            [
                Padding(p=1, w_ratio=[0.2, 0.21], h_ratio=[0.7, 0.71], center=True),
                AlbumentationsEffect(transform=custom_transform),
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
    albumentations_motion_blur_example(),
    albumentations_gaussian_blur_example(),
    albumentations_noise_example(),
    albumentations_brightness_contrast_example(),
    albumentations_rotate_example(),
    albumentations_shift_scale_rotate_example(),
    albumentations_elastic_transform_example(),
    albumentations_grid_distortion_example(),
    albumentations_optical_distortion_example(),
    albumentations_custom_transform_example(),
]
# fmt: on
