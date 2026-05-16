# Import Albumentations effects
from .albumentations_effect import (
    AlbumentationsEffect,
    BrightnessContrast,
    ElasticTransform,
    GaussianBlur,
    GridDistortion,
    Noise,
    OpticalDistortion,
    PoissonNoise,
    Rotate,
    SaltPepperNoise,
    ShiftScaleRotate,
    UniformNoise,
)
from .albumentations_effect import Emboss as AlbumentationsEmboss
from .albumentations_effect import MotionBlur as AlbumentationsMotionBlur
from .base_effect import Effect, Effects, NoEffects
from .curve import Curve
from .dropout_horizontal import DropoutHorizontal
from .dropout_rand import DropoutRand
from .dropout_vertical import DropoutVertical
from .line import Line
from .padding import Padding
from .selector import OneOf
from .text_border import TextBorder

# Create aliases for backward compatibility
Emboss = AlbumentationsEmboss
MotionBlur = AlbumentationsMotionBlur

__all__ = [
    "Effect",
    "Effects",
    "NoEffects",
    "OneOf",
    "DropoutRand",
    "DropoutHorizontal",
    "DropoutVertical",
    "Line",
    "Padding",
    "TextBorder",
    "Curve",
    "AlbumentationsEffect",
    "AlbumentationsEmboss",
    "AlbumentationsMotionBlur",
    "Emboss",
    "MotionBlur",
    "GaussianBlur",
    "Noise",
    "UniformNoise",
    "SaltPepperNoise",
    "PoissonNoise",
    "BrightnessContrast",
    "Rotate",
    "ShiftScaleRotate",
    "ElasticTransform",
    "GridDistortion",
    "OpticalDistortion",
]
