from .base_effect import Effect, Effects, NoEffects
from .selector import OneOf
from .dropout_rand import DropoutRand
from .dropout_horizontal import DropoutHorizontal
from .dropout_vertical import DropoutVertical
from .line import Line
from .padding import Padding

# Import Albumentations effects
from .albumentations_effect import (
    AlbumentationsEffect,
    Emboss as AlbumentationsEmboss,
    MotionBlur as AlbumentationsMotionBlur,
    GaussianBlur,
    Noise,
    BrightnessContrast,
    Rotate,
    ShiftScaleRotate,
    ElasticTransform,
    GridDistortion,
    OpticalDistortion
)

# Import imgaug effects if available (legacy support)
try:
    from .imgaug_effect import ImgAugEffect, Emboss, MotionBlur
    IMGAUG_AVAILABLE = True
except ImportError:
    IMGAUG_AVAILABLE = False
    # Create placeholder classes for backward compatibility
    class ImgAugEffect:
        def __init__(self, *args, **kwargs):
            raise ImportError("imgaug is not installed. Please install imgaug or use Albumentations effects instead.")
    
    class Emboss:
        def __init__(self, *args, **kwargs):
            raise ImportError("imgaug is not installed. Please use AlbumentationsEmboss instead.")
    
    class MotionBlur:
        def __init__(self, *args, **kwargs):
            raise ImportError("imgaug is not installed. Please use AlbumentationsMotionBlur instead.")


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
    "ImgAugEffect",
    "Emboss",
    "MotionBlur",
    "AlbumentationsEffect",
    "AlbumentationsEmboss",
    "AlbumentationsMotionBlur",
    "GaussianBlur",
    "Noise",
    "BrightnessContrast",
    "Rotate",
    "ShiftScaleRotate",
    "ElasticTransform",
    "GridDistortion",
    "OpticalDistortion"
]
