# Migration Guide: From imgaug to Albumentations

This guide helps you migrate from the deprecated imgaug library to Albumentations in the text_renderer project.

## Overview

The text_renderer project now uses Albumentations exclusively for image augmentation. The deprecated imgaug library has been completely removed from the codebase.

## Why Migrate?

- **Active Maintenance**: Albumentations is actively maintained and regularly updated
- **Better Performance**: Generally faster than imgaug
- **Modern API**: Cleaner and more intuitive API design
- **Better Documentation**: Comprehensive documentation and examples
- **Wide Adoption**: Used by many modern computer vision projects

## Migration Steps

### 1. Update Dependencies

Replace imgaug with albumentations in your requirements:

```bash
pip uninstall imgaug
pip install albumentations
```

### 2. Update Imports

Replace imgaug imports with Albumentations imports in your code.

### 3. Update Effect Usage

Replace `ImgAugEffect` with `AlbumentationsEffect` and update parameter names as needed.

## Code Examples

### Before (imgaug - No Longer Supported):

```python
from text_renderer.effect import ImgAugEffect, Emboss, MotionBlur
import imgaug.augmenters as iaa

# Using built-in imgaug effects
ImgAugEffect(aug=iaa.Emboss(alpha=(0.9, 1.0), strength=(1.5, 1.6)))
ImgAugEffect(aug=iaa.MotionBlur(k=(3, 7), angle=(0, 360), direction=(-1.0, 1.0)))

# Using custom imgaug augmenter
custom_aug = iaa.Sequential([
    iaa.GaussianBlur(sigma=(0.0, 0.5)),
    iaa.ContrastNormalization((0.75, 1.5))
])
ImgAugEffect(aug=custom_aug)
```

### After (Albumentations):

```python
from text_renderer.effect import AlbumentationsEffect, AlbumentationsEmboss, AlbumentationsMotionBlur
import albumentations as A

# Using built-in Albumentations effects
AlbumentationsEmboss(alpha=(0.9, 1.0), strength=(1.5, 1.6))
AlbumentationsMotionBlur(blur_limit=(3, 7), p=1.0)

# Using custom Albumentations augmenter
custom_aug = A.Compose([
    A.GaussianBlur(blur_limit=(0, 1), p=1.0),
    A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=1.0)
])
AlbumentationsEffect(aug=custom_aug)
```

## Effect Mapping

| imgaug Effect | Albumentations Effect | Notes |
|---------------|----------------------|-------|
| `iaa.Emboss` | `AlbumentationsEmboss` | Similar parameters |
| `iaa.MotionBlur` | `AlbumentationsMotionBlur` | Different parameter names |
| `iaa.GaussianBlur` | `GaussianBlur` | Similar functionality |
| `iaa.AdditiveGaussianNoise` | `Noise` | Different noise types available |
| `iaa.ContrastNormalization` | `BrightnessContrast` | Combined brightness/contrast |

## Available Albumentations Effects

The following Albumentations effects are available in text_renderer:

- `AlbumentationsEffect` - Generic wrapper for any Albumentations augmenter
- `AlbumentationsEmboss` - Emboss effect
- `AlbumentationsMotionBlur` - Motion blur effect
- `GaussianBlur` - Gaussian blur
- `Noise` - Various noise effects
- `BrightnessContrast` - Brightness and contrast adjustment
- `Rotate` - Rotation
- `ShiftScaleRotate` - Combined shift, scale, and rotate
- `ElasticTransform` - Elastic transformation
- `GridDistortion` - Grid distortion
- `OpticalDistortion` - Optical distortion

## Benefits of Albumentations

1. **Better Performance**: Generally faster than imgaug
2. **Active Development**: Regularly updated with new features
3. **Better Documentation**: Comprehensive docs and examples
4. **Modern API**: Cleaner and more intuitive design
5. **Wide Community**: Large user base and community support

## Complete Example

Here's a complete example showing the migration:

### Before (imgaug - No Longer Supported):

```python
import imgaug.augmenters as iaa
from text_renderer.effect import ImgAugEffect, Effects, Padding

effects = Effects([
    Padding(p=1, w_ratio=[0.2, 0.21], h_ratio=[0.7, 0.71], center=True),
    ImgAugEffect(aug=iaa.Emboss(alpha=(0.9, 1.0), strength=(1.5, 1.6))),
])
```

### After (Albumentations):

```python
from text_renderer.effect import AlbumentationsEmboss, Effects, Padding

effects = Effects([
    Padding(p=1, w_ratio=[0.2, 0.21], h_ratio=[0.7, 0.71], center=True),
    AlbumentationsEmboss(alpha=(0.9, 1.0), strength=(1.5, 1.6)),
])
```

## Need Help?

If you encounter any issues during migration, please:

1. Check the [Albumentations documentation](https://albumentations.ai/docs/)
2. Review the [text_renderer examples](https://github.com/oh-my-ocr/text_renderer/tree/master/example_data)
3. Open an issue on the GitHub repository
