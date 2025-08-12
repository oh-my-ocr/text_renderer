# Migration Guide: From imgaug to Albumentations

This guide helps you migrate from the deprecated imgaug library to Albumentations in the text_renderer project.

## Overview

The text_renderer project now supports both imgaug (legacy) and Albumentations for image augmentation. Albumentations is the recommended choice as it's actively maintained and provides better performance.

## Key Changes

### 1. Import Changes

**Before (imgaug):**
```python
from text_renderer.effect import ImgAugEffect, Emboss, MotionBlur
import imgaug.augmenters as iaa
```

**After (Albumentations):**
```python
from text_renderer.effect import (
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
    OpticalDistortion
)
import albumentations as A
```

### 2. Effect Usage

**Before (imgaug):**
```python
# Using predefined effects
ImgAugEffect(aug=iaa.Emboss(alpha=(0.9, 1.0), strength=(1.5, 1.6)))
ImgAugEffect(aug=iaa.MotionBlur(k=(3, 7), angle=(0, 360), direction=(-1.0, 1.0)))

# Using custom imgaug augmenter
custom_aug = iaa.Sequential([
    iaa.GaussianBlur(sigma=(0, 0.5)),
    iaa.AdditiveGaussianNoise(scale=(0, 0.05*255))
])
ImgAugEffect(aug=custom_aug)
```

**After (Albumentations):**
```python
# Using predefined effects
AlbumentationsEmboss(alpha=(0.9, 1.0), strength=(1.5, 1.6))
AlbumentationsMotionBlur(blur_limit=(3, 7))

# Using custom Albumentations transform
custom_transform = A.Compose([
    A.GaussianBlur(blur_limit=(3, 7), p=1.0),
    A.GaussNoise(var_limit=(10.0, 50.0), p=1.0)
])
AlbumentationsEffect(transform=custom_transform)
```

### 3. Available Effects

#### Direct Replacements

| imgaug Effect | Albumentations Effect | Notes |
|---------------|----------------------|-------|
| `iaa.Emboss` | `AlbumentationsEmboss` | Simplified implementation |
| `iaa.MotionBlur` | `AlbumentationsMotionBlur` | Direct replacement |
| `iaa.GaussianBlur` | `GaussianBlur` | Direct replacement |
| `iaa.AdditiveGaussianNoise` | `Noise` | Direct replacement |
| `iaa.Multiply` | `BrightnessContrast` | Similar functionality |

#### New Effects Available

- `Rotate` - Image rotation
- `ShiftScaleRotate` - Combined shift, scale, and rotate
- `ElasticTransform` - Elastic deformation
- `GridDistortion` - Grid-based distortion
- `OpticalDistortion` - Optical lens distortion

### 4. Parameter Mapping

#### Emboss Effect

**imgaug:**
```python
iaa.Emboss(alpha=(0.9, 1.0), strength=(1.5, 1.6))
```

**Albumentations:**
```python
AlbumentationsEmboss(alpha=(0.9, 1.0), strength=(1.5, 1.6))
```
*Note: The Albumentations emboss is a simplified implementation using brightness/contrast and noise.*

#### Motion Blur

**imgaug:**
```python
iaa.MotionBlur(k=(3, 7), angle=(0, 360), direction=(-1.0, 1.0))
```

**Albumentations:**
```python
AlbumentationsMotionBlur(blur_limit=(3, 7))
```
*Note: Albumentations MotionBlur doesn't support angle and direction parameters directly.*

### 5. Custom Transforms

**imgaug:**
```python
custom_aug = iaa.Sequential([
    iaa.GaussianBlur(sigma=(0, 0.5)),
    iaa.AdditiveGaussianNoise(scale=(0, 0.05*255)),
    iaa.Multiply((0.8, 1.2))
])
ImgAugEffect(aug=custom_aug)
```

**Albumentations:**
```python
custom_transform = A.Compose([
    A.GaussianBlur(blur_limit=(3, 7), p=1.0),
    A.GaussNoise(var_limit=(10.0, 50.0), p=1.0),
    A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=1.0)
])
AlbumentationsEffect(transform=custom_transform)
```

## Migration Steps

### Step 1: Update Dependencies

Replace imgaug with albumentations in your requirements:

```bash
pip uninstall imgaug
pip install albumentations>=1.3.0
```

### Step 2: Update Imports

Replace imgaug imports with Albumentations imports in your code.

### Step 3: Update Effect Usage

Replace `ImgAugEffect` with `AlbumentationsEffect` and update parameter names as needed.

### Step 4: Test Your Code

Run your existing code with the new Albumentations effects to ensure they work as expected.

## Examples

### Complete Example Migration

**Before (imgaug):**
```python
import imgaug.augmenters as iaa
from text_renderer.effect import ImgAugEffect, Effects, Padding

corpus_effects = Effects([
    Padding(p=1, w_ratio=[0.2, 0.21], h_ratio=[0.7, 0.71], center=True),
    ImgAugEffect(aug=iaa.Emboss(alpha=(0.9, 1.0), strength=(1.5, 1.6))),
])
```

**After (Albumentations):**
```python
from text_renderer.effect import AlbumentationsEmboss, Effects, Padding

corpus_effects = Effects([
    Padding(p=1, w_ratio=[0.2, 0.21], h_ratio=[0.7, 0.71], center=True),
    AlbumentationsEmboss(alpha=(0.9, 1.0), strength=(1.5, 1.6)),
])
```

## Benefits of Albumentations

1. **Active Development**: Albumentations is actively maintained and regularly updated
2. **Better Performance**: Generally faster than imgaug
3. **More Transforms**: Rich set of augmentation techniques
4. **Better Documentation**: Comprehensive documentation and examples
5. **Wider Community**: Larger user base and community support

## Backward Compatibility

The original imgaug effects are still available for backward compatibility:

```python
from text_renderer.effect import ImgAugEffect, Emboss, MotionBlur
import imgaug.augmenters as iaa

# This still works
ImgAugEffect(aug=iaa.Emboss(alpha=(0.9, 1.0), strength=(1.5, 1.6)))
```

However, it's recommended to migrate to Albumentations for new projects.

## Need Help?

- Check the [Albumentations documentation](https://albumentations.ai/docs/)
- Look at the [albumentations_example.py](example_data/albumentations_example.py) file for usage examples
- Open an issue on the project repository if you encounter problems
