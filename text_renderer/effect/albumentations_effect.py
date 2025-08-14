from typing import List, Tuple, Union

import albumentations as A
import numpy as np
from PIL import Image

from text_renderer.utils.bbox import BBox
from text_renderer.utils.types import PILImage

from .base_effect import Effect


class AlbumentationsEffect(Effect):
    """
    Apply Albumentations transforms on image.
    """

    def __init__(self, p=1.0, transform: A.BasicTransform = None):
        super().__init__(p)
        self.transform = transform

    def apply(self, img: PILImage, text_bbox: BBox) -> Tuple[PILImage, BBox]:
        if self.transform is None:
            return img, text_bbox

        # Convert PIL image to numpy array
        img_array = np.array(img)
        
        # Check if image has alpha channel (RGBA)
        has_alpha = img_array.shape[-1] == 4
        
        if has_alpha:
            # Convert RGBA to RGB for albumentations
            # Create a white background and composite the RGBA image onto it
            rgb_array = np.zeros((img_array.shape[0], img_array.shape[1], 3), dtype=np.uint8)
            alpha = img_array[:, :, 3:4] / 255.0
            rgb_array = (img_array[:, :, :3] * alpha + (1 - alpha) * 255).astype(np.uint8)
        else:
            rgb_array = img_array

        # Apply transformation
        transformed = self.transform(image=rgb_array)
        transformed_img = transformed["image"]

        # Convert back to PIL image
        if has_alpha:
            # Convert back to RGBA by adding the original alpha channel
            rgba_array = np.zeros((transformed_img.shape[0], transformed_img.shape[1], 4), dtype=np.uint8)
            rgba_array[:, :, :3] = transformed_img
            rgba_array[:, :, 3] = img_array[:, :, 3]  # Preserve original alpha
            return Image.fromarray(rgba_array, mode='RGBA'), text_bbox
        else:
            return Image.fromarray(transformed_img), text_bbox


class Emboss(AlbumentationsEffect):
    def __init__(self, p=1.0, alpha=(0.9, 1.0), strength=(1.5, 1.6)):
        """
        Emboss effect using Albumentations

        Parameters
        ----------
        p: float
            Probability of applying this effect
        alpha: tuple
            Alpha blending factor range
        strength: tuple
            Strength of the emboss effect
        """
        # Note: Albumentations doesn't have a direct emboss transform
        # We'll use a combination of transforms to simulate emboss effect
        # This is a simplified version - you might want to implement a custom emboss
        transform = A.Compose(
            [
                A.RandomBrightnessContrast(
                    brightness_limit=0.1, contrast_limit=0.2, p=1.0
                ),
                A.GaussNoise(p=1.0),
            ]
        )
        super().__init__(p, transform)


class MotionBlur(AlbumentationsEffect):
    def __init__(self, p=1.0, blur_limit=(3, 7), angle=(0, 360), direction=(-1.0, 1.0)):
        """
        Motion blur effect using Albumentations

        Parameters
        ----------
        p: float
            Probability of applying this effect
        blur_limit: tuple
            Range for blur kernel size
        angle: tuple
            Range for blur angle in degrees
        direction: tuple
            Range for blur direction (not directly supported in Albumentations)
        """
        transform = A.MotionBlur(blur_limit=blur_limit, p=1.0)
        super().__init__(p, transform)


class GaussianBlur(AlbumentationsEffect):
    def __init__(self, p=1.0, blur_limit=(3, 7)):
        """
        Gaussian blur effect using Albumentations

        Parameters
        ----------
        p: float
            Probability of applying this effect
        blur_limit: tuple
            Range for blur kernel size
        """
        transform = A.GaussianBlur(blur_limit=blur_limit, p=1.0)
        super().__init__(p, transform)


class Noise(AlbumentationsEffect):
    def __init__(self, p=1.0, var_limit=(10.0, 50.0)):
        """
        Gaussian noise effect using Albumentations

        Parameters
        ----------
        p: float
            Probability of applying this effect
        var_limit: tuple
            Range for noise variance
        """
        transform = A.GaussNoise(p=1.0)
        super().__init__(p, transform)


class UniformNoise(AlbumentationsEffect):
    def __init__(self, p=1.0, intensity_range=(0.1, 0.3)):
        """
        Uniform noise effect using Albumentations

        Parameters
        ----------
        p: float
            Probability of applying this effect
        intensity_range: tuple
            Range for noise intensity (0.0 to 1.0)
        """
        # Use MultiplicativeNoise which adds uniform-like noise
        transform = A.MultiplicativeNoise(
            multiplier=(0.9, 1.1), per_channel=True, p=1.0
        )
        super().__init__(p, transform)


class SaltPepperNoise(AlbumentationsEffect):
    def __init__(self, p=1.0):
        """
        Salt and pepper noise effect using Albumentations

        Parameters
        ----------
        p: float
            Probability of applying this effect
        """
        transform = A.SaltAndPepper(
            p=p, salt_vs_pepper=(0.4, 0.6), amount=(0.02, 0.06)
        )
        super().__init__(p, transform)


class PoissonNoise(AlbumentationsEffect):
    def __init__(self, p=1.0, intensity=(0.1, 0.5), color_shift=(0.01, 0.05)):
        """
        Poisson noise effect using Albumentations ISONoise
        
        ISONoise simulates camera sensor noise which follows a Poisson distribution
        characteristic of photon counting noise in digital imaging.

        Parameters
        ----------
        p: float
            Probability of applying this effect
        intensity: tuple
            Range for noise intensity (0.0 to 1.0)
        color_shift: tuple
            Range for color shift values
        """
        transform = A.ISONoise(
            intensity=intensity,
            color_shift=color_shift,
            p=1.0
        )
        super().__init__(p, transform)


class BrightnessContrast(AlbumentationsEffect):
    def __init__(self, p=1.0, brightness_limit=0.2, contrast_limit=0.2):
        """
        Brightness and contrast adjustment using Albumentations

        Parameters
        ----------
        p: float
            Probability of applying this effect
        brightness_limit: float or tuple
            Range for brightness adjustment
        contrast_limit: float or tuple
            Range for contrast adjustment
        """
        transform = A.RandomBrightnessContrast(
            brightness_limit=brightness_limit, contrast_limit=contrast_limit, p=1.0
        )
        super().__init__(p, transform)


class Rotate(AlbumentationsEffect):
    def __init__(self, p=1.0, limit=10):
        """
        Rotation effect using Albumentations

        Parameters
        ----------
        p: float
            Probability of applying this effect
        limit: int
            Maximum rotation angle in degrees
        """
        transform = A.Rotate(limit=limit, p=1.0)
        super().__init__(p, transform)


class ShiftScaleRotate(AlbumentationsEffect):
    def __init__(self, p=1.0, shift_limit=0.1, scale_limit=0.1, rotate_limit=10):
        """
        Shift, scale, and rotate effect using Albumentations

        Parameters
        ----------
        p: float
            Probability of applying this effect
        shift_limit: float
            Maximum shift as fraction of image size
        scale_limit: float
            Maximum scale change
        rotate_limit: int
            Maximum rotation angle in degrees
        """
        transform = A.ShiftScaleRotate(
            shift_limit=shift_limit,
            scale_limit=scale_limit,
            rotate_limit=rotate_limit,
            p=1.0,
        )
        super().__init__(p, transform)


class ElasticTransform(AlbumentationsEffect):
    def __init__(self, p=1.0, alpha=1, sigma=50, alpha_affine=50):
        """
        Elastic transform effect using Albumentations

        Parameters
        ----------
        p: float
            Probability of applying this effect
        alpha: float
            Elastic deformation parameter
        sigma: float
            Gaussian filter parameter
        alpha_affine: float
            Affine transformation parameter
        """
        transform = A.ElasticTransform(alpha=alpha, sigma=sigma, p=1.0)
        super().__init__(p, transform)


class GridDistortion(AlbumentationsEffect):
    def __init__(self, p=1.0, num_steps=5, distort_limit=0.3):
        """
        Grid distortion effect using Albumentations

        Parameters
        ----------
        p: float
            Probability of applying this effect
        num_steps: int
            Number of grid steps
        distort_limit: float
            Maximum distortion
        """
        transform = A.GridDistortion(
            num_steps=num_steps, distort_limit=distort_limit, p=1.0
        )
        super().__init__(p, transform)


class OpticalDistortion(AlbumentationsEffect):
    def __init__(self, p=1.0, distort_limit=0.05, shift_limit=0.05):
        """
        Optical distortion effect using Albumentations

        Parameters
        ----------
        p: float
            Probability of applying this effect
        distort_limit: float
            Maximum distortion
        shift_limit: float
            Maximum shift
        """
        transform = A.OpticalDistortion(distort_limit=distort_limit, p=1.0)
        super().__init__(p, transform)
