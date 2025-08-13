import typing
from typing import Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from text_renderer.utils.bbox import BBox
from text_renderer.utils.types import PILImage

if typing.TYPE_CHECKING:
    from text_renderer.config import TextColorCfg

from .base_effect import Effect


class TextBorder(Effect):
    def __init__(
        self,
        p=0.5,
        border_width=(1, 3),
        border_color_cfg: "TextColorCfg" = None,
        border_style="solid",  # 'solid', 'dashed', 'dotted'
        blur_radius=0,  # Gaussian blur for border
        # New configuration options
        enable=True,
        fraction=0.5,
        light_enable=True,
        light_fraction=0.5,
        dark_enable=True,
        dark_fraction=0.5,
    ):
        """
        Add border around text characters

        Args:
            p (float): probability to apply effect
            border_width (int, int): border width range
            border_color_cfg (TextColorCfg): border color configuration
            border_style (str): border style - 'solid', 'dashed', 'dotted'
            blur_radius (int): Gaussian blur radius for border (0 for no blur)
            enable (bool): whether to enable text border effect
            fraction (float): fraction of applying text border
            light_enable (bool): whether to enable light border
            light_fraction (float): fraction of light border when enabled
            dark_enable (bool): whether to enable dark border
            dark_fraction (float): fraction of dark border when enabled
        """
        super().__init__(p)
        self.border_width = border_width
        self.border_color_cfg = border_color_cfg
        self.border_style = border_style
        self.blur_radius = blur_radius
        self.enable = enable
        self.fraction = fraction
        self.light_enable = light_enable
        self.light_fraction = light_fraction
        self.dark_enable = dark_enable
        self.dark_fraction = dark_fraction

    def apply(self, img: PILImage, text_bbox: BBox) -> Tuple[PILImage, BBox]:
        """
        Apply text border effect

        Args:
            img: Input image
            text_bbox: Text bounding box

        Returns:
            Modified image and text bounding box
        """
        # Check if effect is enabled
        if not self.enable:
            return img, text_bbox

        # Check fraction probability
        if np.random.random() > self.fraction:
            return img, text_bbox

        # Convert to RGBA if not already
        if img.mode != "RGBA":
            img = img.convert("RGBA")

        # Create a copy of the image
        result_img = img.copy()

        # Get border width
        border_width = np.random.randint(*self.border_width)

        # Get border color based on text color
        border_color = self._get_border_color(img, text_bbox)

        # Create border mask
        border_mask = self._create_border_mask(img, border_width)

        # Apply border based on style
        if self.border_style == "solid":
            result_img = self._apply_solid_border(result_img, border_mask, border_color)
        elif self.border_style == "dashed":
            result_img = self._apply_dashed_border(
                result_img, border_mask, border_color, border_width
            )
        elif self.border_style == "dotted":
            result_img = self._apply_dotted_border(
                result_img, border_mask, border_color, border_width
            )

        # Apply blur if specified
        if self.blur_radius > 0:
            result_img = result_img.filter(
                ImageFilter.GaussianBlur(radius=self.blur_radius)
            )

        return result_img, text_bbox

    def _get_border_color(
        self, img: PILImage, text_bbox: BBox
    ) -> Tuple[int, int, int, int]:
        """Get border color based on text color and configuration"""
        # If custom color config is provided, use it
        if self.border_color_cfg is not None:
            return self.border_color_cfg.get_color(img)

        # Get the dominant text color from the image
        text_color = self._extract_text_color(img)

        # Determine border type based on configuration
        border_type = self._select_border_type()

        if border_type == "light":
            return self._create_light_border_color(text_color)
        elif border_type == "dark":
            return self._create_dark_border_color(text_color)
        else:
            # Default to black border
            return (0, 0, 0, 255)

    def _extract_text_color(self, img: PILImage) -> Tuple[int, int, int]:
        """Extract the dominant text color from the image"""
        # Convert to RGB for color analysis
        rgb_img = img.convert("RGB")
        img_array = np.array(rgb_img)

        # Create a mask for text pixels (assuming text is darker than background)
        gray_img = img.convert("L")
        gray_array = np.array(gray_img)

        # Threshold to find text pixels
        threshold = 128
        text_mask = gray_array < threshold

        if np.sum(text_mask) == 0:
            # If no text pixels found, return black
            return (0, 0, 0)

        # Get text pixels
        text_pixels = img_array[text_mask]

        # Calculate mean color of text pixels
        mean_color = np.mean(text_pixels, axis=0)

        return tuple(map(int, mean_color))

    def _select_border_type(self) -> str:
        """Select border type based on configuration and fractions"""
        if not self.light_enable and not self.dark_enable:
            return "default"

        if not self.light_enable:
            return "dark"

        if not self.dark_enable:
            return "light"

        # Both enabled, use fractions
        rand_val = np.random.random()
        if rand_val < self.light_fraction:
            return "light"
        else:
            return "dark"

    def _create_light_border_color(
        self, text_color: Tuple[int, int, int]
    ) -> Tuple[int, int, int, int]:
        """Create a lighter version of the text color for border"""
        r, g, b = text_color

        # Make the color lighter by increasing RGB values
        light_factor = 1.5  # Make it 50% lighter
        r = min(255, int(r * light_factor))
        g = min(255, int(g * light_factor))
        b = min(255, int(b * light_factor))

        return (r, g, b, 255)

    def _create_dark_border_color(
        self, text_color: Tuple[int, int, int]
    ) -> Tuple[int, int, int, int]:
        """Create a darker version of the text color for border"""
        r, g, b = text_color

        # Make the color darker by decreasing RGB values
        dark_factor = 0.5  # Make it 50% darker
        r = max(0, int(r * dark_factor))
        g = max(0, int(g * dark_factor))
        b = max(0, int(b * dark_factor))

        return (r, g, b, 255)

    def _create_border_mask(self, img: PILImage, border_width: int) -> PILImage:
        """Create a mask for the border area around text"""
        # Convert to grayscale for mask creation
        gray_img = img.convert("L")
        img_array = np.array(gray_img)

        # Create binary mask (text is white, background is black)
        # Assuming text is darker than background (common case)
        threshold = 128
        text_mask = (img_array < threshold).astype(np.uint8) * 255

        # Create border mask by dilating the text mask
        from scipy import ndimage

        border_mask = ndimage.binary_dilation(text_mask, iterations=border_width)
        border_mask = border_mask.astype(np.uint8) * 255

        # Subtract original text mask to get only border area
        border_only = border_mask - text_mask
        border_only = np.clip(border_only, 0, 255)

        return Image.fromarray(border_only, mode="L")

    def _apply_solid_border(
        self,
        img: PILImage,
        border_mask: PILImage,
        border_color: Tuple[int, int, int, int],
    ) -> PILImage:
        """Apply solid border"""
        # Create border image
        border_img = Image.new("RGBA", img.size, border_color)

        # Apply border mask
        border_img.putalpha(border_mask)

        # Composite border with original image
        result = Image.alpha_composite(img, border_img)
        return result

    def _apply_dashed_border(
        self,
        img: PILImage,
        border_mask: PILImage,
        border_color: Tuple[int, int, int, int],
        border_width: int,
    ) -> PILImage:
        """Apply dashed border"""
        # Create dashed pattern
        dash_length = border_width * 2
        gap_length = border_width

        # Create dashed mask
        border_array = np.array(border_mask)
        dashed_mask = np.zeros_like(border_array)

        # Apply dashed pattern
        for i in range(0, border_array.shape[0], dash_length + gap_length):
            end_i = min(i + dash_length, border_array.shape[0])
            dashed_mask[i:end_i] = border_array[i:end_i]

        for j in range(0, border_array.shape[1], dash_length + gap_length):
            end_j = min(j + dash_length, border_array.shape[1])
            dashed_mask[:, j:end_j] = border_array[:, j:end_j]

        dashed_mask_img = Image.fromarray(dashed_mask, mode="L")

        # Apply dashed border
        border_img = Image.new("RGBA", img.size, border_color)
        border_img.putalpha(dashed_mask_img)

        result = Image.alpha_composite(img, border_img)
        return result

    def _apply_dotted_border(
        self,
        img: PILImage,
        border_mask: PILImage,
        border_color: Tuple[int, int, int, int],
        border_width: int,
    ) -> PILImage:
        """Apply dotted border"""
        # Create dotted pattern
        dot_spacing = border_width * 2

        # Create dotted mask
        border_array = np.array(border_mask)
        dotted_mask = np.zeros_like(border_array)

        # Apply dotted pattern
        for i in range(0, border_array.shape[0], dot_spacing):
            for j in range(0, border_array.shape[1], dot_spacing):
                if i < border_array.shape[0] and j < border_array.shape[1]:
                    # Create a small dot
                    dot_size = border_width // 2
                    start_i = max(0, i - dot_size)
                    end_i = min(border_array.shape[0], i + dot_size)
                    start_j = max(0, j - dot_size)
                    end_j = min(border_array.shape[1], j + dot_size)

                    if border_array[i, j] > 0:  # Only add dots where border exists
                        dotted_mask[start_i:end_i, start_j:end_j] = border_array[
                            start_i:end_i, start_j:end_j
                        ]

        dotted_mask_img = Image.fromarray(dotted_mask, mode="L")

        # Apply dotted border
        border_img = Image.new("RGBA", img.size, border_color)
        border_img.putalpha(dotted_mask_img)

        result = Image.alpha_composite(img, border_img)
        return result
