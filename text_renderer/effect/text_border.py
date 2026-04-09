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
            blur_radius (float): Gaussian blur radius for border (0 for no blur)
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

        # Capture input alpha so the bbox update can be based on pixels this
        # effect actually changed, not all pre-existing opaque pixels.
        input_alpha = np.array(img.split()[-1])

        # Create a copy of the image
        result_img = img.copy()

        # Get border width
        border_width = np.random.randint(*self.border_width)

        # Get border color based on text color
        border_color = self._get_border_color(img, text_bbox)

        # Render using the full-image seed so stale upstream bboxes don't
        # drop real glyph pixels from the rendered outline.
        render_border_mask = self._create_border_mask(img, border_width)
        render_style_mask = self._style_border_mask(render_border_mask, border_width)
        render_style_mask = self._blur_mask(render_style_mask)
        border_layer = self._create_border_layer(render_style_mask, border_color)
        result_img = Image.alpha_composite(result_img, border_layer)

        # Update bbox from only the authoritative text content claimed by
        # text_bbox.  Decorations from prior effects can still be bordered
        # in the rendered image, but they must not widen the reported text
        # bbox.  Use a bbox-specific seed mask and derive growth from the
        # added alpha footprint instead of scanning a heuristic window.
        bbox_border_mask = self._create_border_mask(
            img, border_width, text_bbox=text_bbox, filter_decorations=True
        )
        bbox_style_mask = self._style_border_mask(bbox_border_mask, border_width)
        bbox_style_mask = self._blur_mask(bbox_style_mask)
        bbox_border_alpha = self._mask_to_alpha_array(
            bbox_style_mask, border_color[3]
        )
        changed = (bbox_border_alpha > 0) & (input_alpha < 255)
        coords = np.argwhere(changed)
        if coords.size > 0:
            top = min(text_bbox.top, int(coords[:, 0].min()))
            bottom = max(text_bbox.bottom, int(coords[:, 0].max()) + 1)
            left = min(text_bbox.left, int(coords[:, 1].min()))
            right = max(text_bbox.right, int(coords[:, 1].max()) + 1)
            text_bbox = BBox(left, top, right, bottom)

        return result_img, text_bbox

    def _blur_mask(self, border_mask: PILImage) -> PILImage:
        """Blur the alpha mask directly so RGB doesn't mix with black."""
        if self.blur_radius <= 0:
            return border_mask
        return border_mask.filter(ImageFilter.GaussianBlur(radius=self.blur_radius))

    def _style_border_mask(
        self, border_mask: PILImage, border_width: int
    ) -> PILImage:
        """Apply the configured style to a border alpha mask."""
        if self.border_style == "solid":
            return border_mask
        if self.border_style == "dashed":
            return self._create_dashed_border_mask(border_mask, border_width)
        if self.border_style == "dotted":
            return self._create_dotted_border_mask(border_mask, border_width)
        return Image.new("L", border_mask.size, 0)

    def _create_border_layer(
        self,
        border_mask: PILImage,
        border_color: Tuple[int, int, int, int],
    ) -> PILImage:
        """Colorize the border after styling/blur so soft edges keep color."""
        border_layer = Image.new("RGBA", border_mask.size, border_color[:3] + (0,))
        border_layer.putalpha(
            Image.fromarray(
                self._mask_to_alpha_array(border_mask, border_color[3]), mode="L"
            )
        )
        return border_layer

    def _mask_to_alpha_array(
        self, border_mask: PILImage, base_alpha: int
    ) -> np.ndarray:
        """Scale an L-mode mask by the configured border alpha."""
        alpha = np.array(border_mask, dtype=np.uint8)
        if base_alpha >= 255:
            return alpha
        scaled = np.rint(alpha.astype(np.float32) * (base_alpha / 255.0))
        return scaled.astype(np.uint8)

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

    def _create_border_mask(
        self,
        img: PILImage,
        border_width: int,
        text_bbox: BBox = None,
        filter_decorations: bool = False,
    ) -> PILImage:
        """Create a mask for the border area around text."""
        text_mask = self._create_text_mask(
            img, text_bbox=text_bbox, filter_decorations=filter_decorations
        )
        text_mask_array = np.array(text_mask, dtype=np.uint8)

        # Create border mask by dilating the text mask
        from scipy import ndimage

        border_mask = ndimage.binary_dilation(
            text_mask_array > 0, iterations=border_width
        )
        border_mask = border_mask.astype(np.uint8) * 255

        # Subtract original text mask to get only border area
        border_only = border_mask - text_mask_array
        border_only = np.clip(border_only, 0, 255)

        return Image.fromarray(border_only, mode="L")

    def _create_text_mask(
        self,
        img: PILImage,
        text_bbox: BBox = None,
        filter_decorations: bool = False,
    ) -> PILImage:
        """Create the binary text seed mask used for border dilation."""
        gray_img = img.convert("L")
        img_array = np.array(gray_img)
        alpha_array = (
            np.array(img.split()[-1])
            if img.mode == "RGBA"
            else np.full_like(img_array, 255)
        )

        threshold = 128
        text_mask = (img_array < threshold) & (alpha_array > 0)
        if text_bbox is not None:
            text_mask = self._restrict_text_mask_to_bbox(
                text_mask, text_bbox, filter_decorations=filter_decorations
            )

        return Image.fromarray(text_mask.astype(np.uint8) * 255, mode="L")

    def _restrict_text_mask_to_bbox(
        self,
        text_mask: np.ndarray,
        text_bbox: BBox,
        filter_decorations: bool = False,
    ) -> np.ndarray:
        """Clip to text_bbox and optionally suppress thin decoration cores."""
        clipped_bbox = self._clip_bbox_to_image(text_bbox, text_mask.shape[::-1])
        restricted = np.zeros_like(text_mask, dtype=bool)
        if clipped_bbox is None:
            return restricted

        restricted[
            clipped_bbox.top : clipped_bbox.bottom,
            clipped_bbox.left : clipped_bbox.right,
        ] = text_mask[
            clipped_bbox.top : clipped_bbox.bottom,
            clipped_bbox.left : clipped_bbox.right,
        ]

        if not filter_decorations or not restricted.any():
            return restricted

        # A small opening removes 1-2 px horizontal/vertical rules that
        # effects such as Line add without changing text_bbox.  Use the
        # opened mask only to locate the authoritative text core, then keep
        # the original pixels inside that core so thin glyph edges survive.
        from scipy import ndimage

        text_core = ndimage.binary_opening(
            restricted, structure=np.ones((3, 3), dtype=bool)
        )
        if not text_core.any():
            return restricted

        core_coords = np.argwhere(text_core)
        core_top = int(core_coords[:, 0].min())
        core_bottom = int(core_coords[:, 0].max()) + 1
        core_left = int(core_coords[:, 1].min())
        core_right = int(core_coords[:, 1].max()) + 1

        filtered = np.zeros_like(restricted, dtype=bool)
        filtered[core_top:core_bottom, core_left:core_right] = restricted[
            core_top:core_bottom, core_left:core_right
        ]
        return filtered

    def _clip_bbox_to_image(
        self, text_bbox: BBox, size: Tuple[int, int]
    ) -> BBox | None:
        """Clamp bbox bounds to the image before numpy slicing."""
        width, height = size
        left = max(int(text_bbox.left), 0)
        top = max(int(text_bbox.top), 0)
        right = min(int(text_bbox.right), width)
        bottom = min(int(text_bbox.bottom), height)
        if left >= right or top >= bottom:
            return None
        return BBox(left, top, right, bottom)

    def _create_dashed_border_mask(
        self, border_mask: PILImage, border_width: int
    ) -> PILImage:
        """Apply dashed styling to a border mask."""
        dash_length = border_width * 2
        gap_length = border_width

        border_array = np.array(border_mask)
        dashed_mask = np.zeros_like(border_array)

        for i in range(0, border_array.shape[0], dash_length + gap_length):
            end_i = min(i + dash_length, border_array.shape[0])
            dashed_mask[i:end_i] = border_array[i:end_i]

        for j in range(0, border_array.shape[1], dash_length + gap_length):
            end_j = min(j + dash_length, border_array.shape[1])
            dashed_mask[:, j:end_j] = border_array[:, j:end_j]

        return Image.fromarray(dashed_mask, mode="L")

    def _create_dotted_border_mask(
        self, border_mask: PILImage, border_width: int
    ) -> PILImage:
        """Apply dotted styling to a border mask."""
        dot_spacing = border_width * 2

        border_array = np.array(border_mask)
        dotted_mask = np.zeros_like(border_array)

        for i in range(0, border_array.shape[0], dot_spacing):
            for j in range(0, border_array.shape[1], dot_spacing):
                if i < border_array.shape[0] and j < border_array.shape[1]:
                    dot_size = border_width // 2
                    start_i = max(0, i - dot_size)
                    end_i = min(border_array.shape[0], i + dot_size)
                    start_j = max(0, j - dot_size)
                    end_j = min(border_array.shape[1], j + dot_size)

                    if border_array[i, j] > 0:
                        dotted_mask[start_i:end_i, start_j:end_j] = border_array[
                            start_i:end_i, start_j:end_j
                        ]

        return Image.fromarray(dotted_mask, mode="L")

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
