from unittest.mock import patch

from PIL import Image, ImageDraw

from text_renderer.effect.text_border import TextBorder
from text_renderer.utils.bbox import BBox


def _make_text_image(width, height, text_bbox):
    """Create an RGBA image with opaque black text on white-transparent background.

    Uses (255,255,255,0) background so that _create_border_mask's grayscale
    threshold correctly distinguishes text (L=0) from background (L=255).
    """
    img = Image.new("RGBA", (width, height), (255, 255, 255, 0))
    draw = ImageDraw.Draw(img)
    draw.rectangle(
        [text_bbox.left, text_bbox.top, text_bbox.right - 1, text_bbox.bottom - 1],
        fill=(0, 0, 0, 255),
    )
    return img


class TestTextBorderBBox:
    def test_bbox_expands_by_border_width(self):
        """TextBorder should expand text_bbox by the sampled border_width."""
        border = TextBorder(
            p=1.0,
            border_width=(2, 3),  # will sample exactly 2
            enable=True,
            fraction=1.0,
        )
        # 100x40 image, text occupying (10,5)-(90,35)
        text_bbox = BBox(10, 5, 90, 35)
        img = _make_text_image(100, 40, text_bbox)

        with patch("numpy.random.randint", return_value=2):
            result_img, result_bbox = border.apply(img, text_bbox)

        assert result_bbox == BBox(8, 3, 92, 37)

    def test_bbox_clamped_to_image_bounds(self):
        """Expansion should not exceed image dimensions."""
        border = TextBorder(
            p=1.0,
            border_width=(3, 4),
            enable=True,
            fraction=1.0,
        )
        # Text near edges of a small image
        text_bbox = BBox(1, 1, 29, 19)
        img = _make_text_image(30, 20, text_bbox)

        with patch("numpy.random.randint", return_value=3):
            _, result_bbox = border.apply(img, text_bbox)

        assert result_bbox == BBox(0, 0, 30, 20)

    def test_bbox_unchanged_when_disabled(self):
        """When effect is disabled, text_bbox should pass through unchanged."""
        border = TextBorder(p=1.0, enable=False)
        text_bbox = BBox(10, 5, 90, 35)
        img = _make_text_image(100, 40, text_bbox)

        _, result_bbox = border.apply(img, text_bbox)

        assert result_bbox == text_bbox

    def test_bbox_covers_blur_spread(self):
        """With blur, bbox should cover the full blurred alpha footprint."""
        border = TextBorder(
            p=1.0,
            border_width=(2, 3),
            enable=True,
            fraction=1.0,
            blur_radius=3,
        )
        text_bbox = BBox(15, 15, 85, 35)
        img = _make_text_image(100, 50, text_bbox)

        with patch("numpy.random.randint", return_value=2):
            result_img, result_bbox = border.apply(img, text_bbox)

        # Blur spreads alpha past the border_width=2 region
        assert result_bbox.left < text_bbox.left - 2
        assert result_bbox.top < text_bbox.top - 2
        assert result_bbox.right > text_bbox.right + 2
        assert result_bbox.bottom > text_bbox.bottom + 2

        # Returned bbox must match the actual alpha footprint
        import numpy as np

        alpha = np.array(result_img.split()[-1])
        coords = np.argwhere(alpha > 0)
        assert result_bbox.left == int(coords[:, 1].min())
        assert result_bbox.top == int(coords[:, 0].min())
        assert result_bbox.right == int(coords[:, 1].max()) + 1
        assert result_bbox.bottom == int(coords[:, 0].max()) + 1

    def test_bbox_unchanged_when_mask_empty(self):
        """When no text passes the grayscale threshold, bbox should not expand."""
        border = TextBorder(
            p=1.0,
            border_width=(2, 3),
            enable=True,
            fraction=1.0,
        )
        # Light text (L=255 > 128) is not detected by _create_border_mask,
        # so the border mask is empty and no border is drawn.
        text_bbox = BBox(10, 5, 90, 35)
        img = Image.new("RGBA", (100, 40), (0, 0, 0, 0))
        draw = ImageDraw.Draw(img)
        draw.rectangle(
            [text_bbox.left, text_bbox.top, text_bbox.right - 1, text_bbox.bottom - 1],
            fill=(255, 255, 255, 255),
        )

        with patch("numpy.random.randint", return_value=2):
            _, result_bbox = border.apply(img, text_bbox)

        assert result_bbox == text_bbox

    def test_bbox_ignores_non_text_opaque_pixels_outside_text_region(self):
        """Pre-existing non-text opaque pixels shouldn't inflate bbox.

        Only pixels this effect actually modifies (alpha delta > 0)
        contribute to the bbox, so light-colored decorations (L > 128)
        that the grayscale threshold doesn't classify as text produce
        no delta and don't expand the bbox.
        """
        border = TextBorder(
            p=1.0,
            border_width=(2, 3),
            enable=True,
            fraction=1.0,
        )
        text_bbox = BBox(30, 10, 70, 30)
        img = _make_text_image(100, 40, text_bbox)
        # Light-gray (L=200) decorations simulate a prior effect that
        # left opaque non-text pixels far outside text_bbox.  They're
        # above the text threshold (128) so TextBorder doesn't border
        # them, keeping the bbox tight around the actual text.
        draw = ImageDraw.Draw(img)
        draw.rectangle([0, 0, 5, 39], fill=(200, 200, 200, 255))
        draw.rectangle([95, 0, 99, 39], fill=(200, 200, 200, 255))

        with patch("numpy.random.randint", return_value=2):
            _, result_bbox = border.apply(img, text_bbox)

        # bbox should cover text + border (±2), not the far-away decorations
        assert result_bbox.left >= 28
        assert result_bbox.right <= 72

    def test_bbox_does_not_shrink_padded_input(self):
        """Padded text_bbox must not shrink when text occupies a subset."""
        border = TextBorder(
            p=1.0,
            border_width=(2, 3),
            enable=True,
            fraction=1.0,
        )
        # text_bbox is padded (simulating a prior Padding effect)
        text_bbox = BBox(5, 2, 95, 38)
        # Actual text occupies a smaller region
        img = Image.new("RGBA", (100, 40), (0, 0, 0, 0))
        draw = ImageDraw.Draw(img)
        draw.rectangle([20, 10, 79, 29], fill=(0, 0, 0, 255))

        with patch("numpy.random.randint", return_value=2):
            _, result_bbox = border.apply(img, text_bbox)

        assert result_bbox == text_bbox

    def test_border_drawn_around_glyphs_outside_stale_bbox(self):
        """Upstream effects (e.g. Curve.apply) can return a bbox whose
        coordinates don't match the actual glyph positions.  TextBorder
        must still render a border around the real glyph pixels rather
        than dropping anything outside the stale bbox from the mask.

        The returned bbox reflects the authoritative input text_bbox,
        not the full rendered border: TextBorder doesn't try to paper
        over an upstream bbox bug by inflating to cover decorations or
        glyphs the caller didn't claim as text extent."""
        border = TextBorder(
            p=1.0,
            border_width=(2, 3),
            enable=True,
            fraction=1.0,
        )
        # Glyphs at x=[30, 69] but bbox is shifted to (0, 0) — mimics
        # Curve.apply() which translates the bbox but keeps image pixels.
        img = Image.new("RGBA", (100, 40), (255, 255, 255, 0))
        draw = ImageDraw.Draw(img)
        draw.rectangle([30, 10, 69, 29], fill=(0, 0, 0, 255))
        text_bbox = BBox(0, 0, 40, 20)

        import numpy as np

        orig_alpha = np.array(img.split()[-1])

        with patch("numpy.random.randint", return_value=2):
            result_img, _ = border.apply(img, text_bbox)

        result_alpha = np.array(result_img.split()[-1])
        border_added = result_alpha > orig_alpha
        # Border must reach the right edge of actual glyphs (x≈70-72).
        assert border_added[:, 70:73].any()

    def test_bbox_ignores_dark_decorations_outside_text_bbox(self):
        """Upstream effects like Line.apply_horizontal_middle intentionally
        draw dark decorations (strikethrough, underline) without expanding
        text_bbox.  TextBorder borders those dark pixels because the mask
        can't distinguish them from text, but the returned bbox must
        reflect the text extent — not inflate to the full canvas."""
        border = TextBorder(
            p=1.0,
            border_width=(2, 3),
            enable=True,
            fraction=1.0,
        )
        text_bbox = BBox(30, 10, 70, 30)
        img = _make_text_image(100, 40, text_bbox)
        # Dark horizontal line spanning the full image width, simulating
        # Line.apply_horizontal_middle which leaves text_bbox unchanged.
        draw = ImageDraw.Draw(img)
        draw.line((0, 20, 99, 20), fill=(50, 50, 50, 255), width=2)

        with patch("numpy.random.randint", return_value=2):
            _, result_bbox = border.apply(img, text_bbox)

        # Bbox must stay within text_bbox ± border_width (2), not span
        # the full image width due to the decorative line's border.
        assert result_bbox.left >= text_bbox.left - 2
        assert result_bbox.right <= text_bbox.right + 2

    def test_blur_does_not_leak_non_text_opaque_pixels(self):
        """Blur is applied to the border layer, not the composed image,
        so pre-existing opaque non-text pixels (e.g. a light-gray element
        from a prior effect) must not leak into the bbox via Gaussian
        spread."""
        border = TextBorder(
            p=1.0,
            border_width=(2, 3),
            enable=True,
            fraction=1.0,
            blur_radius=3,
        )
        # Light-gray (L=200) rectangle just outside the scan window
        # (scan right = 60 + 2 + 9 = 71).  Without the blur-layer fix,
        # GaussianBlur(radius=3) on the composed image would spread its
        # alpha ~7.5 px into the scan and inflate bbox.right past 70.
        text_bbox = BBox(40, 15, 60, 25)
        img = _make_text_image(100, 40, text_bbox)
        draw = ImageDraw.Draw(img)
        draw.rectangle([72, 10, 76, 29], fill=(200, 200, 200, 255))

        baseline_img = _make_text_image(100, 40, text_bbox)
        with patch("numpy.random.randint", return_value=2):
            _, baseline_bbox = border.apply(baseline_img, text_bbox.copy())
            _, result_bbox = border.apply(img, text_bbox)

        # The light-gray rectangle must not influence the bbox; result
        # should match the text-only baseline exactly.
        assert result_bbox == baseline_bbox

    def test_blurred_colored_border_keeps_halo_color(self):
        """Blur the alpha mask before colorizing so soft border pixels
        keep the configured RGB instead of mixing with black."""
        border = TextBorder(
            p=1.0,
            border_width=(2, 3),
            enable=True,
            fraction=1.0,
            blur_radius=3,
        )
        text_bbox = BBox(20, 10, 80, 30)
        img = _make_text_image(100, 40, text_bbox)
        border_color = (250, 220, 80, 255)

        import numpy as np

        with patch("numpy.random.randint", return_value=2), patch.object(
            border, "_get_border_color", return_value=border_color
        ):
            result_img, _ = border.apply(img, text_bbox)

        result = np.array(result_img)
        orig_alpha = np.array(img.split()[-1])
        halo_pixels = (orig_alpha == 0) & (result[:, :, 3] > 0) & (
            result[:, :, 3] < 255
        )
        assert halo_pixels.any()
        assert np.all(result[:, :, :3][halo_pixels] == np.array(border_color[:3]))

    def test_decorations_inside_loose_bbox_widen_extent_known_limitation(self):
        """Known limitation: a dark decoration drawn inside a loose/padded
        text_bbox (e.g. Line.apply_horizontal_middle through a padded box
        whose text occupies only a sub-region) is bordered AND seeds the
        bbox, widening the reported extent past where the actual text
        ends.  TextBorder cannot distinguish glyph pixels from decoration
        pixels at the pixel level once they're both inside text_bbox.

        Fixing this requires threading an out-of-band decoration mask
        through the Effect pipeline so decoration-emitting effects can
        mark their pixels and TextBorder can subtract them from the
        bbox seed.  Until that lands, this test characterizes the
        current behavior so the limitation is visible and any future
        fix that addresses it must update this test intentionally."""
        border = TextBorder(
            p=1.0,
            border_width=(2, 3),
            enable=True,
            fraction=1.0,
            blur_radius=10,
        )
        text_bbox = BBox(30, 10, 70, 30)
        baseline_img = Image.new("RGBA", (100, 40), (255, 255, 255, 0))
        draw = ImageDraw.Draw(baseline_img)
        draw.rectangle([42, 14, 57, 25], fill=(0, 0, 0, 255))

        decorated_img = baseline_img.copy()
        draw = ImageDraw.Draw(decorated_img)
        draw.line((0, 20, 99, 20), fill=(50, 50, 50, 255), width=2)

        with patch("numpy.random.randint", return_value=2):
            _, baseline_bbox = border.apply(baseline_img, text_bbox.copy())
            _, result_bbox = border.apply(decorated_img, text_bbox.copy())

        # Decoration drives bbox wider than the text-only baseline.
        # When a decoration mask gets threaded through the pipeline,
        # this assertion should flip to result_bbox == baseline_bbox.
        assert result_bbox != baseline_bbox
        assert result_bbox.left < baseline_bbox.left
        assert result_bbox.right > baseline_bbox.right

    def test_border_drawn_when_bbox_top_negative(self):
        """Upstream effects (e.g. Line.apply_top) can return text_bbox with
        negative top/left. The border mask must clamp bbox bounds before
        slicing, otherwise numpy treats -N as wrap-around and the border
        is drawn only on the last N rows/cols."""
        border = TextBorder(
            p=1.0,
            border_width=(2, 3),
            enable=True,
            fraction=1.0,
        )
        # Text covers rows 12-41 of a 43-tall image; bbox is shifted above
        # the canvas to simulate Line.apply_top with out_offset=3.
        img = Image.new("RGBA", (100, 43), (255, 255, 255, 0))
        draw = ImageDraw.Draw(img)
        draw.rectangle([10, 12, 89, 41], fill=(0, 0, 0, 255))
        text_bbox = BBox(0, -3, 100, 43)

        import numpy as np

        orig_alpha = np.array(img.split()[-1])

        with patch("numpy.random.randint", return_value=2):
            result_img, _ = border.apply(img, text_bbox)

        # Border must cover the full rectangle perimeter, not a 3-row strip.
        result_alpha = np.array(result_img.split()[-1])
        pixels_added = int(np.sum(result_alpha > orig_alpha))
        # A border_width=2 ring around an 80x30 rectangle is ~440 px;
        # the wrap-around bug would yield ~90 px.
        assert pixels_added > 300

    def test_fractional_blur_radius_does_not_crash(self):
        """Pillow's GaussianBlur accepts fractional radii; TextBorder
        must not regress that by using blur_radius directly as a numpy
        slice index (which requires int)."""
        border = TextBorder(
            p=1.0,
            border_width=(2, 3),
            enable=True,
            fraction=1.0,
            blur_radius=1.5,
        )
        text_bbox = BBox(10, 5, 90, 35)
        img = _make_text_image(100, 40, text_bbox)

        with patch("numpy.random.randint", return_value=2):
            # Must not raise TypeError on slice indexing.
            _, result_bbox = border.apply(img, text_bbox)

        # Scan window with ceil(1.5*3) = 5 plus border_width 2 = margin 7.
        # The bbox should expand within that margin.
        assert result_bbox.left >= 10 - 7
        assert result_bbox.top >= 5 - 7
        assert result_bbox.right <= 90 + 7
        assert result_bbox.bottom <= 35 + 7

    def test_unknown_border_style_is_noop(self):
        """Unrecognized border_style values must not crash; pre-refactor
        the code silently left the image unchanged, so preserve that
        behavior rather than raising UnboundLocalError."""
        border = TextBorder(
            p=1.0,
            border_width=(2, 3),
            enable=True,
            fraction=1.0,
            border_style="typo_not_a_real_style",
        )
        text_bbox = BBox(10, 5, 90, 35)
        img = _make_text_image(100, 40, text_bbox)

        import numpy as np

        orig_alpha = np.array(img.split()[-1])

        with patch("numpy.random.randint", return_value=2):
            result_img, result_bbox = border.apply(img, text_bbox)

        # No border drawn, bbox unchanged, alpha unchanged
        result_alpha = np.array(result_img.split()[-1])
        assert np.array_equal(result_alpha, orig_alpha)
        assert result_bbox == text_bbox

    def test_bbox_stable_on_fully_opaque_input(self):
        """On a fully opaque image, bbox should not inflate to scan window."""
        border = TextBorder(
            p=1.0,
            border_width=(2, 3),
            enable=True,
            fraction=1.0,
        )
        text_bbox = BBox(20, 10, 80, 30)
        # Fully opaque background (e.g. TextBorder used in render_effects)
        img = Image.new("RGBA", (100, 40), (200, 200, 200, 255))
        draw = ImageDraw.Draw(img)
        draw.rectangle(
            [text_bbox.left, text_bbox.top, text_bbox.right - 1, text_bbox.bottom - 1],
            fill=(0, 0, 0, 255),
        )

        with patch("numpy.random.randint", return_value=2):
            _, result_bbox = border.apply(img, text_bbox)

        # No alpha increase possible on an already-opaque image
        assert result_bbox == text_bbox
