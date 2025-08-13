from typing import List, Tuple

import cv2
import numpy as np
from loguru import logger
from PIL import Image
from PIL.Image import Image as PILImage
from PIL.ImageFont import FreeTypeFont
from tenacity import retry

from text_renderer.bg_manager import BgManager
from text_renderer.config import RenderCfg
from text_renderer.utils import utils
from text_renderer.utils.bbox import BBox
from text_renderer.utils.draw_utils import draw_text_on_bg, transparent_img
from text_renderer.utils.errors import PanicError
from text_renderer.utils.font_text import FontText
from text_renderer.utils.math_utils import PerspectiveTransform
from text_renderer.utils.types import FontColor, is_list


class Render:
    """
    Main text rendering engine for generating synthetic text images.

    This class handles the complete pipeline of text rendering including:
    - Text generation from corpus
    - Font selection and text rendering
    - Background image management
    - Effect application at different stages
    - Perspective transformation
    - Layout management for multi-corpus scenarios

    Args:
        cfg (RenderCfg): Configuration object containing all rendering parameters

    Raises:
        PanicError: If corpus and corpus_effects configuration is inconsistent
    """

    def __init__(self, cfg: RenderCfg):
        self.cfg = cfg
        self.layout = cfg.layout
        if isinstance(cfg.corpus, list) and len(cfg.corpus) == 1:
            self.corpus = cfg.corpus[0]
        else:
            self.corpus = cfg.corpus

        if is_list(self.corpus) and is_list(self.cfg.corpus_effects):
            if len(self.corpus) != len(self.cfg.corpus_effects):
                raise PanicError(
                    f"corpus length({self.corpus}) is not equal to corpus_effects length({self.cfg.corpus_effects})"
                )

        if is_list(self.corpus) and (
            self.cfg.corpus_effects and not is_list(self.cfg.corpus_effects)
        ):
            raise PanicError("corpus is list, corpus_effects is not list")

        if not is_list(self.corpus) and is_list(self.cfg.corpus_effects):
            raise PanicError("corpus_effects is list, corpus is not list")

        self.bg_manager = BgManager(cfg.bg_dir, cfg.pre_load_bg_img)

    @retry
    def __call__(self, *args, **kwargs) -> Tuple[np.ndarray, str]:
        """
        Generate a synthetic text image with the configured settings.

        This method is the main entry point for text rendering. It handles
        the complete pipeline from text generation to final image output.

        Returns:
            Tuple[np.ndarray, str]: A tuple containing:
                - np.ndarray: The generated image as a numpy array (BGR format)
                - str: The text that was rendered

        Raises:
            Exception: Any exception that occurs during rendering process
        """
        try:
            if self._should_apply_layout():
                img, text, cropped_bg, transformed_text_mask = self.gen_multi_corpus()
            else:
                img, text, cropped_bg, transformed_text_mask = self.gen_single_corpus()

            if self.cfg.render_effects is not None:
                img, _ = self.cfg.render_effects.apply_effects(
                    img, BBox.from_size(img.size)
                )

            if self.cfg.return_bg_and_mask:
                gray_text_mask = np.array(transformed_text_mask.convert("L"))
                _, gray_text_mask = cv2.threshold(
                    gray_text_mask, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU
                )
                transformed_text_mask = Image.fromarray(255 - gray_text_mask)

                merge_target = Image.new("RGBA", (img.width * 3, img.height))
                merge_target.paste(img, (0, 0))
                merge_target.paste(cropped_bg, (img.width, 0))
                merge_target.paste(
                    transformed_text_mask,
                    (img.width * 2, 0),
                    mask=transformed_text_mask,
                )

                np_img = np.array(merge_target)
                np_img = cv2.cvtColor(np_img, cv2.COLOR_RGBA2BGR)
                np_img = self.norm(np_img)
            else:
                img = img.convert("RGB")
                np_img = np.array(img)
                np_img = cv2.cvtColor(np_img, cv2.COLOR_RGB2BGR)
                np_img = self.norm(np_img)
            return np_img, text
        except Exception as e:
            logger.exception(e)
            raise e

    def gen_single_corpus(self) -> Tuple[PILImage, str, PILImage, PILImage]:
        """
        Generate text image from a single corpus.

        This method handles the rendering pipeline for a single corpus:
        1. Sample text from corpus
        2. Get background image
        3. Determine text color
        4. Render text mask
        5. Apply corpus effects
        6. Apply perspective transformation
        7. Paste text on background

        Returns:
            Tuple[PILImage, str, PILImage, PILImage]: A tuple containing:
                - PILImage: Final rendered image
                - str: The text that was rendered
                - PILImage: Cropped background image
                - PILImage: Transformed text mask
        """
        font_text = self.corpus.sample()

        bg = self.bg_manager.get_bg()
        if self.cfg.text_color_cfg is not None:
            text_color = self.cfg.text_color_cfg.get_color(bg)

        # corpus text_color has higher priority than RenderCfg.text_color_cfg
        if self.corpus.cfg.text_color_cfg is not None:
            text_color = self.corpus.cfg.text_color_cfg.get_color(bg)

        text_mask = draw_text_on_bg(
            font_text, text_color, char_spacing=self.corpus.cfg.char_spacing
        )

        if self.cfg.corpus_effects is not None:
            text_mask, _ = self.cfg.corpus_effects.apply_effects(
                text_mask, BBox.from_size(text_mask.size)
            )

        if self.cfg.perspective_transform is not None:
            transformer = PerspectiveTransform(self.cfg.perspective_transform)
            # TODO: refactor this, now we must call get_transformed_size to call gen_warp_matrix
            _ = transformer.get_transformed_size(text_mask.size)

            try:
                (
                    transformed_text_mask,
                    transformed_text_pnts,
                ) = transformer.do_warp_perspective(text_mask)
            except Exception as e:
                logger.exception(e)
                logger.error(font_text.font_path, "text", font_text.text)
                raise e
        else:
            transformed_text_mask = text_mask

        img, cropped_bg = self.paste_text_mask_on_bg(bg, transformed_text_mask)

        return img, font_text.text, cropped_bg, transformed_text_mask

    def gen_multi_corpus(self) -> Tuple[PILImage, str, PILImage, PILImage]:
        """
        Generate text image from multiple corpora using layout management.

        This method handles the rendering pipeline for multiple corpora:
        1. Sample text from each corpus
        2. Get background image
        3. Determine text colors
        4. Render text masks for each corpus
        5. Apply corpus effects to each text mask
        6. Use layout to position multiple text masks
        7. Apply perspective transformation
        8. Apply layout effects
        9. Paste merged text on background

        Returns:
            Tuple[PILImage, str, PILImage, PILImage]: A tuple containing:
                - PILImage: Final rendered image
                - str: The merged text that was rendered
                - PILImage: Cropped background image
                - PILImage: Transformed text mask
        """
        font_texts: List[FontText] = [it.sample() for it in self.corpus]

        bg = self.bg_manager.get_bg()

        text_color = None
        if self.cfg.text_color_cfg is not None:
            text_color = self.cfg.text_color_cfg.get_color(bg)

        text_masks, text_bboxes = [], []
        for i in range(len(font_texts)):
            font_text = font_texts[i]

            if text_color is None:
                _text_color = self.corpus[i].cfg.text_color_cfg.get_color(bg)
            else:
                _text_color = text_color
            text_mask = draw_text_on_bg(
                font_text, _text_color, char_spacing=self.corpus[i].cfg.char_spacing
            )

            text_bbox = BBox.from_size(text_mask.size)
            if self.cfg.corpus_effects is not None:
                effects = self.cfg.corpus_effects[i]
                if effects is not None:
                    text_mask, text_bbox = effects.apply_effects(text_mask, text_bbox)
            text_masks.append(text_mask)
            text_bboxes.append(text_bbox)

        text_mask_bboxes, merged_text = self.layout(
            font_texts,
            [it.copy() for it in text_bboxes],
            [BBox.from_size(it.size) for it in text_masks],
        )
        if len(text_mask_bboxes) != len(text_bboxes):
            raise PanicError(
                "points and text_bboxes should have same length after layout output"
            )

        merged_bbox = BBox.from_bboxes(text_mask_bboxes)
        merged_text_mask = transparent_img(merged_bbox.size)
        for text_mask, bbox in zip(text_masks, text_mask_bboxes):
            merged_text_mask.paste(text_mask, bbox.left_top)

        if self.cfg.perspective_transform is not None:
            transformer = PerspectiveTransform(self.cfg.perspective_transform)
            # TODO: refactor this, now we must call get_transformed_size to call gen_warp_matrix
            _ = transformer.get_transformed_size(merged_text_mask.size)

            (
                transformed_text_mask,
                transformed_text_pnts,
            ) = transformer.do_warp_perspective(merged_text_mask)
        else:
            transformed_text_mask = merged_text_mask

        if self.cfg.layout_effects is not None:
            transformed_text_mask, _ = self.cfg.layout_effects.apply_effects(
                transformed_text_mask, BBox.from_size(transformed_text_mask.size)
            )

        img, cropped_bg = self.paste_text_mask_on_bg(bg, transformed_text_mask)

        return img, merged_text, cropped_bg, transformed_text_mask

    def paste_text_mask_on_bg(
        self, bg: PILImage, transformed_text_mask: PILImage
    ) -> Tuple[PILImage, PILImage]:
        """
        Paste the text mask onto a background image at a random position.

        Args:
            bg (PILImage): Background image to paste text onto
            transformed_text_mask (PILImage): Text mask to paste

        Returns:
            Tuple[PILImage, PILImage]: A tuple containing:
                - PILImage: Final image with text pasted on background
                - PILImage: Cropped background image (if return_bg_and_mask is True)
        """
        x_offset, y_offset = utils.random_xy_offset(transformed_text_mask.size, bg.size)
        bg = self.bg_manager.guard_bg_size(bg, transformed_text_mask.size)
        bg = bg.crop(
            (
                x_offset,
                y_offset,
                x_offset + transformed_text_mask.width,
                y_offset + transformed_text_mask.height,
            )
        )
        if self.cfg.return_bg_and_mask:
            _bg = bg.copy()
        else:
            _bg = bg
        bg.paste(transformed_text_mask, (0, 0), mask=transformed_text_mask)
        return bg, _bg

    def get_text_color(self, bg: PILImage, text: str, font: FreeTypeFont) -> FontColor:
        """
        Generate a text color based on background image characteristics.

        This method analyzes the background image and generates a color
        that should provide good contrast for text rendering.

        Args:
            bg (PILImage): Background image to analyze
            text (str): Text to be rendered (not used in current implementation)
            font (FreeTypeFont): Font object (not used in current implementation)

        Returns:
            FontColor: RGBA color tuple for text rendering

        Note:
            This is a TODO method that needs improvement for better color selection.
        """
        # TODO: better get text color
        # text_mask = self.draw_text_on_transparent_bg(text, font)
        np_img = np.array(bg)
        # mean = np.mean(np_img, axis=2)
        mean = np.mean(np_img)

        alpha = np.random.randint(110, 255)
        r = np.random.randint(0, int(mean * 0.7))
        g = np.random.randint(0, int(mean * 0.7))
        b = np.random.randint(0, int(mean * 0.7))
        fg_text_color = (r, g, b, alpha)

        return fg_text_color

    def _should_apply_layout(self) -> bool:
        """
        Determine if layout management should be applied.

        Layout is applied when there are multiple corpora to manage.

        Returns:
            bool: True if layout should be applied, False otherwise
        """
        return isinstance(self.corpus, list) and len(self.corpus) > 1

    def norm(self, image: np.ndarray) -> np.ndarray:
        """
        Normalize the image according to configuration settings.

        This method applies final image processing including:
        - Grayscale conversion (if configured)
        - Height normalization (if configured)

        Args:
            image (np.ndarray): Input image as numpy array

        Returns:
            np.ndarray: Normalized image
        """
        if self.cfg.gray:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        if self.cfg.height != -1 and self.cfg.height != image.shape[0]:
            height, width = image.shape[:2]
            width = int(width // (height / self.cfg.height))
            image = cv2.resize(
                image, (width, self.cfg.height), interpolation=cv2.INTER_CUBIC
            )

        return image
