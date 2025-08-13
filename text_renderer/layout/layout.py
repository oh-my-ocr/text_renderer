"""
Layout management for text rendering operations.

This module provides the base Layout class for managing the positioning
and arrangement of multiple text elements in a single image.
"""

from abc import abstractmethod
from typing import List, Tuple

from text_renderer.utils import FontText
from text_renderer.utils.bbox import BBox


class Layout:
    """
    Abstract base class for text layout management.

    This class defines the interface for layout algorithms that determine
    how multiple text elements should be positioned relative to each other
    in a single image. Layout classes handle both positioning and text merging.

    Subclasses must implement the `apply` method to define specific layout logic.
    """

    def __call__(
        self,
        font_texts: List[FontText],
        text_bboxes: List[BBox],
        text_mask_bboxes: List[BBox],
    ) -> Tuple[List[BBox], str]:
        """
        Apply layout and merge texts.

        This is the main entry point for layout operations. It applies
        the layout algorithm and merges the text content.

        Args:
            font_texts (List[FontText]): List of font text objects
            text_bboxes (List[BBox]): List of text bounding boxes
            text_mask_bboxes (List[BBox]): List of text mask bounding boxes

        Returns:
            Tuple[List[BBox], str]: A tuple containing:
                - List[BBox]: Modified bounding boxes in same coordinate system
                - str: Merged text content
        """
        return self.apply(text_bboxes, text_mask_bboxes), self.merge_texts(font_texts)

    @abstractmethod
    def apply(self, text_bboxes: List[BBox], img_bboxes: List[BBox]) -> List[BBox]:
        """
        Apply layout algorithm to position text elements.

        This method must be implemented by subclasses to define the specific
        layout algorithm for positioning multiple text elements.

        Args:
            text_bboxes (List[BBox]): Text bounding boxes on image
            img_bboxes (List[BBox]): Image bounding boxes

        Returns:
            List[BBox]: Modified image bounding boxes in same coordinate system
        """
        pass

    def merge_texts(self, font_texts: List[FontText]) -> str:
        """
        Merge multiple text elements into a single string.

        This method concatenates the text content from multiple FontText
        objects into a single string. The default implementation simply
        joins all texts together.

        Args:
            font_texts (List[FontText]): List of FontText objects to merge

        Returns:
            str: Merged text content
        """
        return "".join([it.text for it in font_texts])
