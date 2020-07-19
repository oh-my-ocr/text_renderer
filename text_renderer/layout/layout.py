from abc import abstractmethod
from typing import List

from text_renderer.utils import FontText
from text_renderer.utils.bbox import BBox


class Layout:
    def __call__(
        self,
        font_texts: List[FontText],
        text_bboxes: List[BBox],
        text_mask_bboxes: List[BBox],
    ) -> [List[BBox], str]:
        return self.apply(text_bboxes, text_mask_bboxes), self.merge_texts(font_texts)

    @abstractmethod
    def apply(self, text_bboxes: List[BBox], img_bboxes: List[BBox],) -> List[BBox]:
        """

        Parameters
        ----------
        text_bboxes : :obj:`list` of :obj:`BBox`
            Text bbox on image
        img_bboxes : :obj:`list` of :obj:`BBox`
            Image bbox

        Returns
        -------
            :obj:`list` of :obj:`BBox`:
                Modified img_bboxes in same coordinate
        """
        pass

    def merge_texts(self, font_texts: List[FontText]) -> str:
        """
        Output text after merge

        Parameters
        ----------
        font_texts : :obj:`list` of :obj:`BBox`
            FontText

        Returns
        -------
            str:
                Merged text

        """
        return "".join([it.text for it in font_texts])
