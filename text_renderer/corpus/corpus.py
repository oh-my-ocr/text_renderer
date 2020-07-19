import os
from abc import abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

from loguru import logger
from tenacity import retry

from text_renderer.font_manager import FontManager
from text_renderer.config import TextColorCfg, SimpleTextColorCfg
from text_renderer.utils.errors import RetryError, PanicError
from text_renderer.utils import FontText


@dataclass
class CorpusCfg:
    # noinspection pyunresolvedreferences
    """
    Base config for corpus

    args:
        font_dir (path): font files directory
        font_list_file (path): font file names to load from **font_dir**
        font_size (tuple[int, int]): font size in point (min_font_size, max_font_size)
        clip_length (int): clip :func:`~text_renderer.corpus.Corpus.get_text` output. set **-1** disables clip
        text_color_cfg (TextColorCfg): see :class:`~text_renderer.utils.TextColorCfg`
    """
    font_dir: Path
    font_list_file: Path
    font_size: Tuple[int, int]
    clip_length: int = -1
    text_color_cfg: TextColorCfg = SimpleTextColorCfg()

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        if not cls.__doc__:
            cls.__doc__ = CorpusCfg.__doc__
        else:
            cls.__doc__ += CorpusCfg.__doc__


class Corpus:
    """
    Base class of different corpus. See :class:`~text_renderer.corpus.CorpusCfg` for base configs for corpus.
    """

    def __init__(
        self, cfg: "CorpusCfg",
    ):
        self.cfg = cfg
        self.font_manager = FontManager(
            cfg.font_dir, cfg.font_list_file, cfg.font_size,
        )

    @retry()
    def sample(self):
        """
        This method ensures that the selected font supports all characters.

        Returns:
            FontText: A FontText object contains text and font.

        """
        try:
            text = self.get_text()
        except Exception as e:
            err_msg = f"get_text() error: {e}"
            logger.error(err_msg)
            raise RetryError(err_msg)

        if self.cfg.clip_length != -1 and len(text) > self.cfg.clip_length:
            text = text[: self.cfg.clip_length]

        font, support_chars, font_path = self.font_manager.get_font()
        status, intersect = self.font_manager.check_support(text, support_chars)
        if not status:
            err_msg = (
                f"{self.__class__.__name__} {font_path} not support chars: {intersect}"
            )
            logger.error(err_msg)
            raise RetryError(err_msg)

        return FontText(font, text)

    @abstractmethod
    def get_text(self):
        """
        Returns:
            str: text to render
        """
        pass

    @staticmethod
    def filter_by_chars(text, chars_file):
        """
        Filter chars not exist in chars file

        Args:
            text (Union[str, List[str]]): text to filter
            chars_file (Path): one char per line

        Returns:
            Union[str, List[str]]: string(s) removed chars not exist in chars file

        """
        if chars_file is None or not chars_file.exists():
            raise PanicError(f"chars_file not exists: {chars_file}")

        chars = Corpus.load_chars_file(chars_file)

        logger.info("filtering text by chars...")

        filtered_count = 0

        # TODO: find a more efficient way
        if isinstance(text, list):
            out = []
            for t in text:
                _text = ""
                for c in t:
                    if c in chars:
                        _text += c
                    else:
                        filtered_count += 1
                out.append(_text)
        else:
            out = ""
            for c in text:
                if c in chars:
                    out += c
                else:
                    filtered_count += 1
        logger.info(f"filter {filtered_count} chars")
        return out

    @staticmethod
    def load_chars_file(chars_file):
        """

        Args:
            chars_file (Path): one char per line

        Returns:
            Set: chars in file

        """
        with open(str(chars_file), "r", encoding="utf-8") as f:
            lines = f.readlines()
            lines = [it.strip() for it in lines]
            chars = set("".join(lines))
        logger.info(f"load {len(chars)} chars from: {chars_file}.")
        return chars
