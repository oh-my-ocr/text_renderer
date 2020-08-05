import os
from dataclasses import field, dataclass
from pathlib import Path
from typing import Tuple, List

import numpy as np
from loguru import logger

from text_renderer.utils.errors import PanicError

from .corpus import Corpus, CorpusCfg


@dataclass
class CharCorpusCfg(CorpusCfg):
    # noinspection pyunresolvedreferences
    """
    Char corpus config

    args:
        text_paths (List[Path]): Text file paths
        length (Tuple[int, int]): Range of output text length  [min_length, max_length)
        filter_by_chars (bool): If True, filtering text by character set
        chars_file (Path): Character set

    """
    text_paths: List[Path] = field(default_factory=list)
    length: Tuple[int, int] = (5, 10)
    filter_by_chars: bool = False
    chars_file: Path = None


class CharCorpus(Corpus):
    """
    Output contiguous characters of a certain length
    """

    def __init__(
        self, cfg: "CorpusCfg",
    ):
        super().__init__(cfg)

        self.cfg: CharCorpusCfg
        self.text = ""

        if len(self.cfg.text_paths) == 0:
            raise PanicError(f"text_paths must not contain path")

        for p in self.cfg.text_paths:
            if not os.path.exists(p):
                raise PanicError(f"text_path not exists: {p}")

            logger.info(f"load: {p}")
            with open(p, "r", encoding="utf-8") as f:
                self.text += "".join(f.readlines())

        if self.cfg.filter_by_chars:
            self.text = Corpus.filter_by_chars(self.text, self.cfg.chars_file)
            self.font_manager.update_font_support_chars(self.cfg.chars_file)

        if len(self.text) < self.cfg.length[1]:
            raise PanicError("too few texts")

    def get_text(self):
        length = np.random.randint(*self.cfg.length)
        start = np.random.randint(0, len(self.text) - length)
        word = self.text[start : start + length]
        return word
