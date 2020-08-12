from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import numpy as np
from text_renderer.utils.errors import PanicError
from text_renderer.utils.utils import load_chars_file, random_choice

from .corpus import Corpus, CorpusCfg


@dataclass
class RandCorpusCfg(CorpusCfg):
    """
    Random corpus config

    args:
        length (Tuple[int, int]): Range of output text length  [min_length, max_length)
        chars_file (Path): Character set
        filter_font (bool): If True filter font file
                            by intersection of font support chars with chars file
        filter_font_min_support_chars (int): If intersection of font support chars with chars file is lower
                                             than filter_font_min_support_chars, filter this font file.
    """

    length: Tuple[int, int] = (5, 10)
    chars_file: Path = None
    filter_font: bool = False
    filter_font_min_support_chars: int = 100


class RandCorpus(Corpus):
    """
    Randomly selects characters from the character set
    """

    def __init__(self, cfg: "CorpusCfg"):
        super().__init__(cfg)

        self.cfg: RandCorpusCfg
        if self.cfg.chars_file is None or not self.cfg.chars_file.exists():
            raise PanicError(f"chars_file not exists: {self.cfg.chars_file}")

        self.chars = list(load_chars_file(self.cfg.chars_file))

        self.font_manager.update_font_support_chars(self.cfg.chars_file)
        if self.cfg.filter_font:
            self.font_manager.filter_font_path(self.cfg.filter_font_min_support_chars)

    def get_text(self):
        length = np.random.randint(*self.cfg.length)
        chars = random_choice(self.chars, length)
        text = "".join(chars)
        return text
