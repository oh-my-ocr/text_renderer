from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Tuple

import numpy as np
from loguru import logger

from text_renderer.utils.errors import PanicError

from .corpus import Corpus, CorpusCfg


@dataclass
class WordCorpusCfg(CorpusCfg):
    """
    Word corpus config

    args:
        text_paths (List[Path]): Text file paths
        separator (str): word separator of texts and join char in get_text()
        num_word (Tuple[int, int]): Range of output word count  [min_length, max_length)
        filter_by_chars (bool): If True, filtering text by character set
        chars_file (Path): Character set

    """

    text_paths: List[Path] = field(default_factory=list)
    separator: str = " "
    num_word: (int, int) = (1, 5)
    filter_by_chars: bool = False
    chars_file: Path = None


class WordCorpus(Corpus):
    """
    Output contiguous words of a certain length
    """

    def __init__(self, cfg: "CorpusCfg"):
        super().__init__(cfg)

        self.cfg: WordCorpusCfg
        if len(self.cfg.text_paths) == 0:
            raise PanicError("text_paths must not be empty")

        self.words: List[str] = []

        texts = []
        for text_path in self.cfg.text_paths:
            with open(text_path, "r", encoding="utf-8") as f:
                text = f.read()
                texts.append(text)

        if self.cfg.filter_by_chars:
            texts = Corpus.filter_by_chars(texts, self.cfg.chars_file)
            self.font_manager.update_font_support_chars(self.cfg.chars_file)

        for text in texts:
            self.words.extend(text.split(self.cfg.separator))

        logger.info(f"Load {len(self.words)} words")

        if len(self.words) < self.cfg.num_word[1]:
            raise PanicError("too few words")

    def get_text(self):
        self.cfg: WordCorpusCfg
        length = np.random.randint(*self.cfg.num_word)
        start = np.random.randint(0, len(self.words) - length + 1)
        words = self.words[start : start + length]
        word = self.cfg.separator.join(words)
        return word
