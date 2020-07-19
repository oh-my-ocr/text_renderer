from dataclasses import dataclass, field
from pathlib import Path
from typing import List

import numpy as np
from text_renderer.utils.errors import PanicError

from .corpus import Corpus, CorpusCfg


@dataclass
class EnumCorpusCfg(CorpusCfg):
    """
    Enum corpus config

    args:
        text_paths (List[Path]): Text file paths
        items (List[str]): Texts to choice. Only works if text_paths is empty
        filter_by_chars (bool): If True, filtering text by character set
        chars_file (Path): Character set

    """

    text_paths: List[Path] = field(default_factory=list)
    items: List[str] = field(default_factory=list)
    filter_by_chars: bool = False
    chars_file: Path = None


class EnumCorpus(Corpus):
    """
    Output text in text_paths line by line or item from items
    """

    def __init__(self, cfg: "CorpusCfg"):
        super().__init__(cfg)

        self.cfg: EnumCorpusCfg
        if len(self.cfg.text_paths) == 0 and len(self.cfg.items) == 0:
            raise PanicError(f"text_paths or items must not be empty")

        if len(self.cfg.text_paths) != 0 and len(self.cfg.items) != 0:
            raise PanicError(f"only one of text_paths or items can be set")

        self.texts: List[str] = []

        if len(self.cfg.text_paths) != 0:
            for text_path in self.cfg.text_paths:
                with open(str(text_path), "r", encoding="utf-8") as f:
                    for line in f.readlines():
                        self.texts.append(line.strip())
        elif len(self.cfg.items) != 0:
            self.texts = self.cfg.items

        if self.cfg.filter_by_chars:
            self.texts = Corpus.filter_by_chars(self.texts, self.cfg.chars_file)

    def get_text(self):
        return np.random.choice(self.texts)
