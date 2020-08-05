import os
import random
from functools import lru_cache
from itertools import chain
from pathlib import Path
from typing import List, Set, Tuple, Dict

from PIL import ImageFont
from PIL.ImageFont import FreeTypeFont
from fontTools.ttLib import TTFont, TTCollection
from fontTools.unicode import Unicode
from loguru import logger

from text_renderer.utils.errors import PanicError
from text_renderer.utils.utils import load_chars_file


class FontManager:
    def __init__(
        self, font_dir: Path, font_list_file: Path, font_size: Tuple[int, int]
    ):
        assert font_size[0] < font_size[1]
        self.font_size_min = font_size[0]
        self.font_size_max = font_size[1]
        self.font_paths: List[str] = []
        self.font_support_chars_cache: Dict[str, Set] = {}

        with open(str(font_list_file), "r", encoding="utf-8") as f:
            lines = f.readlines()
            lines = [line.strip() for line in lines]

        if len(lines) == 0:
            raise PanicError(f"font list file is empty: {font_list_file}")

        for line in lines:
            font_path = font_dir / line
            if font_path.exists():
                self.font_paths.append(str(font_path))
            else:
                raise PanicError(f"font file not exist: {font_path}")

        self._load_font_support_chars()

    def get_font(self) -> Tuple[FreeTypeFont, Set, str]:
        font_path = random.choice(self.font_paths)
        font_size = random.randint(self.font_size_min, self.font_size_max)

        font = self._get_font(font_path, font_size)
        font_support_chars = self.font_support_chars_cache[font_path]

        return font, font_support_chars, font_path

    def check_support(self, text: str, chars: Set) -> Tuple[bool, Set]:
        # Check whether all chars in text exist in chars
        text_set = set(text)
        intersect = text_set - chars
        status = len(intersect) == 0

        return status, intersect

    def _load_font_support_chars(self):
        for font_path in self.font_paths:
            ttf = self._load_ttfont(font_path)

            chars_int = set()
            for table in ttf["cmap"].tables:
                for k, v in table.cmap.items():
                    chars_int.add(k)

            supported_chars = set([chr(c_int) for c_int in chars_int])

            ttf.close()

            self.font_support_chars_cache[font_path] = supported_chars

    def update_font_support_chars(self, chars_file):
        """
        Although some fonts have a specific character in the cmap, the rendered text is blank on the image.

        Parameters
        ----------
        chars_file: Path
            one char per line
        """
        white_list = [" "]

        removed_chars = []
        chars = load_chars_file(chars_file)
        for font_path in self.font_paths:
            font = self._get_font(font_path, 40)
            for c in chars:
                bbox = font.getmask(c).getbbox()
                if (
                    c not in white_list
                    and bbox is None
                    and c in self.font_support_chars_cache[font_path]
                ):
                    self.font_support_chars_cache[font_path].remove(c)
                    removed_chars.append(c)

            logger.info(f"Remove {len(removed_chars)} empty char mask from font [{font_path}]: {removed_chars}")

    def _load_ttfont(self, font_path: str) -> TTFont:
        """
        Read ttc, ttf, otf font file, return a TTFont object
        """

        # ttc is collection of ttf
        if font_path.endswith("ttc"):
            ttc = TTCollection(font_path)
            # assume all ttfs in ttc file have same supported chars
            return ttc.fonts[0]

        if (
            font_path.endswith("ttf")
            or font_path.endswith("TTF")
            or font_path.endswith("otf")
        ):
            ttf = TTFont(
                font_path, 0, allowVID=0, ignoreDecompileErrors=True, fontNumber=-1
            )

            return ttf

    @lru_cache()
    def _get_font(self, font_path: str, font_size: int) -> FreeTypeFont:
        font = ImageFont.truetype(font_path, font_size)
        return font
