import os
import sys
from pathlib import Path

import pytest
from text_renderer.utils.errors import PanicError
from text_renderer.utils.utils import load_chars_file, SPACE_CHAR

CURRENT_DIR = Path(os.path.abspath(os.path.dirname(__file__)))
DATA_DIR = CURRENT_DIR / "data"


def test_contain_one_space(capsys):
    chars = load_chars_file(DATA_DIR / "one_space.txt")
    assert SPACE_CHAR in chars


def test_contain_two_space():
    with pytest.raises(PanicError, match="Find two space"):
        load_chars_file(DATA_DIR / "two_space.txt")
