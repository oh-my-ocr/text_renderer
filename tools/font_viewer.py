from pathlib import Path

import typer
import streamlit as st

from text_renderer.font_manager import FontManager
from text_renderer.utils import FontText
from text_renderer.utils.draw_utils import draw_text_on_bg


def main(name: str, font_dir: str):
    font_manager = FontManager(Path(font_dir), None, (10, 20))

    font_size = st.sidebar.number_input("Font Size", min_value=30)
    text = st.sidebar.text_input("Input Text", "Hello World")
    text_color = (0, 0, 0, 255)

    # images = {}
    for font_path in font_manager.font_paths:
        font = font_manager._get_font(font_path, font_size)
        font_text = FontText(font, text, font_path)
        text_mask = draw_text_on_bg(font_text, text_color)
        st.text(Path(font_path).name)
        st.image(text_mask)


if __name__ == "__main__":
    try:
        typer.run(main)
    except SystemExit as se:
        if se.code != 0:
            raise
