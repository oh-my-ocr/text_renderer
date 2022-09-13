from pathlib import Path
from typing import List

from typer import Typer, Option
from rich import print
from rich.table import Table

from text_renderer.font_manager import FontManager
from text_renderer.utils.utils import load_chars_file

app = Typer()


@app.command()
def main(font_dir: Path = Option(...), font_list_file: Path = Option(None), char_path: Path = Option(...),
         thresh: int = Option(-1)):
    chars = set(load_chars_file(char_path))
    print(f"{char_path} chars: {len(chars)}")
    font_manager = FontManager(font_dir, font_list_file, font_size=(20, 21))
    font_manager.update_font_support_chars(char_path)

    table = Table()
    table.add_column('font')
    table.add_column('supported')

    rows = []
    for k, all_chars in font_manager.font_support_chars_intersection_with_chars.items():
        count = len(all_chars & chars)
        if count > thresh:
            rows.append((k, count))

    rows.sort(key=lambda x: x[1], reverse=True)
    for row in rows:
        table.add_row(*[row[0], str(row[1])])

    print(table)


if __name__ == "__main__":
    app()
