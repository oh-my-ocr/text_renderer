from dataclasses import dataclass

from PIL.ImageFont import FreeTypeFont


@dataclass
class FontText:
    font: FreeTypeFont
    text: str
    font_path: str

    @property
    def xy(self):
        offset = self.font.getoffset(self.text)
        left, top, right, bottom = self.font.getmask(self.text).getbbox()
        return 0 - offset[0] - left, 0 - offset[1]

    @property
    def offset(self):
        return self.font.getoffset(self.text)

    @property
    def size(self) -> [int, int]:
        """
        Get text size without offset

        Returns:
            width, height
        """
        offset = self.font.getoffset(self.text)
        size = self.font.getsize(self.text)
        width = size[0] - offset[0]
        height = size[1] - offset[1]
        left, top, right, bottom = self.font.getmask(self.text).getbbox()
        return right - left, height
