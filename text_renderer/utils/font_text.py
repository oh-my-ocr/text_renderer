from dataclasses import dataclass

from PIL.ImageFont import FreeTypeFont


@dataclass
class FontText:
    font: FreeTypeFont
    text: str
    font_path: str
    horizontal: bool = True

    @property
    def xy(self):
        # Use getbbox() instead of deprecated getoffset()
        bbox = self.font.getbbox(self.text)
        if bbox[2] > bbox[0] and bbox[3] > bbox[1]:  # Valid bbox
            left, top, right, bottom = bbox
        else:
            # Fallback for empty or invalid bbox
            left, top, right, bottom = 0, 0, 0, self.font.size
        return 0 - left, 0 - top

    @property
    def offset(self):
        # Use getbbox() instead of deprecated getoffset()
        bbox = self.font.getbbox(self.text)
        if bbox[2] > bbox[0] and bbox[3] > bbox[1]:  # Valid bbox
            return (bbox[0], bbox[1])  # Return left, top as offset
        else:
            # Fallback for empty or invalid bbox
            return (0, 0)

    @property
    def size(self) -> [int, int]:
        """
        Get text size without offset

        Returns:
            width, height
        """
        if self.horizontal:
            bbox = self.font.getbbox(self.text)
            if bbox[2] > bbox[0] and bbox[3] > bbox[1]:  # Valid bbox
                size = (bbox[2] - bbox[0], bbox[3] - bbox[1])
            else:
                # Fallback for empty or invalid bbox
                size = (0, self.font.size)
            # Use bbox directly instead of getoffset
            width = size[0]
            height = size[1]
            return width, height
        else:
            widths = []
            heights = []
            for c in self.text:
                bbox = self.font.getbbox(c)
                if bbox[2] > bbox[0] and bbox[3] > bbox[1]:  # Valid bbox
                    char_size = (bbox[2] - bbox[0], bbox[3] - bbox[1])
                else:
                    # Fallback for empty or invalid bbox
                    char_size = (0, self.font.size)
                widths.append(char_size[0])
                heights.append(char_size[1])
            width = max(widths)
            height = sum(heights)
            return height, width
