# Text Renderer
Generate text line images for training deep learning OCR model (e.g. [CRNN](https://github.com/bgshih/crnn)). ![example](./image/example.gif)

- [x] Modular design. You can easily add different components: [Corpus](https://oh-my-ocr.github.io/text_renderer/corpus/index.html), [Effect](https://oh-my-ocr.github.io/text_renderer/effect/index.html), [Layout](https://oh-my-ocr.github.io/text_renderer/layout/index.html).
- [x] Integrate with [imgaug](https://github.com/aleju/imgaug), see [imgaug_example](https://github.com/oh-my-ocr/text_renderer/blob/master/example_data/example.py#L184) for usage.
- [x] Support render multi corpus on image with different effects. [Layout](https://oh-my-ocr.github.io/text_renderer/layout/index.html) is responsible for the layout between multiple corpora
- [x] Support apply effects on different stages of rendering process [corpus_effects](https://oh-my-ocr.github.io/text_renderer/config.html#text_renderer.config.RenderCfg), [layout_effects](https://oh-my-ocr.github.io/text_renderer/config.html#text_renderer.config.RenderCfg), [render_effects](https://oh-my-ocr.github.io/text_renderer/config.html#text_renderer.config.RenderCfg).
- [x] Generate vertical text.
- [x] Support generate `lmdb` dataset which compatible with [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR), see [Dataset](https://oh-my-ocr.github.io/text_renderer/dataset.html)
- [ ] Corpus sampler: helpful to perform character balance

[Documentation](https://oh-my-ocr.github.io/text_renderer/index.html)

## Run Example

Run following command to generate images using example data:

```bash
git clone https://github.com/oh-my-ocr/text_renderer
cd text_renderer
python3 setup.py develop
pip3 install -r docker/requirements.txt
python3 main.py \
    --config example_data/example.py \
    --dataset img \
    --num_processes 2 \
    --log_period 10
```

The data is generated in the `example_data/output` directory. A `labels.json` file contains all annotations in follow format:
```json
{
  "labels": {
    "000000000": "test",
    "000000001": "text2"
  },
  "sizes": {
    "000000000": [
      120,
      32 
    ],
    "000000001": [
      128,
      32 
    ]
  },
  "num-samples": 2
}
```

You can also use `--dataset lmdb` to store image in lmdb file, lmdb file contains follow keys:
- num-samples
- image-000000000
- label-000000000
- size-000000000

You can check config file [example_data/example.py](https://github.com/oh-my-ocr/text_renderer/blob/master/example_data/example.py) to learn how to use text_renderer,
or follow the [Quick Start](https://github.com/oh-my-ocr/text_renderer#quick-start) to learn how to setup configuration
 

## Quick Start
### Prepare file resources
   
- Font files: `.ttf`、`.otf`、`.ttc`
- Background images of any size, either from your business scenario or from publicly available datasets ([COCO](https://cocodataset.org/#home), [VOC](http://host.robots.ox.ac.uk/pascal/VOC/))
- Corpus: text_renderer offers a wide variety of [text sampling methods](https://oh-my-ocr.github.io/text_renderer/corpus/index.html), 
to use these methods, you need to consider the preparation of the corpus from two perspectives：
1. The corpus must be in the target language for which you want to perform OCR recognition
2. The corpus should meets your actual business needs, such as education field, medical field, etc.
- Charset file [Optional but recommend]: OCR models in real-world scenarios (e.g. CRNN) usually support only a limited character set, 
so it's better to filter out characters outside the character set during data generation. 
You can do this by setting the [chars_file](https://oh-my-ocr.github.io/text_renderer/corpus/char_corpus.html) parameter

You can download pre-prepared file resources for this `Quick Start` from here: 

- [simsun.ttf](https://github.com/oh-my-ocr/text_renderer/raw/master/example_data/font/simsun.ttf)
- [background.png](https://github.com/oh-my-ocr/text_renderer/raw/master/example_data/bg/background.png)
- [eng_text.txt](https://github.com/oh-my-ocr/text_renderer/raw/master/example_data/text/eng_text.txt)

Save these resource files in the same directory:
```
workspace
├── bg
│ └── background.png
├── corpus
│ └── eng_text.txt
└── font
    └── simsun.ttf
```

### Create config file
Create a `config.py` file in `workspace` directory. One configuration file must have a `configs` variable, it's 
a list of [GeneratorCfg](https://oh-my-ocr.github.io/text_renderer/config.html#text_renderer.config.GeneratorCfg). 

The complete configuration file is as follows:
```python
import os
from pathlib import Path

from text_renderer.effect import *
from text_renderer.corpus import *
from text_renderer.config import (
    RenderCfg,
    NormPerspectiveTransformCfg,
    GeneratorCfg,
    SimpleTextColorCfg,
)

CURRENT_DIR = Path(os.path.abspath(os.path.dirname(__file__)))


def story_data():
    return GeneratorCfg(
        num_image=10,
        save_dir=CURRENT_DIR / "output",
        render_cfg=RenderCfg(
            bg_dir=CURRENT_DIR / "bg",
            height=32,
            perspective_transform=NormPerspectiveTransformCfg(20, 20, 1.5),
            corpus=WordCorpus(
                WordCorpusCfg(
                    text_paths=[CURRENT_DIR / "corpus" / "eng_text.txt"],
                    font_dir=CURRENT_DIR / "font",
                    font_size=(20, 30),
                    num_word=(2, 3),
                ),
            ),
            corpus_effects=Effects(Line(0.9, thickness=(2, 5))),
            gray=False,
            text_color_cfg=SimpleTextColorCfg(),
        ),
    )


configs = [story_data()]
```

In the above configuration we have done the following things:

1. Specify the location of the resource file
2. Specified text sampling method: 2 or 3 words are randomly selected from the corpus
3. Configured some effects for generation
   - Perspective transformation [NormPerspectiveTransformCfg](https://oh-my-ocr.github.io/text_renderer/_modules/text_renderer/config.html#NormPerspectiveTransformCfg)
   - Random [Line Effect](https://oh-my-ocr.github.io/text_renderer/effect/line.html)
   - Fix output image height to 32
   - Generate color image. `gray=False`, `SimpleTextColorCfg()`
4. Specifies font-related parameters: `font_size`, `font_dir`

### Run 
Run `main.py`, it only has 4 arguments:
- config：Python config file path
- dataset: Dataset format `img` or `lmdb`
- num_processes: Number of processes used
- log_period: Period of log printing. (0, 100)

## All Effect/Layout Examples

Find all effect/layout config example at [link](https://github.com/oh-my-ocr/text_renderer/blob/master/example_data/effect_layout_example.py)

- `bg_and_text_mask`: Three images of the same width are merged together horizontally, 
  it can be used to train GAN model like [EraseNet](https://github.com/lcy0604/EraseNet)

|    | Name                                 | Example                                                                                                                                                                      |
|---:|:-------------------------------------|:-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
|  0 | bg_and_text_mask                     | ![bg_and_text_mask.jpg](https://github.com/oh-my-ocr/text_renderer/raw/master/example_data/effect_layout_image/bg_and_text_mask.jpg)                                         |
|  1 | char_spacing_compact                 | ![char_spacing_compact.jpg](https://github.com/oh-my-ocr/text_renderer/raw/master/example_data/effect_layout_image/char_spacing_compact.jpg)                                 |
|  2 | char_spacing_large                   | ![char_spacing_large.jpg](https://github.com/oh-my-ocr/text_renderer/raw/master/example_data/effect_layout_image/char_spacing_large.jpg)                                     |
|  3 | color_image                          | ![color_image.jpg](https://github.com/oh-my-ocr/text_renderer/raw/master/example_data/effect_layout_image/color_image.jpg)                                                   |
|  4 | curve                                | ![curve.jpg](https://github.com/oh-my-ocr/text_renderer/raw/master/example_data/effect_layout_image/curve.jpg)                                                               |
|  5 | dropout_horizontal                   | ![dropout_horizontal.jpg](https://github.com/oh-my-ocr/text_renderer/raw/master/example_data/effect_layout_image/dropout_horizontal.jpg)                                     |
|  6 | dropout_rand                         | ![dropout_rand.jpg](https://github.com/oh-my-ocr/text_renderer/raw/master/example_data/effect_layout_image/dropout_rand.jpg)                                                 |
|  7 | dropout_vertical                     | ![dropout_vertical.jpg](https://github.com/oh-my-ocr/text_renderer/raw/master/example_data/effect_layout_image/dropout_vertical.jpg)                                         |
|  8 | emboss                               | ![emboss.jpg](https://github.com/oh-my-ocr/text_renderer/raw/master/example_data/effect_layout_image/emboss.jpg)                                                             |
|  9 | extra_text_line_layout               | ![extra_text_line_layout.jpg](https://github.com/oh-my-ocr/text_renderer/raw/master/example_data/effect_layout_image/extra_text_line_layout.jpg)                             |
| 10 | line_bottom                          | ![line_bottom.jpg](https://github.com/oh-my-ocr/text_renderer/raw/master/example_data/effect_layout_image/line_bottom.jpg)                                                   |
| 11 | line_bottom_left                     | ![line_bottom_left.jpg](https://github.com/oh-my-ocr/text_renderer/raw/master/example_data/effect_layout_image/line_bottom_left.jpg)                                         |
| 12 | line_bottom_right                    | ![line_bottom_right.jpg](https://github.com/oh-my-ocr/text_renderer/raw/master/example_data/effect_layout_image/line_bottom_right.jpg)                                       |
| 13 | line_horizontal_middle               | ![line_horizontal_middle.jpg](https://github.com/oh-my-ocr/text_renderer/raw/master/example_data/effect_layout_image/line_horizontal_middle.jpg)                             |
| 14 | line_left                            | ![line_left.jpg](https://github.com/oh-my-ocr/text_renderer/raw/master/example_data/effect_layout_image/line_left.jpg)                                                       |
| 15 | line_right                           | ![line_right.jpg](https://github.com/oh-my-ocr/text_renderer/raw/master/example_data/effect_layout_image/line_right.jpg)                                                     |
| 16 | line_top                             | ![line_top.jpg](https://github.com/oh-my-ocr/text_renderer/raw/master/example_data/effect_layout_image/line_top.jpg)                                                         |
| 17 | line_top_left                        | ![line_top_left.jpg](https://github.com/oh-my-ocr/text_renderer/raw/master/example_data/effect_layout_image/line_top_left.jpg)                                               |
| 18 | line_top_right                       | ![line_top_right.jpg](https://github.com/oh-my-ocr/text_renderer/raw/master/example_data/effect_layout_image/line_top_right.jpg)                                             |
| 19 | line_vertical_middle                 | ![line_vertical_middle.jpg](https://github.com/oh-my-ocr/text_renderer/raw/master/example_data/effect_layout_image/line_vertical_middle.jpg)                                 |
| 20 | padding                              | ![padding.jpg](https://github.com/oh-my-ocr/text_renderer/raw/master/example_data/effect_layout_image/padding.jpg)                                                           |
| 21 | perspective_transform                | ![perspective_transform.jpg](https://github.com/oh-my-ocr/text_renderer/raw/master/example_data/effect_layout_image/perspective_transform.jpg)                               |
| 22 | same_line_layout_different_font_size | ![same_line_layout_different_font_size.jpg](https://github.com/oh-my-ocr/text_renderer/raw/master/example_data/effect_layout_image/same_line_layout_different_font_size.jpg) |
| 23 | vertical_text                        | ![vertical_text.jpg](https://github.com/oh-my-ocr/text_renderer/raw/master/example_data/effect_layout_image/vertical_text.jpg)                                               |


## Contribution

- Corpus: Feel free to contribute more corpus generators to the project, 
  It does not necessarily need to be a generic corpus generator, but can also be a business-specific generator, 
  such as generating ID numbers


## Run in Docker

Build image

```bash
docker build -f docker/Dockerfile -t text_renderer .
```

Config file is provided by `CONFIG` environment.
In `example.py` file, data is generated in `example_data/output` directory,
so we map this directory to the host.

```bash
docker run --rm \
-v `pwd`/example_data/docker_output/:/app/example_data/output \
--env CONFIG=/app/example_data/example.py \
--env DATASET=img \
--env NUM_PROCESSES=2 \
--env LOG_PERIOD=10 \
text_renderer
```

## Build docs

```bash
cd docs
make html
open _build/html/index.html
```

Open `_build/html/index.html`


## Citing text_renderer
If you use text_renderer in your research, please consider use the following BibTeX entry.

```BibTeX
@misc{text_renderer,
  author =       {oh-my-ocr},
  title =        {text_renderer},
  howpublished = {\url{https://github.com/oh-my-ocr/text_renderer}},
  year =         {2021}
}
```
