"""
Text Renderer Package

A modular text rendering library for generating synthetic text images for OCR training.
Supports various text corpora, effects, layouts, and image augmentation techniques.

Main components:
- Corpus: Text generation from different sources (characters, words, enums, etc.)
- Effects: Image augmentation effects (dropout, emboss, motion blur, etc.)
- Layout: Text layout management for multi-corpus rendering
- Render: Main rendering engine for generating text images

Example:
    from text_renderer import Render
    from text_renderer.config import RenderCfg
    
    cfg = RenderCfg(...)
    renderer = Render(cfg)
    image, text = renderer()
"""

__version__ = "0.1.0"
