Text Renderer
=========================================

.. toctree::
   :maxdepth: 1
   :caption: Notes

   config
   note/render_pipeline

.. toctree::
   :maxdepth: 2
   :caption: Modules

   dataset
   corpus/index
   effect/index
   layout/index

Albumentations Effects
----------------------

Text Renderer integrates with Albumentations for image augmentation.
See the Effect section for all available effects. Example usage:

.. code-block:: python

   from text_renderer.effect import AlbumentationsEffect, AlbumentationsEmboss, Effects
   import albumentations as A

   # Built-in effect
   effects = Effects([
       AlbumentationsEmboss(alpha=(0.9, 1.0), strength=(1.5, 1.6)),
   ])

   # Custom Albumentations pipeline
   custom_aug = A.Compose([
       A.GaussianBlur(blur_limit=(0, 1), p=1.0),
       A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=1.0)
   ])
   effects = Effects([
       AlbumentationsEffect(transform=custom_aug),
   ])
