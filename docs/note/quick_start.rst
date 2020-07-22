Quick start
===========

To use text_renderer, you should prepare:

    - Font file: `.ttf` or...
    - Background image
    - Text: Optional. Depends on the :doc:`../corpus/index` ) you use.
    - Character set: Optional. Depends on the :doc:`../corpus/index` ) you use.

Run following command to generate example data:

.. code-block:: bash

    git clone https://github.com/oh-my-ocr/text_renderer
    cd text_renderer
    python3 setup.py develop
    pip3 install -r requirements.txt
    python3 main.py \
        --config example_data/example.py \
        --dataset img \
        --num_processes 2 \
        --log_period 10

The data is generated in the `example_data/output` directory

`main.py` only has 4 arguments:

    - config：Python config file path
    - dataset: Dataset format `img`/`lmdb`
    - num_processes: Number of processes used
    - log_period: Period of log printing. (0, 100)

All parameters related to the example data generation process are all configured in
`example.py <https://github.com/oh-my-ocr/text_renderer/blob/master/example_data/example.py>`_

Check :doc:`../config` and :doc:`../corpus/index` for all configurations


Run in Docker
-------------

Build image

.. code-block:: bash

    docker build -f docker/Dockerfile -t text_renderer .

Config file is provided by `CONFIG` environment.
In `example.py` file, data is generated in `example_data/output` directory,
so we map this directory to the host.

.. code-block:: bash

    docker run --rm \
    -v `pwd`/example_data/docker_output/:/app/example_data/output \
    --env CONFIG=/app/example_data/example.py \
    --env DATASET=img \
    --env NUM_PROCESSES=2 \
    --env LOG_PERIOD=10 \
    text_renderer

Build docs
----------

.. code-block:: bash

    cd docs
    make html

Open `_build/html/index.html`
