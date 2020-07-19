Quick start
===========

.. code-block:: bash

    git clone https://github.com/oh-my-ocr/text_renderer
    cd text_renderer
    python3 setup.py develop
    pip3 install -r requirements.txt
    python3 main.py --config example_data/example.py

The data is generated in the `example_data/output` directory

All parameters related to the generation process are configured in `example.py`.
Check :doc:`../config` for all configurations


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
    text_renderer

Build docs
----------

.. code-block:: bash

    cd docs
    make html

Open `_build/html/index.html`
