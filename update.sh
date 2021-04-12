#!/usr/bin/env bash

set - e

python3 main.py \
--config example_data/effect_layout_example.py \
--dataset img \
--num_processes 1 \
--log_period 100

python3 tools/prepare_effect_layout_example.py


