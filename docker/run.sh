#!/bin/sh

python3 main.py \
--config ${CONFIG} \
--dataset ${DATASET:-img} \
--num_processes ${NUM_PROCESSES:-2} \
--log_period ${LOG_PERIOD:-2} \
