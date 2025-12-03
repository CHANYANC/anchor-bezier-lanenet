#!/bin/bash
set -e

DATA_ROOT=/path/to/CULane
LIST_TRAIN=${DATA_ROOT}/list/train.txt
LIST_VAL=${DATA_ROOT}/list/val.txt
SAVE_DIR=./runs_dual/phase1_curve

python train_dual.py \
        --phase curve_only \
        --data_root "${DATA_ROOT}" \
        --list_train "${LIST_TRAIN}" \
        --list_val "${LIST_VAL}" \
        --save_dir "${SAVE_DIR}" \
        --num_epochs 10
