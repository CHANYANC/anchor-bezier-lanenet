#!/bin/bash
set -e

DATA_ROOT=/path/to/CULane
LIST_TRAIN=${DATA_ROOT}/list/train.txt
LIST_VAL=${DATA_ROOT}/list/val.txt
SAVE_DIR=./runs_dual/phase2_straight
CKPT_PHASE1=./runs_dual/phase1_curve/dual_curve_only_epoch_10.pth

python train_dual.py \
        --phase straight_only \
        --data_root "${DATA_ROOT}" \
        --list_train "${LIST_TRAIN}" \
        --list_val "${LIST_VAL}" \
        --save_dir "${SAVE_DIR}" \
        --resume "${CKPT_PHASE1}" \
        --num_epochs 10
