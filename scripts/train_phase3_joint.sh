#!/bin/bash
set -e

DATA_ROOT=/path/to/CULane
LIST_TRAIN=${DATA_ROOT}/list/train.txt
LIST_VAL=${DATA_ROOT}/list/val.txt
SAVE_DIR=./runs_dual/phase3_joint
CKPT_PHASE2=./runs_dual/phase2_straight/dual_straight_only_epoch_10.pth

python train_dual.py \
        --phase joint \
        --data_root "${DATA_ROOT}" \
        --list_train "${LIST_TRAIN}" \
        --list_val "${LIST_VAL}" \
        --save_dir "${SAVE_DIR}" \
        --resume "${CKPT_PHASE2}" \
        --num_epochs 5
