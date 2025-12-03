#!/bin/bash
set -e

DATA_ROOT=/path/to/CULane
LIST_TRAIN=${DATA_ROOT}/list/train.txt
LIST_VAL=${DATA_ROOT}/list/val.txt
SAVE_DIR=./runs_dual/phase4_route
CKPT_PHASE3=./runs_dual/phase3_joint/dual_joint_epoch_5.pth

python train_dual.py \
        --phase route \
        --data_root "${DATA_ROOT}" \
        --list_train "${LIST_TRAIN}" \
        --list_val "${LIST_VAL}" \
        --save_dir "${SAVE_DIR}" \
        --resume "${CKPT_PHASE3}" \
        --num_epochs 5
