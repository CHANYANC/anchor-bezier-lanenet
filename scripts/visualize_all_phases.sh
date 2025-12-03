#!/bin/bash
set -e

DATA_ROOT=/path/to/CULane
LIST_VAL=${DATA_ROOT}/list/test.txt
OUT_DIR=./vis_all_phases

CKPT_PHASE1=./runs_dual/phase1_curve/dual_curve_only_epoch_10.pth
CKPT_PHASE2=./runs_dual/phase2_straight/dual_straight_only_epoch_10.pth
CKPT_PHASE3=./runs_dual/phase3_joint/dual_joint_epoch_5.pth
CKPT_PHASE4=./runs_dual/phase4_route/dual_route_epoch_5.pth

python visualize_all_phases.py \
        --data_root "${DATA_ROOT}" \
        --list_val "${LIST_VAL}" \
        --ckpt_phase1 "${CKPT_PHASE1}" \
        --ckpt_phase2 "${CKPT_PHASE2}" \
        --ckpt_phase3 "${CKPT_PHASE3}" \
        --ckpt_phase4 "${CKPT_PHASE4}" \
        --out_dir "${OUT_DIR}" \
        --num_vis 50 \
        --num_workers 0
