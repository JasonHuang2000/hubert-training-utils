#!/bin/bash

if [[ $# -ne 2 ]]; then
    echo "Usage: ./finetune.sh [split] [layer]"
    exit 1
fi

split="$1"
layer="$2"
exp_name="${layer}L-${split}h"

source /ntu-jason/.venv/bin/activate

if [[ -d "/ntu-jason/exp/finetune/${exp_name}" ]]; then
    python3 /ntu-jason/s3prl/s3prl/run_downstream.py \
        -m train \
        -e "/ntu-jason/exp/finetune/${exp_name}/dev-clean-best.ckpt"
else
    python3 /ntu-jason/s3prl/s3prl/run_downstream.py \
        -m train \
        -u hubert_local \
        -d asr \
        -n "${exp_name}" \
        -p "/ntu-jason/exp/finetune/${exp_name}" \
        -c "/ntu-jason/s3prl/s3prl/downstream/asr/config.yaml" \
        -k "/ntu-jason/exp/phase2/${exp_name}/checkpoints/checkpoint_best.pt"
fi
