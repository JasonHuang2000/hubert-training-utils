#!/bin/bash

if [[ $# -ne 1 ]]; then
    echo "Usage: ./train_fl.sh [phase]"
    echo "  - phase: HuBERT training phase, could be 1, 1-short or 2."
    exit 1
fi

split=360
layer=2
kmeas_ratio=0.3
phase="$1"

source /ntu-jason/.venv/bin/activate

bash /ntu-jason/hubert-training-utils/scripts/preprocess.sh /LibriSpeech ${split} /ntu-jason/datasets /workspace/preprocessing-${split} 1 ${kmeas_ratio} || exit 1
bash /ntu-jason/hubert-training-utils/scripts/hubert_tiny.sh "$phase" /ntu-jason/hubert-training-utils /workspace/preprocessing-${split} "/ntu-jason/exp/${phase}/FL" /ntu-jason /ntu-jason/ssl-hubert/flute || exit 1
