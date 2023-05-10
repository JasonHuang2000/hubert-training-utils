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

mkdir -p /ntu-jason/exp/${phase}/FL

bash /ntu-jason/hubert-training-utils/scripts/preprocess.sh /LibriSpeech ${split} /ntu-jason/datasets /ntu-jason/preprocessing-${split} 1 ${kmeas_ratio} || exit 1
bash /ntu-jason/hubert-training-utils/scripts/hubert_tiny_fl.sh "$phase" /ntu-jason/hubert-training-utils /ntu-jason/preprocessing-${split} /ntu-jason/exp/${phase}/FL /ntu-jason /ntu-jason/ssl-hubert/flute || exit 1
