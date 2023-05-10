#!/bin/bash

if [[ $# -ne 3 ]]; then
    echo "Usage: ./train.sh [split] [layer] [kmeans_ratio]"
    exit 1
fi

split="$1"
layer="$2"
kmeas_ratio="$3"

source /ntu-jason/.venv/bin/activate

bash /ntu-jason/hubert-training-utils/scripts/preprocess.sh /LibriSpeech ${split} /ntu-jason/datasets /workspace/preprocessing-${split} 1 ${kmeas_ratio} || exit 1
bash /ntu-jason/hubert-training-utils/scripts/hubert_tiny.sh "1-short" /ntu-jason/hubert-training-utils /workspace/preprocessing-${split} /ntu-jason/exp/phase1-short/${layer}L-${split}h || exit 1
