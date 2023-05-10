#!/bin/bash

if [[ $# -ne 3 ]]; then
    echo "Usage: ./train.sh [split] [layer] [kmeans_ratio]"
    exit 1
fi

split="$1"
layer="$2"
kmeas_ratio="$3"

source /ntu-jason/.venv/bin/activate

bash /ntu-jason/hubert-training-utils/scripts/preprocess.sh /LibriSpeech ${split} /ntu-jason/datasets /ntu-jason/preprocessing-${split} 1 ${kmeas_ratio} || exit 1
bash /ntu-jason/hubert-training-utils/scripts/hubert_tiny.sh 1 /ntu-jason/hubert-training-utils /ntu-jason/preprocessing-${split} /ntu-jason/exp/phase1/${layer}L-${split}h || exit 1
echo /ntu-jason/exp/phase1/${layer}L-${split}h/checkpoints/checkpoint_best.pt | \
    bash /ntu-jason/hubert-training-utils/scripts/preprocess.sh /LibriSpeech ${split} /ntu-jason/datasets /ntu-jason/preprocessing-${split} 2 ${kmeas_ratio} || exit 1
bash /ntu-jason/hubert-training-utils/scripts/hubert_tiny.sh 2 /ntu-jason/hubert-training-utils /ntu-jason/preprocessing-${split} /ntu-jason/exp/phase2/${layer}L-${split}h || exit 1
