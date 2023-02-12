#!/bin/bash

if [[ $# -ne 5 ]]; then
    echo "Usage: ./hubert_tiny.sh [phase] [split] [utils_dir] [preprocess_dir] [exp_dir]."
    echo "  - phase: HuBERT training phase, could be 1 or 2."
    echo "  - split: LibriSpeech split use for training, could be 100, 360, 500, or 960."
    echo "  - utils_dir: path to hubert-training-utils directory."
    echo "  - preprocess_dir: path to directory of preprocessing results."
    echo "  - exp_dir: path to expriment directory."
fi

phase="$1"
split="$2"
utils_dir=$(realpath "$3")
preprocess_dir=$(realpath "$4")
exp_dir=$(realpath "$5")

if [[ $phase == "1" ]]; then
    config_name="hubert_tiny_librispeech_p1"
    label_dir="phase1_labels"
    label_rate="100"
elif [[ $phase == "2" ]]; then
    config_name="hubert_tiny_librispeech_p2"
    label_dir="phase2_labels"
    label_rate="50"
else
    echo "Unsupported training phase ${phase}."
    exit 1
fi

pip install --upgrade pip
pip install -r "${utils_dir}/requirements.txt"

cd "$exp_dir"

fairseq-hydra-train \
    --config-dir "${utils_dir}/configs/" \
    --config-name "$config_name" \
    task.data=${preprocess_dir}/tsv-${split} \
    task.label_dir=${preprocess_dir}/${label_dir} \
    task.labels='["km"]' \
    model.label_rate=${label_rate}

