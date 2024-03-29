#!/bin/bash

if [[ $# -ne 4 ]]; then
    echo "Usage: ./hubert_tiny.sh [phase] [split] [utils_dir] [preprocess_dir] [exp_dir]."
    echo "  - phase: HuBERT training phase, could be 1, 2 or 1-short."
    echo "  - utils_dir: path to hubert-training-utils directory."
    echo "  - preprocess_dir: path to directory of preprocessing results."
    echo "  - exp_dir: path to expriment directory."
fi

phase="$1"
utils_dir=$(realpath "$2")
preprocess_dir=$(realpath "$3")
exp_dir=$(realpath "$4")

if [[ $phase == "1" ]]; then
    config_name="hubert_tiny_librispeech_p1"
    label_dir="phase1_labels"
    label_rate="100"
elif [[ $phase == "1-short" ]]; then
    config_name="hubert_tiny_librispeech_p1_short"
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

fairseq-hydra-train \
    --config-dir "${utils_dir}/configs/" \
    --config-name "$config_name" \
    hydra.run.dir=${exp_dir} \
    task.data=${preprocess_dir}/tsv \
    task.label_dir=${preprocess_dir}/${label_dir} \
    task.labels='["km"]' \
    model.label_rate=${label_rate}

