#!/bin/bash

if [[ $# -ne 6 ]]; then
    echo "Usage: ./hubert_tiny_fl.sh [phase] [utils_dir] [preprocess_dir] [exp_dir] [config_dir] [fl_dir]."
    echo "  - phase: HuBERT training phase, could be 1, 1-short or 2."
    echo "  - utils_dir: path to hubert-training-utils directory."
    echo "  - preprocess_dir: path to directory of preprocessing results."
    echo "  - exp_dir: path to expriment directory."
    echo "  - config_dir: path to configuration directory."
    echo "  - fl_dir: path to FLUTE repo."
    exit 1
fi

SCRIPT_DIR=$(dirname "$0")
PROJECT_DIR=$(realpath "${SCRIPT_DIR}/../")
phase="$1"
utils_dir=$(realpath "$2")
preprocess_dir=$(realpath "$3")
exp_dir=$(realpath "$4")
config_dir=$(realpath "$5")
fl_dir=$(realpath "$6")

if [[ $phase == "1" ]]; then
    label_dir="phase1_labels"
    label_rate="100"
    source_file="${PROJECT_DIR}/configs/hubert_tiny_librispeech_p1.yaml"
elif [[ $phase == "1-short" ]]; then
    label_dir="phase1_labels"
    label_rate="100"
    source_file="${PROJECT_DIR}/configs/hubert_tiny_librispeech_p1_short.yaml"
elif [[ $phase == "2" ]]; then
    config_name="hubert_tiny_librispeech_p2"
    label_dir="phase2_labels"
    label_rate="50"
    source_file="${PROJECT_DIR}/configs/hubert_tiny_librispeech_p2.yaml"
else
    echo "Unsupported training phase ${phase}."
    exit 1
fi

# generate experiment configuration
mkdir -p "$exp_dir" "$config_dir"
python3 "${PROJECT_DIR}/gen_config.py" "$exp_dir" "$preprocess_dir" "$label_dir" "$label_rate" "$source_file" "${config_dir}/fairseq_config.yaml"

# run FL
cd $fl_dir
python3 -m torch.distributed.run --nproc_per_node=5 ./e2e_trainer.py \
    -dataPath ./testing \
    -outputPath "${exp_dir}/flute" \
    -config ./experiments/hubert_pretrain/hubert_pretrain_config.yaml \
    -task mlm_bert \
    -backend nccl
