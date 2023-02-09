#!/bin/bash

pip install --upgrade pip
pip install -r /ntu-jason/hubert-traning-utils/requirements.txt

cd /ntu-jason/exp/

fairseq-hydra-train \
    --config-dir /ntu-jason/hubert-training-utils/configs/ \
    --config-name hubert_tiny_librispeech_p2 \
    task.data=/ntu-jason/preprocessing/tsv \
    task.label_dir=/ntu-jason/preprocessing/phase2_labels \
    task.labels='["km"]' \
    model.label_rate=500