#/bin/bash

# Show script usage
if [[ $# -ne 6 ]]; then
    echo "Use this script to automate pre-processing of HuBERT training on LibriSpeech dataset."
    echo ""
    echo "Usage: ./preprocess.sh [root_dir] [split] [data_dir] [result_dir] [phase] [kmeans_ratio]"
    echo "  - root_dir: Path to root of LibriSpeech dataset."
    echo "  - split: The dataset split used for training. Possible values: 100, 360, 500, 960."
    echo "  - data_dir: The target directory for unpacking dataset."
    echo "  - result_dir: The target directory for generated meta files and labels."
    echo "  - phase: The upcoming training phase. Possible values: 1, 2."
    echo "  - kmeans_ratio: The ratio of training data used in K-means training."
    exit 1
fi

SCRIPT_DIR=$(dirname "$0")
PROJECT_DIR=$(realpath "${SCRIPT_DIR}/../")
root_dir=$(realpath "$1")
split="$2"
data_dir=$(realpath "$3")
result_dir=$(realpath "$4")
phase="$5"
kmeans_ratio="$6"
data_split_name=""

if [[ ! -d "$data_dir" ]]; then
    mkdir -p "$data_dir"
fi

# Unpack dataset split
unpack_dataset ()
{
    split_name="$1"
    if [[ ! -d "${data_dir}/LibriSpeech/${split_name}" ]]; then
        echo "Unpacking LibriSpeech ${split_name} split..."
        tar -xf "${root_dir}/${split_name}.tar.gz" --directory ${data_dir}
    else
        echo "LibriSpeech ${split_name} split already found in ${data_dir}. Skipping unpacking process..."
    fi
}

if [[ $split == "100" ]]; then
    unpack_dataset "train-clean-100"
elif [[ $split == "360" ]]; then
    unpack_dataset "train-clean-360"
elif [[ $split == "500" ]]; then
    unpack_dataset "train-other-500"
elif [[ $split == "960" ]]; then
    unpack_dataset "train-clean-100"
    unpack_dataset "train-clean-360"
    unpack_dataset "train-other-500"
else
    echo "Unsupported dataset split: $split"
    exit 1
fi
unpack_dataset "dev-clean"

if [[ ! -d "$result_dir" ]]; then
    mkdir -p "$result_dir"
fi

# Generate tsv meta files
if [[ ! -d "${result_dir}/tsv" ]]; then
    python3 "${PROJECT_DIR}/gen_tsv.py" "${data_dir}/LibriSpeech" "${result_dir}/tsv" "$split" --speaker_sample_ratio=1 --data_sample_ratio=1
else
    echo "Meta files found in ${result_dir}/tsv, skipping generation process."
fi

if [[ $phase == "1" ]]; then
    # Generate MFCC features
    if [[ ! -d "${result_dir}/mfcc" ]]; then
        python3 "${PROJECT_DIR}/simple_kmeans/dump_mfcc_feature.py" "${result_dir}/tsv" train 1 0 "${result_dir}/mfcc"
        python3 "${PROJECT_DIR}/simple_kmeans/dump_mfcc_feature.py" "${result_dir}/tsv" valid 1 0 "${result_dir}/mfcc"
    else
        echo "MFCC features found in ${result_dir}/mfcc, skipping generation process."
    fi

    # Train K-means model
    if [[ ! -f "${result_dir}/phase1_kmeans.pt" ]]; then
        python3 "${PROJECT_DIR}/simple_kmeans/learn_kmeans.py" "${result_dir}/mfcc" train 1 "${result_dir}/phase1_kmeans.pt" 100 --percent "${kmeans_ratio}"
    else
        echo "K-means model checkpoint already exists, skipping training process."
    fi

    # Apply K-means model
    if [[ ! -d "${result_dir}/phase1_labels" ]]; then
        python3 "${PROJECT_DIR}/simple_kmeans/dump_km_label.py" "${result_dir}/mfcc" train "${result_dir}/phase1_kmeans.pt" 1 0 "${result_dir}/phase1_labels"
        python3 "${PROJECT_DIR}/simple_kmeans/dump_km_label.py" "${result_dir}/mfcc" valid "${result_dir}/phase1_kmeans.pt" 1 0 "${result_dir}/phase1_labels"
        mv "${result_dir}/phase1_labels/train_0_1.km"  "${result_dir}/phase1_labels/train.km"
        mv "${result_dir}/phase1_labels/valid_0_1.km"  "${result_dir}/phase1_labels/valid.km"
    else
        echo "K-means labels found in ${result_dir}/phase1_labels, skipping generation process."
    fi

    # Create dummy dict
    for x in $(seq 0 99); do
        echo "$x 1"
    done > "${result_dir}/phase1_labels/dict.km.txt"

elif [[ $phase == "2" ]]; then
    # Generate HuBERT features
    if [[ ! -d "${result_dir}/hubert" ]]; then
        read -p "Please enter Phase-1 HuBERT training checkpoint path: " ckpt_path
        python3 "${PROJECT_DIR}/simple_kmeans/dump_hubert_feature.py" "${result_dir}/tsv" train "$ckpt_path" 0 1 0 "${result_dir}/hubert"
        python3 "${PROJECT_DIR}/simple_kmeans/dump_hubert_feature.py" "${result_dir}/tsv" valid "$ckpt_path" 0 1 0 "${result_dir}/hubert"
    else
        echo "HuBERT features found in ${result_dir}/hubert, skipping generation process."
    fi

    # Train K-means model
    if [[ ! -f "${result_dir}/phase2_kmeans.pt" ]]; then
        python3 "${PROJECT_DIR}/simple_kmeans/learn_kmeans.py" "${result_dir}/hubert" train 1 "${result_dir}/phase2_kmeans.pt" 500 --percent "${kmeans_ratio}"
    else
        echo "K-means model checkpoint already exists, skipping training process."
    fi

    # Apply K-means model
    if [[ ! -d "${result_dir}/phase2_labels" ]]; then
        python3 "${PROJECT_DIR}/simple_kmeans/dump_km_label.py" "${result_dir}/hubert" train "${result_dir}/phase2_kmeans.pt" 1 0 "${result_dir}/phase2_labels"
        python3 "${PROJECT_DIR}/simple_kmeans/dump_km_label.py" "${result_dir}/hubert" valid "${result_dir}/phase2_kmeans.pt" 1 0 "${result_dir}/phase2_labels"
        mv "${result_dir}/phase2_labels/train_0_1.km"  "${result_dir}/phase2_labels/train.km"
        mv "${result_dir}/phase2_labels/valid_0_1.km"  "${result_dir}/phase2_labels/valid.km"
    else
        echo "K-means labels found in ${result_dir}/phase2_labels, skipping generation process."
    fi

    # Create dummy dict
    for x in $(seq 0 499); do
        echo "$x 1"
    done > "${result_dir}/phase2_labels/dict.km.txt"
fi
