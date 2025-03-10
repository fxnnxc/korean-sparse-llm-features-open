# !/bin/bash

DATASET="synthetic"
WINDOW_SIZE=10
MAX_LENGTH=128
LM_NAME="exaone"
LM_SIZE="8b"
DEVICE_MAP="auto"
BATCH_SIZE=4

python script/gather_synthetic_activations.py \
        --dataset "$DATASET" \
        --window_size "$WINDOW_SIZE" \
        --max_length "$MAX_LENGTH" \
        --lm_name "$LM_NAME" \
        --lm_size "$LM_SIZE" \
        --device_map "$DEVICE_MAP" \
        --batch_size "$BATCH_SIZE" \
