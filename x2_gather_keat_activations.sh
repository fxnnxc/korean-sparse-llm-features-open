# !/bin/bash

DATASET="keat"
MAX_LENGTH=128
LM_NAME="exaone"
LM_SIZE="8b"
DEVICE_MAP="auto"
BATCH_SIZE=4

python script/gather_keat_activations.py \
        --dataset "$DATASET" \
        --max_length "$MAX_LENGTH" \
        --lm_name "$LM_NAME" \
        --lm_size "$LM_SIZE" \
        --device_map "$DEVICE_MAP" \
        --batch_size "$BATCH_SIZE" \
