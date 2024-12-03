# !/bin/bash

# Define arrays for parameters
LM_NAME="exaone"
LM_SIZE="8b"
DEVICE_MAP="auto"
BATCH_SIZE=4
SPLIT="train"
total_steps=100000
data="news"

python script/gather_"$data".py \
        --lm_name "$LM_NAME" \
        --lm_size "$LM_SIZE" \
        --device_map "$DEVICE_MAP" \
        --batch_size "$BATCH_SIZE" \
        --split "$SPLIT" \
