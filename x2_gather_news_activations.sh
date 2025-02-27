# !/bin/bash

# Define arrays for parameters
DATASET="news"
MAX_LENGTH=128
LM_NAME="exaone"
LM_SIZE="8b"
LM_CACHE_DIR="cache"
DEVICE_MAP="auto"
OUTPUT_DIR="outputs"
BATCH_SIZE=4

python script/gather_news_activations.py \
        --dataset news \
        --max_length "$MAX_LENGTH" \
        --lm_name "$LM_NAME" \
        --lm_size "$LM_SIZE" \
        --lm_cache_dir "$LM_CACHE_DIR" \
        --device_map "$DEVICE_MAP" \
        --output_dir "$OUTPUT_DIR" \
        --batch_size "$BATCH_SIZE" \
