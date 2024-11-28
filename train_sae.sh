# !/bin/bash

# Define arrays for parameters
# SAE_VERSIONS=("standard" "gated" "gated_anneal" "standard_anneal")
# SAE_VERSIONS=("gated"  )
SAE_VERSIONS=("gated"  )
LM_NAME="exaone"
LM_SIZE="8b"
SPLIT="train"
total_steps=100000
for LANG in "ko"; do
    for quantile in "q2" "q3"; do
        LAYER_QUANTILE=$quantile
        # Loop through SAE versions
        for sae in "${SAE_VERSIONS[@]}"; do
            echo "Training SAE version: $sae"
            python run.py \
                --sae "$sae" \
                --lm_name "$LM_NAME" \
                --lm_size "$LM_SIZE" \
                --split "$SPLIT" \
                --lang "$LANG" \
                --layer_quantile "$LAYER_QUANTILE" \
                --total_steps "$total_steps"
        done
    done
done