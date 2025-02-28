# !/bin/bash

SAE_VERSIONS=("gated")  # ("standard" "gated" "gated_anneal" "standard_anneal")
LM_NAME="exaone"
LM_SIZE="8b"

for DATASET in "keat-ko"; do
    for SAE_VERSION in "${SAE_VERSIONS[@]}"; do
        for LAYER_QUANTILE in "q2" "q3"; do
            echo "Training SAE (dataset: $DATASET, version: $SAE_VERSION, layer quantile: $LAYER_QUANTILE)"
            python ./script/train_sae.py \
                --dataset "$DATASET" \
                --sae_version "$SAE_VERSION" \
                --lm_name "$LM_NAME" \
                --lm_size "$LM_SIZE" \
                --layer_quantile "$LAYER_QUANTILE"
        done
    done
done
