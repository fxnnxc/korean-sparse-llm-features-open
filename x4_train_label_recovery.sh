# !/bin/bash

# SAE_VERSIONS=("standard" "gated")  # ("standard" "gated" "gated_anneal" "standard_anneal")
SAE_VERSIONS=("standard")

for SAE_VERSION in "${SAE_VERSIONS[@]}"; do
    for Q in "2" "3"; do
        echo "Training label recovery (version: $SAE_VERSION, q: $Q)"
        python ./script/train_label_recovery.py \
            --sae_version "$SAE_VERSION" \
            --q "$Q"
    done
done
