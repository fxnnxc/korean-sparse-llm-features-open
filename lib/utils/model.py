import os

import torch

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)


__all__ = [
    'get_exaone',
]


SYSTEM_PROMPT = (
    "A question from a human to an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the human's questions.\n"
)
USER_PROMPT = "What is the capital of France?"
# SYSTEM_PROMPT_TEMPLATE = f"[|system|]{SYSTEM_PROMPT}\n[|user|]\n\n{USER_PROMPT}\n[|assistant|]"
SYSTEM_PROMPT_TEMPLATE = f"[|system|]{SYSTEM_PROMPT}\n[|user|]{USER_PROMPT}\n[|assistant|]"  # (possible fix)
EXAONE_MODEL_SIZES_TO_LAYERS = {
    '8b': 32,
    '13b': 40,
}


def get_exaone(
    lm_size,
    lm_cache_dir,
    device_map='auto',
    precision=None,
    return_tokenizer_only=False,
):
    lm_size = lm_size.upper()
    name = f'LGAI-EXAONE/EXAONE-3.0-7.{lm_size}-Instruct'

    if return_tokenizer_only:
        tokenizer = AutoTokenizer.from_pretrained(
            name,
            cache_dir=None if lm_cache_dir in ['none', None] else lm_cache_dir
        )
        return tokenizer

    if isinstance(device_map, int):
        device_map = custom_device_map(lm_size, device_map)

    precision = torch.bfloat16 if precision is None else precision

    model = AutoModelForCausalLM.from_pretrained(
        name,
        torch_dtype=precision,
        trust_remote_code=True,
        cache_dir=None if lm_cache_dir in ['none', None] else lm_cache_dir,
        device_map=device_map,
        token=os.environ['HF_TOKEN'],
    )
    tokenizer = AutoTokenizer.from_pretrained(
        name,
        cache_dir=None if lm_cache_dir in ['none', None] else lm_cache_dir
    )
    tokenizer.sep_token_id = tokenizer.eos_token_id
    tokenizer.pad_token_id = tokenizer.eos_token_id

    return (
        model,
        tokenizer,
    )


def custom_device_map(lm_size, num_gpus):
    assert num_gpus > 0, "num_gpus must be greater than 0"
    model_size = EXAONE_MODEL_SIZES_TO_LAYERS[lm_size]
    device_map = {
        'transformer.wte': 0,
        'transformer.ln_f': num_gpus - 1,
        'lm_head': num_gpus - 1,
    }
    gpu = 0
    per_block = model_size // num_gpus
    for i in range(1, model_size + 1):
        device_map[f'transformer.h.{i-1}'] = gpu
        if i % per_block == 0 and gpu < num_gpus - 1:
            gpu += 1
    return device_map
