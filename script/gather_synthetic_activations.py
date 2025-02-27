import sys
import pickle
import argparse

import torch
import datasets

from pathlib import Path

from tqdm import tqdm
from omegaconf import OmegaConf

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.append(str(PROJECT_ROOT))
sys.path.append(str(PROJECT_ROOT / 'lib'))

from lib.utils.data import get_dataloder_from_dataset
from lib.utils.fetch import MultipleFetch
from lib.utils.model import get_exaone
from lib.datasets.synthetic import get_synthetic_dataset


def get_flags():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='synthetic')
    parser.add_argument('--window_size', type=int, default=10)
    parser.add_argument('--max_length', type=int, default=128)
    parser.add_argument('--lm_name', type=str, default='exaone')
    parser.add_argument('--lm_size', type=str, default='8b')
    parser.add_argument('--lm_cache_dir', type=str, default=PROJECT_ROOT / 'cache')
    parser.add_argument('--device_map', type=str, default='auto')
    parser.add_argument('--output_dir', type=str, default='outputs')
    parser.add_argument('--batch_size', type=int, default=4)
    flags = parser.parse_args()
    flags = OmegaConf.create(vars(flags))
    return flags


def condition_fn(module, input, output):
    return True


def fetch_fn(module, input, output):
    if not hasattr(module, 'fetched_values'):
        module.fetched_values = []
    module.fetched_values.append(output[0].detach().cpu())  # output[1] is a DynamicCache


def estimate_token_location(text, target, tokenizer, window_size=10, max_length=128):
    tokenizer.padding_side = 'left'
    tokens = tokenizer.encode(text, return_tensors='pt', padding='max_length', max_length=max_length)[0]
    loc = None
    for t in range(1, len(tokens)):
        decoded = tokenizer.decode(tokens[max(0, t - window_size) : t])
        if target in decoded:
            loc = t
            break
    assert loc is not None, f"Target {target} not found in the text: {decoded}"
    return tokens, loc


# dataset of pair of tokens and loc of the target token
def get_dataset(tokenizer, flags):
    dataset = get_synthetic_dataset()
    data = []
    for i in range(len(dataset)):
        text = dataset[i]['ko_text']
        target = dataset[i]['small_category']
        tokens, loc = estimate_token_location(
            text,
            target,
            tokenizer,
            flags.window_size,
            flags.max_length,
        )
        data.append({
            'tokens': tokens,
            'loc': loc,
        })
    dataset = datasets.Dataset.from_list(data)
    return dataset


def main(flags):

    # init
    name = flags.dataset
    llm, tokenizer = get_exaone(  # TODO: what about other LLMs?
        lm_size=flags.lm_size,
        lm_cache_dir=flags.lm_cache_dir,
        device_map=flags.device_map,
    )

    # activations to be recorded
    activations = {
        f'{name}_input_ids': [],
        f'{name}_residual_q1': [],
        f'{name}_residual_q2': [],
        f'{name}_residual_q3': [],
    }

    # register hooks
    num_layers = llm.config.num_layers
    dict_format = {
        'residual_q0': {
            'module': llm.transformer.h[0],
            'condition_fn': condition_fn,
            'fetch_fn': fetch_fn,
        },
        'residual_q1': {
            'module': llm.transformer.h[int(num_layers * 0.25)],
            'condition_fn': condition_fn,
            'fetch_fn': fetch_fn,
        },
        'residual_q2': {
            'module': llm.transformer.h[int(num_layers * 0.5)],
            'condition_fn': condition_fn,
            'fetch_fn': fetch_fn,
        },
        'residual_q3': {
            'module': llm.transformer.h[int(num_layers * 0.75)],
            'condition_fn': condition_fn,
            'fetch_fn': fetch_fn,
        },
    }
    _ = MultipleFetch(dict_format)

    # prepare dataset
    dataset = get_dataset(tokenizer, flags)
    dataloader = get_dataloder_from_dataset(dataset, batch_size=flags.batch_size)

    # fetch and save activations
    print(f"Processing {name} dataset...")
    pbar = tqdm(dataloader)
    for batch in pbar:

        # move to device
        b_tokens = batch['tokens'].to(llm.device)
        b_loc = batch['loc']

        # clear fetched_values (for each batch), but maybe we can do better
        for key in dict_format.keys():
            dict_format[key]['module'].fetched_values = []

        # run forward pass
        _ = llm.forward(input_ids=b_tokens)

        # gather activations
        for b in range(b_tokens.shape[0]):
            tokens = b_tokens[b].detach().cpu()
            loc = int(b_loc[b])
            activations[f'{name}_input_ids'].append(tokens[loc])
            activations[f'{name}_residual_q1'].append(
                dict_format['residual_q1']['module']
                .fetched_values[0][b]  # always 0 since we're clearing fetched_values everytime (maybe better design might exist)
                .detach()
                .cpu()[loc]
            )
            activations[f'{name}_residual_q2'].append(
                dict_format['residual_q2']['module']
                .fetched_values[0][b]
                .detach()
                .cpu()[loc]
            )
            activations[f'{name}_residual_q3'].append(
                dict_format['residual_q3']['module']
                .fetched_values[0][b]
                .detach()
                .cpu()[loc]
            )

    # save activations
    print(f"Saving activations...")
    output_dir = PROJECT_ROOT / flags.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    selected_activations = {
        f'{name}_input_ids': torch.stack(activations[f'{name}_input_ids']),
        f'{name}_residual_q1': torch.stack(activations[f'{name}_residual_q1']),
        f'{name}_residual_q2': torch.stack(activations[f'{name}_residual_q2']),
        f'{name}_residual_q3': torch.stack(activations[f'{name}_residual_q3']),
    }
    with open(output_dir / f'activations_{flags.lm_name}_{flags.lm_size}_{name}.pkl', 'wb') as fpi:
        pickle.dump(selected_activations, fpi, protocol=pickle.HIGHEST_PROTOCOL)

    print("Done!")


if __name__ == '__main__':
    flags = get_flags()
    main(flags)
