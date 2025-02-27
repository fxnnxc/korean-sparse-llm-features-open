import sys
import pickle
import argparse

import torch

from pathlib import Path

from tqdm import tqdm
from omegaconf import OmegaConf

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.append(str(PROJECT_ROOT))
sys.path.append(str(PROJECT_ROOT / 'lib'))

from lib.utils.data import get_dataloder_from_dataset
from lib.utils.fetch import MultipleFetch
from lib.utils.model import get_exaone
from lib.utils.tokenize import get_tokenized_dataset
from lib.datasets.keat import get_keat_dataset


def get_flags():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='keat')
    parser.add_argument('--max_length', type=int, default=128)
    parser.add_argument('--lm_name', type=str, default='exaone')
    parser.add_argument('--lm_size', type=str, default='8b')
    parser.add_argument('--lm_cache_dir', type=str, default=PROJECT_ROOT / 'cache')
    parser.add_argument('--device_map', type=str, default='auto')
    parser.add_argument('--output_dir', type=str, default=PROJECT_ROOT / 'outputs')
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


def main(flags):

    # init
    llm, tokenizer = get_exaone(  # TODO: what about other LLMs?
        lm_size=flags.lm_size,
        lm_cache_dir=flags.lm_cache_dir,
        device_map=flags.device_map,
    )

    # activations to be recorded
    activations = {
        'ko_input_ids': [],
        'en_input_ids': [],
        'ko_residual_q0': [],
        'en_residual_q0': [],
        'ko_residual_q1': [],
        'en_residual_q1': [],
        'ko_residual_q2': [],
        'en_residual_q2': [],
        'ko_residual_q3': [],
        'en_residual_q3': [],
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
    dataset, _ = get_keat_dataset()
    dataset_target = dataset['train']
    dataset_en = get_tokenized_dataset(
        dataset_target,
        tokenizer,
        target_name='en_text',
        output_name='en_input_ids',
        batch_size=flags.batch_size,
        truncation=True,
        max_length=flags.max_length,
        padding='max_length',
    )
    dataset_en = dataset_en.rename_columns({'attention_mask': 'en_attention_mask'})
    dataset_ko = get_tokenized_dataset(
        dataset_target,
        tokenizer,
        target_name='ko_text',
        output_name='ko_input_ids',
        batch_size=flags.batch_size,
        truncation=True,
        max_length=flags.max_length,
        padding='max_length',
    )
    dataset_ko = dataset_ko.rename_columns({'attention_mask': 'ko_attention_mask'})
    dataset = dataset_en.add_column('ko_input_ids', dataset_ko['ko_input_ids'])
    dataset = dataset.add_column('ko_attention_mask', dataset_ko['ko_attention_mask'])
    dataloader = get_dataloder_from_dataset(dataset, batch_size=flags.batch_size)

    # fetch and save activations
    for lang in ('en', 'ko'):
        name = f'keat-{lang}'
        print(f"Processing {name} dataset...")
        pbar = tqdm(dataloader)
        for batch in pbar:

            # move to device
            b_tokens = batch[f'{name}_input_ids'].to(llm.device)
            b_mask = batch[f'{name}_attention_mask'].to(llm.device)

            # clear fetched_values (for each batch), but maybe we can do better
            for k in dict_format.keys():
                dict_format[k]['module'].fetched_values = []

            # run forward pass
            _ = llm.forward(input_ids=b_tokens, attention_mask=b_mask)

            # gather activations
            for b in range(b_tokens.shape[0]):
                tokens = b_tokens[b].detach().cpu()
                activations[f'{name}_input_ids'].append(tokens)
                activations[f'{name}_residual_q0'].append(
                    dict_format['residual_q0']['module']
                    .fetched_values[0][b]  # always 0 since we're clearing fetched_values everytime (maybe better design might exist)
                    .detach()
                    .cpu()
                )
                activations[f'{name}_residual_q1'].append(
                    dict_format['residual_q1']['module']
                    .fetched_values[0][b]
                    .detach()
                    .cpu()
                )
                activations[f'{name}_residual_q2'].append(
                    dict_format['residual_q2']['module']
                    .fetched_values[0][b]
                    .detach()
                    .cpu()
                )
                activations[f'{name}_residual_q3'].append(
                    dict_format['residual_q3']['module']
                    .fetched_values[0][b]
                    .detach()
                    .cpu()
                )

    # save activations
    print(f"Saving activations...")
    output_dir = PROJECT_ROOT / flags.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    for lang in ('en', 'ko'):
        name = f'keat-{lang}'
        selected_activations = {
            f'{name}_input_ids': torch.stack(activations[f'{name}_input_ids']),
            f'{name}_residual_q0': torch.stack(activations[f'{name}_residual_q0']),
            f'{name}_residual_q1': torch.stack(activations[f'{name}_residual_q1']),
            f'{name}_residual_q2': torch.stack(activations[f'{name}_residual_q2']),
            f'{name}_residual_q3': torch.stack(activations[f'{name}_residual_q3']),
        }
        with open(output_dir / f'activations_{flags.lm_name}-{flags.lm_size}_{name}.pkl', 'wb') as fpi:
            pickle.dump(selected_activations, fpi, protocol=pickle.HIGHEST_PROTOCOL)

    print("Done!")


if __name__ == '__main__':
    flags = get_flags()
    main(flags)
