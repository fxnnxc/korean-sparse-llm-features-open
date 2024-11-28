
import torch 
from fetch import MultipleFetch
from dataloader import get_dataloder_from_dataset
from tokenize import get_tokenized_dataset
from load_model import get_exaone
from omegaconf import OmegaConf

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--lm_name", type=str, default="exaone")
parser.add_argument("--lm_size", type=str, default="8b")
parser.add_argument("--lm_cache_dir", type=str, default="/data1/bumjin/datahub")
parser.add_argument("--device_map", type=str, default="auto")
parser.add_argument("--batch_size", type=int, default=4)
parser.add_argument("--split", type=str, default='train')
parser.add_argument("--data", type=str, default='keat')
flags = parser.parse_args()

flags = OmegaConf.create(vars(flags))


llm, tokenizer = get_exaone(lm_name=flags.lm_name, 
                                   lm_size=flags.lm_size, 
                                   lm_cache_dir=flags.lm_cache_dir, 
                                   device_map=flags.device_map)

activations ={'ko_input_ids':[], 
              'en_input_ids':[],
              'ko_residual_q0':[],
              'en_residual_q0':[],
              'ko_residual_q1':[],
              'en_residual_q1':[],
              'ko_residual_q2':[],
              'en_residual_q2':[],
              'ko_residual_q3':[],
              'en_residual_q3':[],
              }

# =========================================================
import torch 
def condition_fn(module, input, output):
    return True

def fetch_fn(module, input, output):
    if not hasattr(module, "fetched_values"):
        module.fetched_values = []
    module.fetched_values.append(output[0].detach().cpu())

# print("==========Fetch [Multiple]==========")
num_layers = llm.config.num_layers
dict_format = {
    f"residual_q0": {
        "module": llm.transformer.h[0], 
        "condition_fn": condition_fn,
        "fetch_fn": fetch_fn
    },
    f"residual_q2": {
        "module": llm.transformer.h[num_layers//2], 
        "condition_fn": condition_fn,
        "fetch_fn": fetch_fn
    },
    f"residual_q1": {
        "module": llm.transformer.h[int(num_layers*0.25)], 
        "condition_fn": condition_fn,
        "fetch_fn": fetch_fn
    },
    f"residual_q3": {
        "module": llm.transformer.h[int(num_layers*0.75)], 
        "condition_fn": condition_fn,
        "fetch_fn": fetch_fn
    },
}
fetch = MultipleFetch(dict_format)
# =========================================================

from keat_small.load_data import load_keat_small
path ="/data1/bumjin/nlp_data/datahub/MRL-2021/dataset/keat"
if flags.data == 'keat':
    datadict, json_data = load_keat_small(path)
    dataset_raw = datadict['train']
else:
    raise ValueError(f"Unknown dataset {flags.data}")
max_length = 128
dataset1 = get_tokenized_dataset(dataset_raw, tokenizer, target_name="en_text", output_name="en_input_ids", batch_size=flags.batch_size,
                                 truncation=True, max_length=max_length, padding='max_length')
dataset1 = dataset1.rename_columns({'attention_mask':'en_attention_mask'})
dataset2 = get_tokenized_dataset(dataset_raw, tokenizer, target_name="ko_text", output_name="ko_input_ids", batch_size=flags.batch_size,
                                 truncation=True, max_length=max_length, padding='max_length')
dataset2 = dataset2.rename_columns({'attention_mask':'ko_attention_mask'})
dataset = dataset1.add_column("ko_input_ids", dataset2['ko_input_ids'])
dataset = dataset.add_column("ko_attention_mask", dataset2['ko_attention_mask'])
print(dataset)

dataloader = get_dataloder_from_dataset(dataset, batch_size=flags.batch_size)
import numpy as np
from tqdm import tqdm

for i in range(2):
    name = "en" if i == 0 else "ko"
    pbar = tqdm(dataloader)
    pbar.set_description(f"Processing Keat {flags.split} {name} data")
    for batch in pbar:
        # clear fetched_values
        for k in dict_format.keys():
            dict_format[k]["module"].fetched_values = []
            
        input_ids = batch[f"{name}_input_ids"]
        outputs = llm.forward(input_ids=input_ids.to(llm.device), 
                                attention_mask=batch[f"{name}_attention_mask"].to(llm.device))
        for i in range(input_ids.shape[0]):
            activations[f'{name}_input_ids'].append(input_ids[i].detach().cpu())  
            # activations[f'{name}_residual_q0'].append(dict_format['residual_q0']['module'].fetched_values[0][i].detach().cpu())
            # activations[f'{name}_residual_q1'].append(dict_format['residual_q1']['module'].fetched_values[0][i].detach().cpu())
            activations[f'{name}_residual_q2'].append(dict_format['residual_q2']['module'].fetched_values[0][i].detach().cpu())
            activations[f'{name}_residual_q3'].append(dict_format['residual_q3']['module'].fetched_values[0][i].detach().cpu())
    
import pickle
import os 
os.makedirs("outputs", exist_ok=True)

for i in range(2):
    name = "en" if i == 0 else "ko"
    with open(f'outputs/activations_{flags.lm_name}_{flags.lm_size}_{flags.split}_{name}.pkl', 'wb') as f:
        selected_activations = {
            f"{name}_input_ids": torch.stack(activations[f"{name}_input_ids"]),
            f"{name}_residual_q2": torch.stack(activations[f"{name}_residual_q2"]),
            f"{name}_residual_q3": torch.stack(activations[f"{name}_residual_q3"]),
        }
        pickle.dump(selected_activations, f, protocol=pickle.HIGHEST_PROTOCOL)
