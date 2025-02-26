import os
import sys
import argparse
import torch
from omegaconf import OmegaConf
from pathlib import Path

base_path = Path(__file__).absolute().parent.parent
sys.path.append(base_path.__str__())
sys.path.append(f"{base_path.__str__()}/lib/")

from lib.utils.data import get_dataloder_from_dataset
from lib.utils.fetch import MultipleFetch
from lib.utils.load_model import get_exaone
from lib.datasets.synthetic import get_synthetic_dataset


PROJECT_ROOT = Path(__file__).parent.parent


parser = argparse.ArgumentParser()
parser.add_argument("--lm_name", type=str, default="exaone")
parser.add_argument("--lm_size", type=str, default="8b")
parser.add_argument("--lm_cache_dir", type=str, default=PROJECT_ROOT / 'cache')
parser.add_argument("--device_map", type=str, default="auto")
parser.add_argument("--batch_size", type=int, default=4)
parser.add_argument("--split", type=str, default="train")
flags = parser.parse_args()

flags = OmegaConf.create(vars(flags))


llm, tokenizer = get_exaone(
    lm_name=flags.lm_name,
    lm_size=flags.lm_size,
    lm_cache_dir=flags.lm_cache_dir,
    device_map=flags.device_map,
)

activations = {
    "synthetic_input_ids": [],
    "synthetic_residual_q2": [],
    "synthetic_residual_q3": [],
}


def condition_fn(module, input, output):
    return True


def fetch_fn(module, input, output):
    if not hasattr(module, "fetched_values"):
        module.fetched_values = []
    module.fetched_values.append(output[0].detach().cpu())


num_layers = llm.config.num_layers
dict_format = {
    f"residual_q0": {
        "module": llm.transformer.h[0],
        "condition_fn": condition_fn,
        "fetch_fn": fetch_fn,
    },
    f"residual_q2": {
        "module": llm.transformer.h[num_layers // 2],
        "condition_fn": condition_fn,
        "fetch_fn": fetch_fn,
    },
    f"residual_q1": {
        "module": llm.transformer.h[int(num_layers * 0.25)],
        "condition_fn": condition_fn,
        "fetch_fn": fetch_fn,
    },
    f"residual_q3": {
        "module": llm.transformer.h[int(num_layers * 0.75)],
        "condition_fn": condition_fn,
        "fetch_fn": fetch_fn,
    },
}
fetch = MultipleFetch(dict_format)


dataset = get_synthetic_dataset()
window_size = 10
max_length = 128


def estimate_token_location(text, target, tokenizer, window_size=5):
    tokenizer.padding_side = "left"
    text_tokens = tokenizer.encode(
        text, return_tensors="pt", padding="max_length", max_length=128
    )[0]

    loc = None
    for t in range(1, len(text_tokens)):
        decoded = tokenizer.decode(text_tokens[max(0, t - window_size) : t])
        if target in decoded:
            loc = t
            break
    assert loc is not None, f"Target {target} not found in the text: {decoded}"
    return text_tokens, loc


text_tokens_list = []
loc_list = []
for i in range(len(dataset)):
    text = dataset[i]["ko_text"]
    target = dataset[i]["small_category"]
    text_tokens, loc = estimate_token_location(text, target, tokenizer, window_size)
    text_tokens_list.append(text_tokens)
    loc_list.append(loc)

dataloader = get_dataloder_from_dataset(dataset, batch_size=flags.batch_size)
from tqdm import tqdm


name = "synthetic"
pbar = tqdm(range(len(dataset)))
pbar.set_description(f"Processing Keat {flags.split} {name} data")
for batch in pbar:
    # clear fetched_values
    for k in dict_format.keys():
        dict_format[k]["module"].fetched_values = []

    input_ids = text_tokens_list[batch].unsqueeze(0).to(llm.device)
    loc = loc_list[batch]
    outputs = llm.forward(input_ids=input_ids)
    for i in range(input_ids.shape[0]):
        activations[f"{name}_input_ids"].append(input_ids[i].detach().cpu()[loc])
        activations[f"{name}_residual_q2"].append(
            dict_format["residual_q2"]["module"]
            .fetched_values[0][i]
            .detach()
            .cpu()[loc, :]
        )
        activations[f"{name}_residual_q3"].append(
            dict_format["residual_q3"]["module"]
            .fetched_values[0][i]
            .detach()
            .cpu()[loc, :]
        )

import pickle
import os

os.makedirs("outputs", exist_ok=True)

name = "synthetic"
with open(
    f"outputs/activations_{flags.lm_name}_{flags.lm_size}_{flags.split}_{name}.pkl",
    "wb",
) as f:
    selected_activations = {
        f"{name}_input_ids": torch.stack(activations[f"{name}_input_ids"]),
        f"{name}_residual_q2": torch.stack(activations[f"{name}_residual_q2"]),
        f"{name}_residual_q3": torch.stack(activations[f"{name}_residual_q3"]),
    }
    pickle.dump(selected_activations, f, protocol=pickle.HIGHEST_PROTOCOL)
