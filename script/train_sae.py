import sys
import pickle
import argparse

import torch
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from pathlib import Path

from tqdm import tqdm
from torch.utils.data import (
    Dataset,
    DataLoader,
)

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.append(str(PROJECT_ROOT))
sys.path.append(str(PROJECT_ROOT / 'lib'))

from lib.models.standard_sae import (  # noqa
    StandardTrainer,
    AutoEncoder,
)
from lib.models.gated_sae import (  # noqa
    GatedTrainer,
    GatedAutoEncoder,
)


class ActivationDataset(Dataset):

    def __init__(self, activations):
        activations = activations.view(-1, activations.shape[-1])
        self.activations = activations.float()

    def __len__(self):
        return len(self.activations)

    def __getitem__(self, idx):
        return self.activations[idx]


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='keat-ko')
    parser.add_argument('--sae_version', type=str, default='standard')
    parser.add_argument('--lm_name', type=str, default='exaone')
    parser.add_argument('--lm_size', type=str, default='8b')
    parser.add_argument('--layer_quantile', type=str, default='q1')
    parser.add_argument('--output_dir', type=str, default=PROJECT_ROOT / 'outputs')
    parser.add_argument('--debug_mode', action='store_true', default=False)
    args = parser.parse_args()
    return args


def visualize_log(log, figsize=(12, 6)):
    steps = sorted(list(log.keys()))
    names = list(log[steps[0]].keys())
    fig, axes = plt.subplots(2, len(names) // 2, figsize=figsize)
    axes = axes.ravel()
    for i, name in enumerate(names):
        values = [log[step][name] for step in steps]
        sns.lineplot(x=steps, y=values, ax=axes[i])
        axes[i].set_title(name)
        if 'loss' in name:
            try:
                axes[i].set_ylim(0, np.quantile(values, 0.8))
            except TypeError:
                pass
    plt.tight_layout()
    return fig, axes


def main(args):

    # iteration settings
    log_steps = 100
    visualize_steps = 100
    save_steps = 10000
    warmup_steps = 1000
    total_steps = 100000

    # hyperparameters
    resample_steps = 10000
    activation_dim = 4096
    dict_size = 100000
    device = 'cuda'
    lr = 1e-4
    l1_penalty = 5e-2
    batch_size = 32
    # anneal_start = 10
    # anneal_end = args.total_steps
    # n_sparsity_updates = 100
    # sparsity_queue_length = 10
    # p_start = 1
    # p_end = 0.1

    # for debug mode
    if args.debug_mode:
        log_steps = 10
        visualize_steps = 100
        save_steps = 100
        warmup_steps = 100
        total_steps = 1000
        resample_steps = 100
        dict_size = 10000

    # resolve paths
    run_dir = args.output_dir / f'sae-{args.sae_version}_{args.lm_name}-{args.lm_size}_{args.dataset}_{args.layer_quantile}'
    run_dir.mkdir(parents=True, exist_ok=True)

    # load activations
    activation_path = args.output_dir / f'activations_{args.lm_name}-{args.lm_size}_{args.dataset}.pkl'
    with open(activation_path, 'rb') as fpi:
        activations_dict = pickle.load(fpi)

    # get dataset
    activations = activations_dict[f'{args.dataset}_residual_{args.layer_quantile}']
    dataset = ActivationDataset(activations)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # load trainer
    if args.sae_version == 'standard':
        trainer = StandardTrainer(
            activation_dim=activation_dim,
            dict_size=dict_size,
            lr=lr,
            l1_penalty=l1_penalty,
            warmup_steps=warmup_steps,  # lr warmup period at start of training and after each resample
            device=device,
            resample_steps=resample_steps,  # how often to resample neurons
        )
    elif args.sae_version == 'gated':
        trainer = GatedTrainer(
            activation_dim=activation_dim,
            dict_size=dict_size,
            lr=lr,
            warmup_steps=warmup_steps,  # lr warmup period at start of training and after each resample
            device=device,
            resample_steps=resample_steps,  # how often to resample neurons
            total_steps=total_steps,
        )
    else:
        raise ValueError(f"Invalid SAE version: {args.sae_version}")

    # init log
    logs = {}

    # set style
    sns.set_style('whitegrid')

    # training loop
    for step, activation in enumerate(tqdm(dataloader)):
        activation = activation.to(device)
        trainer.update(step, activation)

        # logging
        if not (step + 1) % log_steps:
            with torch.no_grad():
                logs[step] = trainer.loss(activation, step, logging=True)[3]

        # visualization
        if not (step + 1) % visualize_steps:
            visualize_log(logs, figsize=(12, 6))
            plt.savefig(run_dir / 'loss.png')
            plt.close()

        # save model
        if not (step + 1) % save_steps:
            model_path = run_dir / 'model.pth'
            torch.save(trainer.ae.state_dict(), model_path)
            torch.save(trainer.ae.state_dict(), run_dir / f'model_{step}.pth')
            with open(run_dir / 'logs.pkl', 'wb') as fpi:
                pickle.dump(logs, fpi)

            # check load
            if args.sae_version == 'standard':
                _ = AutoEncoder.from_pretrained(model_path)
            elif args.sae_version == 'standard_anneal':
                _ = AutoEncoder.from_pretrained(model_path)
            elif args.sae_version == 'gated':
                _ = GatedAutoEncoder.from_pretrained(model_path)
            elif args.sae_version == 'gated_anneal':
                _ = GatedAutoEncoder.from_pretrained(model_path)
            torch.cuda.empty_cache()

        # TODO: what about total steps? what about multiple epochs?
        # if step >= total_steps - 1:
            # break


if __name__ == '__main__':
    args = get_args()
    main(args)
