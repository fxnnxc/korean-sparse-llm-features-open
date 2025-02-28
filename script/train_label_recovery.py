import os
import sys
import pickle
import argparse

import torch
import torch.nn as nn
import numpy as np

from pathlib import Path

from tqdm import tqdm

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.append(str(PROJECT_ROOT))
sys.path.append(str(PROJECT_ROOT / 'lib'))

from lib.models.standard_sae import AutoEncoder  # noqa
from lib.models.gated_sae import GatedAutoEncoder  # noqa
from lib.datasets.synthetic import get_synthetic_dataset  # noqa


class ThreeLayerNet(nn.Module):

    def __init__(self, input_size, depth, num_classes):
        super().__init__()
        self.depth = depth
        self.num_classes = num_classes
        self.hidden_size1 = 64
        self.hidden_size2 = 32
        if depth == 2:
            self.layers = nn.Sequential(
                nn.Linear(input_size, self.hidden_size1),
                nn.ReLU(),
                nn.Linear(self.hidden_size1, num_classes),
            )
        elif depth == 3:
            self.layers = nn.Sequential(
                nn.Linear(input_size, self.hidden_size1),
                nn.ReLU(),
                nn.Linear(self.hidden_size1, self.hidden_size2),
                nn.ReLU(),
                nn.Linear(self.hidden_size2, num_classes),
            )
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.layers(x)
        return x


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--sae_version', type=str, default='gated')
    parser.add_argument('--q', type=int, default=2)
    parser.add_argument('--output_dir', type=str, default=PROJECT_ROOT / 'outputs')
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()
    return args


# TODO: count? isn't this float value?
def get_feature_counts(sae, num_dictionary, activations):
    n_samples = len(activations)
    feature_counts = torch.zeros(n_samples, num_dictionary)
    for s in tqdm(range(n_samples), "Counting features"):
        sample = activations[s].to('cuda')
        _, features = sae(sample, output_features=True)
        feature_counts[s] = torch.sum(features, dim=0).detach().cpu()
    return feature_counts


def main(args):

    # init dataset
    synthetic_dataset = get_synthetic_dataset()

    # get indices of big and small categories
    big_category_indices = {}
    small_category_indices = {}
    for i, row in enumerate(synthetic_dataset):
        big_category = row['big_category']
        small_category = row['small_category']
        if big_category not in big_category_indices:
            big_category_indices[big_category] = []
        if small_category not in small_category_indices:
            small_category_indices[small_category] = []
        big_category_indices[big_category].append(i)
        small_category_indices[small_category].append(i)

    # load activations
    activation_dataset = 'synthetic'
    activations_path = args.output_dir / f'activations_exaone-8b_{activation_dataset}.pkl'
    with open(activations_path, 'rb') as fpi:
        activations = pickle.load(fpi)
    activations = activations[f'{activation_dataset}_residual_q{args.q}']

    # load AE
    device = torch.device(args.device)
    sae_dataset = 'keat-ko'
    sae_path = args.output_dir / f'sae-{args.sae_version}_exaone-8b_{sae_dataset}_q{args.q}'
    if args.sae_version == 'standard':
        ae = AutoEncoder.from_pretrained(f'{sae_path}/model.pth')
    elif args.sae_version == 'gated':
        ae = GatedAutoEncoder.from_pretrained(f'{sae_path}/model.pth')
    ae.to(device)

    # get feature counts
    num_dict = ae.dict_size
    feature_counts = get_feature_counts(ae, num_dict, activations)
    del ae
    torch.cuda.empty_cache()

    # for all cases
    for label_type, category_indices in (
        ('small', small_category_indices),
        ('big', big_category_indices),
    ):
        for depth in (2, 3):
            for top_k in (1, 2, 5, 10, 20, 50):
                print(f"Training with sae-{args.sae_version}_q{args.q} on label: {label_type}, top: {top_k}, depth: {depth}")

                # resolve paths
                save_dir = args.output_dir / f'synthetic_recovery/sae-{args.sae_version}_q{args.q}/{label_type}/top{top_k}/depth{depth}'
                os.makedirs(save_dir, exist_ok=True)

                # get top-k feature indices
                all_indices = []
                for indices in category_indices.values():

                    # select samples
                    selected_feature_counts = feature_counts[indices]
                    mean = torch.mean(selected_feature_counts, dim=0)

                    # select top k features
                    top_k_results = torch.topk(mean, k=top_k, dim=0)
                    top_k_mean_indices = top_k_results.indices

                    all_indices.extend(top_k_mean_indices.tolist())

                # prepare data
                data = np.zeros((len(synthetic_dataset), len(all_indices)))
                for index in range(len(synthetic_dataset)):
                    data[index] = np.asarray(feature_counts[index, all_indices])

                # prepare labels
                small_categories = list(small_category_indices.keys())
                label2lindex = {label: lindex for lindex, label in enumerate(small_categories)}
                lindices = [label2lindex[data['small_category']] for data in synthetic_dataset]
                lindices = np.array(lindices)
                print(f"num of samples: {data.shape[0]}")
                print(f"num of features: {data.shape[1]}")
                print(f"num of classes: {len(small_categories)}")

                # Convert data to PyTorch tensors
                X = torch.FloatTensor(data)
                y = torch.LongTensor(lindices)

                # initialize model, loss, optimizer
                input_size = X.shape[1]
                num_classes = len(small_categories)
                model = ThreeLayerNet(input_size, depth, num_classes)
                criterion = nn.CrossEntropyLoss()
                optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

                # training loop
                epochs = 10000
                losses = []
                eval_steps = 500
                accuracies = []
                pbar = tqdm(range(1, epochs + 1))
                for epoch in pbar:
                    pbar.set_description(f"[E:{epoch:5d}]")

                    # forward pass
                    outputs = model(X)
                    loss = criterion(outputs, y)

                    # backward pass and optimize
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    # save losses
                    losses.append(loss.item())

                    # evaluate
                    if not epoch % eval_steps:
                        with torch.no_grad():
                            logits = model(X)
                            y_hat = torch.argmax(logits, dim=1)
                            accuracy = (y_hat == y).sum().item() / y.size(0)
                        accuracies.append(accuracy)
                        pbar.set_postfix_str(f"A:{accuracy:.3f}, L:{loss.item():.3f}")

                pickle.dump(accuracies, open(save_dir / 'accuracies.pkl', 'wb'))
                pickle.dump(losses, open(save_dir / 'losses.pkl', 'wb'))


if __name__ == '__main__':
    args = get_args()
    main(args)
