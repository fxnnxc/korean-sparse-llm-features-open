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
    parser.add_argument('--sae', type=str, default='gated')
    parser.add_argument('--q', type=int, default=2)
    parser.add_argument('--output_dir', type=str, default=PROJECT_ROOT / 'outputs')
    args = parser.parse_args()
    return args


def get_feature_counts(sae, num_dictionary, activations):
    n_samples = len(activations)
    feature_counts = torch.zeros(n_samples, num_dictionary)
    for s in tqdm(range(n_samples)):
        sample = activations[s].to('cuda')
        _, features = sae(sample, output_features=True)
        feature_counts[s] = torch.sum(features, dim=0).detach().cpu()
    return feature_counts


def main(args):
    activations_path = args.output_dir / 'activations_exaone_8b_synthetic.pkl'
    with open(activations_path, 'rb') as fpi:
        activations = pickle.load(fpi)

    # init dataset
    synthetic_dataset = get_synthetic_dataset()

    # get indices of big and small categories
    big_category_indices = {}
    small_category_indices = {}
    for i, row in enumerate(synthetic_dataset):
        big_category_indices[row['big_category']].append(i)
        small_category_indices[row['small_category']].append(i)

    # =========================================================
    activations_path = "outputs"
    activations_lang = "synthetic"
    lang = "ko"
    activation_name = f"exaone_8b_train_{activations_lang}"
    name = f"exaone_8b_train_{lang}"
    q = args.q
    sae = args.sae
    path = f"{activations_path}/sae/{sae}/{name}_q{q}"

    # =========================================================
    activations = pickle.load(
        open(f"{activations_path}/activations_{activation_name}.pkl", "rb")
    )
    act = activations[f"{activations_lang}_residual_q{q}"]
    if sae == "standard":
        ae = AutoEncoder.from_pretrained(f"{path}/model.pth")
    elif sae == "gated":
        ae = GatedAutoEncoder.from_pretrained(f"{path}/model.pth")

    ae.to("cuda")
    num_dictionary = ae.dict_size
    feature_counts = get_feature_counts(ae, num_dictionary, act)
    del ae
    torch.cuda.empty_cache()
    # =========================================================

    for label_type in ["small", "big"]:
        if label_type == "small":
            category_indices = small_category_indices
        elif label_type == "big":
            category_indices = big_category_indices

        for depth in [2, 3]:
            for top_k in [1, 2, 5, 10, 20, 50]:

                save_path = f"outputs/synthetic/{args.sae}/{args.q}/{label_type}/top{top_k}/depth{depth}"
                os.makedirs(save_path, exist_ok=True)
                print(save_path)

                all_indices = []
                for cat, indices in category_indices.items():
                    # select samples
                    selected_feature_counts = feature_counts[indices]
                    mean = torch.mean(selected_feature_counts, dim=0)

                    # select top k features
                    top_k_results = torch.topk(mean, k=top_k, dim=0)
                    top_k_mean_indices = top_k_results.indices

                    all_indices.extend(top_k_mean_indices.tolist())
                # =========================================================

                sample_data = np.zeros((len(synthetic_dataset), len(all_indices)))
                for i, data in enumerate(synthetic_dataset):
                    sample_data[i] = feature_counts[i, all_indices]

                labels_to_integer = {label: i for i, label in enumerate(small_categories)}
                labels = [labels_to_integer[data["small_category"]] for data in synthetic_dataset]
                labels = np.array(labels)
                print(sample_data.shape, labels.shape)

                # Convert data to PyTorch tensors
                X = torch.FloatTensor(sample_data)
                y = torch.LongTensor(labels)

                # Initialize model, loss, and optimizer
                input_size = X.shape[1]
                num_classes = len(small_categories)

                model = ThreeLayerNet(input_size, depth, num_classes)
                criterion = nn.CrossEntropyLoss()
                optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

                # Training loop
                epochs = 10000
                losses = []
                eval_step = 500
                accuracies = []
                pbar = tqdm(range(epochs))
                for epoch in pbar:
                    # Forward pass
                    outputs = model(X)
                    loss = criterion(outputs, y)

                    # Backward pass and optimize
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    losses.append(loss.item())
                    if epoch % eval_step == 0:
                        # print(f"Epoch {epoch} loss: {loss.item()}")
                        # Calculate accuracy
                        with torch.no_grad():
                            outputs = model(X)
                            _, predicted = torch.max(outputs.data, 1)
                            accuracy = (predicted == y).sum().item() / y.size(0)
                            accuracies.append(accuracy)
                            pbar.set_postfix(accuracy=f"{accuracy:.4f}")
                # =========================================================
                pickle.dump(accuracies, open(f"{save_path}/accuracies.pkl", "wb"))
                pickle.dump(losses, open(f"{save_path}/losses.pkl", "wb"))


if __name__ == '__main__':
    args = get_args()
    main(args)
