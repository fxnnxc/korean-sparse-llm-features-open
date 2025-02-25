import json

import numpy as np
import seaborn as sns
import datasets
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

from typing import Dict
from datasets import Dataset
from pathlib import Path


def load_keat_small(data_dir: Path):
    """
    Load KEAT small dataset from the given directory.

    Args:
        data_dir: Path to the directory containing development.json and evaluation.json
    """
    all_samples = {
        "train": {"en_text": [], "ko_text": [], "categories": []},
        "test": {"en_text": [], "ko_text": [], "categories": []},
    }

    # Ensure data_dir is a Path object
    data_dir = Path(data_dir)

    for name, path in zip(["train", "test"], ["development.json", "evaluation.json"]):
        json_path = data_dir / path
        if not json_path.exists():
            raise FileNotFoundError(f"Dataset file not found: {json_path}")

        with open(json_path, 'r', encoding='utf-8') as f:
            json_data = json.load(f)

        for item in json_data:
            samples = item["text"]
            for sample in samples:
                all_samples[name]["en_text"].append(sample["en_text"])
                all_samples[name]["ko_text"].append(sample["ko_text"])
                all_samples[name]["categories"].append(item["category"])

    # Convert to datasets
    for k, v in all_samples.items():
        all_samples[k] = datasets.Dataset.from_dict(v)
    dataset = datasets.DatasetDict(all_samples)

    return dataset, json_data


def visualize_text_lengths(split2textdataset: Dict[str, Dataset], figsize=(12, 4), bins=30):
    """
    Creates a histogram with smoothed KDE line of text lengths from a dictionary of datasets.

    Args:
        split2textdataset (Dict[str, Dataset]): Dictionary of splits containing text datasets
            e.g., {'train': dataset['text'], 'test': dataset['text']}
        figsize (tuple): Figure size (width, height)
        bins (int): Number of bins for histogram

    Returns:
        matplotlib.figure.Figure: The generated figure
    """
    # Calculate text lengths for each split
    lengths_by_split = {}
    for split, texts in split2textdataset.items():
        # Calculate character lengths
        lengths_by_split[split] = [len(str(text)) for text in texts]

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    # Define colors with train being more vivid red
    colors = {
        'train': '#d62728',  # Vivid red
        'test': '#7fb1d6'    # Light blue
    }
    alphas = {
        'train': 0.4,
        'test': 0.2
    }

    # Plot histograms and smoothed lines
    for split, lengths in lengths_by_split.items():
        # Calculate histogram data for scaling
        counts, bin_edges = np.histogram(lengths, bins=bins)
        max_count = np.max(counts)

        # Plot histogram
        sns.histplot(
            data=lengths,
            label=split,
            alpha=alphas[split],
            stat='count',
            color=colors[split],
            bins=bins,
        )

        # Create smoothed line using KDE
        x_grid = np.linspace(min(lengths), max(lengths), 500)  # More points for smoother curve

        # Use even smaller bandwidth for more detail
        bandwidth = 0.01 * np.std(lengths) * (len(lengths) ** (-1/5))  # Reduced from 0.15 to 0.02
        kde = gaussian_kde(lengths, bw_method=bandwidth)
        kde_values = kde(x_grid)

        # Scale KDE to match histogram height
        scaling_factor = max_count / np.max(kde_values)
        kde_values_scaled = kde_values * scaling_factor

        # Plot smoothed line
        plt.plot(
            x_grid, kde_values_scaled,
            color=colors[split],
            linewidth=2.5 if split == 'train' else 1.5,
            label=f'{split} (smooth)',
            alpha=0.8
        )

        # Add mean line
        mean_length = np.mean(lengths)
        ax.axvline(
            mean_length,
            color=colors[split],
            linestyle='--',
            alpha=0.8,
            linewidth=2 if split == 'train' else 1,
            label=f'{split} mean: {mean_length:.1f}'
        )

    ax.set_title('Distribution of Text Lengths')
    ax.set_xlabel('Text Length (characters)')
    ax.set_ylabel('Frequency (count)')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

    # Add statistics annotation
    stats_text = []
    for split, lengths in lengths_by_split.items():
        percentiles = np.percentile(lengths, [25, 50, 75])
        stats = (
            f'{split}:\n'
            f'  Count: {len(lengths)}\n'
            f'  Mean: {np.mean(lengths):.2f}\n'
            f'  25th %ile: {percentiles[0]:.2f}\n'
            f'  50th %ile: {percentiles[1]:.2f}\n'
            f'  75th %ile: {percentiles[2]:.2f}\n'
            f'  Min: {min(lengths)}\n'
            f'  Max: {max(lengths)}'
        )
        stats_text.append(stats)

    ax.text(
        0.95, 0.95,
        '\n\n'.join(stats_text),
        transform=ax.transAxes,
        verticalalignment='top',
        horizontalalignment='right',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8)
    )

    plt.tight_layout()
    return fig
