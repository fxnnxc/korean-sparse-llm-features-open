import os
import json
import urllib.request
import platform
import subprocess
import zipfile
import shutil

import numpy as np
import seaborn as sns
import datasets
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

from typing import Dict
from pathlib import Path

from datasets import Dataset
from scipy.stats import gaussian_kde


__all__ = [
    'get_keat_dataset',
]


PROJECT_ROOT = Path(__file__).parent.parent.parent.parent


def get_keat_dataset():
    """
    Get KEAT (small) dataset, while downloading and extracting the dataset if it is not already downloaded.
    """

    data = {
        'train': {'en_text': [], 'ko_text': [], 'categories': []},
        'test': {'en_text': [], 'ko_text': [], 'categories': []},
    }

    # download and extract the dataset
    _ = download_and_extract_mrl()
    data_dir = PROJECT_ROOT / 'downloads/MRL-2021/dataset/keat'

    # load the data
    for name, filename in (('train', 'development.json'), ('test', 'evaluation.json')):
        json_path = data_dir / filename
        with open(json_path) as fpi:
            json_data = json.load(fpi)
        for item in json_data:
            samples = item['text']
            for sample in samples:
                data[name]['en_text'].append(sample['en_text'])
                data[name]['ko_text'].append(sample['ko_text'])
                data[name]['categories'].append(item['category'])

    # convert to datasets
    for target, key2rows in data.items():
        data[target] = datasets.Dataset.from_dict(key2rows)
    dataset = datasets.DatasetDict(data)

    return dataset, json_data


def download_and_extract_mrl():
    """
    Downloads and extracts the MRL-2021 repository.
    """

    # set up paths
    download_dir = PROJECT_ROOT / 'downloads'
    download_dir.mkdir(parents=True, exist_ok=True)
    zip_path = download_dir / 'mrl_2021.zip'
    temp_dir = download_dir / 'temp'
    final_dir = download_dir / 'MRL-2021'

    # check if the dataset is already downloaded
    dev_data_exists = False
    try:
        with open(final_dir / 'dataset/keat/development.json') as fpi:
            data = json.load(fpi)
        if len(data) == 700:
            dev_data_exists = True
    except Exception:
        pass
    eval_data_exists = False
    try:
        with open(final_dir / 'dataset/keat/evaluation.json') as fpi:
            data = json.load(fpi)
        if len(data) == 700:
            eval_data_exists = True
    except Exception:
        pass
    if dev_data_exists and eval_data_exists:
        return

    # download the zip file if it doesn't exist
    if not zip_path.exists():
        url = "https://github.com/emorynlp/MRL-2021/archive/refs/heads/master.zip"
        print(f"Downloading MRL-2021 dataset from {url}...")
        urllib.request.urlretrieve(url, zip_path)
        print("Download complete!")

    # create temp directory for extraction
    temp_dir.mkdir(exist_ok=True)

    # extract to temp directory
    print("Extracting files...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(temp_dir)

    # move contents from MRL-2021-master to final location
    master_path = temp_dir / "MRL-2021-master"
    if final_dir.exists():
        shutil.rmtree(final_dir)
    shutil.move(master_path, final_dir)

    # cleanup
    if zip_path.exists():
        os.remove(zip_path)
    if temp_dir.exists():
        shutil.rmtree(temp_dir)
    print("Cleaned up temporary files")

    print(f"MRL-2021 dataset is ready to use at {final_dir}!")


def download_and_install_noto_font(test=True):
    """
    Downloads and installs Noto Sans CJK KR font for matplotlib

    Args:
        test (bool): Whether to save test with image
    """

    # set up paths
    font_url = "https://github.com/googlefonts/noto-cjk/raw/main/Sans/OTF/Korean/NotoSansCJKkr-Regular.otf"
    download_dir = PROJECT_ROOT / 'downloads'
    download_dir.mkdir(parents=True, exist_ok=True)
    font_path = download_dir / 'NotoSansCJKkr-Regular.otf'

    # download font if it doesn't exist
    if not font_path.exists():
        print("Downloading Noto Sans CJK KR font...")
        urllib.request.urlretrieve(font_url, font_path)
        print("Download complete!")

    # get system font directory
    platform_system = platform.system()
    if platform_system == 'Windows':
        font_dir = Path(os.environ['WINDIR']) / 'Fonts'
    elif platform_system == 'Darwin':
        font_dir = Path.home() / 'Library' / 'Fonts'
    else:
        font_dir = Path.home() / '.local' / 'share' / 'fonts'
        font_dir.mkdir(parents=True, exist_ok=True)

    # install font to system directory
    system_font_path = font_dir / font_path.name
    if system_font_path.exists():
        os.remove(system_font_path)
    shutil.copy2(font_path, system_font_path)

    # update font cache on Linux
    if platform_system == 'Linux':
        try:
            subprocess.run(['fc-cache', '-f', '-v'], check=True)
        except subprocess.CalledProcessError:
            print("Warning: Failed to update font cache. Font might not be immediately available.")

    # clear and reload matplotlib font cache
    fm.fontManager.addfont(str(system_font_path))

    # force matplotlib to use the font
    font_names = [f.name for f in fm.fontManager.ttflist]
    print("Available fonts:", [f for f in font_names if 'Noto' in f])

    # try different possible font names
    font_options = ['Noto Sans CJK KR', 'Noto Sans CJK KR Regular', 'NotoSansCJKkr-Regular']
    selected_font = None
    for font in font_options:
        if font in font_names:
            selected_font = font
            break

    if selected_font is None:
        print("Warning: Could not find Noto font in system. Using default font.")
        selected_font = 'DejaVu Sans'

    # configure matplotlib
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = [selected_font, 'DejaVu Sans']

    # test the font
    if test:
        print("Testing font installation...")
        _, ax = plt.subplots(figsize=(10, 2))
        ax.text(0.5, 0.5, '한글 테스트 Text', fontsize=20, ha='center', va='center')
        ax.set_axis_off()


# AI generated code
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
        bandwidth = 0.01 * np.std(lengths) * (len(lengths) ** (-1 / 5))  # Reduced from 0.15 to 0.02
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
