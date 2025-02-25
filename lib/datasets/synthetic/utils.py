import json

import datasets

from pathlib import Path

from .seed import SYNTH_SEED


__all__ = ['get_synthetic_dataset']


PROJECT_ROOT = Path(__file__).parent.parent.parent.parent


def get_synthetic_dataset():
    """
    Get synthetic dataset, while caching the dataset if it is not already cached.
    """

    # check if the dataset is already cached
    cache_path = PROJECT_ROOT / 'cache' / 'synthetic_data.json'
    if cache_path.exists():
        with open(cache_path, 'r', encoding='utf-8') as fpi:
            processed_data = json.load(fpi)['processed_data']
        dataset = datasets.Dataset.from_list(processed_data)
        return dataset

    # create dataset from the data seed dictionary
    processed_data = []
    for big_category, content in SYNTH_SEED.items():
        templates = content['templates']
        categories = content['categories']

        # create all combinations of templates and categories
        for template in templates:
            for small_category in categories:
                ko_text = template.replace('{symbol}', small_category)
                processed_data.append({
                    'big_category': big_category,
                    'small_category': small_category,
                    'ko_text': ko_text,
                })

    # cache the dataset
    with open(cache_path, 'w', encoding='utf-8') as fpo:
        json.dump({'processed_data': processed_data}, fpo, indent=4, ensure_ascii=False)

    dataset = datasets.Dataset.from_list(processed_data)
    return dataset
