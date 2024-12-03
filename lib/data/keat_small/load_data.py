import os
import json
import datasets


def load_keat_small(data_cache_dir):
    all_samples = {
        "train": {"en_text": [], "ko_text": [], "categories": []},
        "test": {"en_text": [], "ko_text": [], "categories": []},
    }
    for name, path in zip(["train", "test"], ["development.json", "evaluation.json"]):
        devel_path = os.path.join(data_cache_dir, path)
        json_data = json.load(open(devel_path))
        for i in range(len(json_data)):
            samples = json_data[i]["text"]
            for j in range(len(samples)):
                all_samples[name]["en_text"].append(samples[j]["en_text"])
                all_samples[name]["ko_text"].append(samples[j]["ko_text"])
                all_samples[name]["categories"].append(json_data[i]["category"])
    for k, v in all_samples.items():
        all_samples[k] = datasets.Dataset.from_dict(v)
    dataset = datasets.DatasetDict(all_samples)

    return dataset, json_data
