from torch.utils.data import DataLoader
from transformers import default_data_collator


def get_dataloder_from_dataset(dataset, batch_size, shuffle=False):
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=default_data_collator,
        pin_memory=True,
    )
    return dataloader
