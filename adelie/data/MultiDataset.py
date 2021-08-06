import torch

from torch.utils.data import Dataset
from typing import Union, Optional, Iterable, Dict, Tuple


class MultiDataset(Dataset):

    def __init__(self, datasets=()):
        self.datasets = []
        for i in datasets:
            self._add_dataset(i)

    def _add_dataset(self, dataset):

        # Check that length of datasets match
        assert all([len(dataset) == len(i) for i in self.datasets])

        # Append dataset
        self.datasets.append(dataset)

    def __len__(self):
        return len(self.datasets[0])

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        return [dataset[idx] for dataset in self.datasets]
