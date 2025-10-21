from dataclasses import dataclass
from typing import Dict
import torch
from torch.utils.data import Dataset

from dtypes.dataset import TorchDatasetType


class LazyTorchDataset(Dataset):
    def __init__(self, data_path):
        super().__init__()
        self.data_path = data_path
        self.data = None


    def _ensure_loaded(self):
        if self.data is None:
            self.data: TorchDatasetType = torch.load(self.data_path, weights_only=False)


    def __len__(self):
        self._ensure_loaded()
        return len(self.data.samples)
 

    def __getitem__(self, idx) -> Dict[str, torch.Tensor]:
        self._ensure_loaded()
        return self.data.samples[idx]