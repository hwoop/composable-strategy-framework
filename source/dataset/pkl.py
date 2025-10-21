import pickle
from dataclasses import dataclass
from typing import Dict, Any
from torch.utils.data import Dataset

class LazyPklDataset(Dataset):
    def __init__(self, data_path):
        super().__init__()
        self.data_path = data_path
        self.data = None

    def _ensure_loaded(self):
        if self.data is None:
            with open(self.data_path, 'rb') as f:
                self.data = pickle.load(f)

    def __len__(self):
        self._ensure_loaded()
        return len(self.data)

    def __getitem__(self, idx) -> Any:
        self._ensure_loaded()
        return self.data[idx]