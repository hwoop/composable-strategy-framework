import torch
from pytorch_lightning import LightningDataModule
from pytorch_lightning.cli import instantiate_class
from torch.utils.data import random_split, DataLoader
from omegaconf import DictConfig, OmegaConf
from typing import DefaultDict, Optional
from hydra.utils import instantiate

class UniversalDataModule(LightningDataModule):
    def __init__(
        self,
        train_dataset_config: DictConfig,
        test_dataset_config: DictConfig,
        val_dataset_config: Optional[DictConfig] = None,
        extra_data_config: Optional[DictConfig] = None,
        val_split_ratio: float = 0.2,
        batch_size: int = 32,
        train_workers: int = 0,
        other_workers: int = 0,
        seed: int = 42,
        collate_fn_config: Optional[DictConfig] = None,
        **kwargs
    ):
        super().__init__()
        self.save_hyperparameters()
        self.collate_fn = None
        self.extra_data = {}


    def prepare_data(self):
        if not self.hparams.extra_data_config:
            return
        
        for name, config in self.hparams.extra_data_config.items():
            self.extra_data[name] = instantiate_class(args=(), init=config)
            print(f"Loaded extra data: '{name}'")


    def setup(self, stage: str):
        # pl_module.cli.instantiate_class는 object를 parameter로 넣을 수 없음..
        if self.hparams.collate_fn_config and self.collate_fn is None:
            self.collate_fn = instantiate(
                self.hparams.collate_fn_config,
                extra_data=self.extra_data 
            )            
        
        if stage in ('fit', 'validate'):
            if self.hparams.val_dataset_config:
                self.train_dataset = instantiate_class(args=(), init=self.hparams.train_dataset_config)
                self.val_dataset = instantiate_class(args=(), init=self.hparams.val_dataset_config)
            else:
                full_train_dataset = instantiate_class(args=(), init=self.hparams.train_dataset_config)
                total_size = len(full_train_dataset)
                val_size = int(total_size * self.hparams.val_split_ratio)
                train_size = total_size - val_size
                
                generator = torch.Generator().manual_seed(self.hparams.seed)
                self.train_dataset, self.val_dataset = random_split(
                    full_train_dataset, [train_size, val_size], generator=generator
                )

        if stage in ('test', 'predict'):
            self.test_dataset = instantiate_class(args=(), init=self.hparams.test_dataset_config)
            

    def _create_dataloader(self, dataset, shuffle=False, num_workers=None):
        if num_workers is None:
            num_workers = self.hparams.other_workers
            
        return DataLoader(
            dataset=dataset,
            batch_size=self.hparams.batch_size,
            num_workers=num_workers,
            shuffle=shuffle,
            collate_fn=self.collate_fn
        )

    def train_dataloader(self):
        return self._create_dataloader(self.train_dataset, shuffle=True, num_workers=self.hparams.train_workers)

    def val_dataloader(self):
        return self._create_dataloader(self.val_dataset)

    def test_dataloader(self):
        return self._create_dataloader(self.test_dataset)

    def predict_dataloader(self):
        return self.test_dataloader()