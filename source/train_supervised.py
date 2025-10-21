import time
from pytorch_lightning import LightningDataModule, LightningModule
from pytorch_lightning.trainer import Trainer
from pytorch_lightning.cli import instantiate_class
import pytorch_lightning as pl

import hydra
from hydra.utils import instantiate

from omegaconf import DictConfig, OmegaConf
import wandb


@hydra.main(config_path="../configs", config_name="train_supervised", version_base="1.3")
def main(cfg: DictConfig):      
    seed = cfg.seed
    if seed is None:
        seed = int(time.time())
        OmegaConf.set_struct(cfg, False)
        cfg.datamodule.init_args.seed = seed
        OmegaConf.set_struct(cfg, True)
          
    pl.seed_everything(cfg.seed)
    print(f"Global seed set to {seed}")
    
    datamodule: LightningDataModule = instantiate_class(args=(), init=cfg.datamodule)
    pl_module: LightningModule = instantiate_class(args=(), init=cfg.pl_module)
    trainer: Trainer = instantiate(cfg.trainer)
    
    trainer.fit(model=pl_module, datamodule=datamodule)
    trainer.test(model=pl_module, datamodule=datamodule, ckpt_path='best')

    
    wandb.finish()

if __name__ == "__main__":
    main()