from pytorch_lightning import LightningModule
import torch
import torch.nn as nn
from typing import Optional, Dict, Any

from strategies.base import BaseStrategy
from torchmetrics import MeanMetric


class ReconstructionStrategy(BaseStrategy):
    def __init__(self, weight: float = 1.0):
        super().__init__()
        self.weight = weight
        self.criterion = nn.MSELoss()


    def calculate(
        self,
        pl_module: LightningModule,
        recon_t: torch.Tensor,
        orig_t: torch.Tensor,
        recon_v: torch.Tensor,
        orig_v: torch.Tensor,
        recon_a: torch.Tensor,
        orig_a: torch.Tensor
    ) -> Optional[torch.Tensor]:
        loss = self.criterion(recon_t, orig_t)
        loss += self.criterion(recon_v, orig_v)
        loss += self.criterion(recon_a, orig_a)
        loss = loss / 3.0
        
        stage = self.get_stage(pl_module)
        pl_module.log(f'{stage}/recon_loss', loss, on_step=(stage == 'train'), on_epoch=(stage != 'train'))
        
        if stage == 'train':
            return loss * self.weight
        else:
            return None
