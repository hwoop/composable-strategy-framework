import torch
from typing import Dict, Tuple, Any
from pytorch_lightning import LightningModule
from torchmetrics import MeanMetric, MetricCollection

from strategies.base import BaseStrategy


class L2Strategy(BaseStrategy):
    def __init__(
        self,
        lambda_weight: float = 5e-5
    ):
        super().__init__()
        self.lambda_weight = lambda_weight
        

    def get_stage(self, pl_module: LightningModule) -> str:
        if pl_module.trainer.training:
            return 'train'
        if pl_module.trainer.validating:
            return 'val'
        if pl_module.trainer.testing:
            return 'test'
        return 'predict'
    
    
    def calculate(
        self,
        logits: torch.Tensor,
        pl_module: LightningModule
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        loss = sum(
            p.pow(2).sum()
            for name, p in pl_module.model.named_parameters()
            if p.requires_grad and 'bias' not in name and 'norm' not in name.lower()
        )

        stage = self.get_stage(pl_module)
        pl_module.log(f'{stage}/l2_loss', loss, on_step=(stage == 'train'), on_epoch=(stage != 'train'))
        
        if stage == 'train':
            return loss * self.lambda_weight
        
        return None
