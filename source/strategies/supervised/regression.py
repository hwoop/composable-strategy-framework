import torch
import torch.nn as nn
from typing import Optional
from pytorch_lightning import LightningModule
from torchmetrics import MeanAbsoluteError, MetricCollection, PearsonCorrCoef

from strategies.base import BaseStrategy


class RegressionStrategy(BaseStrategy):
    def __init__(
        self,
        weight: float = 1.0,
    ):
        super().__init__()
        self.weight = weight
        self.criterion = nn.MSELoss()
        
        metrics = MetricCollection({
            'mae': MeanAbsoluteError(),
            'corr': PearsonCorrCoef(),
        })
        
        self.train_metrics = metrics.clone(prefix='train/rgr_')
        self.val_metrics = metrics.clone(prefix='val/rgr_')
        self.test_metrics = metrics.clone(prefix='test/rgr_')


    def _get_metrics(self, pl_module: LightningModule) -> MetricCollection:
        if pl_module.trainer.training:
            return self.train_metrics
        if pl_module.trainer.validating:
            return self.val_metrics
        return self.test_metrics
    
    
    def calculate(
        self, 
        logits: torch.Tensor,
        labels: torch.Tensor,
        pl_module: LightningModule
    ) -> Optional[torch.Tensor]:
        logits = logits.squeeze(-1) if logits.ndim > 1 else logits
        labels = labels.squeeze(-1) if labels.ndim > 1 else labels
        
        loss = self.criterion(logits, labels.float())
        
        metrics = self._get_metrics(pl_module)
        metrics.update(logits, labels)
        
        stage = self.get_stage(pl_module)
        pl_module.log(f'{stage}/rgr_loss', loss, on_step=(stage == 'train'), on_epoch=(stage != 'train'))
        pl_module.log_dict(metrics, on_step=False, on_epoch=True)

        return loss * self.weight
    