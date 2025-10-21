import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple, Any, List
from pytorch_lightning import LightningModule
from torchmetrics import MetricCollection

from strategies.base import BaseStrategy
from utils.mertics.cosine_simularity import MeanCosineSimilarity


class ContrastiveStrategy(BaseStrategy):
    def __init__(
        self,
        lambda_weight: float = 0.1,
        temperature: float = 0.07
    ):
        super().__init__()
        self.lambda_weight = lambda_weight
        self.temperature = temperature
        self.criterion = nn.CrossEntropyLoss()

        metrics = MetricCollection({
            'cos-sim': MeanCosineSimilarity()
        })
        self.train_metrics = metrics.clone(prefix='train/cl_')
        self.val_metrics = metrics.clone(prefix='val/cl_')
        self.test_metrics = metrics.clone(prefix='test/cl_')
        


    def _get_metrics(self, pl_module: LightningModule) -> MetricCollection:
        if pl_module.trainer.training:
            return self.train_metrics
        if pl_module.trainer.validating:
            return self.val_metrics
        return self.test_metrics


    def calculate(
        self,
        pl_module: LightningModule,
        projection_a: torch.Tensor,
        projection_b: torch.Tensor,
        **kwargs: Any,
    ) -> Optional[torch.Tensor]:
        # (B, T, D) -> (B*T, D)
        if projection_a.dim() == 3:
            B, T, D = projection_a.shape
            z_a = projection_a.reshape(B * T, D)
        else: # (B, D)
            z_a = projection_a

        if projection_b.dim() == 3:
            B, T, D = projection_b.shape
            z_b = projection_b.reshape(B * T, D)
        else: # (B, D)
            z_b = projection_b

        z_a = F.normalize(z_a, dim=-1)
        z_b = F.normalize(z_b, dim=-1)

        sim_matrix = z_a @ z_b.T / self.temperature

        labels = torch.arange(sim_matrix.size(0), device=pl_module.device)

        loss_a_b = self.criterion(sim_matrix, labels)
        loss_b_a = self.criterion(sim_matrix.T, labels)
        loss = (loss_a_b + loss_b_a) / 2
        
        metrics = self._get_metrics(pl_module)
        metrics.update(z_a, z_b)
        
        stage = self.get_stage(pl_module)
        pl_module.log(f'{stage}/cl_loss', loss, on_step=(stage == 'train'), on_epoch=(stage != 'train'))
        pl_module.log_dict(metrics, on_step=False, on_epoch=True)
        
        if pl_module.training:
            return loss * self.lambda_weight
        else:
            return None
