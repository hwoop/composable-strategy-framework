import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
from typing import Optional, Any
from pytorch_lightning import LightningModule
from torchmetrics import MetricCollection

from strategies.base import BaseStrategy
from utils.mertics.cosine_simularity import MeanCosineSimilarity


class BYOLStrategy(BaseStrategy):
    def __init__(
        self, 
        lambda_weight: float = 0.1, 
        momentum: float = 0.996
    ):
        super().__init__()
        self.lambda_weight = lambda_weight
        self.momentum = momentum

        self.online_projector_a: nn.Module = None
        self.online_projector_b: nn.Module = None
        self.target_projector_a: nn.Module = None
        self.target_projector_b: nn.Module = None
        self.predictor_a: nn.Module = None
        self.predictor_b: nn.Module = None

        metrics = MetricCollection({
            'cos-sim': MeanCosineSimilarity()
        })
        self.train_metrics = metrics.clone(prefix='train/byol_')
        self.val_metrics = metrics.clone(prefix='val/byol_')
        self.test_metrics = metrics.clone(prefix='test/byol_')


    def _get_metrics(self, pl_module: LightningModule) -> MetricCollection:
        if pl_module.trainer.training:
            return self.train_metrics
        if pl_module.trainer.validating:
            return self.val_metrics
        return self.test_metrics


    def _build_predictor(self, projector: nn.Module) -> nn.Sequential:
        last_layer = list(projector.modules())[-1]
        if not isinstance(last_layer, nn.Linear):
            raise TypeError("The last layer of the projector must be nn.Linear.")
        
        projection_dim = last_layer.out_features
        return nn.Sequential(
            nn.Linear(projection_dim, 4096),
            nn.LayerNorm(4096),
            nn.ReLU(),
            nn.Linear(4096, projection_dim)
        )

    
    def setup(self, online_projector_a: nn.Module, online_projector_b: nn.Module, device: torch.device, **kwargs):
        if self.online_projector_a is not None: return

        self.online_projector_a = online_projector_a
        self.online_projector_b = online_projector_b

        self.target_projector_a = copy.deepcopy(self.online_projector_a).requires_grad_(False).to(device)
        self.target_projector_b = copy.deepcopy(self.online_projector_b).requires_grad_(False).to(device)
        
        self.predictor_a = self._build_predictor(self.online_projector_a).to(device)
        self.predictor_b = self._build_predictor(self.online_projector_b).to(device)
                

    def calculate(
        self,
        pl_module: LightningModule,
        representation_a: torch.Tensor,
        representation_b: torch.Tensor,
        **kwargs: Any,
    ) -> Optional[torch.Tensor]:
        if self.online_projector_a is None:
             raise RuntimeError("BYOLStrategy has not been set up. Call `setup()` before `calculate`.")

        # Online network
        z_a = F.normalize(self.online_projector_a(representation_a), dim=-1)
        z_b = F.normalize(self.online_projector_b(representation_b), dim=-1)
        q_a = F.normalize(self.predictor_a(z_a), dim=-1)
        q_b = F.normalize(self.predictor_b(z_b), dim=-1)

        # Target network
        with torch.no_grad():
            t_a = F.normalize(self.target_projector_a(representation_a), dim=-1)
            t_b = F.normalize(self.target_projector_b(representation_b), dim=-1)

        loss_1 = 2 - 2 * (q_a * t_b.detach()).sum(dim=-1).mean()
        loss_2 = 2 - 2 * (q_b * t_a.detach()).sum(dim=-1).mean()
        loss = (loss_1 + loss_2) / 2

        metrics = self._get_metrics(pl_module)
        metrics.update(z_a, z_b)

        stage = self.get_stage(pl_module)
        pl_module.log(f'{stage}/byol_loss', loss, on_step=(stage == 'train'), on_epoch=(stage != 'train'))
        pl_module.log_dict(metrics, on_step=False, on_epoch=True)
        
        if pl_module.training:
            return loss * self.lambda_weight
        else:
            return None


    def on_after_optimizer_step(self, pl_module: LightningModule):
        if self.target_projector_a is None: 
            return

        with torch.no_grad():
            for t_p, o_p in zip(self.target_projector_a.parameters(), self.online_projector_a.parameters()):
                t_p.data = self.momentum * t_p.data + (1 - self.momentum) * o_p.data
            
            for t_p, o_p in zip(self.target_projector_b.parameters(), self.online_projector_b.parameters()):
                t_p.data = self.momentum * t_p.data + (1 - self.momentum) * o_p.data
             
                        