import torch
from typing import Optional
from pytorch_lightning import LightningModule

from strategies.base import BaseStrategy


class CMDStrategy(BaseStrategy):
    def __init__(self, weight: float = 1.0, n_moments: int = 5):
        super().__init__()
        self.weight = weight
        self.n_moments = n_moments


    def get_stage(self, pl_module: LightningModule) -> str:
        if pl_module.trainer.training:
            return 'train'
        if pl_module.trainer.validating:
            return 'val'
        if pl_module.trainer.testing:
            return 'test'
        return 'predict'

    def _match_norm(self, x1, x2):
        power = torch.pow(x1 - x2, 2)
        summed = torch.sum(power)
        sqrt = summed ** (0.5)
        return sqrt

    def _scm(self, sx1, sx2, k):
        ss1 = torch.mean(torch.pow(sx1, k), 0)
        ss2 = torch.mean(torch.pow(sx2, k), 0)
        return self._match_norm(ss1, ss2)

    def _cmd_loss(self, x1, x2):
        mx1 = torch.mean(x1, 0)
        mx2 = torch.mean(x2, 0)
        sx1 = x1 - mx1
        sx2 = x2 - mx2
        dm = self._match_norm(mx1, mx2)
        scms = dm
        for i in range(self.n_moments - 1):
            scms += self._scm(sx1, sx2, i + 2)
        return scms

    def calculate(
        self,
        pl_module: LightningModule,
        shared_t: torch.Tensor,
        shared_v: torch.Tensor,
        shared_a: torch.Tensor
    ) -> Optional[torch.Tensor]:
        loss = self._cmd_loss(shared_t, shared_v)
        loss += self._cmd_loss(shared_t, shared_a)
        loss += self._cmd_loss(shared_a, shared_v)
        loss = loss / 3.0
        
        stage = self.get_stage(pl_module)
        pl_module.log(f'{stage}/cmd_loss', loss, on_step=(stage == 'train'), on_epoch=(stage != 'train'))
        
        if stage == 'train':
            return loss * self.weight
        else:
            return None