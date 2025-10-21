import torch
from typing import Optional
from pytorch_lightning import LightningModule
from torchmetrics import MeanMetric

from strategies.base import BaseStrategy


class DifferenceStrategy(BaseStrategy):
    def __init__(self, weight: float = 1.0):
        super().__init__()
        self.weight = weight


    def _diff_loss(self, input1, input2):
        batch_size = input1.size(0)
        input1 = input1.view(batch_size, -1)
        input2 = input2.view(batch_size, -1)

        input1_mean = torch.mean(input1, dim=0, keepdims=True)
        input2_mean = torch.mean(input2, dim=0, keepdims=True)
        input1 = input1 - input1_mean
        input2 = input2 - input2_mean

        input1_l2_norm = torch.norm(input1, p=2, dim=1, keepdim=True).detach()
        input1_l2 = input1.div(input1_l2_norm.expand_as(input1) + 1e-6)

        input2_l2_norm = torch.norm(input2, p=2, dim=1, keepdim=True).detach()
        input2_l2 = input2.div(input2_l2_norm.expand_as(input2) + 1e-6)

        diff_loss = torch.mean((input1_l2.t().mm(input2_l2)).pow(2))
        return diff_loss

    def calculate(
        self,
        pl_module: LightningModule,
        shared_t: torch.Tensor,
        shared_v: torch.Tensor,
        shared_a: torch.Tensor,
        private_t: torch.Tensor,
        private_v: torch.Tensor,
        private_a: torch.Tensor
    ) -> Optional[torch.Tensor]:
        loss = self._diff_loss(private_t, shared_t)
        loss += self._diff_loss(private_v, shared_v)
        loss += self._diff_loss(private_a, shared_a)
        loss += self._diff_loss(private_a, private_t)
        loss += self._diff_loss(private_a, private_v)
        loss += self._diff_loss(private_t, private_v)
        
        stage = self.get_stage(pl_module)
        pl_module.log(f'{stage}/diff_loss', loss, on_step=(stage == 'train'), on_epoch=(stage != 'train'))
        
        if stage == 'train':
            return loss * self.weight
        else:
            return None
