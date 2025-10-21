from abc import ABC, abstractmethod
from typing import Optional
from pytorch_lightning import LightningModule
import torch
import torch.nn as nn


class BaseStrategy(ABC, nn.Module):
    def __init__(self, **kwargs):
        super().__init__()


    @staticmethod
    def get_stage(pl_module: LightningModule) -> str:
        if pl_module.trainer.training:
            return 'train'
        if pl_module.trainer.validating:
            return 'val'
        if pl_module.trainer.testing:
            return 'test'
        return 'predict'
    

    @abstractmethod
    def calculate(
        self, 
        pl_module: LightningModule,
        **kwargs
    ) -> Optional[torch.Tensor]:
        """
        Args:
            model_outputs (Dict[str, torch.Tensor]): 모델의 forward() 결과물.
            batch (Dict[str, Any]): 데이터로더로부터 받은 배치 데이터.
            pl_module (LightningModule): 메인 LightningModule 인스턴스.

        Returns:
            Tuple[torch.Tensor, Dict[str, Any]]: 
            - 최종 계산된 손실 텐서.
            - 로깅을 위한 스칼라 값들이 담긴 딕셔너리.
        """
        raise NotImplementedError


    def on_after_optimizer_step(self, pl_module: LightningModule):
        pass
    
