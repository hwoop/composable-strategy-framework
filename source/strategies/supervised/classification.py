import torch
import torch.nn as nn
from typing import Dict, Optional, Tuple, Any
from pytorch_lightning import LightningModule
from torchmetrics import MetricCollection

from strategies.base import BaseStrategy
from torchmetrics.classification import (
    MulticlassAccuracy, MulticlassF1Score,
    MulticlassPrecision, MulticlassRecall,
    BinaryAccuracy, BinaryAUROC, BinaryAveragePrecision, BinaryF1Score,
    BinaryPrecision, BinaryRecall
)


class ClassificationStrategy(BaseStrategy):
    
    def __init__(
        self,
        weight: float = 1.0,
        num_classes: int = 2
    ):
        super().__init__()
        self.weight = weight
        self.num_classes = num_classes
        self.criterion = nn.CrossEntropyLoss()
        
        metrics = MetricCollection({
            'acc': MulticlassAccuracy(num_classes=num_classes),
            'precision': MulticlassPrecision(num_classes=num_classes, average='macro'),
            'recall': MulticlassRecall(num_classes=num_classes, average='macro'),
            'f1': MulticlassF1Score(num_classes=num_classes, average='macro'),
        })
        
        self.train_metrics = metrics.clone(prefix='train/clf_')
        self.val_metrics = metrics.clone(prefix='val/clf_')
        self.test_metrics = metrics.clone(prefix='test/clf_')


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
        
        loss = self.criterion(logits, labels.long())
        preds = torch.argmax(logits, dim=1)
        
        metrics = self._get_metrics(pl_module)
        metrics.update(preds, labels)
        
        stage = self.get_stage(pl_module)
        pl_module.log(f'{stage}/clf_loss', loss, on_step=(stage == 'train'), on_epoch=(stage != 'train'))
        pl_module.log_dict(metrics, on_step=False, on_epoch=True)

        return loss * self.weight
    

class BinaryClassificationStrategy(ClassificationStrategy):
    
    def __init__(
        self,
        weight: float = 1.0,
        threshold: float = 0.5
    ):
        super(ClassificationStrategy, self).__init__()
        self.weight = weight
        self.threshold = threshold
        self.criterion = nn.BCEWithLogitsLoss()

        metrics = MetricCollection({
            'acc': BinaryAccuracy(threshold=self.threshold),
            'precision': BinaryPrecision(threshold=self.threshold),
            'recall': BinaryRecall(threshold=self.threshold),
            'f1': BinaryF1Score(threshold=self.threshold),
            'auroc': BinaryAUROC(),
            'auprc': BinaryAveragePrecision()
        })        
        
        self.train_metrics = metrics.clone(prefix='train/clf_')
        self.val_metrics = metrics.clone(prefix='val/clf_')
        self.test_metrics = metrics.clone(prefix='test/clf_')


    def calculate(
        self, 
        logits: torch.Tensor,
        labels: torch.Tensor,
        pl_module: LightningModule
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        logits = logits.squeeze(-1) if logits.ndim > 1 else logits
        labels = labels.squeeze(-1) if labels.ndim > 1 else labels
        
        loss = self.criterion(logits, labels.float())
        probs = torch.sigmoid(logits)
        
        metrics = self._get_metrics(pl_module)
        metrics.update(probs, labels.int())

        stage = self.get_stage(pl_module)
        pl_module.log(f'{stage}/clf_loss', loss, on_step=(stage == 'train'), on_epoch=(stage != 'train'))
        pl_module.log_dict(metrics, on_step=False, on_epoch=True)

        return loss * self.weight