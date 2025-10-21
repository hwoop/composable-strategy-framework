from typing import List, Dict, Any, Optional
import torch
import torch.nn as nn
from pytorch_lightning import LightningModule
from pytorch_lightning.cli import instantiate_class
from omegaconf import DictConfig
from pytorch_lightning.loggers import WandbLogger

class ExperimentModule(LightningModule):
    def __init__(
        self,
        model_config: DictConfig,
        optimizer_config: DictConfig,
        strategy_config: DictConfig,
        pretrained_ckpt_path: Optional[str] = None,
        freeze_modules: Optional[list[str]] = None,
        trainable_modules: Optional[List[str]] = None, 
        **kwargs
    ):
        super().__init__()
        self.save_hyperparameters()
        
        self.model: nn.Module = instantiate_class(args=(), init=self.hparams.model_config)
        self.strategies: nn.ModuleDict = nn.ModuleDict({
            name: instantiate_class(args=(), init=cfg['module'])
            for name, cfg in self.hparams.strategy_config.items()
        })

        if pretrained_ckpt_path:
            self._load_pretrained_weights(pretrained_ckpt_path)

        if freeze_modules:
            self._freeze_modules(freeze_modules)


    def setup(self, stage: str):
        self._setup_strategies()


    def configure_optimizers(self):
        # self.named_parameters()를 사용해 model과 strategies의 모든 파라미터를 포함
        param_optimizer_list = [(n, p) for n, p in self.named_parameters() if p.requires_grad]
        
        if not param_optimizer_list:
            raise ValueError("No parameters to optimize. Check model and strategy parameters.")
                    
        no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
        optimizer_config = self.hparams.optimizer_config
        
        optimizer_grouped_parameters = [
            {"params": [p for n, p in param_optimizer_list if not any(nd in n for nd in no_decay)], "weight_decay": optimizer_config.get("weight_decay", 0.0)},
            {"params": [p for n, p in param_optimizer_list if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
        ]
        
        optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=optimizer_config.lr)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=optimizer_config.get("lr_step_size", 5), gamma=optimizer_config.get("lr_gamma", 0.5))
        
        return [optimizer], [scheduler]


    def forward(self, batch: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        return self.model(batch)


    def _step(self, batch: Dict[str, Any], batch_idx: int, stage: str) -> torch.Tensor:
        model_outputs = self(batch)
        available_sources = {"model_outputs": model_outputs, "batch": batch}
        total_loss = torch.tensor(0.0, device=self.device)

        for name, strategy in self.strategies.items():
            cfg = self.hparams.strategy_config[name]
            strategy_inputs = self._prepare_strategy_inputs(cfg, available_sources)
            loss = strategy.calculate(**strategy_inputs, pl_module=self)
            if loss is not None:
                total_loss += loss
                
        self.log(f"{stage}/total_loss", total_loss, on_step=(stage == 'train'), on_epoch=(stage != 'train'), prog_bar=True)                
        return total_loss


    def training_step(self, batch: Dict[str, Any], batch_idx: int) -> torch.Tensor:
        return self._step(batch, batch_idx, 'train')


    def validation_step(self, batch: Dict[str, Any], batch_idx: int):
        self._step(batch, batch_idx, 'val')


    def test_step(self, batch: Dict[str, Any], batch_idx: int):
        self._step(batch, batch_idx, 'test')


    def on_test_epoch_end(self):
        final_metrics = self.trainer.callback_metrics
        
        summary_metrics = {}
        for key, value in final_metrics.items():
            if key.startswith('test/'):
                summary_key = key.replace('test/', 'summary/')
                summary_metrics[summary_key] = value
                
        if not summary_metrics:
            return

        if self.trainer.loggers is not None:
            for logger in self.trainer.loggers:
                if isinstance(logger, WandbLogger):
                    logger.experiment.summary.update(summary_metrics)
                    break
            

    def on_after_optimizer_step(self):
        for strategy in self.strategies.values():
            strategy.on_after_optimizer_step(self)


    def _load_pretrained_weights(self, path: str):
        ckpt = torch.load(path, map_location=self.device)
        model_state_dict = {k.replace("model.", ""): v for k, v in ckpt['state_dict'].items() if k.startswith("model.")}
        self.model.load_state_dict(model_state_dict, strict=False)
        print(f"Pre-trained weights loaded from: {path}")


    def _freeze_modules(self, modules: list[str]):
        for name, param in self.model.named_parameters():
            if any(name.startswith(module_name) for module_name in modules):
                param.requires_grad = False


    def _setup_strategies(self):
        for name, strategy in self.strategies.items():
            if hasattr(strategy, "setup"):
                cfg = self.hparams.strategy_config[name]
                if 'dependencies' not in cfg:
                    continue
                resolved_dependencies = {
                    key: self._resolve_source_value(path, {"self": self})
                    for key, path in cfg.dependencies.items()
                }
                strategy.setup(**resolved_dependencies, device=self.device)


    def _prepare_strategy_inputs(self, cfg: DictConfig, available_sources: Dict[str, Any]) -> Dict[str, Any]:
        strategy_inputs = {}
        if 'inputs' in cfg:
            for arg_name, source_name in cfg.inputs.items():
                strategy_inputs[arg_name] = self._resolve_source_value(source_name, available_sources)
        return strategy_inputs


    def _resolve_source_value(self, source_name: str, available_sources: Dict[str, Any]) -> Optional[Any]:
        parts = source_name.split('.')

        initial_source = available_sources.get(parts[0])
        if initial_source is None:
            return None
            
        return self._recursive_find(initial_source, parts[1:])

    def _recursive_find(self, current_obj: Any, remaining_parts: List[str]) -> Optional[Any]:
        if not remaining_parts:
            return current_obj
        
        next_key = remaining_parts[0]

        if isinstance(current_obj, dict):
            next_obj = current_obj.get(next_key)
        else:
            next_obj = getattr(current_obj, next_key, None)

        if next_obj is None:
            return None
        
        return self._recursive_find(next_obj, remaining_parts[1:])    