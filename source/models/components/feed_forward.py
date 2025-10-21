from typing import List, Optional
from omegaconf import DictConfig
import torch.nn as nn
from pytorch_lightning.cli import instantiate_class


class Identity(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x


class Linear(nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x):
        return self.linear(x)
    

class MLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dims: Optional[List[int]] = None,
        activation_cfg: Optional[DictConfig] = None,
        output_activation_cfg: Optional[DictConfig] = None,
        use_dropout: bool = False,
        dropout_prob: float = 0.5,
    ):
        super().__init__()
        
        layers = []
        
        if not hidden_dims:
            layers.append(nn.Linear(input_dim, output_dim))
            if activation_cfg:
                raise ValueError("activation_cfg must not be provided for single-layered Network.")
        else:
            if activation_cfg is None:
                raise ValueError("activation_cfg must be provided for multi-layered MLP.")
            
            act_fn = instantiate_class(args=(), init=activation_cfg)
            
            # hidden layer
            current_dim = input_dim
            for h_dim in hidden_dims:
                layers.append(nn.Linear(current_dim, h_dim))
                layers.append(act_fn)
                if use_dropout:
                    layers.append(nn.Dropout(dropout_prob))
                current_dim = h_dim
            
            # output layer
            layers.append(nn.Linear(current_dim, output_dim))
        
        if output_activation_cfg:
            output_act_fn = instantiate_class(args=(), init=output_activation_cfg)
            layers.append(output_act_fn)
        
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)