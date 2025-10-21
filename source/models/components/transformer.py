import torch
import torch.nn as nn
from pytorch_lightning.cli import instantiate_class


class TransformerEncoder(nn.Module):
    def __init__(
        self,
        input_dim: int,
        dim_feedforward: int,
        num_heads: int,
        num_layers: int,
        dropout: float,        
        batch_first
    ):
        super().__init__()
        self.input_dim = input_dim
        self.dropout = nn.Dropout(dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=input_dim,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=batch_first
        )

        self.encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=num_layers,
            norm=nn.LayerNorm(input_dim)
        )


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)
    

class PositionalTransformerEncoder(nn.Module):
    def __init__(
        self,
        input_dim: int,
        dim_feedforward: int,
        num_heads: int,
        num_layers: int,
        dropout: float,
        position_embedding_cfg: nn.Module,
        batch_first = True
    ):
        super().__init__()
        self.position_embedding = instantiate_class(args=(), init=position_embedding_cfg)
        self.encoder = TransformerEncoder(
            input_dim=input_dim,
            dim_feedforward=dim_feedforward,
            num_heads=num_heads,
            num_layers=num_layers,
            dropout=dropout,
            batch_first=batch_first
        )
        
        self.dropout = nn.Dropout(dropout)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.position_embedding(x)
        x = self.dropout(x)
        return self.encoder(x)
