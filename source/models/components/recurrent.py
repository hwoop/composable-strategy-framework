import torch
import torch.nn as nn
from typing import Optional


class Embed(nn.Module):
    def __init__(
        self, 
        num_embeddings: int,
        embedding_dim: int,
    ):
        super().__init__()
        self.embed = nn.Embedding(num_embeddings, embedding_dim)
    
    def forward(self, x) -> torch.Tensor:
        return self.embed(x)


class LSTM(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int = 1,
        bias: bool = True,
        batch_first: bool = True,
        dropout: float = 0.0,
        bidirectional: bool = False,
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bias=bias,
            batch_first=batch_first,
            dropout=dropout,
            bidirectional=bidirectional,
        )

    def forward(self, x: torch.Tensor):
        return self.lstm(x)


class GRU(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int = 1,
        bias: bool = True,
        batch_first: bool = True,
        dropout: float = 0.0,
        bidirectional: bool = False,
    ):
        super().__init__()
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bias=bias,
            batch_first=batch_first,
            dropout=dropout,
            bidirectional=bidirectional,
        )

    def forward(self, x: torch.Tensor, hidden: Optional[torch.Tensor] = None):
        return self.gru(x, hidden)