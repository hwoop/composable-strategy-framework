import math
import torch
import torch.nn as nn


class CLSTokenEmbedding(nn.Module):
    def __init__(self, embedding_dim: int, seq_dim: int = 1):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.seq_dim = seq_dim
        self.cls_token = nn.Parameter(torch.zeros(embedding_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        n_dims = x.dim()

        # e.g., (1, 1, D) for 3D
        cls_view_shape = [1] * n_dims
        cls_view_shape[self.seq_dim] = 1
        cls_view_shape[-1] = self.embedding_dim
        # view shape가 (1,1,D) 와 같지 않을 수 있으므로 맨 처음 차원은 1로 고정
        cls_view_shape[0] = 1

        cls_token_reshaped = self.cls_token.view(tuple(cls_view_shape))
        
        # 배치 차원에 맞게 확장
        batch_size = x.size(0)
        expand_shape = list(x.shape)
        expand_shape[0] = batch_size
        expand_shape[self.seq_dim] = 1

        cls_tokens = cls_token_reshaped.expand(tuple(expand_shape))
        
        return torch.cat([cls_tokens, x], dim=self.seq_dim)


class SinusoidalPositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term[:(d_model + 1) // 2])

        pe = pe.unsqueeze(0)  # shape: (1, max_len, d_model)
        self.register_buffer('pe', pe)


    def forward(self, x):
        seq_len = x.size(-1)
        if seq_len > self.pe.size(-1):
            raise ValueError(f"Sequence length {seq_len} exceeds maximum length {self.pe.size(1)}")
        return x + self.pe[:, :seq_len]