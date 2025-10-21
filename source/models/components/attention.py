import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class ScaledDotProductAttention(nn.Module):
    def __init__(self, dropout_prob: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, query, key, value, mask=None):
        # query, key, value: (batch, n_heads, seq_len, d_k)
        d_k = query.size(-1)
        
        # 1. MatMul Q and K^T
        # scores: (batch, n_heads, seq_len, seq_len)
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
        
        # 2. Apply mask (if provided)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9) # or float('-inf')
            
        # 3. Apply softmax to get attention weights
        p_attn = F.softmax(scores, dim=-1)
        
        # 4. Apply dropout
        p_attn = self.dropout(p_attn)
        
        # 5. MatMul with V to get the final output
        # output: (batch, n_heads, seq_len, d_k)
        return torch.matmul(p_attn, value), p_attn


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout_prob: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        # Linear projections for Q, K, V and the final output
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.fc = nn.Linear(d_model, d_model)

        self.attention = ScaledDotProductAttention(dropout_prob=dropout_prob)
        self.dropout = nn.Dropout(dropout_prob)
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, query, key, value, mask=None):
        # query, key, value: (batch, seq_len, d_model)
        batch_size = query.size(0)
        residual = query

        # 1. Pass through linear layers and split into n_heads
        # (batch, seq_len, d_model) -> (batch, n_heads, seq_len, d_k)
        q = self.w_q(query).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        k = self.w_k(key).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        v = self.w_v(value).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)

        # 2. Compute attention
        # x: (batch, n_heads, seq_len, d_k)
        # attn: (batch, n_heads, seq_len, seq_len)
        x, attn = self.attention(q, k, v, mask=mask)

        # 3. Concatenate heads and pass through final linear layer
        # (batch, n_heads, seq_len, d_k) -> (batch, seq_len, d_model)
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        x = self.fc(x)
        
        # 4. Apply dropout and add residual connection
        x = self.dropout(x)
        x += residual
        
        # 5. Apply layer norm
        x = self.layer_norm(x)
        
        return x, attn