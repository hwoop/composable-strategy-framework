import torch
import torch.nn as nn

class LayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-5):
        super().__init__()
        self.layer_norm = nn.LayerNorm(normalized_shape, eps=eps)

    def forward(self, x):
        return self.layer_norm(x)


class BatchNorm1d(nn.Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        super().__init__()
        self.batch_norm = nn.BatchNorm1d(num_features, eps=eps, momentum=momentum)

    def forward(self, x):
        return self.batch_norm(x)


class BatchNorm2d(nn.Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        super().__init__()
        self.batch_norm = nn.BatchNorm2d(num_features, eps=eps, momentum=momentum)

    def forward(self, x):
        return self.batch_norm(x)


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        # (B, Seq_Len, Dim) -> (B, Seq_Len, 1)
        rms = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x * rms

    def forward(self, x):
        # input: (batch, seq_len, dim)
        # weight: (dim) -> (1, 1, dim) for broadcasting
        output = self._norm(x.float()).type_as(x)
        return output * self.weight