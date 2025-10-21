import torch
import torch.nn as nn
from typing import Any, Tuple, Dict


class Identity(nn.Module):
    def forward(self, *args: Tuple[Any, ...], **kwargs: Dict[str, Any]) -> Any:
        if not kwargs:
            # 1-1. 위치 인수가 하나만 들어온 경우, 해당 요소만 반환
            if len(args) == 1:
                return args[0]
            # 1-2. 위치 인수가 여러 개 들어온 경우, 튜플 전체를 반환
            else:
                return args
        
        # 2. 위치 인수가 없는 경우, 키워드 인수 딕셔너리만 반환
        elif not args:
            return kwargs
            
        # 3. 위치 인수와 키워드 인수가 모두 있는 경우, 둘 다 반환
        else:
            return args, kwargs


class Select(nn.Module):
    def __init__(self, dim: int, index: int):
        super().__init__()
        self.dim = dim
        self.index = index

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.select(x, self.dim, self.index)


class Mean(nn.Module):
    def __init__(self, dim=1, keepdim=False):
        super().__init__()
        self.dim = dim
        self.keepdim = keepdim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.mean(x, dim=self.dim, keepdim=self.keepdim)
    
    
class Concat(nn.Module):
    def __init__(self, dim=-1):
        super().__init__()
        self.dim = dim
        
    def forward(self, *x: torch.Tensor) -> torch.Tensor:
        return torch.cat(x, dim=self.dim)


class Add(nn.Module):
    def forward(self, *inputs):
        return sum(inputs)
    
    
class Stack(nn.Module):
    def __init__(self, dim=0):
        super().__init__()
        self.dim = dim

    def forward(self, *x: torch.Tensor) -> torch.Tensor:
        return torch.stack(x, dim=self.dim)
    
    
class Permute(nn.Module):
    """
    텐서의 차원 순서를 변경합니다.
    """
    def __init__(self, dims: list):
        super().__init__()
        self.dims = dims

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.permute(*self.dims)