import torch.nn as nn
from typing import Union, Tuple


class Conv1d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int]],
        stride: Union[int, Tuple[int]] = 1,
        padding: Union[str, int, Tuple[int]] = 0,
        dilation: Union[int, Tuple[int]] = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = 'zeros',
        conv_dim: int = -1,
    ):
        super().__init__()
        self.conv_dim = conv_dim
        self.conv1d = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
            padding_mode=padding_mode,
        )


    def forward(self, x):
        conv_dim = self.conv_dim if self.conv_dim >= 0 else x.dim() + self.conv_dim
        if conv_dim == x.dim() - 1: # (B, C, L)
            return self.conv1d(x)
        
        if x.dim() == 3 and conv_dim == 1:
            # (B, L, C) -> (B, C, L)
            x_permuted = x.permute(0, 2, 1)
            conv_output = self.conv1d(x_permuted)
            # (B, C_out, L) -> (B, L, C_out)
            output = conv_output.permute(0, 2, 1)
            return output
        else:
            raise NotImplementedError(
                f"Automatic permutation for conv_dim={self.conv_dim} "
                f"on a tensor with {x.dim()} dimensions is not implemented."
            )        


class Conv2d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int, int]],
        stride: Union[int, Tuple[int, int]] = 1,
        padding: Union[str, int, Tuple[int, int]] = 0,
        dilation: Union[int, Tuple[int, int]] = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = 'zeros',
    ):
        super().__init__()
        self.conv2d = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
            padding_mode=padding_mode,
        )

    def forward(self, x):
        return self.conv2d(x)