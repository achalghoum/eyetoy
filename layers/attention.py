import math
from abc import ABC, abstractmethod
from typing import Generic, List, Callable, Optional, Type

import natten
import torch
from natten.functional import na2d, na3d, na1d
from torch import nn
from torch.nn import Module, Conv2d, Conv3d, ConvTranspose2d, ConvTranspose3d, \
    Conv1d, Dropout

from generics import ConvType, NAType, SharedScaleNAType
from params import ConvParams, NeighborhoodAttentionParams, HeadParams

natten.use_fused_na()


class NA(ABC, Module, Generic[ConvType]):
    conv_type: Type[ConvType]
    atten_func: Callable

    def __init__(self,
                 attn_params: NeighborhoodAttentionParams,
                 channels: int,
                 is_causal: bool = False,
                 **kwargs):
        super(NA, self).__init__()
        self.attention_window = attn_params.attention_window
        self.attention_stride = attn_params.attention_stride
        self.is_causal = is_causal

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        # Transform query, key, value from NCHW to NHW1C
        kernel_size = min(self.attention_window, *q.shape[2:])
        kernel_size -= 1 if kernel_size % 2 == 0 else 0
        return self.atten_func(query=q,
                               key=k,
                               value=v,
                               kernel_size=kernel_size,
                               dilation=self.attention_stride,
                               is_causal=self.is_causal)


class NA1D(NA[Conv1d]):
    conv_type = Conv1d
    atten_func = staticmethod(na1d)


class NA2D(NA[Conv2d]):
    conv_type = Conv2d
    atten_func = staticmethod(na2d)


class NA3D(NA[Conv3d]):
    conv_type = Conv3d
    atten_func = staticmethod(na3d)


class NA2DTransposed(NA[ConvTranspose2d]):
    conv_type = ConvTranspose2d
    atten_func = staticmethod(na2d)


class NA3DTransposed(NA[ConvTranspose3d]):
    conv_type = ConvTranspose3d
    atten_func = staticmethod(na3d)


class SharedScaleNA(ABC, Module, Generic[ConvType, NAType]):
    conv_type: Type[ConvType]
    na_type: Type[NAType]

    def __init__(self, attn_params: List[NeighborhoodAttentionParams],
                 in_channels: int,
                 intermediate_channels: int,
                 conv_params: ConvParams,
                 dropout: float = 0.2):
        super().__init__()

        self.num_heads = len(attn_params)
        self.qkv_params = conv_params
        self.qkv_params.out_channels = intermediate_channels*self.num_heads*3
        self.qkv_proj = self.conv_type(**conv_params.__dict__)
        self.nas = nn.ModuleList(
            [self.na_type(attn_params=params, channels=intermediate_channels) for params in
             attn_params])
        self.in_channels = in_channels
        self.head_channels = intermediate_channels

    def forward(self, x: torch.Tensor):
        qkv = self.transform_to_nhw1c(self.qkv_proj(x))
        qkv = qkv.split(qkv.shape[-1] // self.num_heads, dim=-1)  # Split into heads
        return [self.transform_from_nhw1c(na(q, k, v)) for i, na in enumerate(self.nas)
                for q, k, v in (qkv[i].split(qkv[i].shape[-1] // 3, dim=-1))]  # Split each head into q,k,v

    @abstractmethod
    def transform_to_nhw1c(self, x: torch.Tensor) -> torch.Tensor:
        raise ValueError(f"Unsupported input dimension: {x.dim()}")

    @abstractmethod
    def transform_from_nhw1c(self, x: torch.Tensor) -> torch.Tensor:
        raise ValueError(f"Unsupported input dimension: {x.dim()}")


class SharedScaleNA1D(SharedScaleNA[Conv1d, NA1D]):
    conv_type = Conv1d
    na_type = NA1D

    def transform_to_nhw1c(self, x: torch.Tensor) -> torch.Tensor:
        return x.permute(0, 2, 1).unsqueeze(2)  # NL1C

    def transform_from_nhw1c(self, x: torch.Tensor) -> torch.Tensor:
        return x.squeeze(2).permute(0, 2, 1)  # NCL


class SharedScaleNA2D(SharedScaleNA[Conv2d, NA2D]):
    conv_type = Conv2d
    na_type = NA2D

    def transform_to_nhw1c(self, x: torch.Tensor) -> torch.Tensor:
        return x.permute(0, 2, 3, 1).unsqueeze(3)  # NHW1C

    def transform_from_nhw1c(self, x: torch.Tensor) -> torch.Tensor:
        return x.squeeze(3).permute(0, 3, 1, 2)  # NCHW


class SharedScaleNA3D(SharedScaleNA[Conv3d, NA3D]):
    conv_type = Conv3d
    na_type = NA3D

    def transform_to_nhw1c(self, x: torch.Tensor) -> torch.Tensor:
        return x.permute(0, 2, 3, 4, 1).unsqueeze(4)  # NDHW1C

    def transform_from_nhw1c(self, x: torch.Tensor) -> torch.Tensor:
        return x.squeeze(4).permute(0, 4, 1, 2, 3)  # NCDHW


class MulitScaleMultiHeadNA(ABC, Module, Generic[ConvType, SharedScaleNAType]):
    attn_type: Type[SharedScaleNAType]
    conv_type: Type[ConvType]

    def __init__(self,
                 head_params: List[HeadParams],
                 intermediate_channels: int,
                 out_channels: int,
                 final_conv_params: ConvParams,
                 scale_factor: int,
                 dropout: float = 0.2):
        super().__init__()
        self.attention_heads = torch.nn.ModuleList([
            self.attn_type(**head_param.__dict__) for head_param in head_params
        ])
        final_conv_params.in_channels = sum(
            param.intermediate_channels * len(param.attn_params) for param in head_params)
        final_conv_params.kernel_size = 1
        final_conv_params.stride = 1
        final_conv_params.out_channels = out_channels
        self.out_proj = self.conv_type(**final_conv_params.__dict__)
        self.intermediate_channels = intermediate_channels
        self.out_channels = out_channels
        self.scale_fn = self._get_scale_fn(scale_factor)
        self.dropout = nn.Dropout(dropout)

    def _get_scale_fn(self, scale_factor):
        if scale_factor == 1:
            return nn.Identity()
        return lambda x: torch.nn.functional.interpolate(x, scale_factor=scale_factor,
                                                         mode='bilinear')

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Compute output size based on scale factor
        input_size = x.shape[2:]  # Assuming NCHW or NCDHW format

        # Process all heads for all batches
        combined_output = torch.cat([torch.nn.functional.interpolate(self.dropout(output), size=input_size,
                                                                     mode='nearest') for head in self.attention_heads for output in head(x)], dim=1)
        combined_output = self.scale_fn(combined_output)
        # Apply final convolution
        return self.out_proj(combined_output)


class MulitScaleMultiHeadNA1D(MulitScaleMultiHeadNA[Conv1d, SharedScaleNA1D]):
    conv_attn_type = SharedScaleNA1D
    conv_type = Conv1d


class MulitScaleMultiHeadNA2D(MulitScaleMultiHeadNA[Conv2d, SharedScaleNA2D]):
    conv_attn_type = SharedScaleNA2D
    conv_type = Conv2d


class MulitScaleMultiHeadNA3D(MulitScaleMultiHeadNA[Conv3d, SharedScaleNA3D]):
    conv_attn_type = SharedScaleNA3D
    conv_type = Conv3d
