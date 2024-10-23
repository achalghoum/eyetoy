import math
from abc import ABC
from typing import Generic, List, Callable, Optional, Type

import natten
import torch
from natten.functional import na2d, na3d, na1d
from torch import nn
from torch.nn import Module, Conv2d, Conv3d, ConvTranspose2d, ConvTranspose3d, \
    Conv1d, Dropout

from generics import ConvType, NAType, SharedConvNAType
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
        self.scale = math.sqrt(channels)
        # Linear Layers for Q, K, V
        self.q_conv = self.conv_type(kernel_size=1, in_channels=channels,
                                     out_channels=channels)

        self.k_conv = self.conv_type(kernel_size=1, in_channels=channels,
                                     out_channels=channels)

        self.v_conv = self.conv_type(kernel_size=1, in_channels=channels,
                                     out_channels=channels)

        self.attention_window = attn_params.attention_window
        self.attention_stride = attn_params.attention_stride
        self.is_causal = is_causal

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Linear Transformations for Q, K, V
        query = self.q_conv(x)
        key = self.k_conv(x)
        value = self.v_conv(x)
        # Transform query, key, value from NCHW to NHW1C
        query = self.transform_to_nhw1c(query)
        key = self.transform_to_nhw1c(key)
        value = self.transform_to_nhw1c(value)
        kernel_size = min(self.attention_window, *x.shape[2:])
        kernel_size -= 1 if kernel_size % 2 == 0 else 0

        return self.transform_from_nhw1c(self.atten_func(query=query, key=key, value=value, kernel_size=kernel_size,
                                 dilation=self.attention_stride,is_causal=self.is_causal))

    def transform_to_nhw1c(self, x: torch.Tensor) -> torch.Tensor:
        # Check if input is 3D, 4D, or 5D (1D, 2D, or 3D data)
        if x.dim() == 3:  # 1D data (NCL)
            return x.permute(0, 2, 1).unsqueeze(2)  # NL1C
        elif x.dim() == 4:  # 2D data (NCHW)
            return x.permute(0, 2, 3, 1).unsqueeze(3)  # NHW1C
        elif x.dim() == 5:  # 3D data (NCDHW)
            return x.permute(0, 2, 3, 4, 1).unsqueeze(4)  # NDHW1C
        else:
            raise ValueError(f"Unsupported input dimension: {x.dim()}")

    def transform_from_nhw1c(self, x: torch.Tensor) -> torch.Tensor:
        # Check if input is 4D, 5D, or 6D (1D, 2D, or 3D data)
        if x.dim() == 4:  # 1D data (NL1C)
            return x.squeeze(2).permute(0, 2, 1)  # NCL
        elif x.dim() == 5:  # 2D data (NHW1C)
            return x.squeeze(3).permute(0, 3, 1, 2)  # NCHW
        elif x.dim() == 6:  # 3D data (NDHW1C)
            return x.squeeze(4).permute(0, 4, 1, 2, 3)  # NCDHW
        else:
            raise ValueError(f"Unsupported input dimension: {x.dim()}")


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


class SharedConvNA(ABC, Module, Generic[ConvType, NAType]):
    conv_type: Type[ConvType]
    na_type: Type[NAType]

    def __init__(self, attn_params: List[NeighborhoodAttentionParams],
                 in_channels: int,
                 intermediate_channels: int,
                 conv_params: ConvParams,
                 dropout: float = 0.2):
        super().__init__()
        self.conv_params = conv_params
        self.conv = self.conv_type(**conv_params.__dict__)
        self.nas = nn.ModuleList(
            [self.na_type(attn_params=params, channels=intermediate_channels) for params in
             attn_params])
        self.in_channels = in_channels
        self.dropout = Dropout(dropout)

    def forward(self, x: torch.Tensor):
        shared_features = self.conv(x)
        return (na(shared_features) for na in self.nas)


class SharedConvNA1D(SharedConvNA[Conv1d, NA1D]):
    conv_type = Conv1d
    na_type = NA1D


class SharedConvNA2D(SharedConvNA[Conv2d, NA2D]):
    conv_type = Conv2d
    na_type = NA2D


class SharedConvNA3D(SharedConvNA[Conv3d, NA3D]):
    conv_type = Conv3d
    na_type = NA3D


class ConvMultiHeadNA(ABC, Module, Generic[ConvType, SharedConvNAType]):
    conv_attn_type: Type[SharedConvNAType]
    conv_type: Type[ConvType]

    def __init__(self,
                 head_params: List[HeadParams],
                 intermediate_channels: int,
                 out_channels: int,
                 final_conv_params: ConvParams,
                 scale_factor: int):
        super().__init__()
        self.attention_heads = torch.nn.ModuleList([
            self.conv_attn_type(**head_param.__dict__) for head_param in head_params
        ])
        final_conv_params.in_channels = sum(
            param.intermediate_channels * len(param.attn_params) for param in head_params)
        final_conv_params.kernel_size = 1
        final_conv_params.stride = 1
        final_conv_params.out_channels = out_channels
        self.final_conv = nn.Sequential(self.conv_type(**final_conv_params.__dict__))
        self.intermediate_channels = intermediate_channels
        self.out_channels = out_channels
        self.dropout = Dropout(0.2)
        self.scale_fn = self._get_scale_fn(scale_factor)
    
    def _get_scale_fn(self, scale_factor):
        if scale_factor == 1:
            return nn.Identity()
        factor = 1.0 / float(scale_factor)
        return lambda x : torch.nn.functional.interpolate(x, scale_factor = factor,
                                                        mode='bilinear')

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Compute output size based on scale factor
        input_size = x.shape[2:]  # Assuming NCHW or NCDHW format

        # Process all heads for all batches
        combined_output = torch.cat([torch.nn.functional.interpolate(output, size=input_size,
                                                        mode='nearest') for head in self.attention_heads for output in head(x)], dim=1)
        combined_output = self.scale_fn(self.dropout(combined_output))
        # Apply final convolution
        return self.final_conv(combined_output)


class ConvMultiHeadNA1D(ConvMultiHeadNA[Conv1d, SharedConvNA1D]):
    conv_attn_type = SharedConvNA1D
    conv_type = Conv1d


class ConvMultiHeadNA2D(ConvMultiHeadNA[Conv2d, SharedConvNA2D]):
    conv_attn_type = SharedConvNA2D
    conv_type = Conv2d


class ConvMultiHeadNA3D(ConvMultiHeadNA[Conv3d, SharedConvNA3D]):
    conv_attn_type = SharedConvNA3D
    conv_type = Conv3d
