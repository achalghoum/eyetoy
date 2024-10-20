import math
from abc import ABC
from typing import Generic, List, Callable, Optional, Type

import natten
import torch
from natten.functional import na2d, na3d, na2d_qk, na3d_qk, na2d_av, na3d_av, na1d, na1d_qk, na1d_av
from torch import nn
from torch.nn import Module, Conv2d, Conv3d, ConvTranspose2d, ConvTranspose3d, \
    Conv1d, Dropout, Sequential, BatchNorm1d, BatchNorm2d, BatchNorm3d

from generics import ConvType, ConvNATType
from params import ConvParams, NeighborhoodAttentionParams, HeadParams

natten.use_fused_na()


class ConvNAT(ABC, Module, Generic[ConvType]):
    conv_type: Type[ConvType]
    atten_func: Callable
    qk_func: Callable
    v_func: Callable
    rpb_fn: Optional[Callable] = None

    def __init__(self, conv_params: ConvParams,
                 attn_params: NeighborhoodAttentionParams,
                 in_channels: int,
                 out_channels: int,
                 intermediate_channels: int,
                 is_causal: bool = False,
                 dropout: float = 0.1,
                 **kwargs):
        super(ConvNAT, self).__init__()
        self.dropout = Dropout(dropout)
        self.scale = math.sqrt(intermediate_channels)
        self.norm = nn.GroupNorm(num_channels=in_channels, num_groups=16)
        # Main Convolutional Layer
        self.conv = Sequential(
            self.conv_type(**conv_params.__dict__),
            nn.GELU()
        )
        # Linear Layers for Q, K, V
        self.q_conv = self.conv_type(kernel_size=1, in_channels=intermediate_channels,
                                     out_channels=intermediate_channels)

        self.k_conv = self.conv_type(kernel_size=1, in_channels=intermediate_channels,
                                     out_channels=intermediate_channels)

        self.v_conv = self.conv_type(kernel_size=1, in_channels=intermediate_channels,
                                     out_channels=intermediate_channels)

        self.attention_window = attn_params.attention_window
        self.attention_stride = attn_params.attention_stride
        self.is_causal = is_causal

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Shared Convolutional Features
        normed_x = self.norm(x)
        shared_features = self.conv(normed_x)
        # Linear Transformations for Q, K, V

        query = self.q_conv(shared_features)
        key = self.k_conv(shared_features)
        value = self.v_conv(shared_features)
        # Transform query, key, value from NCHW to NHW1C
        query = self.transform_to_nhw1c(query)
        key = self.transform_to_nhw1c(key)
        value = self.transform_to_nhw1c(value)
        kernel_size = min(self.attention_window, *shared_features.shape[2:])
        kernel_size -= 1 if kernel_size % 2 == 0 else 0

        output = self.atten_func(query=query, key=key, value=value, kernel_size=kernel_size,
                                 dilation=self.attention_stride,
                                 is_causal=self.is_causal, scale=self.scale)
        return self.transform_from_nhw1c(output)

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


class ConvNAT1D(ConvNAT[Conv1d]):
    conv_type = Conv1d
    atten_func = staticmethod(na1d)
    qk_func = staticmethod(na1d_qk)
    v_func = staticmethod(na1d_av)
    norm_type = BatchNorm1d


class ConvNAT2D(ConvNAT[Conv2d]):
    conv_type = Conv2d
    atten_func = staticmethod(na2d)
    qk_func = staticmethod(na2d_qk)
    v_func = staticmethod(na2d_av)
    norm_type = BatchNorm2d


class ConvNAT3D(ConvNAT[Conv3d]):
    conv_type = Conv3d
    atten_func = staticmethod(na3d)
    qk_func = staticmethod(na3d_qk)
    v_func = staticmethod(na3d_av)
    norm_type = BatchNorm3d


class ConvNAT2DTransposed(ConvNAT[ConvTranspose2d]):
    conv_type = ConvTranspose2d
    atten_func = staticmethod(na2d)
    qk_func = staticmethod(na2d_qk)
    v_func = staticmethod(na2d_av)


class ConvNAT3DTransposed(ConvNAT[ConvTranspose3d]):
    conv_type = ConvTranspose3d
    atten_func = staticmethod(na3d)
    qk_func = staticmethod(na3d_qk)
    v_func = staticmethod(na3d_av)


class ConvMultiHeadNAT(ABC, Module, Generic[ConvType, ConvNATType]):
    conv_attn_type: Type[ConvNATType]
    conv_type: Type[ConvType]

    def __init__(self, num_heads: int, head_params: List[HeadParams], in_channels: int,
                 intermediate_channels: int, out_channels: int, final_conv_params: ConvParams,
                 scale_factor: int, dropout: float = 0.1):
        super().__init__()
        self.attention_heads = torch.nn.ModuleList([
            self.conv_attn_type(**head_param.__dict__) for head_param in head_params
        ])
        final_conv_params.stride =  scale_factor
        final_conv_params.padding = math.ceil(
            float(final_conv_params.kernel_size - final_conv_params.stride) / 2.0)
        self.final_conv = nn.Sequential(self.conv_type(**final_conv_params.__dict__),
                                        nn.GELU())
        self.num_heads = num_heads
        self.in_channels = in_channels
        self.intermediate_channels = intermediate_channels
        self.out_channels = out_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Compute output size based on scale factor
        input_size = x.shape[2:]  # Assuming NCHW or NCDHW format

        # Process all heads for all batches
        head_outputs = [head(x) for head in self.attention_heads]

        # Resize all head outputs to match the scaled input size
        head_outputs = [torch.nn.functional.interpolate(output, size=input_size,
                                                        mode='nearest') for output in head_outputs]

        # Concatenate head outputs along the channel dimension for each batch
        combined_output = torch.cat(head_outputs, dim=1)

        # Apply final convolution
        return self.final_conv(combined_output)


class ConvMultiHeadNAT1D(ConvMultiHeadNAT[Conv1d, ConvNAT1D]):
    conv_attn_type = ConvNAT1D
    conv_type = Conv1d


class ConvMultiHeadNAT2D(ConvMultiHeadNAT[Conv2d, ConvNAT2D]):
    conv_attn_type = ConvNAT2D
    conv_type = Conv2d


class ConvMultiHeadNAT3D(ConvMultiHeadNAT[Conv3d, ConvNAT3D]):
    conv_attn_type = ConvNAT3D
    conv_type = Conv3d
