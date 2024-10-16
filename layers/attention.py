from abc import ABC
from typing import Generic, List, Callable, Optional, Type

import torch
from torch.nn import Module, Conv2d, Conv3d, Parameter, ConvTranspose2d, ConvTranspose3d, \
    Conv1d, Dropout, Sequential, ReLU

from generics import ConvType
from natten.functional import na2d, na3d, na2d_qk, na3d_qk, na2d_av, na3d_av, na1d, na1d_qk, na1d_av
from params import ConvParams, NeighborhoodAttentionParams
from .positional_encoding import  positional_encoding
import natten

natten.use_fused_na()

class ConvNAT(ABC, Module, Generic[ConvType]):
    conv_type: Type[ConvType]
    atten_func: Callable
    qk_func: Callable
    v_func: Callable
    rpb_fn : Optional[Callable] = None
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
        # Main Convolutional Layer
        self.conv = Sequential(self.conv_type(**conv_params.__dict__),
                               ReLU())
        # Linear Layers for Q, K, V
        self.q_conv = self.conv_type(kernel_size=1, in_channels=intermediate_channels,
                                     out_channels=intermediate_channels)
        self.k_conv = self.conv_type(kernel_size=1, in_channels=intermediate_channels,
                                     out_channels=intermediate_channels)
        self.v_conv = self.conv_type(kernel_size=1, in_channels=intermediate_channels,
                                     out_channels=out_channels)
        
        # Initialize weights
        torch.nn.init.xavier_uniform_(self.q_conv.weight)
        torch.nn.init.xavier_uniform_(self.k_conv.weight)
        torch.nn.init.xavier_uniform_(self.v_conv.weight)
        
        # Fixed Attention Parameters
        self.attention_window = attn_params.attention_window
        self.attention_stride = attn_params.attention_stride
        self.is_causal = is_causal

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.dropout(x)
        # Shared Convolutional Features
        shared_features = self.conv(x)
        # Linear Transformations for Q, K, V
        query = self.q_conv(shared_features)
        key = self.k_conv(shared_features)
        value = self.v_conv(shared_features)
        # Transform query, key, value from NCHW to NHW1C
        query = self.transform_to_nhw1c(query)
        key = self.transform_to_nhw1c(key)
        value = self.transform_to_nhw1c(value)
        kernel_size = min(self.attention_window,*shared_features.shape[2:])
        kernel_size -= 1 if kernel_size % 2 == 0 else 0

        output = self.atten_func(query=query, key=key, value=value, kernel_size=kernel_size, dilation=self.attention_stride,
                               is_causal=self.is_causal)
        return self.transform_from_nhw1c(output)

    def nested_attention(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor):
        attn = self.qk_func(query, key, kernel_size=self.attention_window, dilation=self.attention_stride,
                            is_causal=self.is_causal)
        return self.v_func(attn, value, kernel_size=self.attention_window, dilation=self.attention_stride,
                           is_causal=self.is_causal)

    def transform_to_nhw1c(self, x: torch.Tensor) -> torch.Tensor:
        # Check if input is 3D, 4D, or 5D (1D, 2D, or 3D data)
        if x.dim() == 3:  # 1D data (NCL)
            N, C, L = x.shape
            return x.permute(0, 2, 1).unsqueeze(2)  # NL1C
        elif x.dim() == 4:  # 2D data (NCHW)
            N, C, H, W = x.shape
            return x.permute(0, 2, 3, 1).unsqueeze(3)  # NHW1C
        elif x.dim() == 5:  # 3D data (NCDHW)
            N, C, D, H, W = x.shape
            return x.permute(0, 2, 3, 4, 1).unsqueeze(4)  # NDHW1C
        else:
            raise ValueError(f"Unsupported input dimension: {x.dim()}")

    def transform_from_nhw1c(self, x: torch.Tensor) -> torch.Tensor:
        # Check if input is 4D, 5D, or 6D (1D, 2D, or 3D data)
        if x.dim() == 4:  # 1D data (NL1C)
            N, L, _, C = x.shape
            return x.squeeze(2).permute(0, 2, 1)  # NCL
        elif x.dim() == 5:  # 2D data (NHW1C)
            N, H, W, _, C = x.shape
            return x.squeeze(3).permute(0, 3, 1, 2)  # NCHW
        elif x.dim() == 6:  # 3D data (NDHW1C)
            N, D, H, W, _, C = x.shape
            return x.squeeze(4).permute(0, 4, 1, 2, 3)  # NCDHW
        else:
            raise ValueError(f"Unsupported input dimension: {x.dim()}")

class ConvNAT2D(ConvNAT[Conv2d]):
    conv_type = Conv2d
    atten_func = staticmethod(na2d)
    qk_func = staticmethod(na2d_qk)
    v_func = staticmethod(na2d_av)


class ConvNAT3D(ConvNAT[Conv3d]):
    conv_type = Conv3d
    atten_func = staticmethod(na3d)
    qk_func = staticmethod(na3d_qk)
    v_func = staticmethod(na3d_av)


class ConvNAT1D(ConvNAT[Conv1d]):
    conv_type = Conv1d
    atten_func = staticmethod(na1d)
    qk_func = staticmethod(na1d_qk)
    v_func = staticmethod(na1d_av)


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
