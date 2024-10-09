from abc import ABC
from typing import Generic, List, Callable, Type

import torch
from torch.nn import Module, Conv2d, Conv3d, Parameter, ConvTranspose2d, ConvTranspose3d, \
    Conv1d

from generics import ConvType
from natten.functional import na2d, na3d, na2d_qk, na3d_qk, na2d_av, na3d_av, na1d, na1d_qk, na1d_av
from params import ConvParams, NeighborhoodAttentionParams
from positional_encoding import positional_encoding



class ConvNAT(ABC, Module, Generic[ConvType]):
    conv_type: Type[ConvType]
    atten_func: Callable
    qk_func: Callable
    v_func: Callable

    def __init__(self, conv_params: ConvParams,
                 attn_params: NeighborhoodAttentionParams,
                 in_channels: int,
                 out_channels: int,
                 intermediate_channels: int,
                 is_causal: bool = False):
        super().__init__()

        # Shared Convolutional Layer
        self.conv = self.conv_type(**conv_params.__dict__,
                                   in_channels=in_channels,
                                   out_channels=intermediate_channels)

        # Linear Layers for Q, K, V
        self.q_conv = self.conv_type(kernel_size=1, in_channels=intermediate_channels,
                                     out_channels=intermediate_channels)
        self.k_conv = self.conv_type(kernel_size=1, in_channels=intermediate_channels,
                                     out_channels=intermediate_channels)
        self.v_conv = self.conv_type(kernel_size=1, in_channels=intermediate_channels,
                                     out_channels=out_channels)

        # Attention Parameters
        self.dilation = Parameter(torch.tensor(attn_params.min_dilation, dtype=torch.int32),
                                  requires_grad=True)
        self.kernel_size = Parameter(torch.tensor(attn_params.min_kernel_size, dtype=torch.int32),
                                     requires_grad=True)
        self.min_dilation = attn_params.min_dilation
        self.max_dilation = attn_params.max_dilation
        self.min_kernel_size = attn_params.min_kernel_size
        self.max_kernel_size = attn_params.max_kernel_size
        self.is_causal = is_causal

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dilation = int(self.get_dilation())
        kernel_size = int(self.get_kernel_size())

        # Shared Convolutional Features
        shared_features = self.conv(x)

        # Linear Transformations for Q, K, V
        query = self.q_conv(shared_features)
        key = self.k_conv(shared_features)
        value = self.v_conv(shared_features)

        pos_encoding = positional_encoding(query)
        query = torch.cat([query, pos_encoding], dim=-1)
        key = torch.cat([key, pos_encoding], dim=-1)

        if x.is_nested:
            return self.nested_attention(query, key, value, dilation, kernel_size)
        return self.atten_func(query, key, value, kernel_size=kernel_size, dilation=dilation,
                               is_causal=self.is_causal)

    def get_dilation(self):
        if self.training:
            return self.dilation.clamp(self.min_dilation, self.max_dilation)
        else:
            return self.dilation.item()

    def get_kernel_size(self):
        if self.training:
            return self.kernel_size.clamp(self.min_kernel_size, self.max_kernel_size)
        else:
            return self.kernel_size.item()

    def nested_attention(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                         dilation: int, kernel_size: int):
        attn = self.qk_func(query, key, kernel_size=kernel_size, dilation=dilation,
                            is_causal=self.is_causal)
        return self.v_func(attn, value, kernel_size=kernel_size, dilation=dilation,
                           is_causal=self.is_causal)


class ConvNAT2D(ConvNAT[Conv2d]):
    conv_type = Conv2d
    atten_func = na2d
    qk_func = na2d_qk
    v_func = na2d_av


class ConvNAT3D(ConvNAT[Conv3d]):
    conv_type = Conv3d
    atten_func = na3d
    qk_func = na3d_qk
    v_func = na3d_av


class ConvNAT1D(ConvNAT[Conv1d]):
    conv_type = Conv1d
    atten_func = na1d
    qk_func = na1d_qk
    v_func = na1d_av


class ConvNAT2DTransposed(ConvNAT[ConvTranspose2d]):
    conv_type = ConvTranspose2d
    atten_func = na2d
    qk_func = na2d_qk
    v_func = na2d_av


class ConvNAT3DTransposed(ConvNAT[ConvTranspose3d]):
    conv_type = ConvTranspose3d
    atten_func = na3d
    qk_func = na3d_qk
    v_func = na3d_av
