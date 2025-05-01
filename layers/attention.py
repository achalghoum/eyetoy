import math
from abc import ABC, abstractmethod
from typing import Generic, List, Callable, Optional, Type

import natten # type: ignore
import torch
from natten.functional import na2d, na3d, na1d # type: ignore
from torch import nn
from torch.nn import (
    Module,
    Conv2d,
    Conv3d,
    ConvTranspose2d,
    ConvTranspose3d,
    Conv1d,
    Dropout,
)

from generics import ConvType, NAType, SharedScaleNAType
from params import ConvParams, NeighborhoodAttentionParams, HeadParams

natten.use_fused_na()

# Enable KV parallelism
natten.use_kv_parallelism_in_fused_na(True)
natten.set_memory_usage_preference("unrestricted")

class NA(ABC, Module, Generic[ConvType]):
    conv_type: Type[ConvType]
    atten_func: Callable

    def __init__(
        self,
        attn_params: NeighborhoodAttentionParams,
        is_causal: bool = False,
        **kwargs,
    ):
        super(NA, self).__init__()
        self.attention_window = attn_params.attention_window
        self.attention_stride = attn_params.attention_stride
        self.is_causal = is_causal

    def forward(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor
    ) -> torch.Tensor:
        # Transform query, key, value from NCHW to NHW1C
        kernel_size = min(self.attention_window, *q.shape[1:-2])
        kernel_size -= 1 if kernel_size % 2 == 0 else 0
        return self.atten_func(
            query=q,
            key=k,
            value=v,
            kernel_size=kernel_size,
            dilation=self.attention_stride,
            is_causal=self.is_causal,
        )


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

    def __init__(
        self,
        attn_params: NeighborhoodAttentionParams,
        in_channels: int,
        intermediate_channels: int,
        conv_params: ConvParams,
        num_heads: int,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.tokenizer = self.conv_type(**conv_params.__dict__)
        self.qkv_proj = self.conv_type(
            in_channels=conv_params.out_channels, # type: ignore
            out_channels=intermediate_channels * self.num_heads * 3,
            stride=1,
            kernel_size=1,
            padding="same",# type: ignore
        )
        self.na = self.na_type(attn_params=attn_params)
        self.in_channels = in_channels
        self.head_dim = intermediate_channels
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor):
        # Add docstring explaining the transformation
        qkv = self.transform_to_nhw1c(self.qkv_proj(self.dropout(self.tokenizer(x))))
        q = qkv[0]
        k = qkv[1]
        v = qkv[2]
        atten_output = self.na(q, k, v)
        return [
            self.transform_from_nhw1c(o)
            for o in atten_output.chunk(self.num_heads, dim=-2)
        ]

        # Concatenate outputs across heads

    @abstractmethod
    def transform_to_nhw1c(self, x: torch.Tensor) -> torch.Tensor:
        """
        Transforms the input tensor to NHW1C format for attention computation.
        
        Args:
            x (torch.Tensor): The input tensor to be transformed.
            
        Returns:
            torch.Tensor: The transformed tensor in NHW1C format.
            
        Raises:
            ValueError: If the input tensor has an unsupported dimension.
        """
        raise ValueError(f"Unsupported input dimension: {x.dim()}")

    @abstractmethod
    def transform_from_nhw1c(self, x: torch.Tensor) -> torch.Tensor:
        """
        Transforms the input tensor from NHW1C format back to its original format.
        
        Args:
            x (torch.Tensor): The input tensor in NHW1C format to be transformed back.
            
        Returns:
            torch.Tensor: The transformed tensor back to its original format.
            
        Raises:
            ValueError: If the input tensor has an unsupported dimension.
        """
        raise ValueError(f"Unsupported input dimension: {x.dim()}")

class SharedScaleNA1D(SharedScaleNA[Conv1d, NA1D]):
    conv_type = Conv1d
    na_type = NA1D

    def transform_to_nhw1c(self, x: torch.Tensor) -> torch.Tensor:
        B, C, L = x.shape
        return x.reshape(B, L, 3, self.num_heads, self.head_dim).permute(2, 0, 1, 3, 4)

    def transform_from_nhw1c(self, x: torch.Tensor) -> torch.Tensor:
        return x.squeeze(2).permute(0, 2, 1)  # NCL


class SharedScaleNA2D(SharedScaleNA[Conv2d, NA2D]):
    conv_type = Conv2d
    na_type = NA2D

    def transform_to_nhw1c(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        return x.reshape(B, H, W, 3, self.num_heads, self.head_dim).permute(
            3, 0, 1, 2, 4, 5
        )

    def transform_from_nhw1c(self, x: torch.Tensor) -> torch.Tensor:
        return x.squeeze(3).permute(0, 3, 1, 2)  # NCHW


class SharedScaleNA3D(SharedScaleNA[Conv3d, NA3D]):
    conv_type = Conv3d
    na_type = NA3D

    def transform_to_nhw1c(self, x: torch.Tensor) -> torch.Tensor:
        B, D, H, W, C = x.shape
        return x.reshape(B, D, H, W, 3, self.num_heads, self.head_dim).permute(
            4, 0, 5, 1, 2, 3, 6
        )

    def transform_from_nhw1c(self, x: torch.Tensor) -> torch.Tensor:
        return x.squeeze(4).permute(0, 4, 1, 2, 3)  # NCDHW


class MultiScaleMultiHeadNA(ABC, Module, Generic[ConvType, SharedScaleNAType]):
    attn_type: Type[SharedScaleNAType]
    conv_type: Type[ConvType]

    def __init__(
        self,
        head_params: List[HeadParams],
        intermediate_channels: int,
        out_channels: int,
        final_conv_params: ConvParams,
        scale_factor: int,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.attention_heads = torch.nn.ModuleList(
            [self.attn_type(**head_param.__dict__) for head_param in head_params]
        )
        final_conv_params.in_channels = sum(
            param.intermediate_channels * param.num_heads for param in head_params
        )
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
        return lambda x: torch.nn.functional.interpolate(
            x, scale_factor=scale_factor, mode="bilinear"
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Compute output size based on scale factor
        input_size = x.shape[2:]  # Assuming NCHW or NCDHW format

        # Process all heads for all batches
        combined_output = torch.cat(
            [
                torch.nn.functional.interpolate(
                    self.dropout(output), size=input_size, mode="nearest"
                )
                for head in self.attention_heads
                for output in head(x)
            ],
            dim=1,
        )
        combined_output = self.scale_fn(combined_output)
        # Apply final convolution
        return self.out_proj(combined_output)


class MultiScaleMultiHeadNA1D(MultiScaleMultiHeadNA[Conv1d, SharedScaleNA1D]):
    attn_type = SharedScaleNA1D
    conv_type = Conv1d


class MultiScaleMultiHeadNA2D(MultiScaleMultiHeadNA[Conv2d, SharedScaleNA2D]):
    attn_type = SharedScaleNA2D
    conv_type = Conv2d


class MultiScaleMultiHeadNA3D(MultiScaleMultiHeadNA[Conv3d, SharedScaleNA3D]):
    attn_type = SharedScaleNA3D
    conv_type = Conv3d
