from abc import ABC, abstractmethod
from typing import Generic, List, Optional, Type, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from generics import ConvType
from layers.transformer import (
    MSNATTransformer1D,
    MSNATTransformer2D,
    MSNATTransformer3D,
)
from layers.norms import LayerNorm2d, LayerNorm3d
from params import TransformerParams


class MSNATDiffusionBlock(ABC, nn.Module, Generic[ConvType]):
    """
    Base class for Multi-Scale Neighborhood Attention Transformer Diffusion Block.
    """

    conv_type: Type[ConvType]
    transformer_type: Type[nn.Module]
    norm_type: Type[nn.Module]

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        transformer_params: List[TransformerParams],
        time_dim: int,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.time_mlp = nn.Sequential(nn.SiLU(), nn.Linear(time_dim, out_channels))

        # Transformer stack
        self.transformers = nn.ModuleList(
            [self.transformer_type(**param.__dict__) for param in transformer_params]
        )

        # Apply final convolution if channels don't match
        self.proj = (
            self.conv_type(in_channels, out_channels, kernel_size=1)
            if in_channels != out_channels
            else nn.Identity()
        )

        self.dropout = nn.Dropout(dropout)
        self.norm = self.norm_type(out_channels)

    def forward(self, x: Tensor, t: Tensor) -> Tensor:
        # Transform time embedding
        time_emb = self.time_mlp(t)

        # Apply projection if needed
        x = self.proj(x)

        # Process through transformer stack
        for transformer in self.transformers:
            x = transformer(x)

        # Apply normalization and time embedding
        x = self.norm(x)

        # Add time embedding with appropriate reshaping
        x = self._add_time_embedding(x, time_emb)

        return x

    @abstractmethod
    def _add_time_embedding(self, x: Tensor, time_emb: Tensor) -> Tensor:
        """
        Add time embedding to the input tensor with appropriate reshaping.
        To be implemented by subclasses for different dimensions.
        """
        pass


class MSNATDiffusionBlock1D(MSNATDiffusionBlock[nn.Conv1d]):
    """1D implementation of Multi-Scale Neighborhood Attention Transformer Diffusion Block."""

    conv_type = nn.Conv1d
    transformer_type = MSNATTransformer1D
    norm_type = nn.LayerNorm

    def _add_time_embedding(self, x: Tensor, time_emb: Tensor) -> Tensor:
        # Reshape time embedding for 1D data: [B, C] -> [B, C, 1]
        time_emb = time_emb.unsqueeze(-1)
        return x + time_emb


class MSNATDiffusionBlock2D(MSNATDiffusionBlock[nn.Conv2d]):
    """2D implementation of Multi-Scale Neighborhood Attention Transformer Diffusion Block."""

    conv_type = nn.Conv2d
    transformer_type = MSNATTransformer2D
    norm_type = LayerNorm2d

    def _add_time_embedding(self, x: Tensor, time_emb: Tensor) -> Tensor:
        # Reshape time embedding for 2D data: [B, C] -> [B, C, 1, 1]
        time_emb = time_emb.unsqueeze(-1).unsqueeze(-1)
        return x + time_emb


class MSNATDiffusionBlock3D(MSNATDiffusionBlock[nn.Conv3d]):
    """3D implementation of Multi-Scale Neighborhood Attention Transformer Diffusion Block."""

    conv_type = nn.Conv3d
    transformer_type = MSNATTransformer3D
    norm_type = LayerNorm3d

    def _add_time_embedding(self, x: Tensor, time_emb: Tensor) -> Tensor:
        # Reshape time embedding for 3D data: [B, C] -> [B, C, 1, 1, 1]
        time_emb = time_emb.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        return x + time_emb
