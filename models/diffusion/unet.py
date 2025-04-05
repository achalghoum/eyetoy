from abc import ABC, abstractmethod
from typing import Generic, Type, List, Optional

import torch
from torch import nn as nn, Tensor

from generics import ConvType
from models.diffusion import (
    MSNATDiffusionBlock1D,
    MSNATDiffusionBlock2D,
    MSNATDiffusionBlock3D,
)
from layers.embeddings import TimestepEmbedding
from params import TransformerParams


class MSNATUNet(ABC, nn.Module, Generic[ConvType]):
    """
    Abstract UNet architecture using Multi-Scale Neighborhood Attention Transformers.
    """

    conv_type: Type[ConvType]
    diffusion_block_type: Type[nn.Module]

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        channels: List[int],
        transformer_params_per_level: List[List[TransformerParams]],
        time_dim: int,
        num_classes: Optional[int] = None,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.time_dim = time_dim
        self.is_conditional = num_classes is not None

        # Timestep embedding
        self.time_embedding = TimestepEmbedding(time_dim)

        # Class embedding for conditional generation
        if self.is_conditional and num_classes is not None:
            self.class_embedding = nn.Embedding(num_classes, time_dim)

        # Initial projection
        self.init_conv = self.conv_type(
            in_channels, channels[0], kernel_size=3, padding=1
        )

        # Down blocks
        self.down_blocks = nn.ModuleList()
        self.pool = self._get_pool_layer()

        for i in range(len(channels) - 1):
            self.down_blocks.append(
                self.diffusion_block_type(
                    channels[i],
                    channels[i],
                    transformer_params_per_level[i],
                    time_dim,
                    dropout,
                )
            )
            # Add second block for skip connection
            self.down_blocks.append(
                self.diffusion_block_type(
                    channels[i],
                    channels[i + 1],
                    transformer_params_per_level[i],
                    time_dim,
                    dropout,
                )
            )

        # Middle block
        mid_channels = channels[-1]
        self.mid_block = self.diffusion_block_type(
            mid_channels,
            mid_channels,
            transformer_params_per_level[-1],
            time_dim,
            dropout,
        )

        # Up blocks
        self.up_blocks = nn.ModuleList()
        self.upsample = self._get_upsample_layer()

        for i in range(len(channels) - 1, 0, -1):
            self.up_blocks.append(
                self.diffusion_block_type(
                    channels[i] + channels[i - 1],  # Skip connection
                    channels[i - 1],
                    transformer_params_per_level[i - 1],
                    time_dim,
                    dropout,
                )
            )
            # Add second block without skip connection
            self.up_blocks.append(
                self.diffusion_block_type(
                    channels[i - 1],
                    channels[i - 1],
                    transformer_params_per_level[i - 1],
                    time_dim,
                    dropout,
                )
            )

        # Final convolution
        self.final_conv = nn.Sequential(
            self.conv_type(channels[0], channels[0], kernel_size=3, padding=1),
            nn.GELU(),
            self.conv_type(channels[0], out_channels, kernel_size=1),
        )

    @abstractmethod
    def _get_pool_layer(self) -> nn.Module:
        """Return the appropriate pooling layer for the dimension."""
        pass

    @abstractmethod
    def _get_upsample_layer(self) -> nn.Module:
        """Return the appropriate upsampling layer for the dimension."""
        pass

    @abstractmethod
    def _cat_skip_connection(self, x: Tensor, skip: Tensor) -> Tensor:
        """Concatenate skip connection with current features along the channel dimension."""
        pass

    def forward(self, x: Tensor, time: Tensor, y: Optional[Tensor] = None) -> Tensor:
        # Time embedding
        t = self.time_embedding(time)

        # Add class conditioning if available
        if self.is_conditional and y is not None:
            t = t + self.class_embedding(y)

        # Initial convolution
        h = self.init_conv(x)

        # Store skip connections
        skips = [h]

        # Down path
        for i in range(0, len(self.down_blocks), 2):
            h = self.down_blocks[i](h, t)
            skips.append(h)
            h = self.down_blocks[i + 1](h, t)
            if i < len(self.down_blocks) - 2:  # Don't pool after last down block
                h = self.pool(h)

        # Middle block
        h = self.mid_block(h, t)

        # Up path
        for i in range(0, len(self.up_blocks), 2):
            if i > 0:  # Don't upsample on first up block
                h = self.upsample(h)
            skip_idx = len(skips) - i // 2 - 1
            h = self._cat_skip_connection(h, skips[skip_idx])
            h = self.up_blocks[i](h, t)
            h = self.up_blocks[i + 1](h, t)

        # Final convolution
        return self.final_conv(h)


class MSNATUNet1D(MSNATUNet[nn.Conv1d]):
    """1D implementation of the MSNAT U-Net architecture."""

    conv_type = nn.Conv1d
    diffusion_block_type = MSNATDiffusionBlock1D

    def _get_pool_layer(self) -> nn.Module:
        return nn.MaxPool1d(kernel_size=2)

    def _get_upsample_layer(self) -> nn.Module:
        return nn.Upsample(scale_factor=2, mode="linear", align_corners=False)

    def _cat_skip_connection(self, x: Tensor, skip: Tensor) -> Tensor:
        return torch.cat([x, skip], dim=1)


class MSNATUNet2D(MSNATUNet[nn.Conv2d]):
    """2D implementation of the MSNAT U-Net architecture."""

    conv_type = nn.Conv2d
    diffusion_block_type = MSNATDiffusionBlock2D

    def _get_pool_layer(self) -> nn.Module:
        return nn.MaxPool2d(kernel_size=2)

    def _get_upsample_layer(self) -> nn.Module:
        return nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)

    def _cat_skip_connection(self, x: Tensor, skip: Tensor) -> Tensor:
        return torch.cat([x, skip], dim=1)


class MSNATUNet3D(MSNATUNet[nn.Conv3d]):
    """3D implementation of the MSNAT U-Net architecture."""

    conv_type = nn.Conv3d
    diffusion_block_type = MSNATDiffusionBlock3D

    def _get_pool_layer(self) -> nn.Module:
        return nn.MaxPool3d(kernel_size=2)

    def _get_upsample_layer(self) -> nn.Module:
        return nn.Upsample(scale_factor=2, mode="trilinear", align_corners=False)

    def _cat_skip_connection(self, x: Tensor, skip: Tensor) -> Tensor:
        return torch.cat([x, skip], dim=1)
