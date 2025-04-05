from typing import Optional, Tuple, List, Union, Literal

import torch
import math
from torch import nn as nn, Tensor


class SinusoidalPositionEmbeddings(nn.Module):
    """
    Sinusoidal position embeddings for timestep encoding.
    """

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, time: Tensor) -> Tensor:
        device = time.device
        half_dim = self.dim // 2
        embeddings = torch.log(torch.tensor(10000)) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


class TimestepEmbedding(nn.Module):
    """
    Timestep embedding layer.
    """

    def __init__(self, time_dim: int, device: Optional[torch.device] = None):
        super().__init__()
        self.time_dim = time_dim
        self.device = device

        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_dim // 4),
            nn.Linear(time_dim // 4, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )

    def forward(self, t: Tensor) -> Tensor:
        return self.time_mlp(t)


class PositionalEmbedding(nn.Module):
    """
    Base class for positional embeddings that can be used across different dimensions.
    """

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def _compute_sin_cos_embedding(self, position: Tensor, dim: int) -> Tensor:
        """
        Compute sinusoidal embedding for a position tensor.

        Args:
            position: Position tensor to embed
            dim: Embedding dimension (should be even)

        Returns:
            Sinusoidal embedding tensor
        """
        half_dim = dim // 2
        div_term = torch.exp(
            torch.arange(0, dim, 2, device=position.device) * -(math.log(10000.0) / dim)
        )

        # Reshape for proper broadcasting
        if position.dim() > 1:
            # Add dimensions for proper broadcasting with div_term
            for _ in range(position.dim() - 1):
                div_term = div_term.unsqueeze(0)

        # Compute sin/cos embeddings
        pe_sin = torch.sin(position.unsqueeze(-1) * div_term)
        pe_cos = torch.cos(position.unsqueeze(-1) * div_term)

        # Interleave sin and cos
        pe = torch.empty(*position.shape, dim, device=position.device)
        pe[..., 0::2] = pe_sin
        pe[..., 1::2] = pe_cos

        return pe


class PositionalEmbedding1D(PositionalEmbedding):
    """
    1D positional embedding layer. Takes sequence positions and returns embeddings.
    """

    def __init__(self, dim: int):
        super().__init__(dim)

    def forward(self, x: Tensor) -> Tensor:
        """
        Generate positional embeddings for a 1D input tensor.

        Args:
            x: Input tensor of shape [B, C, L]

        Returns:
            Positional embedding tensor of shape [B, D, L]
        """
        batch_size, _, seq_len = x.shape

        # Generate position tensor
        positions = (
            torch.arange(seq_len, device=x.device).unsqueeze(0).repeat(batch_size, 1)
        )

        # Compute embedding
        pe = self._compute_sin_cos_embedding(positions, self.dim)

        # Reshape to match expected [B, D, L] format
        return pe.transpose(-1, -2)


class PositionalEmbedding2D(PositionalEmbedding):
    """
    2D positional embedding layer. Takes spatial positions and returns embeddings.
    """

    def __init__(self, dim: int):
        super().__init__(dim)
        assert dim % 2 == 0, "Dimension must be even for 2D embeddings"
        self.dim_per_axis = dim // 2

    def forward(self, x: Tensor) -> Tensor:
        """
        Generate positional embeddings for a 2D input tensor.

        Args:
            x: Input tensor of shape [B, C, H, W]

        Returns:
            Positional embedding tensor of shape [B, D, H, W]
        """
        batch_size, _, height, width = x.shape

        # Generate position tensors for height and width
        y_position = torch.arange(height, device=x.device).unsqueeze(1).repeat(1, width)
        x_position = torch.arange(width, device=x.device).unsqueeze(0).repeat(height, 1)

        # Repeat for batch dimension
        y_position = y_position.unsqueeze(0).repeat(batch_size, 1, 1)
        x_position = x_position.unsqueeze(0).repeat(batch_size, 1, 1)

        # Compute embeddings for each axis
        pe_y = self._compute_sin_cos_embedding(y_position, self.dim_per_axis)
        pe_x = self._compute_sin_cos_embedding(x_position, self.dim_per_axis)

        # Concatenate along the embedding dimension
        pe = torch.cat([pe_y, pe_x], dim=-1)

        # Permute to [B, D, H, W] format
        return pe.permute(0, 3, 1, 2)


class PositionalEmbedding3D(PositionalEmbedding):
    """
    3D positional embedding layer. Takes volumetric positions and returns embeddings.
    """

    def __init__(self, dim: int):
        super().__init__(dim)
        assert dim % 3 == 0, "Dimension must be divisible by 3 for 3D embeddings"
        self.dim_per_axis = dim // 3

    def forward(self, x: Tensor) -> Tensor:
        """
        Generate positional embeddings for a 3D input tensor.

        Args:
            x: Input tensor of shape [B, C, D, H, W]

        Returns:
            Positional embedding tensor of shape [B, emb_dim, D, H, W]
        """
        batch_size, _, depth, height, width = x.shape

        # Generate position tensors for each dimension
        z_position = (
            torch.arange(depth, device=x.device)
            .view(depth, 1, 1)
            .repeat(1, height, width)
        )
        y_position = (
            torch.arange(height, device=x.device)
            .view(1, height, 1)
            .repeat(depth, 1, width)
        )
        x_position = (
            torch.arange(width, device=x.device)
            .view(1, 1, width)
            .repeat(depth, height, 1)
        )

        # Add batch dimension
        z_position = z_position.unsqueeze(0).repeat(batch_size, 1, 1, 1)
        y_position = y_position.unsqueeze(0).repeat(batch_size, 1, 1, 1)
        x_position = x_position.unsqueeze(0).repeat(batch_size, 1, 1, 1)

        # Compute embeddings for each axis
        pe_z = self._compute_sin_cos_embedding(z_position, self.dim_per_axis)
        pe_y = self._compute_sin_cos_embedding(y_position, self.dim_per_axis)
        pe_x = self._compute_sin_cos_embedding(x_position, self.dim_per_axis)

        # Concatenate along the embedding dimension
        pe = torch.cat([pe_z, pe_y, pe_x], dim=-1)

        # Permute to [B, D, depth, H, W] format
        return pe.permute(0, 4, 1, 2, 3)


def get_positional_embedding(dim: int, embedding_dim: int) -> PositionalEmbedding:
    """
    Factory function to create a positional embedding layer based on the dimension.

    Args:
        dim: Dimension of the input (1, 2, or 3)
        embedding_dim: Dimension of the embedding

    Returns:
        Appropriate positional embedding layer
    """
    if dim == 1:
        return PositionalEmbedding1D(embedding_dim)
    elif dim == 2:
        return PositionalEmbedding2D(embedding_dim)
    elif dim == 3:
        return PositionalEmbedding3D(embedding_dim)
    else:
        raise ValueError(f"Unsupported dimension: {dim}. Must be 1, 2, or 3.")
