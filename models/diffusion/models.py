from abc import ABC, abstractmethod
from typing import List, Optional, Tuple, Union, Literal, Generic, Type

import torch
import torch.nn as nn
from torch import Tensor

from generics import ConvType
from models.diffusion import MSNATUNet
from models.diffusion.scheduler import DDPMScheduler
from models.diffusion.unet import MSNATUNet1D, MSNATUNet2D, MSNATUNet3D
from params import TransformerParams, create_diffusion_transformer_params


class DiffusionModel(ABC, nn.Module, Generic[ConvType]):
    """
    Abstract diffusion model for generating samples.
    """

    unet_type: Type[MSNATUNet]

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        channels: List[int],
        transformer_params_per_level: List[List[TransformerParams]],
        resolution: Union[int, List[int]],
        timesteps: int = 1000,
        beta_schedule: Literal["linear", "cosine", "quadratic"] = "linear",
        beta_start: float = 1e-4,
        beta_end: float = 2e-2,
        num_classes: Optional[int] = None,
        device: Union[str, torch.device] = (
            "cuda" if torch.cuda.is_available() else "cpu"
        ),
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.channels = channels
        self.timesteps = timesteps
        self.is_conditional = num_classes is not None

        # Convert resolution to list if it's a single int
        if isinstance(resolution, int):
            self.resolution = [resolution] * self.get_dim()
        else:
            self.resolution = resolution

        # Create UNet model
        self.model = self.unet_type(
            in_channels=in_channels,
            out_channels=out_channels,
            channels=channels,
            transformer_params_per_level=transformer_params_per_level,
            time_dim=channels[0] * 4,  # Usually 4x base channels for time embedding
            num_classes=num_classes,
        )

        # Create scheduler
        self.scheduler = DDPMScheduler(
            num_timesteps=timesteps,
            beta_schedule=beta_schedule,
            beta_start=beta_start,
            beta_end=beta_end,
        )

        # Move to device
        if isinstance(device, str):
            device = torch.device(device)
        self.device = device
        self.to(device)

    @abstractmethod
    def get_dim(self) -> int:
        """Return the dimension of the model (1, 2, or 3)."""
        pass

    def forward(self, x: Tensor, t: Tensor, y: Optional[Tensor] = None) -> Tensor:
        """
        Forward pass predicting the noise.

        Args:
            x: Input tensor of shape [B, C, ...], noisy sample
            t: Timestep tensor of shape [B]
            y: Optional class labels for conditional generation

        Returns:
            Predicted noise
        """
        return self.model(x, t, y)

    def sample(
        self,
        shape: Tuple[int, ...],
        y: Optional[Tensor] = None,
        device: Optional[torch.device] = None,
    ) -> Tensor:
        """
        Sample from the diffusion model.

        Args:
            shape: Shape of the sample to generate [B, C, ...]
            y: Optional class labels for conditional generation
            device: Device to use for generation

        Returns:
            Generated sample
        """
        if device is None:
            device = self.device

        # Start from random noise
        x = torch.randn(shape, device=device)

        # Gradually denoise
        for i in range(self.timesteps - 1, -1, -1):
            t = torch.full((shape[0],), i, device=device, dtype=torch.long)

            with torch.no_grad():
                noise_pred = self(x, t, y)
                x = self.scheduler.step(noise_pred, i, x)

        return x

    def sample_sequence(
        self,
        shape: Tuple[int, ...],
        device: Optional[torch.device] = None,
        num_timesteps: int = 10,
        y: Optional[Tensor] = None,
        guidance_scale: float = 1.0,
    ) -> List[Tensor]:
        """
        Generate a sample sequence showing the denoising process.

        Args:
            shape: Shape of the sample to generate [B, C, ...]
            device: Device to use for generation
            num_timesteps: Number of timesteps to include in the sequence
            y: Optional class labels for conditional generation
            guidance_scale: Guidance scale for classifier-free guidance

        Returns:
            List of tensors at different timesteps during generation
        """
        if device is None:
            device = self.device

        # Initialize with random noise
        x = torch.randn(shape, device=device)

        # Calculate timesteps to save
        save_steps = torch.linspace(0, self.timesteps - 1, num_timesteps).long()
        samples = []

        # Reverse diffusion process
        for i in range(self.timesteps - 1, -1, -1):
            t = torch.full((shape[0],), i, device=device, dtype=torch.long)

            with torch.no_grad():
                # Get model prediction for denoising
                if guidance_scale > 1.0 and self.is_conditional:
                    # For classifier-free guidance
                    noise_pred_cond = self(x, t, y=y)
                    noise_pred_uncond = self(x, t, y=None)
                    noise_pred = noise_pred_uncond + guidance_scale * (
                        noise_pred_cond - noise_pred_uncond
                    )
                else:
                    noise_pred = self(x, t, y=y)

                # Perform sampling step
                x = self.scheduler.step(noise_pred, i, x)

            # Save intermediate step if in timesteps_to_save
            if i in save_steps:
                samples.append(x.clone())

        # Make sure we have the final sample
        if x is not samples[-1]:
            samples.append(x.clone())

        return samples


class DiffusionModel1D(DiffusionModel[nn.Conv1d]):
    """1D implementation of the diffusion model."""

    unet_type = MSNATUNet1D

    def get_dim(self) -> int:
        return 1


class DiffusionModel2D(DiffusionModel[nn.Conv2d]):
    """2D implementation of the diffusion model."""

    unet_type = MSNATUNet2D

    def get_dim(self) -> int:
        return 2


class DiffusionModel3D(DiffusionModel[nn.Conv3d]):
    """3D implementation of the diffusion model."""

    unet_type = MSNATUNet3D

    def get_dim(self) -> int:
        return 3


class DiffusionAutoencoder(nn.Module):
    """
    Autoencoder that uses diffusion model as latent encoder.
    """

    def __init__(
        self,
        encoder: nn.Module,
        diffusion_model: DiffusionModel,
        decoder: nn.Module,
        latent_dim: int,
    ):
        super().__init__()
        self.encoder = encoder
        self.diffusion_model = diffusion_model
        self.decoder = decoder
        self.latent_dim = latent_dim

    def encode(self, x: Tensor) -> Tensor:
        """Encode input to latent space."""
        return self.encoder(x)

    def sample_latent(
        self, shape: Tuple[int, ...], device: Optional[torch.device] = None
    ) -> Tensor:
        """Sample from latent space using diffusion model."""
        return self.diffusion_model.sample(shape, device=device)

    def decode(self, z: Tensor) -> Tensor:
        """Decode from latent space."""
        return self.decoder(z)

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        """Forward pass through the autoencoder."""
        z = self.encode(x)
        return self.decode(z), z


def create_diffusion_model(
    dim: Literal[1, 2, 3],
    in_channels: int,
    out_channels: int,
    base_channels: int = 64,
    channel_multipliers: List[int] = [1, 2, 4, 8],
    transformer_params_per_level: Optional[List[List[TransformerParams]]] = None,
    resolution: Union[int, List[int]] = 32,
    timesteps: int = 1000,
    beta_schedule: Literal["linear", "cosine", "quadratic"] = "linear",
    num_classes: Optional[int] = None,
    device: Union[str, torch.device] = "cuda" if torch.cuda.is_available() else "cpu",
) -> DiffusionModel:
    """
    Factory function to create a diffusion model with the appropriate dimension.

    Args:
        dim: Dimensionality (1, 2, or 3)
        in_channels: Number of input channels
        out_channels: Number of output channels
        base_channels: Base number of channels
        channel_multipliers: Multipliers for channels at each level
        transformer_params_per_level: Parameters for transformers at each level
        resolution: Resolution of the input/output (can be a single int or list of ints)
        timesteps: Number of timesteps for the diffusion process
        beta_schedule: Schedule for the noise variance
        num_classes: Number of classes for conditional generation
        device: Device to use

    Returns:
        Diffusion model of the appropriate dimension
    """
    # Create channels list
    channels = [base_channels * m for m in channel_multipliers]

    # Create transformer parameters if not provided
    if transformer_params_per_level is None:
        transformer_params_per_level = create_diffusion_transformer_params(
            dim=dim, channels=channels
        )

    # Create model based on dimension
    if dim == 1:
        return DiffusionModel1D(
            in_channels=in_channels,
            out_channels=out_channels,
            channels=channels,
            transformer_params_per_level=transformer_params_per_level,
            resolution=resolution,
            timesteps=timesteps,
            beta_schedule=beta_schedule,
            num_classes=num_classes,
            device=device,
        )
    elif dim == 2:
        return DiffusionModel2D(
            in_channels=in_channels,
            out_channels=out_channels,
            channels=channels,
            transformer_params_per_level=transformer_params_per_level,
            resolution=resolution,
            timesteps=timesteps,
            beta_schedule=beta_schedule,
            num_classes=num_classes,
            device=device,
        )
    elif dim == 3:
        return DiffusionModel3D(
            in_channels=in_channels,
            out_channels=out_channels,
            channels=channels,
            transformer_params_per_level=transformer_params_per_level,
            resolution=resolution,
            timesteps=timesteps,
            beta_schedule=beta_schedule,
            num_classes=num_classes,
            device=device,
        )
    else:
        raise ValueError(f"Unsupported dimension: {dim}")
