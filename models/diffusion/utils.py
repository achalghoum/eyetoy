import math
from typing import Callable, List, Literal, Optional, Tuple, Union, Any, cast

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.animation import FuncAnimation
from matplotlib.figure import Figure
from torch import Tensor

from models.diffusion.models import DiffusionModel


def create_beta_schedule(
    schedule_type: Literal["linear", "cosine", "quadratic"],
    timesteps: int,
    start: float = 1e-4,
    end: float = 2e-2,
) -> Tensor:
    """
    Create a beta schedule for the diffusion process.

    Args:
        schedule_type: Type of beta schedule ('linear', 'cosine', or 'quadratic')
        timesteps: Number of timesteps in the diffusion process
        start: Starting value for beta
        end: Ending value for beta

    Returns:
        Tensor of beta values for each timestep
    """
    if schedule_type == "linear":
        return torch.linspace(start, end, timesteps)
    elif schedule_type == "cosine":
        steps = timesteps + 1
        x = torch.linspace(0, timesteps, steps)
        alphas_cumprod = torch.cos(((x / timesteps) + 0.008) / 1.008 * math.pi / 2) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return torch.clip(betas, 0, 0.999)
    elif schedule_type == "quadratic":
        return torch.linspace(start**0.5, end**0.5, timesteps) ** 2
    else:
        raise ValueError(f"Unknown schedule type: {schedule_type}")


def visualize_sample(
    sample: Tensor,
    title: str = "Generated Sample",
    figsize: Tuple[int, int] = (12, 12),
    slice_idx: Optional[int] = None,
    cmap: str = "viridis",
) -> Figure:
    """
    Visualize a sample from the diffusion model for any dimension.

    Args:
        sample: Generated sample [B, C, ...]
        title: Figure title
        figsize: Figure size
        slice_idx: For 3D samples, the slice index to visualize
        cmap: Colormap for visualization

    Returns:
        Matplotlib figure
    """
    sample_np = sample.detach().cpu().numpy()

    # Get first sample from batch
    sample_np = sample_np[0]

    # Get dimensionality
    dim = len(sample_np.shape) - 1  # Subtract channel dimension

    fig = plt.figure(figsize=figsize)

    if dim == 1:
        # 1D sample [C, L]
        if sample_np.shape[0] == 1:
            plt.plot(sample_np[0])
        else:
            for c in range(sample_np.shape[0]):
                plt.plot(sample_np[c], label=f"Channel {c}")
            plt.legend()

    elif dim == 2:
        # 2D sample [C, H, W]
        if sample_np.shape[0] == 1:
            plt.imshow(sample_np[0], cmap=cmap)
            plt.colorbar()
        else:
            # Multi-channel image: display each channel
            n_channels = min(sample_np.shape[0], 3)
            for i in range(n_channels):
                plt.subplot(1, n_channels, i + 1)
                plt.imshow(sample_np[i], cmap=cmap)
                plt.title(f"Channel {i}")
                plt.colorbar()

    elif dim == 3:
        # 3D sample [C, D, H, W]
        if slice_idx is None:
            slice_idx = sample_np.shape[1] // 2  # Middle slice

        if sample_np.shape[0] == 1:
            plt.imshow(sample_np[0, slice_idx], cmap=cmap)
            plt.colorbar()
            plt.title(f"{title} - Slice {slice_idx}")
        else:
            # Multi-channel volume: display each channel
            n_channels = min(sample_np.shape[0], 3)
            for i in range(n_channels):
                plt.subplot(1, n_channels, i + 1)
                plt.imshow(sample_np[i, slice_idx], cmap=cmap)
                plt.title(f"Channel {i}, Slice {slice_idx}")
                plt.colorbar()
    else:
        raise ValueError(f"Unsupported dimensionality: {dim}")

    plt.tight_layout()
    return fig


def sample_timestep(
    model: DiffusionModel,
    x: Tensor,
    t: Tensor,
    *,
    guidance_scale: float = 1.0,
    class_labels: Optional[Tensor] = None,
) -> Tensor:
    """
    Sample from the model at a specific timestep.

    Args:
        model: Diffusion model
        x: Input tensor [B, C, ...], noisy sample
        t: Timestep tensor [B]
        guidance_scale: Guidance scale for classifier-free guidance
        class_labels: Optional class labels for conditional generation

    Returns:
        Noise prediction
    """
    if guidance_scale > 1.0 and model.is_conditional:
        # For classifier-free guidance
        # Run both conditional and unconditional forward passes
        # x_cond refers to using the class_labels
        # x_uncond refers to using no class label (or a special null token)

        noise_pred_cond = model(x, t, y=class_labels)
        noise_pred_uncond = model(x, t, y=None)

        # Apply classifier-free guidance
        return noise_pred_uncond + guidance_scale * (
            noise_pred_cond - noise_pred_uncond
        )
    else:
        return model(x, t, y=class_labels)


def generate_sample(
    model: DiffusionModel,
    n_samples: int = 1,
    guidance_scale: float = 1.0,
    class_labels: Optional[Tensor] = None,
    device: Optional[torch.device] = None,
    slice_idx: Optional[int] = None,
) -> Tuple[Tensor, Figure]:
    """
    Generate a sample using the diffusion model and visualize it.

    Args:
        model: Diffusion model
        n_samples: Number of samples to generate
        guidance_scale: Guidance scale for classifier-free guidance
        class_labels: Optional class labels for conditional generation
        device: Device to use for generation
        slice_idx: For 3D samples, the index of the slice to show

    Returns:
        Tuple of (generated sample, visualization figure)
    """
    if device is None:
        device = model.device

    # Determine shape based on model parameters
    shape = [n_samples, model.out_channels]
    if isinstance(model.resolution, list):
        shape.extend(model.resolution)
    else:
        shape.extend([model.resolution] * model.get_dim())

    # Generate sample
    sample = model.sample(shape=tuple(shape), y=class_labels, device=device)

    # Visualize sample
    fig = visualize_sample(sample, slice_idx=slice_idx)

    return sample, fig


def create_animation(
    samples: List[Tensor],
    filename: str = "diffusion_animation.gif",
    interval: int = 100,
    slice_idx: Optional[int] = None,
    cmap: str = "viridis",
) -> None:
    """
    Create an animation from a sequence of samples.

    Args:
        samples: List of samples at different timesteps
        filename: Output filename
        interval: Interval between frames (milliseconds)
        slice_idx: For 3D samples, the slice index to visualize
        cmap: Colormap for visualization
    """
    fig = plt.figure(figsize=(10, 10))

    # Determine dimensionality
    sample = samples[0]
    sample_np = sample.detach().cpu().numpy()[0]  # First sample in batch
    dim = len(sample_np.shape) - 1  # Subtract channel dimension

    def update(frame: int) -> None:
        plt.clf()
        sample = samples[frame]
        sample_np = sample.detach().cpu().numpy()[0]

        if dim == 1:
            # 1D sample [C, L]
            if sample_np.shape[0] == 1:
                plt.plot(sample_np[0])
            else:
                for c in range(sample_np.shape[0]):
                    plt.plot(sample_np[c], label=f"Channel {c}")
                plt.legend()

        elif dim == 2:
            # 2D sample [C, H, W]
            if sample_np.shape[0] == 1:
                plt.imshow(sample_np[0], cmap=cmap)
                plt.colorbar()
            else:
                # Multi-channel image: display first channel
                plt.imshow(sample_np[0], cmap=cmap)
                plt.colorbar()

        elif dim == 3:
            # 3D sample [C, D, H, W]
            s_idx = slice_idx if slice_idx is not None else sample_np.shape[1] // 2

            if sample_np.shape[0] == 1:
                plt.imshow(sample_np[0, s_idx], cmap=cmap)
                plt.colorbar()
                plt.title(f"Timestep {frame} - Slice {s_idx}")
            else:
                # Multi-channel volume: display first channel
                plt.imshow(sample_np[0, s_idx], cmap=cmap)
                plt.colorbar()
                plt.title(f"Channel 0, Timestep {frame}, Slice {s_idx}")

        plt.tight_layout()

    # Create animation
    anim = FuncAnimation(fig, update, frames=len(samples), interval=interval)

    # Save animation
    if filename.endswith(".gif"):
        anim.save(filename, writer="pillow")
    else:
        anim.save(filename)

    plt.close(fig)
