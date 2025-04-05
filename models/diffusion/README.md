# Dimension-Agnostic Diffusion Models

This module implements dimension-agnostic diffusion models for 1D, 2D, and 3D data generation. The architecture is based on Multi-Scale Neighborhood Attention Transformers (MSNAT) for efficient modeling of long-range dependencies.

## Features

- Supports 1D, 2D, and 3D data generation using the same model architecture
- Implements denoising diffusion probabilistic models (DDPM)
- Uses transformer-based architectures with attention mechanisms
- Provides utilities for visualization and sample generation
- Supports conditional generation
- Includes autoencoder variant for latent space generation

## Architecture Overview

The implementation includes:

1. **Diffusion Blocks**: Dimension-specific blocks that use multi-scale neighborhood attention
2. **UNet Architecture**: A U-Net architecture with skip connections and attention at multiple scales
3. **Diffusion Model**: Core implementation of the diffusion process for data generation
4. **Autoencoder**: Optional variant that uses the diffusion model in latent space

## Generating Samples

You can generate samples using the provided example script:

```bash
python -m models.diffusion.examples --dim 2 --resolution 32 --channels 1 --base-channels 64 --timesteps 1000 --create-animation
```

### Options:

- `--dim`: Dimensionality (1, 2, or 3)
- `--resolution`: Resolution of generated samples
- `--channels`: Number of channels in generated samples
- `--base-channels`: Base number of channels for the model
- `--timesteps`: Number of diffusion timesteps
- `--num-samples`: Number of samples to generate
- `--device`: Device to use (cuda/cpu)
- `--create-animation`: Create an animation of the diffusion process
- `--animation-interval`: Interval between frames
- `--output-file`: Output file for the animation

## Implementation Details

The implementation is organized as follows:

1. `blocks.py`: Diffusion blocks for different dimensions
2. `models.py`: Core diffusion model and autoencoder implementations
3. `utils.py`: Utility functions for visualization and sampling
4. `examples.py`: Example usage and demonstration

## Example Flow

1. Define transformer parameters for each level based on dimensions
2. Create the diffusion model with appropriate dimension settings
3. Generate samples using the model
4. Visualize the generated samples and optionally save animations

## Requirements

- PyTorch
- Matplotlib (for visualization)

## Future Work

- Add training capabilities
- Implement more efficient sampling strategies (e.g., DDIM)
- Support for text-to-image/3D conditioning
- Memory efficiency improvements for large resolutions 