import argparse

import matplotlib.pyplot as plt
import torch
import numpy as np

from params import create_diffusion_transformer_params
from models.diffusion.models import (
    create_diffusion_model,
)
from models.diffusion.utils import generate_sample, visualize_sample, create_animation


def main(args):
    """
    Main function to demonstrate diffusion model across different dimensions.

    Args:
        args: Command line arguments
    """
    # Create transformer parameters based on dimensionality
    transformer_params_per_level = create_diffusion_transformer_params(
        dim=args.dim, channels=[args.channels * (2**i) for i in range(args.num_levels)]
    )

    # Create and initialize the diffusion model
    model = create_diffusion_model(
        dim=args.dim,
        in_channels=args.in_channels,
        out_channels=args.out_channels,
        base_channels=args.channels,
        channel_multipliers=[1, 2, 4, 8][: args.num_levels],
        transformer_params_per_level=transformer_params_per_level,
        resolution=args.resolution,
        timesteps=args.timesteps,
        device=args.device,
    )

    # Generate samples
    samples = []
    for i in range(args.num_samples):
        print(f"Generating sample {i+1}/{args.num_samples}...")
        sample, _ = generate_sample(
            model=model, n_samples=1, device=args.device
        )
        samples.append(sample)

        # Visualize the sample
        if args.output_file:
            output_file = args.output_file.replace(".mp4", f"_{i+1}.mp4")
            if args.animate:
                # Create animation of the generation process
                sample_sequence = model.sample_sequence(
                    shape=(1, args.out_channels) + (args.resolution,) * args.dim,
                    device=torch.device(args.device),
                )
                create_animation(
                    samples=sample_sequence,
                    filename=output_file,
                    interval=1000 // args.fps,
                )
            else:
                # Just visualize the final sample
                fig = visualize_sample(sample)
                plt.savefig(output_file.replace(".mp4", ".png"))
                plt.close(fig)

    return samples


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate samples using dimension-agnostic diffusion model"
    )
    parser.add_argument(
        "--dim",
        type=int,
        default=3,
        choices=[1, 2, 3],
        help="Dimensionality (1D, 2D, or 3D)",
    )
    parser.add_argument(
        "--resolution", type=int, default=32, help="Resolution of generated sample"
    )
    parser.add_argument(
        "--in_channels", type=int, default=1, help="Number of input channels"
    )
    parser.add_argument(
        "--out_channels", type=int, default=1, help="Number of output channels"
    )
    parser.add_argument("--channels", type=int, default=32, help="Base channel count")
    parser.add_argument(
        "--num_levels", type=int, default=4, help="Number of U-Net levels"
    )
    parser.add_argument(
        "--timesteps", type=int, default=1000, help="Number of diffusion timesteps"
    )
    parser.add_argument(
        "--num_samples", type=int, default=1, help="Number of samples to generate"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use (cuda or cpu)",
    )
    parser.add_argument(
        "--output_file", type=str, default="sample.mp4", help="Output file path"
    )
    parser.add_argument(
        "--animate",
        action="store_true",
        help="Create animation of the generation process",
    )
    parser.add_argument(
        "--fps", type=int, default=30, help="Frames per second for animation"
    )

    args = parser.parse_args()
    main(args)
