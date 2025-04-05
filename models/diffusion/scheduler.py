import math
from typing import Literal, Optional, Tuple

import torch
from torch import Tensor
from torch.nn import functional as F


class DDPMScheduler:
    """
    Denoising Diffusion Probabilistic Model scheduler for the forward and reverse processes.
    """

    def __init__(
        self,
        num_timesteps: int,
        beta_schedule: Literal["linear", "cosine", "quadratic"] = "linear",
        beta_start: float = 1e-4,
        beta_end: float = 2e-2,
    ):
        self.num_timesteps = num_timesteps

        # Define beta schedule
        if beta_schedule == "linear":
            betas = torch.linspace(beta_start, beta_end, num_timesteps)
        elif beta_schedule == "cosine":
            steps = num_timesteps + 1
            x = torch.linspace(0, num_timesteps, steps)
            alphas_cumprod = (
                torch.cos(((x / num_timesteps) + 0.008) / 1.008 * math.pi / 2) ** 2
            )
            alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
            betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
            betas = torch.clip(betas, 0, 0.999)
        elif beta_schedule == "quadratic":
            betas = torch.linspace(beta_start**0.5, beta_end**0.5, num_timesteps) ** 2
        else:
            raise ValueError(f"Unknown beta schedule: {beta_schedule}")

        self.betas = betas

        # Pre-compute diffusion process constants
        self.alphas = 1 - betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)

        # Calculations for diffusion q(x_t | x_{t-1}) and others
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1 - self.alphas_cumprod)
        self.log_one_minus_alphas_cumprod = torch.log(1 - self.alphas_cumprod)
        self.sqrt_recip_alphas_cumprod = torch.sqrt(1 / self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = torch.sqrt(1 / self.alphas_cumprod - 1)

        # Calculations for posterior q(x_{t-1} | x_t, x_0)
        self.posterior_variance = (
            betas * (1 - self.alphas_cumprod_prev) / (1 - self.alphas_cumprod)
        )
        self.posterior_log_variance_clipped = torch.log(
            torch.cat([self.posterior_variance[1:2], self.posterior_variance[1:]])
        )
        self.posterior_mean_coef1 = (
            betas * torch.sqrt(self.alphas_cumprod_prev) / (1 - self.alphas_cumprod)
        )
        self.posterior_mean_coef2 = (
            (1 - self.alphas_cumprod_prev)
            * torch.sqrt(self.alphas)
            / (1 - self.alphas_cumprod)
        )

    def q_sample(
        self, x_0: Tensor, t: Tensor, noise: Optional[Tensor] = None
    ) -> Tensor:
        """
        Forward diffusion process: q(x_t | x_0)

        Args:
            x_0: Clean data
            t: Timestep
            noise: Optional noise to add

        Returns:
            Noisy sample x_t
        """
        if noise is None:
            noise = torch.randn_like(x_0)

        sqrt_alphas_cumprod_t = self._extract_into_tensor(
            self.sqrt_alphas_cumprod, t, x_0.shape
        )
        sqrt_one_minus_alphas_cumprod_t = self._extract_into_tensor(
            self.sqrt_one_minus_alphas_cumprod, t, x_0.shape
        )

        return sqrt_alphas_cumprod_t * x_0 + sqrt_one_minus_alphas_cumprod_t * noise

    def _predict_x0_from_eps(self, x_t: Tensor, t: Tensor, eps: Tensor) -> Tensor:
        """
        Predict x_0 from the noise eps.
        """
        sqrt_recip_alphas_cumprod_t = self._extract_into_tensor(
            self.sqrt_recip_alphas_cumprod, t, x_t.shape
        )
        sqrt_recipm1_alphas_cumprod_t = self._extract_into_tensor(
            self.sqrt_recipm1_alphas_cumprod, t, x_t.shape
        )

        return sqrt_recip_alphas_cumprod_t * x_t - sqrt_recipm1_alphas_cumprod_t * eps

    def q_posterior_mean_variance(
        self, x_0: Tensor, x_t: Tensor, t: Tensor
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Compute the mean and variance of the diffusion posterior q(x_{t-1} | x_t, x_0).
        """
        posterior_mean_coef1_t = self._extract_into_tensor(
            self.posterior_mean_coef1, t, x_t.shape
        )
        posterior_mean_coef2_t = self._extract_into_tensor(
            self.posterior_mean_coef2, t, x_t.shape
        )

        posterior_mean = posterior_mean_coef1_t * x_0 + posterior_mean_coef2_t * x_t
        posterior_variance_t = self._extract_into_tensor(
            self.posterior_variance, t, x_t.shape
        )
        posterior_log_variance_clipped_t = self._extract_into_tensor(
            self.posterior_log_variance_clipped, t, x_t.shape
        )

        return posterior_mean, posterior_variance_t, posterior_log_variance_clipped_t

    def p_mean_variance(
        self, model_output: Tensor, x_t: Tensor, t: Tensor, clip_denoised: bool = True
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Apply the model to predict the mean and variance of p(x_{t-1} | x_t).
        """
        # Predict x_0
        pred_x0 = self._predict_x0_from_eps(x_t, t, model_output)

        # Clip x_0 to [-1, 1]
        if clip_denoised:
            pred_x0 = torch.clamp(pred_x0, -1.0, 1.0)

        # Get posterior mean and variance
        model_mean, posterior_variance, posterior_log_variance = (
            self.q_posterior_mean_variance(pred_x0, x_t, t)
        )

        return model_mean, posterior_variance, posterior_log_variance

    def step(
        self,
        model_output: Tensor,
        timestep: int,
        sample: Tensor,
        clip_denoised: bool = True,
    ) -> Tensor:
        """
        Predict the sample at the previous timestep by reversing the process.

        Args:
            model_output: Predicted noise output
            timestep: Current discrete timestep in the diffusion chain
            sample: Current noisy sample x_t
            clip_denoised: Whether to clip denoised signal to [-1, 1]

        Returns:
            The predicted previous sample x_{t-1}
        """
        t = torch.full(
            (sample.shape[0],), timestep, device=sample.device, dtype=torch.long
        )

        # Get mean and variance
        model_mean, posterior_variance, posterior_log_variance = self.p_mean_variance(
            model_output=model_output, x_t=sample, t=t, clip_denoised=clip_denoised
        )

        # No variance during inference
        if timestep == 0:
            return model_mean
        else:
            # Add noise
            noise = torch.randn_like(sample)
            return model_mean + torch.exp(0.5 * posterior_log_variance) * noise

    def _extract_into_tensor(
        self, arr: Tensor, t: Tensor, broadcast_shape: Tuple[int, ...]
    ) -> Tensor:
        """
        Helper function to extract values from arr indexed by t and broadcast to the specified shape.
        """
        out = arr.gather(-1, t)
        while len(out.shape) < len(broadcast_shape):
            out = out[..., None]
        return out.expand(broadcast_shape)
