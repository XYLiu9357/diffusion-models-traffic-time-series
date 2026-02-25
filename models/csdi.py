# models/csdi.py

import torch
import torch.nn as nn

from .unet import SpatioTemporalUNet


class CSDI(nn.Module):
    """
    CSDI model for time series forecasting/imputation.
    Wraps a SpatioTemporalUNet that expects input [B, T, N, F] (here F=3).
    """

    def __init__(self, unet: SpatioTemporalUNet):
        super().__init__()
        self.unet = unet
        # Verify the U‑Net has 3 input channels
        if self.unet.input_proj.in_channels != 3:
            raise ValueError("CSDI requires the U‑Net to have in_channels=3")

    def forward(self, noisy_target, timesteps, cond, mask):
        """
        Args:
            noisy_target: [B, T, N, 1] – noisy future steps
            timesteps: [B] – diffusion timesteps
            cond: [B, T, N, 1] – conditioning (repeated past)
            mask: [B, T, N, 1] – 0 for missing (future), 1 for observed
        Returns:
            predicted noise: [B, T, N, 1]
        """
        # Concatenate along feature dimension → [B, T, N, 3]
        x = torch.cat([noisy_target, cond, mask], dim=-1)
        # Pass directly to U‑Net (it will permute internally)
        out = self.unet(x, timesteps)
        return out
