import logging

import numpy as np
import torch
from torch.utils.data import Dataset

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class TrafficDataset(Dataset):
    """PyTorch dataset for traffic forecasting and imputation."""

    def __init__(
        self,
        data: np.ndarray,  # [T, N, F] normalized data
        input_len: int = 12,  # Number of past steps (1 hour at 5-min)
        output_len: int = 12,  # Number of future steps (1 hour)
        stride: int = 1,  # Step between windows
        mode: str = "forecast",  # "forecast" or "imputation"
        corruption_rate: float = 0.0,  # For imputation: % of values to mask
    ):
        self.data = torch.FloatTensor(data)
        self.input_len = input_len
        self.output_len = output_len
        self.stride = stride
        self.mode = mode
        self.corruption_rate = corruption_rate

        # Create window indices
        self.indices = []
        T = data.shape[0]
        for i in range(0, T - input_len - output_len + 1, stride):
            self.indices.append(i)

        logger.info(f"Created dataset with {len(self.indices)} windows")

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        start = self.indices[idx]

        # Extract windows
        past = self.data[start : start + self.input_len]  # [input_len, N, 1]
        future = self.data[
            start + self.input_len : start + self.input_len + self.output_len
        ]  # [output_len, N, 1]

        if self.mode == "forecast":
            return {
                "past": past,  # Conditioning context
                "future": future,  # Target to generate
                "idx": idx,
            }

        elif self.mode == "imputation":
            # Create corruption mask
            mask = torch.ones_like(future)
            if self.corruption_rate > 0:
                # Random missing mask
                mask = torch.bernoulli(
                    torch.ones_like(future) * (1 - self.corruption_rate)
                )

            # Corrupted version: observed values stay, missing become NaN (will be handled in loss)
            corrupted = future.clone()
            corrupted[mask == 0] = float("nan")

            return {
                "past": past,
                "future": future,  # Ground truth
                "corrupted": corrupted,  # With missing values
                "mask": mask,  # 1=observed, 0=missing
                "idx": idx,
            }

    def get_stats(self):
        """Return dataset statistics."""
        return {
            "num_windows": len(self.indices),
            "input_len": self.input_len,
            "output_len": self.output_len,
            "data_shape": self.data.shape,
            "mode": self.mode,
        }
