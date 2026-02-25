import logging

import numpy as np
import torch
from torch.utils.data import Dataset

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# data/dataset.py

import logging

import numpy as np
import torch
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


class TrafficDataset(Dataset):
    def __init__(
        self,
        data: np.ndarray,  # [T, N, F] normalized data
        input_len: int = 12,
        output_len: int = 12,
        stride: int = 1,
        mode: str = "forecast",  # "forecast" or "imputation"
        corruption_rate: float = 0.0,  # for imputation: fraction of values to mask
        include_cond: bool = False,  # for forecast: whether to return cond and mask
    ):
        self.data = torch.FloatTensor(data)
        self.input_len = input_len
        self.output_len = output_len
        self.stride = stride
        self.mode = mode
        self.corruption_rate = corruption_rate
        self.include_cond = include_cond

        # Create window indices
        self.indices = []
        T = data.shape[0]
        # Need enough steps for input_len + output_len
        for i in range(0, T - input_len - output_len + 1, stride):
            self.indices.append(i)

        logger.info(f"Created dataset with {len(self.indices)} windows, mode={mode}")

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        start = self.indices[idx]
        past = self.data[start : start + self.input_len]  # [in_len, N, F]
        future = self.data[
            start + self.input_len : start + self.input_len + self.output_len
        ]
        combined = torch.cat([past, future], dim=0)  # [in_len+out_len, N, F]

        if self.mode == "forecast":
            item = {"past": past, "future": future, "idx": idx}
            if self.include_cond:
                # Conditioning: repeat last observed value across future steps
                last_obs = past[-1:]  # [1, N, F]
                cond = last_obs.repeat(self.output_len, 1, 1)  # [out_len, N, F]
                mask = torch.zeros_like(future)  # 0 = missing (to be generated)
                item["cond"] = cond
                item["mask"] = mask
            return item

        elif self.mode == "imputation":
            # Generate random mask over the combined window
            # 1 = observed, 0 = missing
            mask = torch.bernoulli(
                torch.ones_like(combined) * (1 - self.corruption_rate)
            )
            # Corrupted version: observed values stay, missing become 0.0
            corrupted = combined.clone()
            corrupted[mask == 0] = 0.0
            return {
                "combined": combined,  # ground truth
                "corrupted": corrupted,  # input to model (zeros for missing)
                "mask": mask,  # 1=observed, 0=missing
                "idx": idx,
            }
        else:
            raise ValueError(f"Unknown mode: {self.mode}")

    def get_stats(self):
        return {
            "num_windows": len(self.indices),
            "input_len": self.input_len,
            "output_len": self.output_len,
            "data_shape": self.data.shape,
            "mode": self.mode,
        }
