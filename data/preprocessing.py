"""
Preprocessing utilities for traffic data:
- Missing value imputation
- Normalization (z‑score)
- Chronological splitting
- Inverse transformation
"""

import logging

import numpy as np

logger = logging.getLogger(__name__)


def impute_missing(data: np.ndarray, method: str = "mean") -> np.ndarray:
    """
    Impute missing (NaN) values in the data array.

    Args:
        data: Array of shape (T, N, F) – time steps, sensors, features.
        method: Imputation method. Currently only "mean" (per‑sensor mean).

    Returns:
        imputed_data: Array of same shape with NaNs replaced.
    """
    if method == "mean":
        # Compute per‑sensor mean ignoring NaNs
        col_mean = np.nanmean(data, axis=0, keepdims=True)  # shape (1, N, F)
        nan_mask = np.isnan(data)
        if nan_mask.any():
            logger.info(f"Imputing {nan_mask.sum()} missing values with column means")
            data = np.where(nan_mask, col_mean, data)
    else:
        raise NotImplementedError(f"Imputation method '{method}' not implemented.")
    return data


def normalize(data: np.ndarray, method: str = "zscore", return_params: bool = True):
    """
    Normalize the data along the sensor dimension.

    Args:
        data: Array of shape (T, N, F).
        method: Normalization method. Currently only "zscore".
        return_params: If True, return (normalized_data, mean, std).

    Returns:
        If return_params is True:
            normalized_data, mean, std
        else:
            normalized_data
    """
    if method == "zscore":
        mean = np.mean(data, axis=0, keepdims=True)  # shape (1, N, F)
        std = np.std(data, axis=0, keepdims=True)
        # Avoid division by zero for constant sensors
        std[std == 0] = 1.0
        normalized = (data - mean) / std
        logger.info(
            f"Normalized data range: [{normalized.min():.3f}, {normalized.max():.3f}]"
        )
        if return_params:
            return normalized, mean, std
        return normalized
    else:
        raise NotImplementedError(f"Normalization method '{method}' not implemented.")


def inverse_normalize(
    normalized_data: np.ndarray, mean: np.ndarray, std: np.ndarray
) -> np.ndarray:
    """
    Revert z‑score normalization to obtain original‑scale data.

    Args:
        normalized_data: Array of shape (..., N, F).
        mean, std: Arrays of shape (1, N, F) as returned by normalize().

    Returns:
        data: Array in original scale.
    """
    return normalized_data * std + mean


def split_data(data: np.ndarray, train_ratio: float = 0.7, val_ratio: float = 0.1):
    """
    Chronologically split data into train, validation, and test sets.

    Args:
        data: Array of shape (T, N, F).
        train_ratio: Fraction for training.
        val_ratio: Fraction for validation (test gets the remainder).

    Returns:
        train_data, val_data, test_data: Each a view of the original array.
    """
    T = data.shape[0]
    train_idx = int(T * train_ratio)
    val_idx = int(T * (train_ratio + val_ratio))

    train = data[:train_idx]
    val = data[train_idx:val_idx]
    test = data[val_idx:]

    logger.info(f"Data split: train {train.shape}, val {val.shape}, test {test.shape}")
    return train, val, test
