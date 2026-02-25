#!/usr/bin/env python
"""
Evaluation script for trained TimeGrad forecasting model.
"""

import argparse
import logging
import pathlib

import matplotlib.pyplot as plt
import numpy as np
import torch
from diffusers import DDIMScheduler
from torch.utils.data import DataLoader

from data.dataset import TrafficDataset
from data.preprocessing import inverse_normalize
from models.timegrad import TimeGrad
from utils.samplers import forecast_timegrad

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Paths
DATA_DIR = pathlib.Path("data")
DEFAULT_PROCESSED = DATA_DIR / "processed_metr_la_timegrad.npz"
DEFAULT_CHECKPOINT = pathlib.Path(".") / "timegrad_forecast.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_test_data(processed_file):
    """Load test data and normalization parameters."""
    data = np.load(processed_file)
    test_data = data["test_data"]  # [T_test, N, 1]
    mean = data["mean"]  # [1, N, 1]
    std = data["std"]  # [1, N, 1]
    adj_mx = data["adj_mx"]
    sensor_ids = data["sensor_ids"]
    return test_data, mean, std, adj_mx, sensor_ids


def create_test_loader(test_data, batch_size=16):
    """Create a DataLoader for test windows (forecast mode)."""
    dataset = TrafficDataset(
        test_data,
        input_len=12,
        output_len=12,
        stride=1,
        mode="forecast",
        include_cond=False,  # TimeGrad does not use cond
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    return loader, dataset


def compute_metrics(pred, target):
    """
    pred: [n_samples, B, T, N, 1] or [B, T, N, 1] (mean)
    target: [B, T, N, 1]
    Returns dict with MAE, RMSE.
    """
    if pred.ndim == 5:
        pred_mean = pred.mean(dim=0)
    else:
        pred_mean = pred

    mae = torch.mean(torch.abs(pred_mean - target)).item()
    rmse = torch.sqrt(torch.mean((pred_mean - target) ** 2)).item()
    return {"MAE": mae, "RMSE": rmse}


def plot_predictions(pred, target, past, num_sensors=5):
    """
    Plot predictions vs ground truth for forecasting.
    pred: [T, N, 1] or [n_samples, T, N, 1] (if multiple)
    target: [T, N, 1]
    past: [T_past, N, 1] (observed)
    """
    plt.figure(figsize=(12, 8))
    T_future = target.shape[0]
    T_past = past.shape[0]
    time_axis_past = np.arange(-T_past, 0)
    time_axis_future = np.arange(T_future)

    for i in range(min(num_sensors, target.shape[1])):
        plt.subplot(num_sensors, 1, i + 1)
        # Plot past (observed)
        plt.plot(time_axis_past, past[:, i, 0].cpu(), "b-", label="Past (observed)")
        # Plot future ground truth
        plt.plot(time_axis_future, target[:, i, 0].cpu(), "g-", label="True future")
        # Plot predictions
        if pred.ndim == 3:  # single sample
            plt.plot(time_axis_future, pred[:, i, 0].cpu(), "r--", label="Predicted")
        else:
            pred_mean = pred.mean(dim=0)[:, i, 0].cpu()
            plt.plot(time_axis_future, pred_mean, "r--", label="Predicted (mean)")
            pred_std = pred.std(dim=0)[:, i, 0].cpu()
            plt.fill_between(
                time_axis_future,
                pred_mean - 1.96 * pred_std,
                pred_mean + 1.96 * pred_std,
                color="r",
                alpha=0.2,
                label="95% CI",
            )
        plt.ylabel(f"Sensor {i}")
        plt.legend(loc="upper right")
    plt.xlabel("Time steps (relative)")
    plt.tight_layout()
    plt.show()


def main():
    parser = argparse.ArgumentParser(description="Evaluate TimeGrad model.")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=str(DEFAULT_CHECKPOINT),
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--processed_file",
        type=str,
        default=str(DEFAULT_PROCESSED),
        help="Path to processed data .npz file",
    )
    parser.add_argument(
        "--batch_size", type=int, default=16, help="Batch size for evaluation"
    )
    parser.add_argument(
        "--n_samples",
        type=int,
        default=5,
        help="Number of samples for probabilistic evaluation",
    )
    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("EVALUATING TIMEGRAD MODEL")
    logger.info("=" * 60)

    # 1. Load test data and parameters
    test_data, mean, std, adj_mx, sensor_ids = load_test_data(args.processed_file)
    logger.info(f"Test data shape: {test_data.shape}")

    # 2. Create test loader
    test_loader, dataset = create_test_loader(test_data, batch_size=args.batch_size)
    logger.info(f"Number of test windows: {len(test_loader.dataset)}")

    # 3. Load model
    # Determine hidden_dim from checkpoint? For simplicity, assume same as training (64)
    model = TimeGrad(
        num_sensors=207,
        hidden_dim=64,
        rnn_layers=1,
        input_dim=1,
    )
    model.load_state_dict(torch.load(args.checkpoint, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    logger.info(f"Model loaded from {args.checkpoint}")

    # 4. Diffusion scheduler (same as training)
    scheduler = DDIMScheduler(
        num_train_timesteps=1000,
        beta_schedule="linear",
        prediction_type="epsilon",
    )
    num_inference_steps = 50
    scheduler.set_timesteps(num_inference_steps)

    # 5. Evaluation loop
    all_metrics = {"MAE": [], "RMSE": []}
    sample_batch_for_plot = None
    sample_pred_for_plot = None

    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            past = batch["past"].to(DEVICE)  # [B, T_in, N, 1]
            target = batch["future"].to(DEVICE)  # [B, T_out, N, 1]

            # Generate forecasts
            samples = forecast_timegrad(
                model,
                scheduler,
                past,
                T_out=12,
                num_inference_steps=num_inference_steps,
                num_samples=args.n_samples,
            )  # [n_samples, B, T_out, N, 1]

            # Compute metrics (using mean prediction)
            metrics = compute_metrics(samples, target)
            all_metrics["MAE"].append(metrics["MAE"])
            all_metrics["RMSE"].append(metrics["RMSE"])

            # Save a batch for plotting (first batch)
            if batch_idx == 0:
                sample_batch_for_plot = (past.cpu(), target.cpu())
                sample_pred_for_plot = samples.cpu()  # [n_samples, B, T_out, N, 1]

    # Aggregate metrics
    final_mae = np.mean(all_metrics["MAE"])
    final_rmse = np.mean(all_metrics["RMSE"])
    logger.info(f"Test MAE: {final_mae:.4f} (normalized scale)")
    logger.info(f"Test RMSE: {final_rmse:.4f} (normalized scale)")

    # Convert to original scale (mph)
    avg_std = np.mean(std)
    logger.info(f"Average std: {avg_std:.2f} mph")
    logger.info(f"Approx. MAE in mph: {final_mae * avg_std:.2f}")
    logger.info(f"Approx. RMSE in mph: {final_rmse * avg_std:.2f}")

    # Plot predictions for a few examples
    if sample_batch_for_plot is not None:
        past_batch, target_batch = sample_batch_for_plot
        pred_batch = sample_pred_for_plot  # [n_samples, B, T_out, N, 1]

        for ex in range(min(2, past_batch.shape[0])):
            logger.info(f"Plotting example {ex}...")
            past_ex = past_batch[ex]  # [T_in, N, 1]
            target_ex = target_batch[ex]  # [T_out, N, 1]
            pred_ex = pred_batch[:, ex, ...]  # [n_samples, T_out, N, 1]
            plot_predictions(pred_ex, target_ex, past_ex, num_sensors=3)


if __name__ == "__main__":
    main()
