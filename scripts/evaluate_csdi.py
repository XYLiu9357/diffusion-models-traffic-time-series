#!/usr/bin/env python
"""
Evaluation script for trained CSDI model.
Supports both forecasting and imputation tasks.
"""

import argparse
import logging
import pathlib

import matplotlib.pyplot as plt
import numpy as np
import torch
from diffusers import DDIMScheduler
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.dataset import TrafficDataset
from data.preprocessing import inverse_normalize
from models.csdi import CSDI
from models.unet import SpatioTemporalUNet
from utils.samplers import forecast_csdi, impute_csdi

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Paths
DATA_DIR = pathlib.Path("data")
DEFAULT_FORECAST_PROCESSED = DATA_DIR / "processed_metr_la_csdi.npz"
DEFAULT_IMPUTATION_PROCESSED = DATA_DIR / "processed_metr_la_imputation.npz"
DEFAULT_FORECAST_CHECKPOINT = pathlib.Path("checkpoints") / "csdi_metrla.pth"
DEFAULT_IMPUTATION_CHECKPOINT = pathlib.Path("checkpoints") / "csdi_imputation.pth"
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


def create_test_loader(test_data, task, batch_size=16, corruption_rate=0.15):
    """Create a DataLoader for test windows according to task."""
    if task == "forecast":
        dataset = TrafficDataset(
            test_data,
            input_len=12,
            output_len=12,
            stride=1,
            mode="forecast",
            include_cond=True,
        )
    elif task == "imputation":
        dataset = TrafficDataset(
            test_data,
            input_len=12,
            output_len=12,
            stride=1,
            mode="imputation",
            corruption_rate=corruption_rate,  # same as training; mask not used in evaluation? We'll keep fixed for consistency
        )
    else:
        raise ValueError(f"Unknown task: {task}")

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    return loader, dataset


def compute_metrics(pred, target, mask=None):
    """
    Compute MAE and RMSE.
    If mask is provided, compute only on positions where mask == 0 (missing).
    pred: [n_samples, B, T, N, 1] or [B, T, N, 1] (mean)
    target: [B, T, N, 1]
    mask: [B, T, N, 1] or None (1 = observed, 0 = missing)
    """
    if pred.ndim == 5:
        pred_mean = pred.mean(dim=0)
    else:
        pred_mean = pred

    if mask is not None:
        # Select only missing positions (mask == 0)
        missing_mask = (mask == 0).float()
        # If no missing positions in this batch, return zeros (or skip)
        if missing_mask.sum() == 0:
            return {"MAE": 0.0, "RMSE": 0.0}
        mae = (torch.abs(pred_mean - target) * missing_mask).sum() / missing_mask.sum()
        rmse = torch.sqrt(
            ((pred_mean - target) ** 2 * missing_mask).sum() / missing_mask.sum()
        )
    else:
        mae = torch.mean(torch.abs(pred_mean - target))
        rmse = torch.sqrt(torch.mean((pred_mean - target) ** 2))

    return {"MAE": mae.item(), "RMSE": rmse.item()}


def plot_predictions(pred, target, context, task, num_sensors=5):
    """
    Plot predictions vs ground truth.
    For forecast: context is past (observed).
    For imputation: context is corrupted input (observed values, zeros for missing).
    """
    plt.figure(figsize=(12, 8))
    T_total = target.shape[0]
    # For imputation, we have no separate past; context length equals target length
    if task == "forecast":
        T_past = context.shape[0]
        T_future = target.shape[0]
        time_axis_past = np.arange(-T_past, 0)
        time_axis_future = np.arange(T_future)
    else:  # imputation
        T_context = context.shape[0]  # same as target
        time_axis = np.arange(T_context)

    for i in range(min(num_sensors, target.shape[1])):
        plt.subplot(num_sensors, 1, i + 1)

        if task == "forecast":
            # Plot past (ground truth)
            plt.plot(
                time_axis_past, context[:, i, 0].cpu(), "b-", label="Past (observed)"
            )
            # Plot future ground truth
            plt.plot(time_axis_future, target[:, i, 0].cpu(), "g-", label="True future")
            # Plot predictions
            if pred.ndim == 3:  # single sample
                plt.plot(
                    time_axis_future, pred[:, i, 0].cpu(), "r--", label="Predicted"
                )
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
        else:  # imputation
            # Plot corrupted input (observed values, zeros for missing)
            plt.plot(time_axis, context[:, i, 0].cpu(), "b-", label="Corrupted input")
            # Plot ground truth
            plt.plot(time_axis, target[:, i, 0].cpu(), "g-", label="True values")
            # Plot imputed (mean and CI)
            if pred.ndim == 3:
                plt.plot(time_axis, pred[:, i, 0].cpu(), "r--", label="Imputed")
            else:
                pred_mean = pred.mean(dim=0)[:, i, 0].cpu()
                plt.plot(time_axis, pred_mean, "r--", label="Imputed (mean)")
                pred_std = pred.std(dim=0)[:, i, 0].cpu()
                plt.fill_between(
                    time_axis,
                    pred_mean - 1.96 * pred_std,
                    pred_mean + 1.96 * pred_std,
                    color="r",
                    alpha=0.2,
                    label="95% CI",
                )
            plt.ylabel(f"Sensor {i}")

        plt.legend(loc="upper right")

    if task == "forecast":
        plt.xlabel("Time steps (relative)")
    else:
        plt.xlabel("Time step index")
    plt.tight_layout()
    plt.show()


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate CSDI model on forecast or imputation."
    )
    parser.add_argument(
        "--task",
        type=str,
        choices=["forecast", "imputation"],
        default="forecast",
        help="Task to evaluate: forecast or imputation",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to model checkpoint. If not provided, uses default based on task.",
    )
    parser.add_argument(
        "--processed_file",
        type=str,
        default=None,
        help="Path to processed data .npz file. If not provided, uses default based on task.",
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
    parser.add_argument(
        "--corruption_rate",
        type=float,
        default=0.15,
        help="Corruption rate for imputation (only used if task=imputation)",
    )
    args = parser.parse_args()

    # Set defaults based on task
    if args.processed_file is None:
        args.processed_file = (
            DEFAULT_FORECAST_PROCESSED
            if args.task == "forecast"
            else DEFAULT_IMPUTATION_PROCESSED
        )
    if args.checkpoint is None:
        args.checkpoint = (
            DEFAULT_FORECAST_CHECKPOINT
            if args.task == "forecast"
            else DEFAULT_IMPUTATION_CHECKPOINT
        )

    logger.info("=" * 60)
    logger.info(f"EVALUATING CSDI MODEL ON {args.task.upper()} TASK")
    logger.info("=" * 60)

    # 1. Load test data and parameters
    test_data, mean, std, adj_mx, sensor_ids = load_test_data(args.processed_file)
    logger.info(f"Test data shape: {test_data.shape}")

    # 2. Create test loader
    test_loader, dataset = create_test_loader(
        test_data,
        args.task,
        batch_size=args.batch_size,
        corruption_rate=args.corruption_rate,
    )
    logger.info(f"Number of test windows: {len(test_loader.dataset)}")

    # 3. Load model
    # Note: time_steps depends on task: 12 for forecast, 24 for imputation (combined window)
    time_steps = 12 if args.task == "forecast" else 24
    unet = SpatioTemporalUNet(
        adj_mx=adj_mx,
        in_channels=3,
        out_channels=1,
        time_steps=time_steps,
        num_sensors=adj_mx.shape[0],
        hidden_dims=[64, 128, 256],
        use_gat=False,
        debug=False,
    )
    model = CSDI(unet)
    model.load_state_dict(torch.load(args.checkpoint, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    logger.info(f"Model loaded from {args.checkpoint}")

    # 4. Diffusion scheduler (use DDIM for faster sampling)
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
    logger.info(f"Samples per batch: {args.n_samples}")

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(test_loader, desc="Evaluating")):
            if args.task == "forecast":
                cond = batch["cond"].to(DEVICE)
                mask = batch["mask"].to(DEVICE)
                target = batch["future"].to(DEVICE)
            else:  # imputation
                combined = batch["combined"].to(DEVICE)
                corrupted = batch["corrupted"].to(DEVICE)
                mask = batch["mask"].to(DEVICE)
                # For imputation, the target is the full combined sequence
                target = combined
                # Conditioning is the corrupted input
                cond = corrupted

            # Generate samples
            samples = []
            sampler_fn = forecast_csdi if args.task == "forecast" else impute_csdi

            # Generate all samples in one go
            samples = sampler_fn(
                model,
                scheduler,
                cond,
                mask,
                num_steps=num_inference_steps,
                n_samples=args.n_samples,
            )  # shape: [n_samples, B, T, N, 1]

            # Compute metrics (using mean prediction)
            metrics = compute_metrics(samples, target, mask=mask)
            all_metrics["MAE"].append(metrics["MAE"])
            all_metrics["RMSE"].append(metrics["RMSE"])

            # Save a batch for plotting (first batch)
            if batch_idx == 0:
                if args.task == "forecast":
                    sample_batch_for_plot = (cond.cpu(), target.cpu())
                else:
                    # For imputation, we want to show corrupted input and ground truth
                    sample_batch_for_plot = (corrupted.cpu(), target.cpu())
                sample_pred_for_plot = samples.cpu()  # [n_samples, B, T, N, 1]

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

    # Plot predictions
    if sample_batch_for_plot is not None:
        context_batch, target_batch = sample_batch_for_plot
        pred_batch = sample_pred_for_plot

        # Pick a couple of examples
        for ex in range(min(2, context_batch.shape[0])):
            logger.info(f"Plotting example {ex}...")
            context_ex = context_batch[ex]  # [T, N, 1] or [T_cond, N, 1]
            target_ex = target_batch[ex]  # [T, N, 1]
            pred_ex = pred_batch[:, ex, ...]  # [n_samples, T, N, 1]
            plot_predictions(pred_ex, target_ex, context_ex, args.task, num_sensors=3)


if __name__ == "__main__":
    main()
