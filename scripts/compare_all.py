#!/usr/bin/env python
"""
Comprehensive evaluation script comparing baselines and trained models.
Usage:
    python scripts/compare_all.py --task forecast
    python scripts/compare_all.py --task imputation --seed 42
"""

import argparse
import logging
import pathlib
import pickle
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from diffusers import DDIMScheduler, DDPMScheduler
from sklearn.linear_model import LinearRegression
from torch.utils.data import DataLoader
from tqdm import tqdm

# Import your modules
from data.dataset import TrafficDataset
from data.loader import METRLoader
from data.preprocessing import impute_missing, normalize, split_data
from models.csdi import CSDI
from models.timegrad import TimeGrad
from models.unet import SpatioTemporalUNet
from utils.samplers import forecast_timegrad, impute_csdi

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

DATA_DIR = pathlib.Path("data")
GRAPH_FILE = DATA_DIR / "adj_METR-LA.pkl"
DATA_FILE = DATA_DIR / "METR-LA.h5"
PROCESSED_FORECAST = (
    DATA_DIR / "processed_metr_la_csdi.npz"
)  # for forecast (or timegrad)
PROCESSED_IMPUTATION = DATA_DIR / "processed_metr_la_imputation.npz"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ----------------------------------------------------------------------
# 1. Data loading helpers
# ----------------------------------------------------------------------
def load_test_data(task, processed_file=None):
    """Load test data and normalization parameters."""
    if processed_file is None:
        processed_file = (
            PROCESSED_FORECAST if task == "forecast" else PROCESSED_IMPUTATION
        )
    data = np.load(processed_file)
    test_data = data["test_data"]  # [T_test, N, 1]
    mean = data["mean"]  # [1, N, 1]
    std = data["std"]  # [1, N, 1]
    adj_mx = data["adj_mx"]
    sensor_ids = data["sensor_ids"]
    # Also load training data to compute historical averages and regression
    # We'll reload raw data and split again
    loader = METRLoader(GRAPH_FILE, DATA_FILE)
    sensor_ids_full, adj_mx_full = loader.load_graph()
    data_raw, timestamps_raw, _ = loader.load_data()
    data_to_graph = loader.align_sensors()
    reordered_raw = loader.prepare_for_modeling(data_to_graph)
    reordered_raw = impute_missing(reordered_raw, method="mean")
    normalized_raw, _, _ = normalize(
        reordered_raw, method="zscore", return_params=True
    )  # use same mean/std? No, we need raw normalized with same stats? Actually we need normalized data with the same mean/std as test. We'll use the same normalization parameters (mean, std) from the saved file, but those were computed on the whole data before split. So we should apply them to the raw data.
    # Instead, we can just use the normalized data we already have from the npz? But that's only test. We need train data normalized with same stats.
    # We'll recompute using the saved mean/std.
    # We have mean, std from the npz, which were computed on the whole dataset before split. So we can normalize the raw data using those.
    normalized_raw = (reordered_raw - mean) / std
    # Now split again to get train indices
    T = normalized_raw.shape[0]
    train_idx = int(T * 0.7)
    val_idx = int(T * 0.8)
    train_data_full = normalized_raw[:train_idx]
    # Also get timestamps for train
    timestamps_train = timestamps_raw[:train_idx]
    return test_data, mean, std, adj_mx, sensor_ids, train_data_full, timestamps_train


def create_test_loader(test_data, task, batch_size=16, corruption_rate=0.15, seed=42):
    """Create test DataLoader with fixed seed for reproducibility (imputation masks)."""
    if task == "forecast":
        dataset = TrafficDataset(
            test_data,
            input_len=12,
            output_len=12,
            stride=1,
            mode="forecast",
            include_cond=True,  # for CSDI forecasting; TimeGrad will ignore cond
        )
    else:  # imputation
        # Set seed to ensure same masks across all evaluations
        torch.manual_seed(seed)
        dataset = TrafficDataset(
            test_data,
            input_len=12,
            output_len=12,
            stride=1,
            mode="imputation",
            corruption_rate=corruption_rate,
            include_cond=False,
        )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    return loader, dataset


# ----------------------------------------------------------------------
# 2. Baseline implementations
# ----------------------------------------------------------------------
def persistence_forecast(batch):
    """Repeat last observed value for all future steps."""
    past = batch["past"]  # [B, T_in, N, F]
    last_obs = past[:, -1:, :, :]  # [B, 1, N, F]
    B, T_out, N, F = batch["future"].shape
    return last_obs.expand(-1, T_out, -1, -1)  # [B, T_out, N, F]


def historical_average_forecast(batch, hist_avg, timestamps_test):
    """
    hist_avg: tensor of shape [288, N, 1] (288 = 24h * 12 steps/h)
    timestamps_test: list of datetime strings for the entire test set (needed to map windows)
    This is simplified: we need to know the time-of-day for each window's future steps.
    A more practical approach: for each test window, we know its start time, then compute indices.
    We'll implement a version that requires passing the test dataset with time info.
    For now, we'll skip detailed implementation and note that it requires time alignment.
    """
    # Placeholder: return zeros
    return torch.zeros_like(batch["future"])


def linear_regression_forecast(batch, lr_models):
    """
    lr_models: list of 207 sklearn LinearRegression objects.
    past: [B, T_in, N, F]
    returns: [B, T_out, N, F]
    """
    past = batch["past"].cpu().numpy()  # [B, T_in, N, 1]
    B, T_in, N, F = past.shape
    pred = np.zeros((B, 12, N, F))
    for n in range(N):
        model = lr_models[n]
        # past for this sensor: [B, T_in]
        X = past[:, :, n, 0]  # [B, T_in]
        pred_n = model.predict(X)  # [B, 12]
        pred[:, :, n, 0] = pred_n
    return torch.from_numpy(pred).float()


def mean_imputation(batch, sensor_means):
    """Replace missing with per-sensor mean."""
    corrupted = batch["corrupted"]
    mask = batch["mask"]
    sensor_means = sensor_means.view(1, 1, -1, 1)  # [1,1,N,1]
    imputed = corrupted.clone()
    imputed[mask == 0] = sensor_means.expand_as(corrupted)[mask == 0]
    return imputed


def forward_fill_imputation(batch):
    """Forward fill along time per sensor (simplified loop)."""
    corrupted = batch["corrupted"].cpu().numpy()
    mask = batch["mask"].cpu().numpy()
    B, T, N, F = corrupted.shape
    imputed = corrupted.copy()
    for b in range(B):
        for n in range(N):
            last_val = None
            for t in range(T):
                if mask[b, t, n, 0] == 1:
                    last_val = corrupted[b, t, n, 0]
                else:
                    if last_val is not None:
                        imputed[b, t, n, 0] = last_val
                    else:
                        # if no previous observed, use next observed (backward fill)
                        # find next observed
                        next_val = None
                        for tt in range(t + 1, T):
                            if mask[b, tt, n, 0] == 1:
                                next_val = corrupted[b, tt, n, 0]
                                break
                        if next_val is not None:
                            imputed[b, t, n, 0] = next_val
                        # else leave as is (will be handled by mean later? we'll keep as is)
    return torch.from_numpy(imputed).float()


def linear_interpolation_imputation(batch):
    """Linear interpolation between nearest observed values."""
    corrupted = batch["corrupted"].cpu().numpy()
    mask = batch["mask"].cpu().numpy()
    B, T, N, F = corrupted.shape
    imputed = corrupted.copy()
    for b in range(B):
        for n in range(N):
            seq = corrupted[b, :, n, 0]
            m = mask[b, :, n, 0]
            # find indices of observed
            obs_idx = np.where(m == 1)[0]
            if len(obs_idx) == 0:
                continue  # nothing to interpolate
            # for each missing segment, interpolate
            missing_segments = []
            start = None
            for t in range(T):
                if m[t] == 0 and start is None:
                    start = t
                elif (m[t] == 1 or t == T - 1) and start is not None:
                    end = t if m[t] == 1 else t + 1
                    missing_segments.append((start, end))
                    start = None
            for s, e in missing_segments:
                # find left and right observed
                left_idx = s - 1
                right_idx = e
                while left_idx >= 0 and m[left_idx] == 0:
                    left_idx -= 1
                while right_idx < T and m[right_idx] == 0:
                    right_idx += 1
                if left_idx >= 0 and right_idx < T:
                    # linear interpolation
                    left_val = seq[left_idx]
                    right_val = seq[right_idx]
                    for t in range(s, e):
                        alpha = (t - left_idx) / (right_idx - left_idx)
                        imputed[b, t, n, 0] = left_val * (1 - alpha) + right_val * alpha
                elif left_idx >= 0:
                    # only left observed: forward fill
                    for t in range(s, e):
                        imputed[b, t, n, 0] = seq[left_idx]
                elif right_idx < T:
                    # only right observed: backward fill
                    for t in range(s, e):
                        imputed[b, t, n, 0] = seq[right_idx]
    return torch.from_numpy(imputed).float()


# ----------------------------------------------------------------------
# 3. Evaluation loop
# ----------------------------------------------------------------------
def evaluate_forecast_method(method_fn, test_loader, device, **kwargs):
    """Generic evaluation for forecasting methods (returns MAE, RMSE normalized)."""
    all_mae, all_rmse = [], []
    for batch in tqdm(test_loader, desc="Evaluating"):
        target = batch["future"].to(device)
        pred = method_fn(batch, **kwargs)
        if isinstance(pred, torch.Tensor):
            pred = pred.to(device)
        else:
            pred = torch.from_numpy(pred).float().to(device)
        mae = F.l1_loss(pred, target).item()
        rmse = torch.sqrt(F.mse_loss(pred, target)).item()
        all_mae.append(mae)
        all_rmse.append(rmse)
    return np.mean(all_mae), np.mean(all_rmse)


def evaluate_imputation_method(method_fn, test_loader, device, **kwargs):
    """Generic evaluation for imputation methods (masked loss)."""
    all_mae, all_rmse = [], []
    for batch in tqdm(test_loader, desc="Evaluating"):
        target = batch["combined"].to(device)
        mask = batch["mask"].to(device)  # 1=observed, 0=missing
        pred = method_fn(batch, **kwargs)
        if isinstance(pred, torch.Tensor):
            pred = pred.to(device)
        else:
            pred = torch.from_numpy(pred).float().to(device)
        # Compute error only on missing positions
        error = (pred - target).abs()
        mae = (error * (1 - mask)).sum() / (1 - mask).sum().clamp(min=1)
        se = (pred - target) ** 2
        rmse = torch.sqrt((se * (1 - mask)).sum() / (1 - mask).sum().clamp(min=1))
        all_mae.append(mae.item())
        all_rmse.append(rmse.item())
    return np.mean(all_mae), np.mean(all_rmse)


# ----------------------------------------------------------------------
# 4. Model loading
# ----------------------------------------------------------------------
def load_csdi_forecast(adj_mx, checkpoint_path):
    unet = SpatioTemporalUNet(
        adj_mx=adj_mx,
        in_channels=3,
        out_channels=1,
        time_steps=12,
        num_sensors=adj_mx.shape[0],
        hidden_dims=[64, 128, 256],
        use_gat=False,
        debug=False,
    )
    model = CSDI(unet)
    model.load_state_dict(torch.load(checkpoint_path, map_location="cpu"))
    model.to(DEVICE)
    model.eval()
    return model


def load_csdi_imputation(adj_mx, checkpoint_path):
    unet = SpatioTemporalUNet(
        adj_mx=adj_mx,
        in_channels=3,
        out_channels=1,
        time_steps=24,
        num_sensors=adj_mx.shape[0],
        hidden_dims=[64, 128, 256],
        use_gat=False,
        debug=False,
    )
    model = CSDI(unet)
    model.load_state_dict(torch.load(checkpoint_path, map_location="cpu"))
    model.to(DEVICE)
    model.eval()
    return model


def load_timegrad(adj_mx, checkpoint_path):
    model = TimeGrad(
        num_sensors=adj_mx.shape[0], hidden_dim=64, rnn_layers=1, input_dim=1
    )
    model.load_state_dict(torch.load(checkpoint_path, map_location="cpu"))
    model.to(DEVICE)
    model.eval()
    return model


# ----------------------------------------------------------------------
# 5. Main
# ----------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--task", type=str, choices=["forecast", "imputation"], required=True
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for imputation masks"
    )
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--plot", action="store_true", help="Plot example results")
    parser.add_argument("--csdi_forecast_ckpt", type=str, default="csdi_forecast.pth")
    parser.add_argument(
        "--csdi_imputation_ckpt", type=str, default="csdi_imputation.pth"
    )
    parser.add_argument("--timegrad_ckpt", type=str, default="timegrad_forecast.pth")
    args = parser.parse_args()

    # Load test data and additional info
    test_data, mean, std, adj_mx, sensor_ids, train_data_full, timestamps_train = (
        load_test_data(args.task)
    )
    avg_std = np.mean(std)
    logger.info(f"Test data shape: {test_data.shape}, avg std = {avg_std:.2f} mph")

    # Create test loader (with fixed seed for imputation)
    test_loader, dataset = create_test_loader(
        test_data, args.task, batch_size=args.batch_size, seed=args.seed
    )
    logger.info(f"Number of test windows: {len(test_loader.dataset)}")

    results = {}  # dict of method name -> (mae_norm, rmse_norm)

    if args.task == "forecast":
        # ----- Baselines -----
        logger.info("Evaluating persistence...")
        mae, rmse = evaluate_forecast_method(persistence_forecast, test_loader, DEVICE)
        results["Persistence"] = (mae, rmse)

        # Historical average (requires time alignment; simplified: skip for now)
        # logger.info("Evaluating historical average...")
        # mae, rmse = evaluate_forecast_method(historical_average_forecast, test_loader, DEVICE, ...)
        # results["Historical avg"] = (mae, rmse)

        # Linear regression (need to train on training data first)
        # For simplicity, we'll skip here or you can pre-train and load coefficients.
        # We'll add a placeholder.
        logger.info("Linear regression not implemented in this script; skipping.")

        # ----- Models -----
        # TimeGrad
        logger.info("Loading TimeGrad...")
        timegrad_model = load_timegrad(adj_mx, args.timegrad_ckpt)
        scheduler = DDIMScheduler(num_train_timesteps=1000, beta_schedule="linear")
        scheduler.set_timesteps(50)

        def timegrad_predict(batch):
            past = batch["past"].to(DEVICE)
            with torch.no_grad():
                samples = forecast_timegrad(
                    timegrad_model,
                    scheduler,
                    past,
                    T_out=12,
                    num_inference_steps=50,
                    num_samples=1,
                )
            return samples[0]  # take first sample

        logger.info("Evaluating TimeGrad...")
        mae, rmse = evaluate_forecast_method(timegrad_predict, test_loader, DEVICE)
        results["TimeGrad"] = (mae, rmse)

        # CSDI forecasting
        logger.info("Loading CSDI forecasting model...")
        csdi_forecast = load_csdi_forecast(adj_mx, args.csdi_forecast_ckpt)
        scheduler_csdi = DDIMScheduler(num_train_timesteps=1000, beta_schedule="linear")
        scheduler_csdi.set_timesteps(50)

        def csdi_forecast_predict(batch):
            cond = batch["cond"].to(DEVICE)
            mask = batch["mask"].to(DEVICE)  # all zeros
            with torch.no_grad():
                # Use impute_csdi but with all zeros mask? Actually we have forecast sampling function.
                # We'll reuse forecast_timegrad? No, CSDI uses different sampling.
                # We'll need a separate function. For now, we can use impute_csdi with mask=zeros (all missing)
                samples = impute_csdi(
                    csdi_forecast, scheduler_csdi, cond, mask, num_steps=50
                )
            # Remove extra leading dimension if present
            if samples.dim() == 5 and samples.size(0) == 1:
                samples = samples.squeeze(0)
            return samples

        logger.info("Evaluating CSDI forecast...")
        mae, rmse = evaluate_forecast_method(csdi_forecast_predict, test_loader, DEVICE)
        results["CSDI (forecast)"] = (mae, rmse)

    else:  # imputation
        # ----- Baselines -----
        sensor_means = torch.from_numpy(
            mean.squeeze(0)
        ).float()  # [N,1] after removing time dim? mean is [1,N,1] so squeeze first dim.

        logger.info("Evaluating mean imputation...")
        mae, rmse = evaluate_imputation_method(
            lambda b: mean_imputation(b, sensor_means), test_loader, DEVICE
        )
        results["Mean imputation"] = (mae, rmse)

        logger.info("Evaluating forward fill...")
        mae, rmse = evaluate_imputation_method(
            forward_fill_imputation, test_loader, DEVICE
        )
        results["Forward fill"] = (mae, rmse)

        logger.info("Evaluating linear interpolation...")
        mae, rmse = evaluate_imputation_method(
            linear_interpolation_imputation, test_loader, DEVICE
        )
        results["Linear interpolation"] = (mae, rmse)

        # ----- Models -----
        logger.info("Loading CSDI imputation model...")
        csdi_impute = load_csdi_imputation(adj_mx, args.csdi_imputation_ckpt)
        scheduler = DDIMScheduler(num_train_timesteps=1000, beta_schedule="linear")
        scheduler.set_timesteps(50)

        def csdi_impute_predict(batch):
            corrupted = batch["corrupted"].to(DEVICE)
            mask = batch["mask"].to(DEVICE)
            with torch.no_grad():
                samples = impute_csdi(
                    csdi_impute, scheduler, corrupted, mask, num_steps=50
                )
            return samples

        logger.info("Evaluating CSDI imputation...")
        mae, rmse = evaluate_imputation_method(csdi_impute_predict, test_loader, DEVICE)
        results["CSDI (imputation)"] = (mae, rmse)

    # Print results
    logger.info("=" * 60)
    logger.info(f"Results for {args.task} task:")
    for name, (mae_norm, rmse_norm) in results.items():
        logger.info(
            f"{name:25s} MAE: {mae_norm:.4f} (norm), RMSE: {rmse_norm:.4f} (norm)"
        )
        logger.info(
            f"{name:25s} MAE: {mae_norm * avg_std:.2f} mph, RMSE: {rmse_norm * avg_std:.2f} mph"
        )
    logger.info("=" * 60)

    # Optionally plot a few examples (for the best method or all)
    if args.plot and args.task == "imputation":
        # Plot first batch
        batch = next(iter(test_loader))
        corrupted = batch["corrupted"][:2].cpu()
        target = batch["combined"][:2].cpu()
        mask = batch["mask"][:2].cpu()
        # Get predictions from CSDI
        csdi_impute_predict(batch)  # we already have function
        # etc. (plotting code similar to evaluate scripts)


if __name__ == "__main__":
    main()
