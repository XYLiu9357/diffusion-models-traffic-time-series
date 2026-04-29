#!/usr/bin/env python
"""
Comprehensive evaluation script comparing baselines and trained models.
Usage:
    python scripts/compare_all.py --task forecast
    python scripts/compare_all.py --task imputation --corruption_rate 0.5 --seed 42
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
PROCESSED_FORECAST = DATA_DIR / "processed_metr_la_csdi.npz"
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
    loader = METRLoader(GRAPH_FILE, DATA_FILE)
    sensor_ids_full, adj_mx_full = loader.load_graph()
    data_raw, timestamps_raw, _ = loader.load_data()
    data_to_graph = loader.align_sensors()
    reordered_raw = loader.prepare_for_modeling(data_to_graph)
    reordered_raw = impute_missing(reordered_raw, method="mean")
    normalized_raw = (reordered_raw - mean) / std
    T = normalized_raw.shape[0]
    train_idx = int(T * 0.7)
    train_data_full = normalized_raw[:train_idx]
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
            include_cond=True,
        )
    else:  # imputation
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
# 2. Baseline implementations (unchanged)
# ----------------------------------------------------------------------
def persistence_forecast(batch):
    past = batch["past"]
    last_obs = past[:, -1:, :, :]
    B, T_out, N, F = batch["future"].shape
    return last_obs.expand(-1, T_out, -1, -1)


def historical_average_forecast(batch, hist_avg, timestamps_test):
    return torch.zeros_like(batch["future"])


def linear_regression_forecast(batch, lr_models):
    past = batch["past"].cpu().numpy()
    B, T_in, N, F = past.shape
    pred = np.zeros((B, 12, N, F))
    for n in range(N):
        model = lr_models[n]
        X = past[:, :, n, 0]
        pred_n = model.predict(X)
        pred[:, :, n, 0] = pred_n
    return torch.from_numpy(pred).float()


# ------------------------------------------------------------------
# Fixed mean imputation: fill missing with 0 (normalized mean)
# ------------------------------------------------------------------
def mean_imputation(batch):
    corrupted = batch["corrupted"]
    mask = batch["mask"]
    imputed = corrupted.clone()
    imputed[mask == 0] = 0.0  # zero is the mean of normalized data
    return imputed


def forward_fill_imputation(batch):
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
                        next_val = None
                        for tt in range(t + 1, T):
                            if mask[b, tt, n, 0] == 1:
                                next_val = corrupted[b, tt, n, 0]
                                break
                        if next_val is not None:
                            imputed[b, t, n, 0] = next_val
    return torch.from_numpy(imputed).float()


def linear_interpolation_imputation(batch):
    corrupted = batch["corrupted"].cpu().numpy()
    mask = batch["mask"].cpu().numpy()
    B, T, N, F = corrupted.shape
    imputed = corrupted.copy()
    for b in range(B):
        for n in range(N):
            seq = corrupted[b, :, n, 0]
            m = mask[b, :, n, 0]
            obs_idx = np.where(m == 1)[0]
            if len(obs_idx) == 0:
                continue
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
                left_idx = s - 1
                right_idx = e
                while left_idx >= 0 and m[left_idx] == 0:
                    left_idx -= 1
                while right_idx < T and m[right_idx] == 0:
                    right_idx += 1
                if left_idx >= 0 and right_idx < T:
                    left_val = seq[left_idx]
                    right_val = seq[right_idx]
                    for t in range(s, e):
                        alpha = (t - left_idx) / (right_idx - left_idx)
                        imputed[b, t, n, 0] = left_val * (1 - alpha) + right_val * alpha
                elif left_idx >= 0:
                    for t in range(s, e):
                        imputed[b, t, n, 0] = seq[left_idx]
                elif right_idx < T:
                    for t in range(s, e):
                        imputed[b, t, n, 0] = seq[right_idx]
    return torch.from_numpy(imputed).float()


# ----------------------------------------------------------------------
# 3. Evaluation loops (unchanged)
# ----------------------------------------------------------------------
def evaluate_forecast_method(method_fn, test_loader, device, **kwargs):
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
    all_mae, all_rmse = [], []
    for batch in tqdm(test_loader, desc="Evaluating"):
        target = batch["combined"].to(device)
        mask = batch["mask"].to(device)
        pred = method_fn(batch, **kwargs)
        if isinstance(pred, torch.Tensor):
            pred = pred.to(device)
        else:
            pred = torch.from_numpy(pred).float().to(device)
        error = (pred - target).abs()
        mae = (error * (1 - mask)).sum() / (1 - mask).sum().clamp(min=1)
        se = (pred - target) ** 2
        rmse = torch.sqrt((se * (1 - mask)).sum() / (1 - mask).sum().clamp(min=1))
        all_mae.append(mae.item())
        all_rmse.append(rmse.item())
    return np.mean(all_mae), np.mean(all_rmse)


# ----------------------------------------------------------------------
# 4. Model loading (unchanged)
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
    # New argument for corruption rate
    parser.add_argument(
        "--corruption_rate",
        type=float,
        default=0.15,
        help="Corruption rate for imputation (only used if task=imputation)",
    )
    args = parser.parse_args()

    # Load test data and additional info
    test_data, mean, std, adj_mx, sensor_ids, train_data_full, timestamps_train = (
        load_test_data(args.task)
    )
    avg_std = np.mean(std)
    logger.info(f"Test data shape: {test_data.shape}, avg std = {avg_std:.2f} mph")

    # Create test loader, passing the corruption rate
    test_loader, dataset = create_test_loader(
        test_data,
        args.task,
        batch_size=args.batch_size,
        corruption_rate=args.corruption_rate,
        seed=args.seed,
    )
    logger.info(f"Number of test windows: {len(test_loader.dataset)}")
    if args.task == "imputation":
        logger.info(f"Using corruption rate: {args.corruption_rate}")

    results = {}

    if args.task == "forecast":
        # ----- Baselines -----
        logger.info("Evaluating persistence...")
        mae, rmse = evaluate_forecast_method(persistence_forecast, test_loader, DEVICE)
        results["Persistence"] = (mae, rmse)

        # Historical average (requires time alignment; not implemented here)
        # logger.info("Evaluating historical average...")
        # mae, rmse = evaluate_forecast_method(historical_average_forecast, test_loader, DEVICE, ...)
        # results["Historical avg"] = (mae, rmse)

        # Linear regression (needs training on training set; not implemented here)
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
            # samples shape: [1, B, T_out, N, 1] → remove sample dimension
            return samples[0]

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
            mask = batch["mask"].to(DEVICE)  # all zeros for forecast
            with torch.no_grad():
                samples = impute_csdi(
                    csdi_forecast, scheduler_csdi, cond, mask, num_steps=50
                )
            # impute_csdi returns [n_samples, B, T, N, F] with n_samples=1
            return samples.squeeze(0)

        logger.info("Evaluating CSDI forecast...")
        mae, rmse = evaluate_forecast_method(csdi_forecast_predict, test_loader, DEVICE)
        results["CSDI (forecast)"] = (mae, rmse)

    else:  # imputation
        logger.info("Evaluating mean imputation...")
        mae, rmse = evaluate_imputation_method(mean_imputation, test_loader, DEVICE)
        results["Mean imputation"] = (mae, rmse)

        # ------------------------------------------------------------------
        # Forward fill and linear interpolation remain unchanged
        # ------------------------------------------------------------------
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

        # ------------------------------------------------------------------
        # CSDI imputation – fix shape mismatch
        # ------------------------------------------------------------------
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
            # impute_csdi returns [n_samples, B, T, N, F] with n_samples=1 by default
            return samples.squeeze(0)  # remove the sample dimension → [B, T, N, F]

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

    if args.plot and args.task == "imputation":
        # (plotting code unchanged; you can add it as in your original)
        pass


if __name__ == "__main__":
    main()
