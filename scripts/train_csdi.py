#!/usr/bin/env python
"""
Training script for CSDI model.
Supports both forecasting and imputation tasks.
"""

import argparse
import logging
import pathlib

import numpy as np
import torch
import torch.nn.functional as F
from diffusers import DDPMScheduler
from torch.utils.data import DataLoader

from data.dataset import TrafficDataset
from data.loader import METRLoader
from data.preprocessing import impute_missing, normalize, split_data
from models.csdi import CSDI
from models.unet import SpatioTemporalUNet
from training.csdi_forecast_trainer import CSDIForecastTrainer
from training.csdi_imputation_trainer import CSDIImputationTrainer

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Paths
DATA_DIR = pathlib.Path("data")
GRAPH_FILE = DATA_DIR / "adj_METR-LA.pkl"
DATA_FILE = DATA_DIR / "METR-LA.h5"
PROCESSED_FILE_FORECAST = DATA_DIR / "processed_metr_la_csdi.npz"
PROCESSED_FILE_IMPUTATION = DATA_DIR / "processed_metr_la_imputation.npz"


def main():
    parser = argparse.ArgumentParser(description="Train CSDI model.")
    parser.add_argument(
        "--task",
        type=str,
        choices=["forecast", "imputation"],
        default="forecast",
        help="Task to train for: forecast or imputation",
    )
    parser.add_argument(
        "--corruption_rate",
        type=float,
        default=0.15,
        help="Corruption rate for imputation (ignored for forecast)",
    )
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument(
        "--num_epochs", type=int, default=50, help="Number of training epochs"
    )
    args = parser.parse_args()

    # Set task-specific parameters
    if args.task == "forecast":
        mode = "forecast"
        include_cond = True
        corruption_rate = 0.0  # not used
        time_steps = 12
        processed_file = PROCESSED_FILE_FORECAST
        trainer_class = CSDIForecastTrainer
        model_save_name = "csdi_forecast.pth"
    else:  # imputation
        mode = "imputation"
        include_cond = False
        corruption_rate = args.corruption_rate
        time_steps = 24
        processed_file = PROCESSED_FILE_IMPUTATION
        trainer_class = CSDIImputationTrainer
        model_save_name = "csdi_imputation.pth"

    logger.info("=" * 60)
    logger.info(f"CSDI TRAINING FOR TASK: {args.task.upper()}")
    logger.info("=" * 60)

    # 1. Load and preprocess data
    loader = METRLoader(GRAPH_FILE, DATA_FILE)
    sensor_ids, adj_mx = loader.load_graph()
    data, timestamps, data_sensor_ids = loader.load_data()
    data_to_graph = loader.align_sensors()
    reordered_data = loader.prepare_for_modeling(data_to_graph)

    # 2. Impute missing (if any)
    reordered_data = impute_missing(reordered_data, method="mean")

    # 3. Normalize
    normalized_data, mean, std = normalize(
        reordered_data, method="zscore", return_params=True
    )

    # 4. Split chronologically
    train_data, val_data, test_data = split_data(
        normalized_data, train_ratio=0.7, val_ratio=0.1
    )

    # 5. Create datasets
    train_dataset = TrafficDataset(
        train_data,
        input_len=12,
        output_len=12,
        stride=1,
        mode=mode,
        include_cond=include_cond,
        corruption_rate=corruption_rate,
    )
    val_dataset = TrafficDataset(
        val_data,
        input_len=12,
        output_len=12,
        stride=1,
        mode=mode,
        include_cond=include_cond,
        corruption_rate=corruption_rate,
    )

    # 6. Data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )

    logger.info(f"Train loader: {len(train_loader)} batches/epoch")
    logger.info(f"Val loader: {len(val_loader)} batches")

    # 7. Create U-Net with appropriate time_steps
    unet = SpatioTemporalUNet(
        adj_mx=adj_mx,
        in_channels=3,  # noisy + cond + mask
        out_channels=1,
        time_steps=time_steps,
        num_sensors=207,
        hidden_dims=[64, 128, 256],
        use_gat=False,
        debug=False,
    )

    # 8. Wrap in CSDI
    model = CSDI(unet)

    # 9. Diffusion scheduler
    scheduler = DDPMScheduler(
        num_train_timesteps=1000,
        beta_schedule="linear",
        prediction_type="epsilon",
    )

    # 10. Trainer
    trainer = trainer_class(
        model=model,
        scheduler=scheduler,
        train_loader=train_loader,
        val_loader=val_loader,
        learning_rate=1e-4,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )

    # 11. Quick forward test
    logger.info("Testing forward pass...")
    sample_batch = next(iter(train_loader))
    if args.task == "forecast":
        future = sample_batch["future"].to(trainer.device)
        cond = sample_batch["cond"].to(trainer.device)
        mask = sample_batch["mask"].to(trainer.device)
        target = future
    else:  # imputation
        combined = sample_batch["combined"].to(trainer.device)
        corrupted = sample_batch["corrupted"].to(trainer.device)
        mask = sample_batch["mask"].to(trainer.device)
        target = combined
        cond = corrupted

    t = torch.randint(0, 1000, (target.shape[0],), device=trainer.device)
    with torch.no_grad():
        noise = torch.randn_like(target)
        noisy = trainer.scheduler.add_noise(target, noise, t)
        pred = model(noisy, t, cond, mask)
        loss = F.mse_loss(pred, noise, reduction="none")
        loss = (loss * (1 - mask)).mean()
    logger.info(f"Forward pass successful! Masked loss: {loss.item():.6f}")

    # 12. Save processed data for later use
    np.savez(
        processed_file,
        train_data=train_data,
        val_data=val_data,
        test_data=test_data,
        mean=mean,
        std=std,
        adj_mx=adj_mx,
        sensor_ids=sensor_ids,
        data_to_graph_idx=data_to_graph,
    )
    logger.info(f"Processed data saved to {processed_file}")

    # 13. Train
    logger.info("Starting training...")
    trainer.train(num_epochs=args.num_epochs, validate_every=5)

    logger.info("=" * 60)
    logger.info(f"CSDI {args.task.upper()} TRAINING COMPLETE")
    logger.info("=" * 60)

    # Save the trained model
    model_save_path = pathlib.Path(".") / model_save_name
    model_save_path.parent.mkdir(exist_ok=True)
    torch.save(trainer.model.state_dict(), model_save_path)
    logger.info(f"Model saved to {model_save_path}")
    return trainer


if __name__ == "__main__":
    trainer = main()
