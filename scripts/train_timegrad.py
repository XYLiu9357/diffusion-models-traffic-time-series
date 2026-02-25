#!/usr/bin/env python
"""
Training script for TimeGrad forecasting model.
"""

import argparse
import logging
import pathlib

import numpy as np
import torch
from diffusers import DDPMScheduler
from torch.utils.data import DataLoader

from data.dataset import TrafficDataset
from data.loader import METRLoader
from data.preprocessing import impute_missing, normalize, split_data
from models.timegrad import TimeGrad
from training.timegrad_trainer import TimeGradTrainer

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

DATA_DIR = pathlib.Path("data")
GRAPH_FILE = DATA_DIR / "adj_METR-LA.pkl"
DATA_FILE = DATA_DIR / "METR-LA.h5"
PROCESSED_FILE = DATA_DIR / "processed_metr_la_timegrad.npz"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_epochs", type=int, default=50)
    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument("--rnn_layers", type=int, default=1)
    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("TIMEGRAD TRAINING")
    logger.info("=" * 60)

    # 1. Load and preprocess data (same as before)
    loader = METRLoader(GRAPH_FILE, DATA_FILE)
    sensor_ids, adj_mx = loader.load_graph()
    data, timestamps, data_sensor_ids = loader.load_data()
    data_to_graph = loader.align_sensors()
    reordered_data = loader.prepare_for_modeling(data_to_graph)

    reordered_data = impute_missing(reordered_data, method="mean")
    normalized_data, mean, std = normalize(
        reordered_data, method="zscore", return_params=True
    )
    train_data, val_data, test_data = split_data(
        normalized_data, train_ratio=0.7, val_ratio=0.1
    )

    # 2. Create datasets (forecast mode, no cond needed)
    train_dataset = TrafficDataset(
        train_data,
        input_len=12,
        output_len=12,
        stride=1,
        mode="forecast",
        include_cond=False,
    )
    val_dataset = TrafficDataset(
        val_data,
        input_len=12,
        output_len=12,
        stride=1,
        mode="forecast",
        include_cond=False,
    )

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

    # 3. Create model
    model = TimeGrad(
        num_sensors=207,
        hidden_dim=args.hidden_dim,
        rnn_layers=args.rnn_layers,
        input_dim=1,
    )

    # 4. Scheduler
    scheduler = DDPMScheduler(
        num_train_timesteps=1000, beta_schedule="linear", prediction_type="epsilon"
    )

    # 5. Trainer
    trainer = TimeGradTrainer(
        model=model,
        scheduler=scheduler,
        train_loader=train_loader,
        val_loader=val_loader,
        learning_rate=1e-4,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )

    # 6. Quick test
    logger.info("Testing forward pass...")
    sample_batch = next(iter(train_loader))
    past = sample_batch["past"].to(trainer.device)
    future = sample_batch["future"].to(trainer.device)
    # Run one training step manually? We'll rely on trainer's validation.
    # For a quick test, we can run a single step.
    t = torch.randint(0, 1000, (past.shape[0],), device=trainer.device)
    # We'll just check that model.encode_past works
    h = model.encode_past(past)
    logger.info(f"Past encoding shape: {h.shape}")  # Should be [batch*207, hidden_dim]

    # 7. Save processed data
    np.savez(
        PROCESSED_FILE,
        train_data=train_data,
        val_data=val_data,
        test_data=test_data,
        mean=mean,
        std=std,
        adj_mx=adj_mx,
        sensor_ids=sensor_ids,
        data_to_graph_idx=data_to_graph,
    )
    logger.info(f"Processed data saved to {PROCESSED_FILE}")

    # 8. Train
    logger.info("Starting training...")
    trainer.train(num_epochs=args.num_epochs, validate_every=5)

    # 9. Save model
    model_save_path = pathlib.Path(".") / "timegrad_forecast.pth"
    torch.save(trainer.model.state_dict(), model_save_path)
    logger.info(f"Model saved to {model_save_path}")

    return trainer


if __name__ == "__main__":
    trainer = main()
