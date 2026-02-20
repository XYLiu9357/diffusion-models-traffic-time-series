"""
Training script for forecasting with the spatio‑temporal U‑Net.
"""

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
from models.unet import SpatioTemporalUNet
from training.trainer import DiffusionTrainer

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Paths
DATA_DIR = pathlib.Path("data")
GRAPH_FILE = DATA_DIR / "adj_METR-LA.pkl"
DATA_FILE = DATA_DIR / "METR-LA.h5"
PROCESSED_FILE = DATA_DIR / "processed_metr_la.npz"


def main():
    logger.info("=" * 60)
    logger.info("FORECASTING TRAINING (PHASE 1)")
    logger.info("=" * 60)

    # 1. Load raw data and align sensors
    loader = METRLoader(GRAPH_FILE, DATA_FILE)
    sensor_ids, adj_mx = loader.load_graph()
    data, timestamps, data_sensor_ids = loader.load_data()
    data_to_graph = loader.align_sensors()
    reordered_data = loader.prepare_for_modeling(data_to_graph)

    # 2. Impute missing values (if any)
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
        mode="forecast",
    )
    val_dataset = TrafficDataset(
        val_data,
        input_len=12,
        output_len=12,
        stride=1,
        mode="forecast",
    )

    # 6. Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=16,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=16,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )

    logger.info(f"Train loader: {len(train_loader)} batches per epoch")
    logger.info(f"Val loader: {len(val_loader)} batches")

    # 7. Initialize model
    model = SpatioTemporalUNet(
        adj_mx=adj_mx,
        in_channels=1,
        time_steps=12,
        num_sensors=207,
        hidden_dims=[64, 128, 256],
        use_gat=False,
        debug=False,  # set to True if you want detailed prints
    )

    # 8. Diffusion scheduler
    scheduler = DDPMScheduler(
        num_train_timesteps=1000,
        beta_schedule="linear",
        prediction_type="epsilon",
    )

    # 9. Trainer
    trainer = DiffusionTrainer(
        model=model,
        scheduler=scheduler,
        train_loader=train_loader,
        val_loader=val_loader,
        learning_rate=1e-4,
    )

    # 10. Quick test forward pass
    logger.info("Testing forward pass...")
    sample_batch = next(iter(train_loader))
    future = sample_batch["future"].to(trainer.device)
    t = torch.randint(0, 1000, (future.shape[0],), device=trainer.device)
    with torch.no_grad():
        noise = torch.randn_like(future)
        noisy = trainer.scheduler.add_noise(future, noise, t)
        pred = trainer.model(noisy, t)
    loss = F.mse_loss(pred, noise).item()
    logger.info(f"Forward pass successful! Loss: {loss:.6f}")

    # 11. Save processed data for later phases
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

    # 12. Quick training test (optional)
    logger.info("Running quick training test (5 epochs)...")
    trainer.train(num_epochs=5, validate_every=2)

    logger.info("=" * 60)
    logger.info("TRAINING COMPLETE")
    logger.info("=" * 60)

    return trainer


if __name__ == "__main__":
    trainer = main()
