"""
Phase 1: Foundation & Data Pipeline for METR-LA Traffic Diffusion Models
Author: Your Name
Date: February 2026

This module implements:
1. Data loading and preprocessing for METR-LA
2. Sensor alignment between graph and data
3. Sliding window dataset creation
4. Shared U-Net backbone with graph convolutions
5. Basic training utilities
"""

import logging
import pathlib
import pickle
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Union

import h5py
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers import DDPMScheduler, UNet2DModel
from torch.utils.data import DataLoader, Dataset
from torch_geometric.nn import GATConv, GCNConv
from torch_geometric.utils import from_networkx

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


# ============================================================================
# Part 1: Data Loading and Preprocessing
# ============================================================================


class METRLoader:
    """Load and preprocess METR-LA traffic data."""

    def __init__(self, graph_path: pathlib.Path, data_path: pathlib.Path):
        self.graph_path = graph_path
        self.data_path = data_path
        self.sensor_ids = None
        self.adj_mx = None
        self.data = None
        self.timestamps = None
        self.data_sensor_ids = None

    def load_graph(self) -> Tuple[List[str], np.ndarray]:
        """Load graph adjacency matrix and sensor IDs."""
        with open(self.graph_path, "rb") as f:
            sensor_ids, sensor_id_to_ind, adj_mx = pickle.load(f, encoding="latin1")

        self.sensor_ids = sensor_ids
        self.adj_mx = adj_mx

        logger.info(f"Graph loaded: {len(sensor_ids)} sensors")
        logger.info(f"Adjacency matrix shape: {adj_mx.shape}")
        logger.info(f"Sparsity: {(adj_mx == 0).sum() / adj_mx.size * 100:.1f}% zeros")
        logger.info(f"Weight range: [{adj_mx.min():.3f}, {adj_mx.max():.3f}]")

        return sensor_ids, adj_mx

    def load_data(self) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """Load traffic speed data from HDF5."""
        with h5py.File(self.data_path, "r") as f:
            # METR-LA is stored as pandas DataFrame in HDF5
            df_group = f["df"]

            # Extract data
            speeds = df_group["block0_values"][:]  # [34272, 207]
            sensor_ids_bytes = df_group["axis0"][:]  # [207]
            timestamp_ns = df_group["axis1"][:]  # [34272]

        # Convert sensor IDs from bytes to strings
        self.data_sensor_ids = [sid.decode("utf-8") for sid in sensor_ids_bytes]

        # Convert nanosecond timestamps to datetime strings
        timestamps = []
        for ns in timestamp_ns:
            dt = datetime(1970, 1, 1) + timedelta(seconds=ns / 1e9)
            timestamps.append(dt.strftime("%Y-%m-%d %H:%M:%S"))
        self.timestamps = np.array(timestamps)

        # Add feature dimension: [T, N] -> [T, N, 1]
        self.data = speeds.reshape(speeds.shape[0], speeds.shape[1], 1)

        logger.info(f"Data loaded: {self.data.shape} (timesteps, sensors, features)")
        logger.info(f"Time range: {self.timestamps[0]} to {self.timestamps[-1]}")
        logger.info(
            f"Speed range: [{np.nanmin(self.data):.1f}, {np.nanmax(self.data):.1f}] mph"
        )
        logger.info(
            f"Missing values: {np.isnan(self.data).sum() / self.data.size * 100:.2f}%"
        )

        return self.data, self.timestamps, self.data_sensor_ids

    def align_sensors(self) -> np.ndarray:
        """
        Create mapping from data indices to graph indices.

        Returns:
            data_to_graph_idx: Array of shape [207] where value at position i
                               gives the corresponding graph index for data sensor i.
        """
        if self.sensor_ids is None or self.data_sensor_ids is None:
            raise ValueError(
                "Load graph and data first using load_graph() and load_data()"
            )

        # Create mapping from sensor ID to graph index
        id_to_graph = {sid: idx for idx, sid in enumerate(self.sensor_ids)}

        # Map each data sensor to its graph index
        data_to_graph = []
        missing_sensors = []

        for i, sid in enumerate(self.data_sensor_ids):
            if sid in id_to_graph:
                data_to_graph.append(id_to_graph[sid])
            else:
                missing_sensors.append(sid)
                data_to_graph.append(-1)  # Placeholder for missing

        data_to_graph = np.array(data_to_graph)

        if missing_sensors:
            logger.warning(f"Found {len(missing_sensors)} sensors in data not in graph")
            logger.warning(f"First few missing: {missing_sensors[:5]}")

        logger.info(
            f"Sensor alignment complete. Valid mappings: {(data_to_graph >= 0).sum()}/{len(data_to_graph)}"
        )

        return data_to_graph

    def prepare_for_modeling(self, data_to_graph_idx: np.ndarray):
        """
        Reorder data sensors to match graph order for consistent graph convolutions.

        Args:
            data_to_graph_idx: Mapping from data indices to graph indices

        Returns:
            reordered_data: Data with sensors ordered according to graph
        """
        # Create inverse mapping: graph index -> data index
        graph_to_data = {}
        for data_idx, graph_idx in enumerate(data_to_graph_idx):
            if graph_idx >= 0:
                graph_to_data[graph_idx] = data_idx

        # Reorder data along sensor dimension
        reordered_data = np.zeros_like(self.data)
        valid_sensors = 0

        for graph_idx in range(len(self.sensor_ids)):
            if graph_idx in graph_to_data:
                data_idx = graph_to_data[graph_idx]
                reordered_data[:, graph_idx, :] = self.data[:, data_idx, :]
                valid_sensors += 1
            else:
                # Fill with NaN for missing sensors (should not happen in METR-LA)
                reordered_data[:, graph_idx, :] = np.nan

        logger.info(
            f"Data reordered: {valid_sensors}/{len(self.sensor_ids)} sensors aligned"
        )

        return reordered_data


# ============================================================================
# Part 2: PyTorch Dataset with Sliding Windows
# ============================================================================


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


# ============================================================================
# Part 3: Graph-Enhanced U-Net (Shared Backbone)
# ============================================================================


class GraphConvBlock(nn.Module):
    """Graph convolution block for spatial processing."""

    def __init__(self, in_channels, out_channels, adj_mx, use_gat=False):
        super().__init__()
        self.adj_mx = adj_mx
        self.use_gat = use_gat

        # Convert adjacency matrix to edge index for PyTorch Geometric
        self.edge_index = self._adj_to_edge_index(adj_mx)

        if use_gat:
            self.conv = GATConv(in_channels, out_channels, heads=4, concat=False)
        else:
            self.conv = GCNConv(in_channels, out_channels)

        self.norm = nn.LayerNorm(out_channels)
        self.activation = nn.ReLU()

    def _adj_to_edge_index(self, adj_mx):
        """Convert dense adjacency to edge_index for PyG."""
        adj = torch.FloatTensor(adj_mx)
        edge_index = adj.nonzero().t().contiguous()
        return edge_index

    def forward(self, x):
        """
        Args:
            x: [batch, time, nodes, channels] or [batch*nodes, channels] after reshape
        """
        batch_size, time_steps, num_nodes, channels = x.shape

        # Reshape to [batch*time, nodes, channels] for graph conv
        x_reshaped = x.permute(
            0, 1, 3, 2
        ).contiguous()  # [batch, time, channels, nodes]
        x_reshaped = x_reshaped.view(
            -1, channels, num_nodes
        )  # [batch*time, channels, nodes]
        x_reshaped = x_reshaped.permute(0, 2, 1)  # [batch*time, nodes, channels]

        # Apply graph convolution
        out = self.conv(x_reshaped, self.edge_index.to(x.device))

        # Reshape back
        out = out.view(batch_size, time_steps, num_nodes, -1)
        out = self.norm(out)
        out = self.activation(out)

        return out


class SpatioTemporalUNet(nn.Module):
    """
    Shared U-Net backbone with:
    - Temporal convolutions (along time dimension)
    - Graph convolutions (along sensor dimension)
    - Skip connections for U-Net structure
    """

    def __init__(
        self,
        adj_mx: np.ndarray,  # Adjacency matrix [N, N]
        in_channels: int = 1,  # Traffic speed (1 feature)
        time_steps: int = 12,  # Input time steps
        num_sensors: int = 207,  # Number of sensors
        hidden_dims: List[int] = [64, 128, 256],  # Hidden channels
        use_gat: bool = False,
    ):
        super().__init__()
        self.adj_mx = adj_mx
        self.time_steps = time_steps
        self.num_sensors = num_sensors

        # Register adjacency matrix as buffer (not trainable)
        self.register_buffer("adj_mx_tensor", torch.FloatTensor(adj_mx))

        # Initial projection
        self.input_proj = nn.Conv2d(
            in_channels, hidden_dims[0], kernel_size=(3, 3), padding=(1, 1)
        )

        # Encoder (downsampling)
        self.encoders = nn.ModuleList()
        self.graph_convs_enc = nn.ModuleList()

        in_dim = hidden_dims[0]
        for h_dim in hidden_dims[1:]:
            # Temporal downsampling (along time)
            self.encoders.append(
                nn.Conv2d(
                    in_dim, h_dim, kernel_size=(3, 1), stride=(2, 1), padding=(1, 0)
                )
            )
            # Spatial graph conv
            self.graph_convs_enc.append(GraphConvBlock(h_dim, h_dim, adj_mx, use_gat))
            in_dim = h_dim

        # Bottleneck
        self.bottleneck_time = nn.Conv2d(
            hidden_dims[-1], hidden_dims[-1], kernel_size=(3, 1), padding=(1, 1)
        )
        self.bottleneck_graph = GraphConvBlock(
            hidden_dims[-1], hidden_dims[-1], adj_mx, use_gat
        )

        # Decoder (upsampling)
        self.decoders = nn.ModuleList()
        self.graph_convs_dec = nn.ModuleList()
        self.skip_convs = nn.ModuleList()

        in_dim = hidden_dims[-1]
        for h_dim in reversed(hidden_dims[:-1]):
            # Temporal upsampling
            self.decoders.append(
                nn.ConvTranspose2d(
                    in_dim,
                    h_dim,
                    kernel_size=(3, 1),
                    stride=(2, 1),
                    padding=(1, 0),
                    output_padding=(1, 0),
                )
            )
            # Spatial graph conv
            self.graph_convs_dec.append(
                GraphConvBlock(h_dim * 2, h_dim, adj_mx, use_gat)
            )
            # Skip connection projection
            self.skip_convs.append(nn.Conv2d(h_dim, h_dim, kernel_size=1))
            in_dim = h_dim

        # Output projection
        self.output_proj = nn.Conv2d(
            hidden_dims[0], in_channels, kernel_size=(3, 3), padding=(1, 1)
        )

        # Time embedding for diffusion timestep
        self.time_embed = nn.Sequential(
            nn.Linear(1, hidden_dims[0]),
            nn.ReLU(),
            nn.Linear(hidden_dims[0], hidden_dims[0]),
        )

        logger.info(f"Initialized SpatioTemporalUNet with:")
        logger.info(f"  - Input: [batch, {time_steps}, {num_sensors}, {in_channels}]")
        logger.info(f"  - Hidden dims: {hidden_dims}")
        logger.info(f"  - Using GAT: {use_gat}")

    def forward(self, x, timesteps, cond=None):
        """
        Args:
            x: Noisy input [batch, time, sensors, features]
            timesteps: Diffusion timesteps [batch]
            cond: Optional conditioning vector (for forecasting/imputation)

        Returns:
            Predicted noise [batch, time, sensors, features]
        """
        # CRITICAL: Ensure timesteps are on the same device as x
        if timesteps.device != x.device:
            print(f"Moving timesteps from {timesteps.device} to {x.device}")
            timesteps = timesteps.to(x.device)

        # Also ensure timesteps are float32
        timesteps = timesteps.float()

        # Reshape to [batch, features, time, sensors] for 2D convs
        x = x.permute(0, 3, 1, 2).contiguous()  # [B, F, T, N]

        # Add time embedding
        t_emb = self.time_embed(timesteps.unsqueeze(-1))  # [B, H]
        t_emb = t_emb.unsqueeze(-1).unsqueeze(-1)  # [B, H, 1, 1]

        # Ensure t_emb is on same device as x
        if t_emb.device != x.device:
            t_emb = t_emb.to(x.device)

        # Initial projection
        h = self.input_proj(x)  # [B, H1, T, N]
        h = h + t_emb  # Inject time info

        # Store skip connections
        skips = []

        # Encoder
        for encoder, graph_conv in zip(self.encoders, self.graph_convs_enc):
            # Temporal downsampling
            h = encoder(h)  # [B, H_next, T/2, N]

            # Spatial graph conv - need to ensure all tensors on same device
            h = h.permute(0, 2, 3, 1).contiguous()  # [B, T, N, H]

            # Ensure graph conv's edge_index is on correct device
            if hasattr(graph_conv, "edge_index"):
                if graph_conv.edge_index.device != h.device:
                    graph_conv.edge_index = graph_conv.edge_index.to(h.device)

            h = graph_conv(h)  # [B, T, N, H]
            h = h.permute(0, 3, 1, 2).contiguous()  # [B, H, T, N]
            skips.append(h)

        # Bottleneck
        h = self.bottleneck_time(h)
        h = h.permute(0, 2, 3, 1).contiguous()

        # Ensure bottleneck graph conv's edge_index is on correct device
        if hasattr(self.bottleneck_graph, "edge_index"):
            if self.bottleneck_graph.edge_index.device != h.device:
                self.bottleneck_graph.edge_index = self.bottleneck_graph.edge_index.to(
                    h.device
                )

        h = self.bottleneck_graph(h)
        h = h.permute(0, 3, 1, 2).contiguous()

        # Decoder
        for decoder, graph_conv, skip_conv in zip(
            self.decoders, self.graph_convs_dec, self.skip_convs
        ):
            # Temporal upsampling
            h = decoder(h)  # [B, H_prev, T*2, N]

            # Get corresponding skip connection
            skip = skips.pop()

            # Align channels if needed
            skip = skip_conv(skip)

            # Concatenate skip connection
            h = torch.cat([h, skip], dim=1)  # [B, 2*H_prev, T, N]
            h = h.permute(0, 2, 3, 1).contiguous()  # [B, T, N, 2*H]

            # Ensure graph conv's edge_index is on correct device
            if hasattr(graph_conv, "edge_index"):
                if graph_conv.edge_index.device != h.device:
                    graph_conv.edge_index = graph_conv.edge_index.to(h.device)

            h = graph_conv(h)  # [B, T, N, H]
            h = h.permute(0, 3, 1, 2).contiguous()  # [B, H, T, N]

        # Output projection
        out = self.output_proj(h)  # [B, F, T, N]

        # Reshape back to original format
        out = out.permute(0, 2, 3, 1).contiguous()  # [B, T, N, F]

        return out

    def get_trainable_params(self):
        """Return parameter count for logging."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ============================================================================
# Part 4: Training Utilities and Visualization
# ============================================================================


class DiffusionTrainer:
    """Basic training utilities for diffusion models."""

    def __init__(
        self,
        model: nn.Module,
        scheduler: DDPMScheduler,
        train_loader: DataLoader,
        val_loader: DataLoader,
        learning_rate: float = 1e-4,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        self.model = model.to(device)
        self.scheduler = scheduler
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device

        self.optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
        self.losses = []
        self.val_losses = []

        logger.info(f"Trainer initialized on {device}")
        logger.info(f"Model parameters: {model.get_trainable_params():,}")

    def train_epoch(self):
        """Train for one epoch."""
        self.model.train()
        epoch_loss = 0

        for batch in self.train_loader:
            # Move to device
            future = batch["future"].to(self.device)  # [B, T, N, F]

            # Sample random timesteps
            batch_size = future.shape[0]
            t = torch.randint(
                0,
                self.scheduler.config.num_train_timesteps,
                (batch_size,),
                device=self.device,
            ).long()

            # Add noise
            noise = torch.randn_like(future)
            noisy_future = self.scheduler.add_noise(future, noise, t)

            # Get conditioning from past if available
            if "past" in batch:
                cond = batch["past"].to(self.device)
                # For now, simple conditioning (will enhance in Phase 2)
                # We just concatenate past along time for conditioning
                # In Phase 2, this will be replaced with proper encoding
                cond = cond.mean(dim=1, keepdim=True)  # Simple pooling
            else:
                cond = None

            # Predict noise
            noise_pred = self.model(noisy_future, t, cond)

            # Compute loss
            loss = F.mse_loss(noise_pred, noise)

            # Backward
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()

            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(self.train_loader)
        self.losses.append(avg_loss)

        return avg_loss

    def validate(self):
        """Run validation."""
        self.model.eval()
        val_loss = 0

        with torch.no_grad():
            for batch in self.val_loader:
                future = batch["future"].to(self.device)

                # Sample fixed timesteps for validation
                t = torch.full(
                    (future.shape[0],),
                    self.scheduler.config.num_train_timesteps // 2,
                    device=self.device,
                ).long()

                noise = torch.randn_like(future)
                noisy_future = self.scheduler.add_noise(future, noise, t)

                if "past" in batch:
                    cond = batch["past"].to(self.device)
                    cond = cond.mean(dim=1, keepdim=True)
                else:
                    cond = None

                noise_pred = self.model(noisy_future, t, cond)
                loss = F.mse_loss(noise_pred, noise)

                val_loss += loss.item()

        avg_val_loss = val_loss / len(self.val_loader)
        self.val_losses.append(avg_val_loss)

        return avg_val_loss

    def train(self, num_epochs: int = 100, validate_every: int = 5):
        """Full training loop."""
        logger.info(f"Starting training for {num_epochs} epochs")

        for epoch in range(num_epochs):
            train_loss = self.train_epoch()

            if epoch % validate_every == 0:
                val_loss = self.validate()
                logger.info(
                    f"Epoch {epoch}: train_loss={train_loss:.6f}, val_loss={val_loss:.6f}"
                )
            else:
                logger.info(f"Epoch {epoch}: train_loss={train_loss:.6f}")

        logger.info("Training complete!")

    def plot_losses(self):
        """Plot training and validation losses."""
        plt.figure(figsize=(10, 5))
        plt.plot(self.losses, label="Train Loss")
        plt.plot(self.val_losses, label="Validation Loss")
        plt.xlabel("Epoch")
        plt.ylabel("MSE Loss")
        plt.title("Training Progress")
        plt.legend()
        plt.grid(True)
        plt.show()


# ============================================================================
# Part 5: Main Execution Pipeline
# ============================================================================


def main():
    """Run Phase 1: Load data, create datasets, initialize model."""

    logger.info("=" * 60)
    logger.info("PHASE 1: FOUNDATION & DATA PIPELINE")
    logger.info("=" * 60)

    # 1. Load and preprocess data
    loader = METRLoader(GRAPH_FILE, DATA_FILE)

    # Load graph
    sensor_ids, adj_mx = loader.load_graph()

    # Load data
    data, timestamps, data_sensor_ids = loader.load_data()

    # Align sensors
    data_to_graph = loader.align_sensors()

    # Reorder data to match graph order
    reordered_data = loader.prepare_for_modeling(data_to_graph)

    # 2. Handle missing values (simple mean imputation for now)
    # In Phase 2, this will be handled by the imputation model itself
    nan_mask = np.isnan(reordered_data)
    if nan_mask.any():
        logger.info(f"Imputing {nan_mask.sum()} missing values with column means")
        col_mean = np.nanmean(reordered_data, axis=0, keepdims=True)
        reordered_data = np.where(nan_mask, col_mean, reordered_data)

    # 3. Normalize data (z-score per sensor)
    mean = np.mean(reordered_data, axis=0, keepdims=True)
    std = np.std(reordered_data, axis=0, keepdims=True)
    std[std == 0] = 1.0
    normalized_data = (reordered_data - mean) / std

    logger.info(
        f"Normalized data range: [{normalized_data.min():.3f}, {normalized_data.max():.3f}]"
    )

    # 4. Split data chronologically
    T = normalized_data.shape[0]
    train_ratio, val_ratio = 0.7, 0.1
    train_idx = int(T * train_ratio)
    val_idx = int(T * (train_ratio + val_ratio))

    train_data = normalized_data[:train_idx]
    val_data = normalized_data[train_idx:val_idx]
    test_data = normalized_data[val_idx:]

    logger.info(
        f"Data split: train {train_data.shape}, val {val_data.shape}, test {test_data.shape}"
    )

    # 5. Create datasets
    train_dataset = TrafficDataset(
        train_data,
        input_len=12,
        output_len=12,
        stride=1,
        mode="forecast",  # Start with forecasting for Phase 1
    )

    val_dataset = TrafficDataset(
        val_data, input_len=12, output_len=12, stride=1, mode="forecast"
    )

    # 6. Create data loaders
    train_loader = DataLoader(
        train_dataset, batch_size=16, shuffle=True, num_workers=2, pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset, batch_size=16, shuffle=False, num_workers=2, pin_memory=True
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
        use_gat=False,  # Start with GCN, can experiment with GAT later
    )

    # 8. Initialize diffusion scheduler
    scheduler = DDPMScheduler(
        num_train_timesteps=1000, beta_schedule="linear", prediction_type="epsilon"
    )

    # 9. Initialize trainer
    trainer = DiffusionTrainer(
        model=model,
        scheduler=scheduler,
        train_loader=train_loader,
        val_loader=val_loader,
        learning_rate=1e-4,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )

    logger.info("Testing forward pass...")
    sample_batch = next(iter(train_loader))

    # Move to device
    future = sample_batch["future"].to(trainer.device)
    print(f"Future device: {future.device}")

    # Create timesteps on correct device
    t = torch.randint(0, 1000, (future.shape[0],), device=trainer.device)
    print(f"Timesteps device: {t.device}")

    with torch.no_grad():
        noise = torch.randn_like(future)
        print(f"Noise device: {noise.device}")

        noisy = trainer.scheduler.add_noise(future, noise, t)
        print(f"Noisy device: {noisy.device}")

        # Move model to device if not already
        trainer.model = trainer.model.to(trainer.device)

        pred = trainer.model(noisy, t)
        print(f"Pred device: {pred.device}")

    logger.info(f"Forward pass successful!")
    logger.info(f"  Input shape: {noisy.shape}")
    logger.info(f"  Output shape: {pred.shape}")
    logger.info(f"  Loss: {F.mse_loss(pred, noise).item():.6f}")

    # 11. Save processed data for future phases
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

    # 12. Optional: Quick training test (remove for actual training)
    logger.info("Running quick training test (5 epochs)...")
    trainer.train(num_epochs=5, validate_every=2)

    logger.info("=" * 60)
    logger.info("PHASE 1 COMPLETE")
    logger.info("=" * 60)

    return trainer


if __name__ == "__main__":
    trainer = main()
