import logging
import pathlib

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers import DDPMScheduler
from torch.utils.data import DataLoader

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


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

    def train(
        self,
        num_epochs: int = 100,
        validate_every: int = 5,
        save_every=10,
        checkpoint_dir="checkpoints",
    ):
        """Full training loop."""
        logger.info(f"Starting training for {num_epochs} epochs")
        checkpoint_dir = pathlib.Path(checkpoint_dir)
        checkpoint_dir.mkdir(exist_ok=True)

        for epoch in range(num_epochs):
            train_loss = self.train_epoch()

            if epoch % validate_every == 0:
                val_loss = self.validate()
                logger.info(
                    f"Epoch {epoch}: train_loss={train_loss:.6f}, val_loss={val_loss:.6f}"
                )
            else:
                logger.info(f"Epoch {epoch}: train_loss={train_loss:.6f}")

            if (epoch + 1) % save_every == 0:
                checkpoint_path = checkpoint_dir / f"epoch_{epoch}.pth"
                torch.save(self.model.state_dict(), checkpoint_path)
                logger.info(f"Checkpoint saved at {checkpoint_path}")
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


if __name__ == "__main__":
    pass
