# training/csdi_trainer.py

import torch
import torch.nn.functional as F

from .trainer import DiffusionTrainer


class CSDIForecastTrainer(DiffusionTrainer):
    """
    Trainer for CSDI forecasting.
    The mask indicates observed (1) vs missing (0) positions.
    """

    def train_epoch(self):
        self.model.train()
        epoch_loss = 0

        for batch in self.train_loader:
            # Move data to device
            future = batch["future"].to(self.device)
            cond = batch["cond"].to(self.device)
            mask = batch["mask"].to(self.device)  # 0 for missing (future)

            batch_size = future.shape[0]
            t = torch.randint(
                0,
                self.scheduler.config.num_train_timesteps,
                (batch_size,),
                device=self.device,
            ).long()

            noise = torch.randn_like(future)
            noisy_future = self.scheduler.add_noise(future, noise, t)

            noise_pred = self.model(noisy_future, t, cond, mask)

            # Masked loss: only on positions where mask == 0 (i.e., missing)
            loss = F.mse_loss(noise_pred, noise, reduction="none")
            loss = (loss * (1 - mask)).mean()

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()

            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(self.train_loader)
        self.losses.append(avg_loss)
        return avg_loss

    def validate(self):
        self.model.eval()
        val_loss = 0

        with torch.no_grad():
            for batch in self.val_loader:
                future = batch["future"].to(self.device)
                cond = batch["cond"].to(self.device)
                mask = batch["mask"].to(self.device)

                t = torch.full(
                    (future.shape[0],),
                    self.scheduler.config.num_train_timesteps // 2,
                    device=self.device,
                ).long()

                noise = torch.randn_like(future)
                noisy_future = self.scheduler.add_noise(future, noise, t)

                noise_pred = self.model(noisy_future, t, cond, mask)

                loss = F.mse_loss(noise_pred, noise, reduction="none")
                loss = (loss * (1 - mask)).mean()
                val_loss += loss.item()

        avg_val_loss = val_loss / len(self.val_loader)
        self.val_losses.append(avg_val_loss)
        return avg_val_loss
