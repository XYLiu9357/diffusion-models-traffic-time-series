# training/imputation_trainer.py

import torch
import torch.nn.functional as F

from .trainer import DiffusionTrainer


class CSDIImputationTrainer(DiffusionTrainer):
    """
    Trainer for CSDI imputation.
    Expects batches with keys: "combined", "corrupted", "mask".
    """

    def train_epoch(self):
        self.model.train()
        epoch_loss = 0

        for batch in self.train_loader:
            combined = batch["combined"].to(self.device)  # ground truth
            corrupted = batch["corrupted"].to(
                self.device
            )  # input (observed values, zeros elsewhere)
            mask = batch["mask"].to(self.device)  # 1=observed, 0=missing

            batch_size = combined.shape[0]
            t = torch.randint(
                0,
                self.scheduler.config.num_train_timesteps,
                (batch_size,),
                device=self.device,
            ).long()

            noise = torch.randn_like(combined)
            noisy_combined = self.scheduler.add_noise(combined, noise, t)

            # Model forward: noisy_target, timesteps, cond (corrupted), mask
            noise_pred = self.model(noisy_combined, t, corrupted, mask)

            # Masked loss: only on missing positions (mask == 0)
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
                combined = batch["combined"].to(self.device)
                corrupted = batch["corrupted"].to(self.device)
                mask = batch["mask"].to(self.device)

                t = torch.full(
                    (combined.shape[0],),
                    self.scheduler.config.num_train_timesteps // 2,
                    device=self.device,
                ).long()

                noise = torch.randn_like(combined)
                noisy_combined = self.scheduler.add_noise(combined, noise, t)

                noise_pred = self.model(noisy_combined, t, corrupted, mask)

                loss = F.mse_loss(noise_pred, noise, reduction="none")
                loss = (loss * (1 - mask)).mean()
                val_loss += loss.item()

        avg_val_loss = val_loss / len(self.val_loader)
        self.val_losses.append(avg_val_loss)
        return avg_val_loss
