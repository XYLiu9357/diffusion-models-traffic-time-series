"""
TimeGrad trainer.
"""

import torch

from .trainer import DiffusionTrainer


class TimeGradTrainer(DiffusionTrainer):
    """
    Trainer for TimeGrad.
    Expects batches with "past" and "future".
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def train_epoch(self):
        self.model.train()
        epoch_loss = 0

        for batch in self.train_loader:
            past = batch["past"].to(self.device)  # [B, T_in, N, 1]
            future = batch["future"].to(self.device)  # [B, T_out, N, 1]

            B, T_in, N, F = past.shape
            T_out = future.shape[1]

            # Encode past: flatten sensors into batch dimension
            past_flat = past.permute(0, 2, 1, 3).reshape(
                B * N, T_in, F
            )  # [B*N, T_in, 1]

            # Initialize RNN hidden states (3‑D)
            h_full = torch.zeros(
                self.model.rnn.num_layers,
                B * N,
                self.model.hidden_dim,
                device=self.device,
            )
            c_full = torch.zeros(
                self.model.rnn.num_layers,
                B * N,
                self.model.hidden_dim,
                device=self.device,
            )

            # Run RNN over past
            _, (h_full, c_full) = self.model.rnn(past_flat, (h_full, c_full))

            step_loss = 0.0

            for t in range(T_out):
                target_t = future[:, t, :, :]  # [B, N, F]

                # Sample random diffusion timestep
                t_step = torch.randint(
                    0,
                    self.scheduler.config.num_train_timesteps,
                    (B,),
                    device=self.device,
                ).long()

                # Add noise
                noise = torch.randn_like(target_t)
                noisy_t = self.scheduler.add_noise(target_t, noise, t_step)

                # Prepare for decoder: flatten sensors
                noisy_flat = noisy_t.reshape(B * N, F)  # [B*N, F]
                h_last = h_full[-1]  # [B*N, hidden] (last layer)

                # Predict noise
                noise_pred_flat = self.model.decoder(
                    noisy_flat, h_last, t_step
                )  # [B*N, F]
                noise_pred = noise_pred_flat.reshape(B, N, F)

                # Loss
                loss = torch.nn.functional.mse_loss(noise_pred, noise)
                step_loss += loss

                # Teacher forcing: update RNN with ground truth target_t
                input_t = target_t.reshape(B * N, 1, F)  # [B*N, 1, F]
                _, (h_full, c_full) = self.model.rnn(input_t, (h_full, c_full))

            # Average loss over future steps
            avg_step_loss = step_loss / T_out

            self.optimizer.zero_grad()
            avg_step_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()

            epoch_loss += avg_step_loss.item()

        return epoch_loss / len(self.train_loader)

    def validate(self):
        self.model.eval()
        val_loss = 0

        with torch.no_grad():
            for batch in self.val_loader:
                past = batch["past"].to(self.device)
                future = batch["future"].to(self.device)

                B, T_in, N, F = past.shape
                T_out = future.shape[1]

                # Encode past
                past_flat = past.permute(0, 2, 1, 3).reshape(B * N, T_in, F)
                h_full = torch.zeros(
                    self.model.rnn.num_layers,
                    B * N,
                    self.model.hidden_dim,
                    device=self.device,
                )
                c_full = torch.zeros(
                    self.model.rnn.num_layers,
                    B * N,
                    self.model.hidden_dim,
                    device=self.device,
                )
                _, (h_full, c_full) = self.model.rnn(past_flat, (h_full, c_full))

                step_loss = 0.0

                for t in range(T_out):
                    target_t = future[:, t, :, :]

                    # Fixed timestep for validation (e.g., middle)
                    t_step = torch.full(
                        (B,),
                        self.scheduler.config.num_train_timesteps // 2,
                        device=self.device,
                    ).long()

                    noise = torch.randn_like(target_t)
                    noisy_t = self.scheduler.add_noise(target_t, noise, t_step)

                    noisy_flat = noisy_t.reshape(B * N, F)
                    h_last = h_full[-1]
                    noise_pred_flat = self.model.decoder(noisy_flat, h_last, t_step)
                    noise_pred = noise_pred_flat.reshape(B, N, F)

                    loss = torch.nn.functional.mse_loss(noise_pred, noise)
                    step_loss += loss

                    # Update RNN with ground truth
                    input_t = target_t.reshape(B * N, 1, F)
                    _, (h_full, c_full) = self.model.rnn(input_t, (h_full, c_full))

                val_loss += (step_loss / T_out).item()

        return val_loss / len(self.val_loader)
