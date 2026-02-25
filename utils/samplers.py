# utils/samplers.py
import torch


@torch.no_grad()
def forecast_csdi(model, scheduler, cond, mask, num_steps=None, n_samples=1):
    """
    Fast probabilistic forecasting using DDIM.
    Generates multiple future samples in one batched run.

    Args:
        model: CSDI model
        scheduler: DDIM scheduler (or compatible)
        cond: observed past values [B, T, N, F] (with zeros for future steps)
        mask: binary mask, 1 for observed positions, 0 for future positions [B, T, N, F]
        num_steps: number of diffusion steps to use (if None, uses scheduler default)
        n_samples: number of independent samples to generate per batch element

    Returns:
        samples: tensor of shape [n_samples, B, T, N, F]
    """
    model.eval()
    device = cond.device
    B, T, N, F = cond.shape

    if num_steps is not None:
        scheduler.set_timesteps(num_steps)

    timesteps = scheduler.timesteps.to(device)  # [num_steps]

    # Repeat cond and mask n_samples times along batch dimension
    cond_multi = cond.repeat(n_samples, 1, 1, 1)  # [n_samples*B, T, N, F]
    mask_multi = mask.repeat(n_samples, 1, 1, 1)  # [n_samples*B, T, N, F]

    # Start from pure noise at missing positions, keep observed fixed
    x = torch.randn(n_samples * B, T, N, F, device=device)
    x = x * (1 - mask_multi) + cond_multi * mask_multi

    for t in timesteps:
        t_batch = t.expand(n_samples * B).long()

        with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
            noise_pred = model(x, t_batch, cond_multi, mask_multi)

        # Scheduler step on CPU (standard for diffusers)
        x = scheduler.step(noise_pred.cpu(), t.cpu(), x.cpu()).prev_sample.to(device)

        # Re-apply observed values after each step
        x = x * (1 - mask_multi) + cond_multi * mask_multi

    # Reshape to [n_samples, B, T, N, F]
    return x.view(n_samples, B, T, N, F)


@torch.no_grad()
def impute_csdi(model, scheduler, corrupted, mask, num_steps=None, n_samples=1):
    """
    Fast imputation using DDIM. If n_samples > 1, generates multiple samples in one batched run.
    Returns tensor of shape [n_samples, B, T, N, F].
    """
    model.eval()
    device = corrupted.device
    B, T, N, F = corrupted.shape

    if num_steps is not None:
        scheduler.set_timesteps(num_steps)

    timesteps = scheduler.timesteps.to(device)

    # Repeat data n_samples times along a new batch dimension
    corrupted_multi = corrupted.repeat(n_samples, 1, 1, 1)  # [n_samples*B, T, N, F]
    mask_multi = mask.repeat(n_samples, 1, 1, 1)  # [n_samples*B, T, N, F]

    # Start from noise for each replica
    x = torch.randn(n_samples * B, T, N, F, device=device)
    x = x * (1 - mask_multi) + corrupted_multi * mask_multi

    for t in timesteps:
        t_batch = t.expand(n_samples * B).long()

        with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
            noise_pred = model(x, t_batch, corrupted_multi, mask_multi)

        # Scheduler step on CPU
        x = scheduler.step(noise_pred.cpu(), t.cpu(), x.cpu()).prev_sample.to(device)

        # Re-apply observed values
        x = x * (1 - mask_multi) + corrupted_multi * mask_multi

    # Reshape to [n_samples, B, T, N, F]
    return x.view(n_samples, B, T, N, F)


def forecast_timegrad(
    model,
    scheduler,
    past,
    T_out=12,
    num_inference_steps=50,
    num_samples=1,
):
    """
    Forecasting with TimeGrad using DDIM scheduler (CPU‑based scheduler).
    past: [B, T_in, N, 1]
    returns: [num_samples, B, T_out, N, 1]
    """
    model.eval()
    device = past.device

    # Scheduler stays on CPU
    scheduler.set_timesteps(num_inference_steps)
    timesteps = scheduler.timesteps  # pre‑computed timesteps (on CPU)

    B, T_in, N, F = past.shape

    # Encode past
    past_flat = past.permute(0, 2, 1, 3).reshape(B * N, T_in, F)
    h_full = torch.zeros(model.rnn.num_layers, B * N, model.hidden_dim, device=device)
    c_full = torch.zeros(model.rnn.num_layers, B * N, model.hidden_dim, device=device)
    _, (h_full, c_full) = model.rnn(past_flat, (h_full, c_full))

    all_samples = []

    for _ in range(num_samples):
        h_sample = h_full.clone()
        c_sample = c_full.clone()
        generated = torch.zeros(B, T_out, N, F, device=device)

        for t in range(T_out):
            # Start from pure noise
            x = torch.randn(B, N, F, device=device)

            # Reverse diffusion using pre‑computed timesteps
            for step in timesteps:  # step is a scalar on CPU
                timestep = torch.full((B,), step, device=device, dtype=torch.long)

                x_flat = x.reshape(B * N, F)
                hidden_flat = h_sample[-1]

                noise_pred_flat = model.decoder(x_flat, hidden_flat, timestep)
                noise_pred = noise_pred_flat.reshape(B, N, F)

                # Move to CPU for scheduler step
                x_cpu = x.cpu()
                noise_pred_cpu = noise_pred.cpu()
                # scheduler.step expects CPU tensors and scalar timestep
                x_cpu = scheduler.step(noise_pred_cpu, step, x_cpu).prev_sample
                x = x_cpu.to(device)

            generated[:, t, :, :] = x

            # Update RNN with generated value
            input_t = x.reshape(B * N, 1, F)
            _, (h_sample, c_sample) = model.rnn(input_t, (h_sample, c_sample))

        all_samples.append(generated.unsqueeze(0))

    return torch.cat(all_samples, dim=0)
