# Diffusion Models for Urban Traffic Time Series

## Introduction

Urban traffic management relies on accurate modeling of spatiotemporal sensor data, yet two fundamental challenges persist: probabilistic forecasting under uncertainty, and reconstruction of missing values caused by sensor failures.

While these tasks are typically addressed with separate architectures, we propose a unified framework based on denoising diffusion probabilistic models that can handle both problems through conditional generation. We adapt CSDI - a diffusion model for time series imputation - with graph convolutional networks to capture spatial dependencies among traffic sensors, and compare it against TimeGrad, an autoregressive diffusion model for forecasting.

Experiments on the METR-LA traffic dataset reveal a striking contrast: CSDI with graph convolutions achieves state-of-the-art imputation performance, reducing MAE by 35% over traditional methods (1.98 mph vs. 3.06 mph), with an ablation study confirming that spatial modeling is critical (removing graph convolutions more than doubles error). However, for forecasting, both diffusion models underperform a simple persistence baseline, suggesting that short-term traffic predictability is dominated by recent observations and that these architectures may require task-specific modifications.

Our results demonstrate that diffusion models offer a powerful unified approach for traffic imputation while highlighting the need for improved forecasting designs and more comprehensive baselines in future work.

For more details, see the full paper [here](docs/diffusion_models_for_urban_traffic_time_series.pdf).

## Dependencies

The project requires the following libraries to execute:
- `diffusers`
- `matplotlib`
- `numpy`
- `torch`
- `scikit-learn`

## Use

The framework uses CSDI for imputation tasks and TimeGrad for forecast tasks, but CSDI can, in principle, be used as a non-autoregressive forecasting approach as well. See [Makefile](Makefile) for more details.

## License

Distributed under the MIT license.
