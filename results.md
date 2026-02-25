# Imputation

0.15 corruption rate.

python -m scripts.evaluate_csdi --task imputation --checkpoint csdi_imputation.pth
2026-02-23 12:48:11,026 - numexpr.utils - INFO - NumExpr defaulting to 10 threads.
2026-02-23 12:48:11,188 - __main__ - INFO - ============================================================
2026-02-23 12:48:11,188 - __main__ - INFO - EVALUATING CSDI MODEL ON IMPUTATION TASK
2026-02-23 12:48:11,188 - __main__ - INFO - ============================================================
2026-02-23 12:48:11,212 - __main__ - INFO - Test data shape: (6855, 207, 1)
2026-02-23 12:48:11,215 - data.dataset - INFO - Created dataset with 6832 windows, mode=imputation
2026-02-23 12:48:11,216 - __main__ - INFO - Number of test windows: 6832
2026-02-23 12:48:11,487 - __main__ - INFO - Model loaded from csdi_imputation.pth
2026-02-23 12:48:11,487 - __main__ - INFO - Samples per batch: 5
Evaluating:   0%|                                                                                                                                          | 0/427 [00:00<?, ?it/s]2026-02-23 12:48:14,364 - numexpr.utils - INFO - NumExpr defaulting to 10 threads.
2026-02-23 12:48:14,398 - numexpr.utils - INFO - NumExpr defaulting to 10 threads.
Evaluating: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 427/427 [1:00:02<00:00,  8.44s/it]
2026-02-24 12:30:05,759 - __main__ - INFO - Test MAE: 0.6001 (normalized scale)
2026-02-24 12:30:05,760 - __main__ - INFO - Test RMSE: 0.7802 (normalized scale)
2026-02-24 12:30:05,761 - __main__ - INFO - Average std: 19.19 mph
2026-02-24 12:30:05,761 - __main__ - INFO - Approx. MAE in mph: 11.51
2026-02-24 12:30:05,762 - __main__ - INFO - Approx. RMSE in mph: 14.97
2026-02-24 12:30:05,762 - __main__ - INFO - Plotting example 0...
2026-02-24 12:30:11,675 - __main__ - INFO - Plotting example 1...

# CSDI Forecast

2026-02-24 16:15:03,752 - numexpr.utils - INFO - NumExpr defaulting to 10 threads.
2026-02-24 16:15:03,916 - __main__ - INFO - ============================================================
2026-02-24 16:15:03,916 - __main__ - INFO - EVALUATING CSDI MODEL ON FORECAST TASK
2026-02-24 16:15:03,916 - __main__ - INFO - ============================================================
2026-02-24 16:15:03,937 - __main__ - INFO - Test data shape: (6855, 207, 1)
2026-02-24 16:15:03,958 - data.dataset - INFO - Created dataset with 6832 windows, mode=forecast
2026-02-24 16:15:03,958 - __main__ - INFO - Number of test windows: 6832
2026-02-24 16:15:04,269 - __main__ - INFO - Model loaded from csdi_metrla.pth
2026-02-24 16:15:04,273 - __main__ - INFO - Samples per batch: 5
Evaluating:   0%|                                                                                                                                          | 0/427 [00:00<?, ?it/s]2026-02-24 16:15:06,479 - numexpr.utils - INFO - NumExpr defaulting to 10 threads.
2026-02-24 16:15:06,500 - numexpr.utils - INFO - NumExpr defaulting to 10 threads.
Evaluating: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 427/427 [31:11<00:00,  4.38s/it]
2026-02-24 16:46:15,910 - __main__ - INFO - Test MAE: 0.9160 (normalized scale)
2026-02-24 16:46:15,910 - __main__ - INFO - Test RMSE: 1.1682 (normalized scale)
2026-02-24 16:46:15,910 - __main__ - INFO - Average std: 19.19 mph
2026-02-24 16:46:15,910 - __main__ - INFO - Approx. MAE in mph: 17.57
2026-02-24 16:46:15,911 - __main__ - INFO - Approx. RMSE in mph: 22.41
2026-02-24 16:46:15,911 - __main__ - INFO - Plotting example 0...
2026-02-24 16:47:32,186 - __main__ - INFO - Plotting example 1...

# Timegrad Forecast

2026-02-24 23:19:37,829 - __main__ - INFO - Test MAE: 0.9166 (normalized scale)
2026-02-24 23:19:37,829 - __main__ - INFO - Test RMSE: 1.1688 (normalized scale)
2026-02-24 23:19:37,830 - __main__ - INFO - Average std: 19.19 mph
2026-02-24 23:19:37,830 - __main__ - INFO - Approx. MAE in mph: 17.59
2026-02-24 23:19:37,830 - __main__ - INFO - Approx. RMSE in mph: 22.42
2026-02-24 23:19:37,830 - __main__ - INFO - Plotting example 0...
2026-02-25 01:53:15,389 - __main__ - INFO - Plotting example 1...

# Compare All

## Forecast

2026-02-25 05:50:02,105 - __main__ - INFO - ============================================================
2026-02-25 05:50:02,105 - __main__ - INFO - Results for forecast task:
2026-02-25 05:50:02,105 - __main__ - INFO - Persistence               MAE: 0.2841 (norm), RMSE: 0.5327 (norm)
2026-02-25 05:50:02,105 - __main__ - INFO - Persistence               MAE: 5.45 mph, RMSE: 10.22 mph
2026-02-25 05:50:02,105 - __main__ - INFO - TimeGrad                  MAE: 0.6733 (norm), RMSE: 1.0047 (norm)
2026-02-25 05:50:02,105 - __main__ - INFO - TimeGrad                  MAE: 12.92 mph, RMSE: 19.28 mph
2026-02-25 05:50:02,105 - __main__ - INFO - CSDI (forecast)           MAE: 1.0738 (norm), RMSE: 1.3045 (norm)
2026-02-25 05:50:02,105 - __main__ - INFO - CSDI (forecast)           MAE: 20.60 mph, RMSE: 25.03 mph
2026-02-25 05:50:02,105 - __main__ - INFO - ============================================================
