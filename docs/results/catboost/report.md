# CatBoost comparison

Selection used validation only; holdout was evaluated afterwards.

## validation

| Model | Days | MAE | RMSE | WAPE % | Horizon-total WAPE % |
|---|---:|---:|---:|---:|---:|
| catboost_d4 | 7 | 5.141 | 7.010 | 30.60 | 21.42 |
| catboost_d4 | 14 | 5.557 | 7.638 | 32.33 | 22.02 |
| catboost_d4 | 30 | 5.593 | 7.646 | 34.73 | 18.23 |
| catboost_d6 | 7 | 5.272 | 7.157 | 31.38 | 22.10 |
| catboost_d6 | 14 | 5.710 | 7.816 | 33.22 | 23.19 |
| catboost_d6 | 30 | 5.649 | 7.740 | 35.08 | 18.83 |
| mean_28 | 7 | 5.118 | 7.023 | 30.46 | 15.95 |
| mean_28 | 14 | 5.584 | 7.531 | 32.49 | 17.05 |
| mean_28 | 30 | 5.774 | 7.606 | 35.86 | 13.24 |
| seasonal_naive_7 | 7 | 5.814 | 7.665 | 34.61 | 20.89 |
| seasonal_naive_7 | 14 | 6.360 | 8.569 | 37.00 | 22.40 |
| seasonal_naive_7 | 30 | 6.833 | 9.171 | 42.43 | 22.40 |

## holdout

| Model | Days | MAE | RMSE | WAPE % | Horizon-total WAPE % |
|---|---:|---:|---:|---:|---:|
| catboost_d4 | 7 | 4.172 | 5.391 | 31.34 | 8.95 |
| catboost_d4 | 14 | 5.246 | 6.568 | 46.63 | 25.97 |
| catboost_d4 | 30 | 5.102 | 6.766 | 36.73 | 10.19 |
| mean_28 | 7 | 5.126 | 6.814 | 38.50 | 17.62 |
| mean_28 | 14 | 6.057 | 7.720 | 53.84 | 37.49 |
| mean_28 | 30 | 5.432 | 7.117 | 39.11 | 13.15 |
| seasonal_naive_7 | 7 | 6.129 | 8.084 | 46.03 | 16.42 |
| seasonal_naive_7 | 14 | 7.236 | 9.704 | 64.32 | 37.90 |
| seasonal_naive_7 | 30 | 6.563 | 8.877 | 47.25 | 16.92 |

![Holdout forecast](holdout_forecast.png)
