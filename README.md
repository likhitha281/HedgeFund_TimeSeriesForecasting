# Hedge Fund Time Series Forecasting

---

## Overview

A multi-horizon financial time series forecasting pipeline built for a hedge fund prediction competition. The system forecasts a return-like target (`y_target`) across **4 forecast horizons (1, 3, 10, 25 steps)** for thousands of anonymized financial series, evaluated using **weighted RMSE**.

### Results Summary

| Model | H=1 | H=3 | H=10 | H=25 |
|-------|-----|-----|------|------|
| Naive (per-series median) | 0.001260 | 0.002125 | 0.003344 | 0.004227 |
| LightGBM baseline | 0.001258 | 0.001929 | 0.002733 | 0.003193 |
| LightGBM improved | 0.000942 | 0.001529 | 0.002392 | 0.003046 |
| **Ensemble (final)** | **0.000941** | **0.001529** | **0.002392** | **0.003048** |

---

## Project Structure

```
project/
├── train.parquet              # Training data (5.3M rows × 94 cols)
├── test.parquet               # Test data (1.4M rows × 92 cols)
├── improved_pipeline.py       # Main modeling pipeline
├── submission_improved.csv    # Output predictions (competition format)
├── models/                    # Saved model files
│   ├── lgbm_h1.pkl
│   ├── lgbm_h3.pkl
│   ├── lgbm_h10.pkl
│   ├── lgbm_h25.pkl
│   ├── xgb_h1.pkl
│   └── xgb_h10.pkl
├── comparison_baselines.png   # Model comparison chart
├── feature_importance_improved.png
├── rmse_vs_horizon.png
└── README.md
```

---

## Requirements

### Python Version
Python 3.8+

### Install Dependencies

```bash
pip install pandas numpy lightgbm xgboost scikit-learn \
            scipy statsmodels pyarrow matplotlib seaborn joblib
```

### Optional (for extensions)

```bash
# PyTorch — for MLP neural ensemble layer
pip install torch --index-url https://download.pytorch.org/whl/cpu

# CatBoost — additional ensemble model
pip install catboost

# Optuna — hyperparameter tuning
pip install optuna
```

---

## Usage

### 1. Basic Run (fastest, ~90 min on CPU)

Place `train.parquet` and `test.parquet` in the same directory as the script, then:

```bash
python improved_pipeline.py
```

This runs with:
- ARIMA features **OFF** (fast mode)
- MLP ensemble **ON** (if PyTorch is installed)
- XGBoost skipped for H=3 and H=25 (not beneficial based on experiments)

### 2. Enable ARIMA Features (+20–40 min)

Open `improved_pipeline.py` and set the flag at the top:

```python
USE_ARIMA_FEATURES = True   # line ~46
```

ARIMA(1,0,1) residuals are computed per series and added as a lagged feature (`arima_resid_lag1`), capturing structure that gradient boosting misses.

### 3. Full Run with All Extensions (~2.5 hrs on CPU)

```python
USE_ARIMA_FEATURES = True
# Ensure PyTorch is installed for MLP layer
```

---

## Pipeline Architecture

```
train.parquet / test.parquet
        │
        ▼
┌─────────────────────────────┐
│   Feature Engineering       │  110 features total:
│   • Lags: 1,2,3,5,7,10,15  │  - 7 lag features
│   • Rolling: mean/std/      │  - 16 rolling statistics
│     max/min (5,10,20,30)    │  - 2 momentum diffs
│   • Fourier: sin/cos        │  - 4 Fourier features
│     periods 7 & 30          │  - 2 positional features
│   • Momentum diffs          │  - 1 cluster label
│   • ts_position, horizon_log│  - 79 raw features
└────────────┬────────────────┘
             │
             ▼
┌─────────────────────────────┐
│  K-Means Cluster-Aware      │  k=8 clusters on series stats
│  Per-Series Normalization   │  y_norm = clip((y - median)/std, -10, 10)
└────────────┬────────────────┘
             │
             ▼
┌─────────────────────────────┐
│  Time-Based Train/Val Split │  85% train (ts ≤ 3061)
│                             │  15% val   (ts > 3061)
└────────────┬────────────────┘
             │
        ┌────┴────┐
        ▼         ▼
  ┌──────────┐ ┌──────────┐
  │ LightGBM │ │ XGBoost  │   Trained separately per horizon
  │ Huber    │ │ Squared  │   Horizon-specific params
  │ loss     │ │ error    │   (H=3, H=25: LightGBM only)
  └────┬─────┘ └────┬─────┘
       │             │
       └──────┬──────┘
              ▼
  ┌───────────────────────┐
  │  1/RMSE Weighted      │   Weights learned from val performance
  │  Ensemble             │   H=3, H=25: weight [1.0, 0.0]
  └───────────┬───────────┘
              │
              ▼
  ┌───────────────────────┐
  │  MLP Neural Layer     │   Optional: 3-layer MLP blends GBM preds
  │  (if PyTorch avail.)  │   + top features. Only used if it improves.
  └───────────┬───────────┘
              │
              ▼
   submission_improved.csv
```

### Cross-Horizon Knowledge Transfer

For H=3, H=10, H=25: the H=1 LightGBM prediction is added as an input feature (`xh_lgbm_h1`). This transfers short-term momentum signal to longer-horizon models.

```
H=1 model → prediction → used as feature in H=3, H=10, H=25 models
```

---

## Key Design Decisions

### Why Huber Loss?
`y_target` has kurtosis ≈ 289 (extremely heavy tails). Huber loss is robust to outliers unlike MSE, and produces better calibrated predictions on the bulk of the distribution.

### Why Per-Series Normalization?
Series span 12 orders of magnitude in variance. Without normalization, the model is dominated by high-weight, high-variance series and fails on quieter ones.

### Why Drop feature_w through feature_z?
These 4 features are 38.6% missing in **test** but only 0.12% missing in **train** — a train/test distribution shift that would cause the model to rely on a signal unavailable at inference time.

### Why Skip XGBoost on H=3 and H=25?
Experimental results showed XGBoost received weight ≈ 0 on H=3 and actively hurt H=25 (RMSE 0.004733 vs LightGBM's 0.003046). Skipping it saves ~15 min with no accuracy loss.

### Why 1/RMSE Weighting Instead of Meta-Learner CV?
The 3-fold CV meta-learner spent ~90 min learning weights that were mathematically nearly identical to 1/RMSE weights (XGBoost weight was already 0 on H=3). The simpler approach saves time with negligible accuracy difference.

---

## Horizon-Specific LightGBM Parameters

| Parameter | H=1 | H=3 | H=10 | H=25 |
|-----------|-----|-----|------|------|
| `num_leaves` | 127 | 191 | 511 | 511 |
| `learning_rate` | 0.01 | 0.01 | 0.008 | 0.008 |
| `min_child_samples` | 50 | 50 | 20 | 15 |
| `reg_lambda` | 1.0 | 0.5 | 0.2 | 0.1 |
| `n_estimators` (max) | 3000 | 3000 | 3000 | 3000 |
| Early stopping patience | 100 | 100 | 100 | 100 |

Short horizons use fewer leaves and more regularization (smoother targets).
Long horizons need more capacity (`num_leaves=511`) and less regularization to fit complex patterns.

---

## Output Files

| File | Description |
|------|-------------|
| `submission_improved.csv` | Test predictions in competition format (`id;prediction`) |
| `models/lgbm_h{1,3,10,25}.pkl` | Saved LightGBM models |
| `models/xgb_h{1,10}.pkl` | Saved XGBoost models (H=1 and H=10 only) |
| `comparison_baselines.png` | Bar chart comparing all models per horizon |
| `feature_importance_improved.png` | Top-25 features per horizon |
| `rmse_vs_horizon.png` | RMSE trend across horizons for all models |

---

## Estimated Runtime (CPU)

| Configuration | Time |
|---------------|------|
| LightGBM only (no XGBoost, no MLP) | ~50 min |
| Default (LightGBM + XGBoost H1/H10 + MLP) | ~90 min |
| With ARIMA features | ~2.5 hrs |

> **Tip:** Set `n_jobs=-1` in LightGBM params and `n_jobs=-1` in XGBoost to use all CPU cores automatically.

---

## EDA Findings That Shaped the Design

| Finding | Design Impact |
|---------|--------------|
| 63.7% of series stationary (ADF test) | Normalize in-place rather than differencing |
| ACF significant up to lag 15 | Lags extended to [1,2,3,5,7,10,15] |
| y_target kurtosis ≈ 289 | Huber loss instead of MSE |
| feature_w–z: 38.6% missing in test | Dropped from feature set |
| Weight spans 12 orders of magnitude | Used as sample_weight everywhere |
| feature_bz, feature_cd highest corr. | Top raw features confirmed by importance plots |

---

## References

1. Ke et al. (2017). *LightGBM: A Highly Efficient Gradient Boosting Decision Tree.* NeurIPS.
2. Chen & Guestrin (2016). *XGBoost: A Scalable Tree Boosting System.* KDD.
3. Makridakis et al. (2022). *M5 Accuracy Competition.* International Journal of Forecasting.
4. Wolpert (1992). *Stacked Generalization.* Neural Networks.
5. Montero-Manso & Hyndman (2021). *Forecasting Groups of Time Series.* IJF.
