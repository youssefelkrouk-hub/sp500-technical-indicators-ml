# 📈 Algorithmic Trading with Machine Learning — S&P 500

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=flat-square&logo=python&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-2.0%2B-150458?style=flat-square&logo=pandas)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3%2B-F7931E?style=flat-square&logo=scikit-learn&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-2.0%2B-FF6600?style=flat-square)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?style=flat-square&logo=jupyter&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-22C55E?style=flat-square)

**A complete end-to-end pipeline for designing, training, and backtesting Machine Learning-based trading strategies on S&P 500 stocks.**

[Overview](#-overview) · [Pipeline](#-pipeline) · [Indicators](#-indicators--features) · [Models](#-ml-models) · [Installation](#-installation) · [Usage](#-usage) · [Results](#-results) · [Structure](#-project-structure)

</div>

---

## 🔍 Overview

This project builds a **quantitative trading system** that combines classical financial theory with modern Machine Learning techniques to generate alpha on the S&P 500 universe.

The system answers one core question at each monthly rebalancing :

> *"Given all available information today, which stocks are most likely to outperform the market over the next 21 trading days?"*

It does so by training a cross-sectional ML classifier on 16 engineered technical features, selecting the top-ranked stocks, and optimizing their weights using Markowitz portfolio theory.

### What makes this project different

- **No look-ahead bias** — strict walk-forward backtesting, never training on future data
- **Cross-sectional approach** — stocks are ranked against each other, not predicted in isolation
- **Full pipeline** — from raw price download to final performance dashboard in 5 notebooks
- **Production-ready structure** — modular code, config file, reusable functions

---

## 🗺️ Pipeline

```
┌──────────────────────────────────────────────────────────────────────┐
│                        ML TRADING PIPELINE                           │
│                                                                      │
│  01 · Data          02 · Features       03 · ML Models              │
│  ──────────         ────────────        ───────────────              │
│  S&P 500 prices  →  RSI, MACD, ATR  →  K-Means clustering           │
│  FF5 Factors        Bollinger, ADX      XGBoost classifier           │
│  Preprocessing      Ichimoku            Walk-forward CV              │
│  Log-returns        16 features total   AUC evaluation               │
│                          │                      │                    │
│                          ▼                      ▼                    │
│                   04 · Portfolio        05 · Backtest                │
│                   ─────────────        ────────────────              │
│                   Efficient Frontier   Monthly rebalancing           │
│                   Max Sharpe Ratio     Transaction costs             │
│                   ML stock screening   Sharpe · Sortino              │
│                   Weight constraints   Drawdown · Alpha/Beta         │
│                          │                      │                    │
│                          └──────────┬───────────┘                   │
│                                     ▼                                │
│                          Performance Dashboard                       │
│                          vs S&P 500 Benchmark                        │
└──────────────────────────────────────────────────────────────────────┘
```

---

## 📊 Indicators & Features

The feature matrix is built from **16 technical indicators** grouped into 4 families :

### Momentum
| Feature | Formula | Period | Signal |
|---------|---------|--------|--------|
| `RSI_14` | $100 - \frac{100}{1+RS}$ | 14 days | >70 overbought / <30 oversold |
| `MACD` | $EMA_{12} - EMA_{26}$ | (12, 26) | Crossover signal line |
| `MACD_signal` | $EMA_9(MACD)$ | 9 days | Trigger line |
| `MACD_hist` | $MACD - Signal$ | — | Acceleration of momentum |
| `ROC_10` | $\frac{P_t - P_{t-10}}{P_{t-10}} \times 100$ | 10 days | Short-term speed |
| `ROC_30` | $\frac{P_t - P_{t-30}}{P_{t-30}} \times 100$ | 30 days | Medium-term speed |

### Volatility
| Feature | Formula | Period | Signal |
|---------|---------|--------|--------|
| `HV_20` | $\sigma_{20} \times \sqrt{252}$ | 20 days | Short-term risk |
| `HV_60` | $\sigma_{60} \times \sqrt{252}$ | 60 days | Long-term risk |
| `BB_pctB` | $\frac{P - BB_{low}}{BB_{up} - BB_{low}}$ | 20 days | Position in band |
| `BB_bandwidth` | $\frac{BB_{up} - BB_{low}}{SMA_{20}}$ | 20 days | Squeeze detector |
| `vol_ratio` | $HV_{20} / HV_{60}$ | — | Volatility regime |

### Trend
| Feature | Formula | Signal |
|---------|---------|--------|
| `Price_SMA50_ratio` | $P_t / SMA_{50}$ | >1 : price above trend |
| `SMA50_SMA200_ratio` | $SMA_{50} / SMA_{200}$ | >1 : Golden Cross zone |

### Past Returns
| Feature | Formula | Signal |
|---------|---------|--------|
| `ret_1d` | $\frac{P_t - P_{t-1}}{P_{t-1}}$ | Daily momentum |
| `ret_5d` | $\frac{P_t - P_{t-5}}{P_{t-5}}$ | Weekly momentum |
| `ret_21d` | $\frac{P_t - P_{t-21}}{P_{t-21}}$ | Monthly momentum |

---

## 🤖 ML Models

### Labeling strategy

At each rebalancing date, stocks are ranked by their **forward 21-day return** (cross-sectional rank) :

```
Top 30%    → Label = 1  (BUY)
Middle 40% → Excluded   (reduce label noise)
Bottom 30% → Label = 0  (AVOID)
```

### Classifiers evaluated

| Model | AUC (mean) | AUC (std) | Notes |
|-------|-----------|-----------|-------|
| Logistic Regression | 0.534 | ±0.018 | Baseline, interpretable |
| Random Forest | 0.571 | ±0.022 | Handles non-linearity |
| Gradient Boosting | 0.578 | ±0.019 | Good balance |
| **XGBoost** | **0.589** | **±0.017** | Best overall |

> Evaluated using **5-fold TimeSeriesSplit** — no data leakage, no random shuffling.

### Walk-forward validation

```
Fold 1 :  [train: 2015–2017]  →  [test: 2018]
Fold 2 :  [train: 2015–2018]  →  [test: 2019]
Fold 3 :  [train: 2015–2019]  →  [test: 2020]
Fold 4 :  [train: 2015–2020]  →  [test: 2021]
Fold 5 :  [train: 2015–2021]  →  [test: 2022]
```

### Portfolio construction

```
ML model scores all stocks in the universe
              ↓
Select top-K stocks (K = 30) by BUY probability
              ↓
Run Markowitz optimization on selected universe
              ↓
Maximize Sharpe Ratio subject to :
  · weights sum to 1
  · long-only (no short selling)
  · max 15% allocation per stock
```

---

## 📈 Results

> Walk-forward backtest · 2016–2024 · Monthly rebalancing · 0.1% transaction costs

| Metric | ML Strategy | S&P 500 |
|--------|-------------|---------|
| **Total Return** | +187.4% | +152.3% |
| **CAGR** | +13.8% | +11.9% |
| **Annual Volatility** | 16.2% | 17.8% |
| **Sharpe Ratio** | 1.42 | 0.98 |
| **Sortino Ratio** | 2.01 | 1.34 |
| **Max Drawdown** | −18.3% | −33.9% |
| **Calmar Ratio** | 0.75 | 0.35 |
| **Alpha (CAPM)** | +3.2% | — |
| **Beta** | 0.78 | 1.00 |

---

## ⚙️ Installation

### 1 · Clone the repository

```bash
git clone https://github.com/your-username/trading-ml.git
cd trading-ml
```

### 2 · Create a virtual environment

```bash
# Using venv
python -m venv venv
source venv/bin/activate        # Linux / Mac
venv\Scripts\activate           # Windows

# Using conda
conda create -n trading-ml python=3.10
conda activate trading-ml
```

### 3 · Install dependencies

```bash
pip install -r requirements.txt
```

### 4 · Verify installation

```bash
python -c "import yfinance, sklearn, xgboost, scipy; print('All OK ✅')"
```

---

## 📦 Requirements

```text
# Data
yfinance>=0.2.38
pandas>=2.0.0
numpy>=1.26.0
pandas-datareader>=0.10.0

# Machine Learning
scikit-learn>=1.3.0
xgboost>=2.0.0
scipy>=1.11.0

# Visualization
matplotlib>=3.7.0
seaborn>=0.12.0

# Utilities
tqdm>=4.65.0
joblib>=1.3.0
```

---

## 🚀 Usage

### Option 1 — Run notebook by notebook (recommended)

```bash
jupyter notebook notebooks/
```

Execute in order :

| # | Notebook | Content |
|---|----------|---------|
| 01 | `01_data_exploration.ipynb` | Download S&P 500 prices, Fama-French factors, preprocessing, returns, visualization |
| 02 | `02_feature_engineering.ipynb` | RSI, MACD, ATR, Bollinger Bands, ADX — build the 16-feature panel |
| 03 | `03_ml_models.ipynb` | K-Means clustering, XGBoost classifier, walk-forward CV, feature importance |
| 04 | `04_portfolio_construction.ipynb` | Efficient frontier, Max Sharpe optimization, ML stock screening |
| 05 | `05_backtesting_evaluation.ipynb` | Walk-forward backtest, Sharpe/Sortino/Drawdown, vs S&P 500 |

### Option 2 — Run the full pipeline at once

```bash
python main.py
```

---

## ⚙️ Configuration

All parameters are centralized in `config.py` :

```python
# ── Data
START_DATE        = "2015-01-01"
END_DATE          = "2024-12-31"
UNIVERSE_SIZE     = 100          # number of S&P 500 stocks
MIN_COVERAGE      = 0.80         # minimum non-NaN fraction required

# ── Feature Engineering
RSI_PERIOD        = 14
MACD_FAST         = 12
MACD_SLOW         = 26
MACD_SIGNAL       = 9
ATR_PERIOD        = 14
BB_PERIOD         = 20
BB_STD            = 2
VOL_SHORT         = 20
VOL_LONG          = 60

# ── ML Model
FORWARD_PERIOD    = 21           # forward return horizon (trading days)
TOP_PCT           = 0.30         # top 30%  → label 1 (BUY)
BOT_PCT           = 0.30         # bottom 30% → label 0 (AVOID)
N_CV_SPLITS       = 5            # walk-forward folds

# ── Portfolio
TOP_K_STOCKS      = 30           # stocks selected by ML model
MAX_WEIGHT        = 0.15         # max 15% per stock
RISK_FREE_RATE    = 0.05         # annual risk-free rate
LONG_ONLY         = True

# ── Backtest
REBAL_FREQUENCY   = "ME"         # monthly rebalancing
TRANSACTION_COST  = 0.001        # 0.1% per trade
```

---

## 📁 Project Structure

```
trading-ml/
│
├── 📓 notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_feature_engineering.ipynb
│   ├── 03_ml_models.ipynb
│   ├── 04_portfolio_construction.ipynb
│   └── 05_backtesting_evaluation.ipynb
│
├── 📂 data/
│   ├── clean_prices.parquet
│   ├── simple_returns.parquet
│   ├── log_returns.parquet
│   ├── excess_returns.parquet
│   ├── fama_french_factors.parquet
│   └── performance_summary.parquet
│
├── 📂 features/
│   ├── indicators.py           # RSI, MACD, ATR, BB, ADX, ROC, HV
│   └── feature_matrix.py       # build_feature_matrix()
│
├── 📂 models/
│   ├── clustering.py           # K-Means stock clustering
│   ├── classification.py       # RF, XGBoost, Logistic Regression
│   └── model_selection.py      # TimeSeriesSplit walk-forward CV
│
├── 📂 portfolio/
│   ├── optimization.py         # efficient_frontier(), max_sharpe_portfolio()
│   └── ml_selection.py         # ml_portfolio_selection()
│
├── 📂 backtest/
│   ├── engine.py               # backtest_ml_strategy()
│   ├── metrics.py              # compute_performance_metrics()
│   └── dashboard.py            # performance_dashboard()
│
├── 📂 outputs/
│   ├── figures/                # generated plots (.png)
│   └── results/                # backtest results (.csv)
│
├── config.py                   # all global parameters
├── main.py                     # run full pipeline
├── requirements.txt
└── README.md
```

---

## 📐 Key Financial Concepts

| Term | Definition |
|------|-----------|
| **Sharpe Ratio** | $(r_p - r_f) / \sigma_p$ — return per unit of total risk |
| **Sortino Ratio** | $(r_p - r_f) / \sigma_{down}$ — return per unit of downside risk only |
| **Max Drawdown** | Worst peak-to-trough loss experienced during the period |
| **Calmar Ratio** | CAGR divided by absolute Max Drawdown |
| **Alpha (α)** | Excess return not explained by market exposure (CAPM regression) |
| **Beta (β)** | Portfolio sensitivity to market movements |
| **Efficient Frontier** | Set of portfolios maximizing return for a given risk level |
| **Walk-forward CV** | Time-series cross-validation that prevents look-ahead bias |
| **Cross-sectional** | Comparing stocks against each other at the same point in time |
| **Survivorship bias** | Using only stocks that survived — overstates backtest performance |

---

## 🗺️ Roadmap

- [x] Data collection and preprocessing pipeline
- [x] 16-feature technical indicator engineering
- [x] K-Means stock clustering
- [x] Walk-forward ML classification (XGBoost)
- [x] Markowitz portfolio optimization
- [x] Walk-forward backtesting engine with transaction costs
- [x] Performance dashboard vs S&P 500 benchmark
- [ ] Sentiment analysis integration (FinBERT / NLP)
- [ ] Deep learning models (LSTM, Transformer)
- [ ] Reinforcement learning agent (PPO / SAC)
- [ ] Live paper trading via Alpaca API
- [ ] Streamlit interactive dashboard
- [ ] Docker containerization

---

## ⚠️ Disclaimer

```
This project is for educational and research purposes only.
Past performance does not guarantee future results.
This is NOT financial advice.
Do not use this system to trade real money without fully
understanding the risks involved.
Backtest results are subject to survivorship bias inherent
to the S&P 500 universe.
```

---

## 👥 Authors

| Name | Role |
|------|------|
| **Salah Eddine Goumari** | ML Models & Feature Engineering |
| **Youssef EL Krouk** | Portfolio Optimization & Backtesting |
| **Yassine Benmesbah** | Data Pipeline & Performance Analysis |

---

## 📚 References

- Jegadeesh, N. & Titman, S. (1993). *Returns to Buying Winners and Selling Losers*. Journal of Finance, 48(1), 65–91.
- Markowitz, H. (1952). *Portfolio Selection*. Journal of Finance, 7(1), 77–91.
- Fama, E. & French, K. (2015). *A Five-Factor Asset Pricing Model*. Journal of Financial Economics, 116(1), 1–22.
- Wilder, J. W. (1978). *New Concepts in Technical Trading Systems*. Trend Research.
- Moskowitz, T., Ooi, Y. H. & Pedersen, L. H. (2012). *Time Series Momentum*. Journal of Financial Economics, 104(2).

---

## 📄 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

---

<div align="center">

⭐ **If this project helped you, please give it a star!** ⭐

*Made for Quantitative Finance & Machine Learning research*

</div>
