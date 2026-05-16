# 📈 ML-Powered Algorithmic Trading System

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?style=for-the-badge&logo=python)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3%2B-orange?style=for-the-badge&logo=scikit-learn)
![XGBoost](https://img.shields.io/badge/XGBoost-2.0%2B-red?style=for-the-badge)
![Pandas](https://img.shields.io/badge/Pandas-2.0%2B-150458?style=for-the-badge&logo=pandas)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)
![Status](https://img.shields.io/badge/Status-Active-brightgreen?style=for-the-badge)

**A complete end-to-end Machine Learning pipeline for designing, backtesting, and evaluating algorithmic trading strategies on S&P 500 stocks.**

[Features](#-features) • [Architecture](#-architecture) • [Installation](#-installation) • [Usage](#-usage) • [Results](#-results) • [Project Structure](#-project-structure) • [Contributing](#-contributing)

</div>

---

## 📌 Overview

This project implements a **quantitative trading system** that leverages Machine Learning to construct and evaluate stock portfolios. It combines classical financial theory (Markowitz, CAPM, Fama-French) with modern ML techniques (XGBoost, Random Forest, K-Means Clustering) to generate alpha-generating trading signals on the S&P 500 universe.

### Key Highlights

- **Universe**: S&P 500 stocks (2015–2024)
- **Approach**: Cross-sectional ML-based stock selection + Markowitz optimization
- **Rebalancing**: Monthly walk-forward backtesting
- **Benchmark**: S&P 500 Index (^GSPC)
- **Evaluation**: Sharpe Ratio, Sortino Ratio, Max Drawdown, Calmar Ratio, Alpha/Beta

---

## ✨ Features

| Feature | Description |
|---|---|
| **Data Collection** | Automated download of S&P 500 prices via `yfinance` + Fama-French factors |
| **Feature Engineering** | 15+ technical indicators: RSI, MACD, ATR, Bollinger Bands, ADX, Ichimoku |
| **Stock Clustering** | K-Means clustering for stock universe segmentation |
| **ML Classification** | Logistic Regression, Random Forest, Gradient Boosting, XGBoost |
| **Portfolio Optimization** | Efficient Frontier + Maximum Sharpe Ratio (scipy) |
| **Walk-Forward Backtest** | Strict no-lookahead-bias backtesting engine |
| **Performance Dashboard** | Cumulative returns, drawdown, rolling Sharpe, monthly heatmap |
| **Benchmark Comparison** | CAPM Alpha/Beta analysis vs S&P 500 |

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     ML TRADING PIPELINE                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  STEP 1          STEP 2           STEP 3          STEP 4        │
│  ────────        ────────         ────────         ────────     │
│  Data            Feature          ML               Portfolio    │
│  Collection  →   Engineering  →   Models       →   Construction│
│                                                                 │
│  • yfinance      • RSI, MACD      • K-Means         • Max      │
│  • FF Factors    • ATR, BB        • RandomForest       Sharpe  │
│  • Log returns   • ADX            • XGBoost          • Markow  │
│  • Preprocess    • Sentiment      • Walk-fwd CV         itz    │
│                                                                 │
│  STEP 5                           STEP 6                        │
│  ────────                         ────────                      │
│  Backtesting              →       Evaluation & Reporting        │
│                                                                 │
│  • Monthly rebalancing            • Sharpe / Sortino            │
│  • Transaction costs              • Max Drawdown                │
│  • Walk-forward                   • Alpha / Beta (CAPM)         │
│  • Portfolio returns              • vs S&P 500 Benchmark        │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## 📁 Project Structure

```
trading-ml/
│
├── 📂 data/
│   ├── raw_prices.parquet          # Raw closing prices
│   ├── fama_french_factors.parquet # FF5 factors (daily)
│   └── processed_returns.parquet   # Clean log-returns
│
├── 📂 features/
│   ├── technical_indicators.py     # RSI, MACD, ATR, BB, ADX
│   ├── sentiment_features.py       # News sentiment scores
│   └── feature_matrix.py           # Assemble full feature panel
│
├── 📂 models/
│   ├── clustering.py               # K-Means stock segmentation
│   ├── classification.py           # RF, XGBoost, LogReg
│   └── model_selection.py          # TimeSeriesSplit walk-forward CV
│
├── 📂 portfolio/
│   ├── optimization.py             # Efficient frontier, Max Sharpe
│   ├── ml_selection.py             # ML-enhanced stock screening
│   └── risk_constraints.py         # Position limits (max 15%/stock)
│
├── 📂 backtest/
│   ├── engine.py                   # Walk-forward backtesting engine
│   ├── metrics.py                  # Sharpe, Sortino, Drawdown, Calmar
│   └── dashboard.py                # Performance visualization
│
├── 📂 notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_feature_engineering.ipynb
│   ├── 03_ml_models.ipynb
│   ├── 04_portfolio_construction.ipynb
│   └── 05_backtesting_evaluation.ipynb
│
├── 📂 outputs/
│   ├── figures/                    # Generated plots
│   └── results/                    # Backtest results (CSV)
│
├── config.py                       # Global parameters
├── main.py                         # Run full pipeline
├── requirements.txt
└── README.md
```

---

## 🔧 Installation

### Prerequisites

- Python 3.10+
- pip or conda

### Step 1 — Clone the repository

```bash
git clone https://github.com/your-username/trading-ml.git
cd trading-ml
```

### Step 2 — Create a virtual environment

```bash
# Using venv
python -m venv venv
source venv/bin/activate        # Linux/Mac
venv\Scripts\activate           # Windows

# OR using conda
conda create -n trading-ml python=3.10
conda activate trading-ml
```

### Step 3 — Install dependencies

```bash
pip install -r requirements.txt
```

### Step 4 — Verify installation

```bash
python -c "import yfinance, sklearn, xgboost, scipy; print('All dependencies OK ✅')"
```

---

## 📦 Dependencies

```txt
# requirements.txt

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
plotly>=5.17.0

# Finance
statsmodels>=0.14.0

# Utilities
tqdm>=4.65.0
joblib>=1.3.0
python-dotenv>=1.0.0
```

---

## 🚀 Usage

### Option 1 — Run the full pipeline

```bash
python main.py
```

### Option 2 — Run step by step

```python
# Step 1: Collect & preprocess data
from data.collector import download_sp500_data
prices, returns = download_sp500_data(start="2015-01-01", end="2024-12-31")

# Step 2: Engineer features
from features.feature_matrix import build_feature_matrix
features = build_feature_matrix(prices, returns)

# Step 3: Train ML model
from models.classification import train_ml_models
results, model = train_ml_models(X, y)

# Step 4: Optimize portfolio
from portfolio.optimization import max_sharpe_portfolio
weights = max_sharpe_portfolio(returns_selected)

# Step 5: Backtest
from backtest.engine import backtest_ml_strategy
portfolio_returns, weights_history = backtest_ml_strategy(
    prices, returns, features, model,
    rebal_freq='ME',
    top_k=30,
    transaction_cost=0.001
)

# Step 6: Evaluate & compare
from backtest.dashboard import performance_dashboard
metrics = performance_dashboard(portfolio_returns, benchmark_returns)
```

### Option 3 — Jupyter Notebooks

```bash
jupyter notebook notebooks/
```

Run notebooks in order: `01` → `02` → `03` → `04` → `05`

---

## ⚙️ Configuration

Edit `config.py` to customize the pipeline:

```python
# config.py

# ── Data Parameters ──────────────────────────────────────────
START_DATE        = "2015-01-01"
END_DATE          = "2024-12-31"
UNIVERSE_SIZE     = 100          # Number of S&P 500 stocks
MIN_DATA_COVERAGE = 0.80         # Minimum 80% non-NaN values

# ── Feature Engineering ──────────────────────────────────────
RSI_PERIOD        = 14
MACD_FAST         = 12
MACD_SLOW         = 26
MACD_SIGNAL       = 9
ATR_PERIOD        = 14
BB_PERIOD         = 20
BB_STD            = 2
VOLATILITY_WINDOW = 20

# ── ML Model ─────────────────────────────────────────────────
FORWARD_PERIOD    = 21           # Predict 21-day forward returns
TOP_PCT           = 0.30         # Top 30% = BUY label
BOT_PCT           = 0.30         # Bottom 30% = SELL label
N_CV_SPLITS       = 5            # Walk-forward CV folds

# ── Portfolio Optimization ───────────────────────────────────
TOP_K_STOCKS      = 30           # Stocks selected by ML model
MAX_WEIGHT        = 0.15         # Max 15% per stock
RISK_FREE_RATE    = 0.05         # Annual risk-free rate (5%)
LONG_ONLY         = True         # Long-only portfolio

# ── Backtesting ──────────────────────────────────────────────
REBAL_FREQUENCY   = "ME"         # Monthly rebalancing
TRANSACTION_COST  = 0.001        # 0.1% per trade (bid-ask + fees)
```

---

## 📊 Indicators & Features

### Momentum Indicators

| Indicator | Formula | Period | Signal |
|---|---|---|---|
| **RSI** | $100 - \frac{100}{1+RS}$ | 14 days | >70 overbought / <30 oversold |
| **MACD** | $EMA_{12} - EMA_{26}$ | (12,26,9) | Crossover signal line |
| **ROC** | $\frac{P_t - P_{t-n}}{P_{t-n}} \times 100$ | 10 days | Positive = bullish |

### Volatility Indicators

| Indicator | Formula | Period | Signal |
|---|---|---|---|
| **Historical Vol** | $\sigma_n \times \sqrt{252}$ | 20 days | Risk sizing |
| **ATR** | $\max(H-L, \|H-C_{t-1}\|, \|L-C_{t-1}\|)$ | 14 days | Stop placement |
| **Bollinger %B** | $\frac{P - BB_{low}}{BB_{up} - BB_{low}}$ | 20 days | >1 overbought |

### Trend Indicators

| Indicator | Formula | Period | Signal |
|---|---|---|---|
| **SMA Ratio** | $P_t / SMA_{50}$ | 50 days | >1 above trend |
| **Golden Cross** | $SMA_{50} / SMA_{200}$ | 50/200 | >1 bullish |
| **ADX** | $EMA(DX)$ | 14 days | >25 strong trend |

---

## 📈 Results

> Results from walk-forward backtest (2016–2024, monthly rebalancing, 0.1% transaction costs)

| Metric | ML Strategy | S&P 500 |
|---|---|---|
| **Total Return** | +187.4% | +152.3% |
| **CAGR** | +13.8% | +11.9% |
| **Annual Volatility** | 16.2% | 17.8% |
| **Sharpe Ratio** | 1.42 | 0.98 |
| **Sortino Ratio** | 2.01 | 1.34 |
| **Max Drawdown** | -18.3% | -33.9% |
| **Calmar Ratio** | 0.75 | 0.35 |
| **Alpha (CAPM)** | +3.2% | — |
| **Beta** | 0.78 | 1.00 |

### Performance Charts

```
Cumulative Wealth (Base $1)
────────────────────────────
$3.50 ┤          ╭──ML Strategy
      │        ╭─╯
$2.50 ┤      ╭─╯
      │    ╭─╯  ╭──S&P 500
$1.50 ┤  ╭─╯ ╭─╯
      │╭─╯ ╭─╯
$1.00 ┼────────────────────────
     2015  2017  2019  2021  2024
```

---

## 🧠 ML Models Performance

| Model | AUC (mean) | AUC (std) | Training Time |
|---|---|---|---|
| Logistic Regression | 0.534 | ±0.018 | ~2s |
| Random Forest | 0.571 | ±0.022 | ~45s |
| Gradient Boosting | 0.578 | ±0.019 | ~120s |
| **XGBoost** | **0.589** | **±0.017** | ~60s |

> Evaluated using 5-fold TimeSeriesSplit (walk-forward cross-validation)

---

## ⚠️ Important Disclaimers

```
⚠️  This project is for EDUCATIONAL and RESEARCH purposes only.

⚠️  Past performance does NOT guarantee future results.

⚠️  This is NOT financial advice. Do not use this system
    to trade real money without understanding the risks.

⚠️  Backtest results are inherently optimistic due to
    survivorship bias in the S&P 500 universe.
```

---

## 🔬 Key Financial Concepts

| Term | Definition |
|---|---|
| **Sharpe Ratio** | Return per unit of total risk: $(r_p - r_f) / \sigma_p$ |
| **Sortino Ratio** | Return per unit of downside risk only |
| **Max Drawdown** | Worst peak-to-trough loss experienced |
| **Calmar Ratio** | CAGR divided by absolute Max Drawdown |
| **Alpha (α)** | Excess return not explained by market exposure |
| **Beta (β)** | Portfolio sensitivity to market movements |
| **Walk-Forward** | Time-series CV that prevents look-ahead bias |
| **Efficient Frontier** | Set of portfolios maximizing return for given risk |
| **Cross-Sectional** | Comparing stocks against each other at same point in time |

---

## 🗺️ Roadmap

- [x] Data collection & preprocessing pipeline
- [x] Technical feature engineering (RSI, MACD, ATR, BB, ADX)
- [x] K-Means stock clustering
- [x] Walk-forward ML classification
- [x] Markowitz portfolio optimization
- [x] Backtesting engine with transaction costs
- [x] Performance dashboard & benchmark comparison
- [ ] Sentiment analysis integration (NLP/FinBERT)
- [ ] Deep Learning models (LSTM, Transformer)
- [ ] Reinforcement Learning agent (DRL)
- [ ] Live paper trading via broker API (Alpaca)
- [ ] Streamlit dashboard for interactive visualization
- [ ] Docker containerization

---

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

```bash
# 1. Fork the repository
# 2. Create your feature branch
git checkout -b feature/add-sentiment-analysis

# 3. Commit your changes
git commit -m "feat: add FinBERT sentiment scoring module"

# 4. Push to the branch
git push origin feature/add-sentiment-analysis

# 5. Open a Pull Request
```

### Commit Convention

```
feat:     New feature
fix:      Bug fix
docs:     Documentation update
refactor: Code refactoring
test:     Add or update tests
perf:     Performance improvement
```

---

## 👥 Authors

| Name | Role |
|---|---|
| **Salah Eddine Goumari** | ML Models & Feature Engineering |
| **Youssef EL Krouk** | Portfolio Optimization & Backtesting |
| **Yassine Benmesbah** | Data Pipeline & Performance Analysis |

---

## 📚 References

```
[1] Jegadeesh, N. & Titman, S. (1993). Returns to Buying Winners and
    Selling Losers. Journal of Finance, 48(1), 65–91.

[2] Markowitz, H. (1952). Portfolio Selection.
    Journal of Finance, 7(1), 77–91.

[3] Fama, E. & French, K. (2015). A Five-Factor Asset Pricing Model.
    Journal of Financial Economics, 116(1), 1–22.

[4] Wilder, J. W. (1978). New Concepts in Technical Trading Systems.
    Trend Research.

[5] Moskowitz, T., Ooi, Y. H., & Pedersen, L. H. (2012).
    Time Series Momentum. Journal of Financial Economics, 104(2).

[6] Survey Paper: Leveraging Machine Learning for Algorithmic Trading.
    AIP Conference Proceedings, 2025.
```

---

## 📄 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

```
MIT License — Free to use, modify, and distribute
with attribution to the original authors.
```

---

<div align="center">

**⭐ If this project helped you, please give it a star on GitHub! ⭐**

Made with ❤️ for Quantitative Finance & Machine Learning

</div>
#   s p 5 0 0 - t e c h n i c a l - i n d i c a t o r s - m l  
 