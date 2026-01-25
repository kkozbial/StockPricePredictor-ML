# Stock Market Data Pipeline

End-to-end machine learning pipeline for stock market analysis - from raw API data collection through DuckDB storage to bankruptcy prediction models.

## Overview

This project demonstrates a complete data engineering and machine learning workflow:
- **Data Collection**: Automated fetching from multiple APIs (SEC EDGAR, FRED, yfinance, Stooq)
- **Data Storage**: Three-layer DuckDB architecture (raw → cleaned → staging)
- **Feature Engineering**: Financial ratios, Altman Z-score, macroeconomic indicators
- **ML Models**: Bankruptcy prediction using Random Forest, XGBoost, Logistic Regression

## Features

- **Price Data**: Historical OHLCV from yfinance (US) and Stooq (Poland)
- **Financial Reports**: SEC EDGAR API with automatic 10-K/10-Q parsing
- **Macroeconomic Data**: FRED API (US) and Eurostat/DBnomics (Poland)
- **Company Metadata**: ~12k SEC-registered companies with SIC filtering
- **Incremental Updates**: Smart fetching to avoid redundant API calls
- **Bankruptcy Module**: ML-based risk scoring with interpretable features

## Project Structure

```
Stock_market/
├── config/                     # Configuration files
│   ├── settings.yaml.template  # Template (copy to settings.yaml)
│   ├── logging.conf            # Logging configuration
│   └── tickers_*.csv/xlsx      # Ticker lists
├── src/
│   ├── data_fetch/             # API data fetchers
│   ├── database/               # DuckDB schema and loaders
│   ├── preprocessing/          # Data transformation
│   ├── bankruptcy/             # ML models for bankruptcy prediction
│   ├── analysis/               # Visualization and statistics
│   └── utils/                  # Helper functions
├── tests/                      # Unit tests
├── main.py                     # CLI entry point
└── requirements.txt            # Python dependencies
```

## Database Architecture

### Schema `main` (raw data)
| Table | Description |
|-------|-------------|
| `prices` | OHLCV price data |
| `financials` | SEC financial reports |
| `macro` | Macroeconomic indicators |
| `company_metadata` | SEC company information |
| `company_status` | Active/delisted status |

### Schema `staging` (ML-ready)
| Table | Description |
|-------|-------------|
| `master_dataset` | Monthly granularity, ~88 features |
| `macro_normalized` | Standardized macro indicators |

## Installation

```bash
# Clone and setup
git clone https://github.com/yourusername/stock-market-pipeline.git
cd stock-market-pipeline
python -m venv venv
venv\Scripts\activate  # Windows
pip install -r requirements.txt

# Configure API keys
cp config/settings.yaml.template config/settings.yaml
# Edit settings.yaml with your FRED API key
```

## Usage

```bash
# Build database from scratch
python main.py --steps build

# Update existing database
python main.py --steps update

# Fetch specific data modules
python main.py --steps fetch --fetch-modules prices financials

# Run analysis and generate reports
python main.py --steps analysis
```

## ML Module - Bankruptcy Prediction

```python
from src.bankruptcy.train import train_model, compare_models
from src.bankruptcy.data_loader import load_bankruptcy_data

# Load training data
X, y = load_bankruptcy_data()

# Compare models
results = compare_models(X, y)

# Train selected model
result = train_model(X, y, model_type="random_forest")
print(f"ROC-AUC: {result.roc_auc:.3f}")
```

## Requirements

- Python 3.10+
- FRED API key (free): https://fred.stlouisfed.org/docs/api/api_key.html

## Tech Stack

- **Database**: DuckDB
- **Data Processing**: pandas, NumPy
- **ML**: scikit-learn, XGBoost
- **Visualization**: matplotlib, seaborn
- **APIs**: yfinance, fredapi, requests

## License

MIT
