# Stock Market Data Pipeline

Pipeline do analizy danych giełdowych - od pobierania surowych danych z API, przez przechowywanie w DuckDB, po modele predykcji bankructwa.

## Opis projektu

Projekt łączy data engineering i machine learning:
- **Pobieranie danych**: Automatyczne pobieranie z SEC EDGAR, FRED, yfinance, Stooq
- **Baza danych**: Trójwarstwowa architektura DuckDB (raw → cleaned → staging)
- **Feature engineering**: Wskaźniki finansowe, Altman Z-score, dane makroekonomiczne
- **Modele ML**: Predykcja bankructwa (Random Forest, XGBoost, Logistic Regression)

## Funkcjonalności

- **Dane cenowe**: Historyczne OHLCV z yfinance (USA) i Stooq (Polska)
- **Raporty finansowe**: SEC EDGAR API z automatycznym parsowaniem 10-K/10-Q
- **Dane makroekonomiczne**: FRED API (USA) i Eurostat/DBnomics (Polska)
- **Metadane spółek**: ~12k firm zarejestrowanych w SEC z filtrowaniem SIC
- **Aktualizacje przyrostowe**: Inteligentne pobieranie bez zbędnych wywołań API
- **Moduł bankructwa**: Scoring ryzyka z interpretowalnymi cechami

## Struktura projektu

```
Stock_market/
├── config/                     # Pliki konfiguracyjne
│   ├── settings.yaml.template  # Szablon (skopiuj do settings.yaml)
│   ├── logging.conf            # Konfiguracja logowania
│   └── tickers_*.csv/xlsx      # Listy tickerów
├── src/
│   ├── data_fetch/             # Pobieranie danych z API
│   ├── database/               # Schemat DuckDB i loadery
│   ├── preprocessing/          # Transformacja danych
│   ├── bankruptcy/             # Modele ML do predykcji bankructwa
│   ├── analysis/               # Wizualizacja i statystyki
│   └── utils/                  # Funkcje pomocnicze
├── tests/                      # Testy jednostkowe
├── main.py                     # Punkt wejścia CLI
└── requirements.txt            # Zależności Python
```

## Architektura bazy danych

### Schema `main` (surowe dane)
| Tabela | Opis |
|--------|------|
| `prices` | Dane cenowe OHLCV |
| `financials` | Raporty finansowe SEC |
| `macro` | Wskaźniki makroekonomiczne |
| `company_metadata` | Informacje o spółkach SEC |
| `company_status` | Status aktywny/wycofany |

### Schema `staging` (dane do ML)
| Tabela | Opis |
|--------|------|
| `master_dataset` | Granulacja miesięczna, ~88 cech |
| `macro_normalized` | Znormalizowane wskaźniki makro |

## Instalacja

```bash
# Klonowanie i konfiguracja
git clone https://github.com/kkozbial/StockPricePredictor-ML.git
cd StockPricePredictor-ML
python -m venv venv
venv\Scripts\activate  # Windows
pip install -r requirements.txt

# Konfiguracja kluczy API
cp config/settings.yaml.template config/settings.yaml
# Edytuj settings.yaml i wpisz klucz FRED API
```

## Użycie

```bash
# Zbuduj bazę od zera
python main.py --steps build

# Zaktualizuj istniejącą bazę
python main.py --steps update

# Pobierz wybrane moduły danych
python main.py --steps fetch --fetch-modules prices financials

# Uruchom analizy i generuj raporty
python main.py --steps analysis
```

## Moduł ML - Predykcja bankructwa

```python
from src.bankruptcy.train import train_model, compare_models
from src.bankruptcy.data_loader import load_bankruptcy_data

# Załaduj dane treningowe
X, y = load_bankruptcy_data()

# Porównaj modele
results = compare_models(X, y)

# Trenuj wybrany model
result = train_model(X, y, model_type="random_forest")
print(f"ROC-AUC: {result.roc_auc:.3f}")
```

## Wymagania

- Python 3.10+
- Klucz FRED API (darmowy): https://fred.stlouisfed.org/docs/api/api_key.html

## Technologie

- **Baza danych**: DuckDB
- **Przetwarzanie danych**: pandas, NumPy
- **ML**: scikit-learn, XGBoost
- **Wizualizacja**: matplotlib, seaborn
- **API**: yfinance, fredapi, requests

## Licencja

MIT
