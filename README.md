# Stock Market Data Pipeline

Pipeline do analizy danych giełdowych - od pobierania surowych danych z API, przez przechowywanie w DuckDB, po modele uczenia maszynowego.

## Opis projektu

Kompleksowy pipeline łączący pobieranie danych z wielu źródeł, transformację i modelowanie predykcyjne:
- **Integracja danych**: Pobieranie z publicznych API (SEC, FRED, Yahoo Finance) z obsługą cache'owania
- **Inżynieria danych**: Trójwarstwowa architektura bazy danych (raw → cleaned → staging) w DuckDB
- **Feature engineering**: Konstruowanie cech finansowych i ekonomicznych dla modeli ML
- **Machine Learning**: Modele predykcyjne z walidacją (Random Forest, XGBoost, Logistic Regression)

## Funkcjonalności

- **Integracja danych z wielu źródeł**: Agregacja danych z rynku akcji, rynków finansowych i danych makroekonomicznych
- **System ETL**: Automatyzacja pobierania, czyszczenia i transformacji surowych danych
- **Skalowalna architektura**: Zdolność do obsługi dużych zbiorów danych z optymalizacją wydajności
- **Feature engineering**: Wskaźniki finansowe i makroekonomiczne, Altman Z-score
- **Modele predykcyjne**: Implementacja i walidacja algorytmów ML

## Struktura projektu

```
Stock_market/
├── config/                     # Konfiguracja i metadane
│   ├── settings.yaml.template  # Szablon (skopiuj do settings.yaml)
│   ├── logging.conf            # Konfiguracja logowania
│   └── tickers_list.csv        # Lista tickerów do analizy
├── src/
│   ├── data_fetch/             # Pobieranie danych z API (yfinance, SEC, FRED, Stooq)
│   ├── database/               # Schemat DuckDB (raw, cleaned, staging) + loadery
│   ├── preprocessing/          # Transformacja: normalizacja + feature engineering
│   ├── bankruptcy/             # Modele ML (Random Forest, XGBoost, LR)
│   ├── analysis/               # Statystyki, wizualizacje, raporty
│   └── utils/                  # Helpery: config, logging, API, I/O
├── data/                       # Dane
│   ├── raw/                    # Surowe dane z API (JSON)
│   ├── processed/              # Wyczyszczone dane (CSV)
│   └── cache/                  # Cache SEC filingów (nie usuwać)
├── logs/                       # Logi pipeline'u
├── reports/                    # Raporty i analizy
├── outputs/                    # Wyniki: predykcje, modele ML
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

### Schema `cleaned` (dane wyczyszczone)
| Tabela | Opis |
|--------|------|
| `prices_clean` | Ceny bez duplikatów i outlierów |
| `financials_clean` | Finanse z walidacją |
| `macro_clean` | Makro znormalizowana |

### Schema `staging` (dane do ML)
| Tabela | Opis |
|--------|------|
| `master_dataset` | Granulacja miesięczna, ~88 cech (gotowe do ML) |
| `macro_normalized` | Wskaźniki makro znormalizowane |
| `features_engineered` | Wskaźniki z feature engineering |

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
# Domyślnie: fetch + update (tryb interaktywny)
python main.py

# Zbuduj bazę od zera
python main.py --steps build

# Zaktualizuj istniejącą bazę (auto preprocessing)
python main.py --steps update

# Pobierz dane (wybrane moduły: prices, financials, shares, dividends, sectors, macro)
python main.py --steps fetch --fetch-modules prices financials

# Przetwórz dane (normalizacja + feature engineering)
python main.py --steps preprocess

# Uruchom analizy i raporty
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
