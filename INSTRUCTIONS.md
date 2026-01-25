# Instrukcja użycia

## Instalacja

```bash
# Środowisko
python -m venv venv
venv\Scripts\activate          # Windows
pip install -r requirements.txt

# Konfiguracja API
copy config\settings.yaml.template config\settings.yaml
# Edytuj settings.yaml i wpisz klucz FRED API
```

## Podstawowe komendy

```bash
# Zbuduj bazę od zera
python main.py --steps build

# Zaktualizuj bazę
python main.py --steps update

# Tylko pobierz dane
python main.py --steps fetch

# Tylko preprocessing
python main.py --steps preprocess

# Analizy i wykresy
python main.py --steps analysis
```

## Opcje

```bash
# Wybrane tickery
python main.py --steps fetch --tickers AAPL GOOGL MSFT

# Wybrane moduły: prices, financials, shares, dividends, sectors, macro
python main.py --steps fetch --fetch-modules prices financials

# Metadane SEC
python main.py --fetch-metadata
```

## Uruchamianie modułów

```bash
python -m src.analysis.visualization
python -m src.analysis.correlations
python -m src.analysis.descriptive_stats
```

## Rozwiązywanie problemów

**"ImportError: relative import"** - użyj `python -m src.modul` zamiast `python src/modul.py`
