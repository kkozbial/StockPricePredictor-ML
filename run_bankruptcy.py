"""Skrypt do eksportu wyników predykcji bankructwa do Excela."""
from src.bankruptcy.train import train_model

# Trenuj model
result = train_model(model_name="xgboost")

# Sortuj po prawdopodobieństwie i zapisz
predictions = result.predictions.sort_values("y_prob", ascending=False)
predictions.to_excel("bankruptcy_predictions.xlsx", index=False)

print(f"\nZapisano {len(predictions)} wierszy do bankruptcy_predictions.xlsx")
print(predictions.head(20))
