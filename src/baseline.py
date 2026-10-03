# baseline.py
# Simple benchmark: predict each week with the demand of the same week last year (seasonal naive).
# The ML model has to beat this number to be worth using.

import numpy as np
import pandas as pd
from pathlib import Path

base = Path(__file__).resolve().parent.parent
data = base / "data"

test_weeks = 12   # same test period as train_model.py

weekly = pd.read_csv(data / "weekly_features.csv", parse_dates=["Week_Start"])

# Lag_52 = units sold 52 weeks earlier for the same SKU
df = weekly.dropna(subset=["Lag_52", "Units_Sold"])

last_weeks = sorted(df["Week_Start"].unique())[-test_weeks:]
test = df[df["Week_Start"].isin(last_weeks)]

actual = test["Units_Sold"]
predicted = test["Lag_52"]

mae = np.abs(actual - predicted).mean()
rmse = np.sqrt(((actual - predicted) ** 2).mean())
wape = np.abs(actual - predicted).sum() / actual.sum() * 100

print("SEASONAL NAIVE BASELINE (same week last year)")
print(f"MAE  : {mae:.2f}")
print(f"RMSE : {rmse:.2f}")
print(f"WAPE : {wape:.2f}%")
print("\ntest period:", test["Week_Start"].min().date(), "to", test["Week_Start"].max().date())
print("rows:", len(test))