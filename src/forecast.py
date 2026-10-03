# forecast.py
# Predicts the next 8 weeks for every SKU. Each predicted week is fed back in as history
# so the following week can use it for its lag features.

import numpy as np
import pandas as pd
import joblib
from pathlib import Path

base = Path(__file__).resolve().parent.parent
data = base / "data"
models = base / "models"
outputs = base / "outputs"
outputs.mkdir(exist_ok=True)

forecast_weeks = 8

# Future promotions / holidays are not known yet.
# "last_year" copies the same week of last year (promotions repeat at the same time each year),
# "none" assumes no promotion and no holiday.
future_events = "last_year"

weekly = pd.read_csv(data / "weekly_features.csv", parse_dates=["Week_Start"])
model = joblib.load(models / "demand_model.pkl")
features = joblib.load(models / "feature_list.pkl")

last_week = weekly["Week_Start"].max()
future = pd.date_range(last_week + pd.Timedelta(weeks=1), periods=forecast_weeks, freq="W-MON")
print("last actual week:", last_week.date())
print("forecasting:", future.min().date(), "to", future.max().date())

rows = []

for sku, d in weekly.groupby("SKU"):
    d = d.sort_values("Week_Start")
    history = list(d["Units_Sold"])
    events = d.set_index("Week_Start")[["Promo_Days", "Holiday_Days"]]
    code = int(d["SKU_Code"].iloc[0])

    for week in future:
        promo, holiday = 0, 0
        if future_events == "last_year":
            last_year = week - pd.Timedelta(weeks=52)
            if last_year in events.index:
                promo = events.loc[last_year, "Promo_Days"]
                holiday = events.loc[last_year, "Holiday_Days"]

        x = pd.DataFrame({
            "Lag_1": [history[-1]],
            "Lag_2": [history[-2]],
            "Lag_4": [history[-4]],
            "Rolling_4": [np.mean(history[-4:])],
            "Week_of_Year": [int(week.isocalendar().week)],
            "Month": [week.month],
            "Promo_Days": [promo],
            "Holiday_Days": [holiday],
            "SKU_Code": [code],
        })

        pred = max(float(model.predict(x[features])[0]), 0)
        history.append(pred)
        rows.append({"Week_Start": week, "SKU": sku,
                     "Predicted_Units": round(pred, 2), "Promo_Days_Assumed": promo})

result = pd.DataFrame(rows)
result.to_csv(outputs / "forecasts.csv", index=False)

print("rows:", len(result), "| SKUs:", result["SKU"].nunique())
print("\ntotal predicted units per week:")
print(result.groupby("Week_Start")["Predicted_Units"].sum().round(0))
print("\nsaved to outputs/forecasts.csv")