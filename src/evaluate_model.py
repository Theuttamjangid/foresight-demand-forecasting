# evaluate_model.py
# Honest check of the model. The score in train_model.py uses real last-week values as lags
# (one step ahead). In real use we forecast several weeks ahead using our own predictions,
# so errors pile up. Here we copy that situation:
#   1. hide the last 8 weeks
#   2. train only on the weeks before them
#   3. forecast the 8 weeks recursively
#   4. compare with what really happened, and with the seasonal naive baseline

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.ensemble import HistGradientBoostingRegressor

base = Path(__file__).resolve().parent.parent
data = base / "data"
outputs = base / "outputs"
outputs.mkdir(exist_ok=True)

horizon = 8

features = ["Lag_1", "Lag_2", "Lag_4", "Rolling_4", "Week_of_Year",
            "Month", "Promo_Days", "Holiday_Days", "SKU_Code"]

weekly = pd.read_csv(data / "weekly_features.csv", parse_dates=["Week_Start"])

weeks = sorted(weekly["Week_Start"].unique())
cutoff = weeks[-horizon]          # first hidden week
test_weeks = weeks[-horizon:]

# 1. train only on the past
past = weekly[weekly["Week_Start"] < cutoff].dropna(subset=features + ["Units_Sold"])

model = HistGradientBoostingRegressor(
    max_iter=300, learning_rate=0.05, max_leaf_nodes=15,
    categorical_features=["SKU_Code"], random_state=42)
model.fit(past[features], past["Units_Sold"])

# 2. recursive forecast for the hidden weeks
rows = []
for sku, d in weekly.groupby("SKU"):
    d = d.sort_values("Week_Start")
    history = list(d[d["Week_Start"] < cutoff]["Units_Sold"])
    events = d.set_index("Week_Start")[["Promo_Days", "Holiday_Days"]]
    code = int(d["SKU_Code"].iloc[0])

    for week in test_weeks:
        week = pd.Timestamp(week)
        # promo / holiday assumed from the same week last year, like in forecast.py
        ly = week - pd.Timedelta(weeks=52)
        promo = events.loc[ly, "Promo_Days"] if ly in events.index else 0
        holiday = events.loc[ly, "Holiday_Days"] if ly in events.index else 0

        x = pd.DataFrame({
            "Lag_1": [history[-1]], "Lag_2": [history[-2]], "Lag_4": [history[-4]],
            "Rolling_4": [np.mean(history[-4:])],
            "Week_of_Year": [int(week.isocalendar().week)], "Month": [week.month],
            "Promo_Days": [promo], "Holiday_Days": [holiday], "SKU_Code": [code]})

        pred = max(float(model.predict(x[features])[0]), 0)
        history.append(pred)
        rows.append({"Week_Start": week, "SKU": sku, "Predicted": pred})

pred_df = pd.DataFrame(rows)

# 3. compare with actuals and the baseline
actual_df = weekly[weekly["Week_Start"].isin(test_weeks)][["Week_Start", "SKU", "Units_Sold", "Lag_52"]]
result = pred_df.merge(actual_df, on=["Week_Start", "SKU"])
result["Step"] = result.groupby("SKU").cumcount() + 1      # 1 = first week ahead ... 8


def scores(actual, predicted):
    mae = np.abs(actual - predicted).mean()
    rmse = np.sqrt(((actual - predicted) ** 2).mean())
    wape = np.abs(actual - predicted).sum() / actual.sum() * 100
    return mae, rmse, wape


print("recursive backtest:", pd.Timestamp(test_weeks[0]).date(), "to", pd.Timestamp(test_weeks[-1]).date())
print("training data ends before", pd.Timestamp(cutoff).date())
print()

m = scores(result["Units_Sold"], result["Predicted"])
b = scores(result["Units_Sold"], result["Lag_52"])

print(f"{'':10s}{'MAE':>8s}{'RMSE':>8s}{'WAPE %':>9s}")
print(f"{'model':10s}{m[0]:8.2f}{m[1]:8.2f}{m[2]:9.2f}")
print(f"{'baseline':10s}{b[0]:8.2f}{b[1]:8.2f}{b[2]:9.2f}")

print("\nWAPE by weeks ahead:")
by_step = result.groupby("Step").apply(
    lambda g: np.abs(g["Units_Sold"] - g["Predicted"]).sum() / g["Units_Sold"].sum() * 100,
    include_groups=False)
print(by_step.round(2))

result.to_csv(outputs / "backtest_results.csv", index=False)
print("\nsaved to outputs/backtest_results.csv")