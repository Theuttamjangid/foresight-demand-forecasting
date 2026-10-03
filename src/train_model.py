# train_model.py
# Trains the weekly demand model, checks it on the last 12 weeks and compares with a simple baseline.

import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

base = Path(__file__).resolve().parent.parent
data = base / "data"
models = base / "models"
models.mkdir(exist_ok=True)

features = ["Lag_1", "Lag_2", "Lag_4", "Rolling_4", "Week_of_Year",
            "Month", "Promo_Days", "Holiday_Days", "SKU_Code"]
target = "Units_Sold"
test_weeks = 12


def make_model():
    return HistGradientBoostingRegressor(
        max_iter=300, learning_rate=0.05, max_leaf_nodes=15,
        categorical_features=["SKU_Code"], random_state=42)


def show_scores(title, actual, predicted):
    mae = mean_absolute_error(actual, predicted)
    rmse = np.sqrt(mean_squared_error(actual, predicted))
    wape = np.abs(actual - predicted).sum() / actual.sum() * 100
    print(f"{title}: MAE {mae:.2f} | RMSE {rmse:.2f} | WAPE {wape:.2f}%")


weekly = pd.read_csv(data / "weekly_features.csv", parse_dates=["Week_Start"])
df = weekly.dropna(subset=features + [target])

# split by time, last 12 weeks are the test set
all_weeks = sorted(df["Week_Start"].unique())
split_week = all_weeks[-test_weeks]
train = df[df["Week_Start"] < split_week]
test = df[df["Week_Start"] >= split_week]

print("train:", train["Week_Start"].min().date(), "to", train["Week_Start"].max().date(), "-", len(train), "rows")
print("test :", test["Week_Start"].min().date(), "to", test["Week_Start"].max().date(), "-", len(test), "rows")

model = make_model()
model.fit(train[features], train[target])
pred = np.maximum(model.predict(test[features]), 0)

print()
show_scores("model   ", test[target].values, pred)
show_scores("baseline", test[target].values, test["Lag_52"].values)

# final model uses all the data
final_model = make_model()
final_model.fit(df[features], df[target])

joblib.dump(final_model, models / "demand_model.pkl")
joblib.dump(features, models / "feature_list.pkl")
print("\nmodel saved in models folder")