# feature_engineering.py
# Converts daily sales into weekly data per SKU and adds lag / rolling features.

import pandas as pd
from pathlib import Path

base = Path(__file__).resolve().parent.parent
data = base / "data"

sales = pd.read_csv(data / "sales_clean.csv", parse_dates=["Date"])
calendar = pd.read_csv(data / "calendar_clean.csv", parse_dates=["Date"])

df = sales.merge(calendar[["Date", "is_holiday"]], on="Date", how="left")

# week starts on Monday
df["Week_Start"] = df["Date"] - pd.to_timedelta(df["Date"].dt.dayofweek, unit="D")

weekly = df.groupby(["SKU", "Week_Start"]).agg(
    Units_Sold=("Units_Sold", "sum"),
    Promo_Days=("Promotion", "sum"),
    Holiday_Days=("is_holiday", "sum"),
    Days=("Date", "count"),
).reset_index()

# the last week of the data has only 3 days, so drop incomplete weeks
print("incomplete SKU-weeks dropped:", (weekly["Days"] < 7).sum())
weekly = weekly[weekly["Days"] == 7].drop(columns="Days")
weekly = weekly.sort_values(["SKU", "Week_Start"]).reset_index(drop=True)

# calendar features
weekly["Week_of_Year"] = weekly["Week_Start"].dt.isocalendar().week.astype(int)
weekly["Month"] = weekly["Week_Start"].dt.month
weekly["SKU_Code"] = weekly["SKU"].str.extract(r"(\d+)")[0].astype(int)

# lag features (shift keeps each row looking only at the past of its own SKU)
units = weekly.groupby("SKU")["Units_Sold"]
weekly["Lag_1"] = units.shift(1)
weekly["Lag_2"] = units.shift(2)
weekly["Lag_4"] = units.shift(4)
weekly["Rolling_4"] = units.transform(lambda x: x.shift(1).rolling(4).mean())

# same week last year - only used for the baseline comparison
weekly["Lag_52"] = units.shift(52)

# the NaN rows at the start of each SKU are left as they are (they are not zero demand)
weekly.to_csv(data / "weekly_features.csv", index=False)

print("rows:", len(weekly), "| columns:", len(weekly.columns))
print("weeks per SKU:", weekly.groupby("SKU").size().unique())
print("from", weekly["Week_Start"].min().date(), "to", weekly["Week_Start"].max().date())
