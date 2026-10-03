# inventory_analysis.py
# Combines the forecast with the latest inventory snapshot to see which SKUs may run short
# (or are overstocked).

import numpy as np
import pandas as pd
from pathlib import Path

base = Path(__file__).resolve().parent.parent
data = base / "data"
outputs = base / "outputs"

# how many weeks of the forecast to use for the average daily demand
# (lead times here are 3-14 days, so the near term matters most)
weeks_used = 4
overstock_days = 90

inventory = pd.read_csv(data / "inventory_clean.csv", parse_dates=["Snapshot_Date"])
forecast = pd.read_csv(outputs / "forecasts.csv", parse_dates=["Week_Start"])

# latest snapshot only
latest = inventory["Snapshot_Date"].max()
stock = inventory[inventory["Snapshot_Date"] == latest].copy()
print("inventory snapshot:", latest.date(), "|", len(stock), "SKUs")

# average daily demand from the first few forecast weeks
first_weeks = sorted(forecast["Week_Start"].unique())[:weeks_used]
demand = (forecast[forecast["Week_Start"].isin(first_weeks)]
          .groupby("SKU")["Predicted_Units"].sum().reset_index())
demand["Avg_Daily_Demand"] = demand["Predicted_Units"] / (weeks_used * 7)

df = stock.merge(demand[["SKU", "Avg_Daily_Demand"]], on="SKU", how="left")
print("SKUs without forecast:", df["Avg_Daily_Demand"].isnull().sum())

# days of cover (avoid dividing by zero)
df["Days_of_Cover"] = np.where(df["Avg_Daily_Demand"] > 0,
                               df["Current_Stock"] / df["Avg_Daily_Demand"], np.nan)

# what we will need before a new order could arrive
df["Lead_Time_Demand"] = df["Avg_Daily_Demand"] * df["Lead_Time_Days"]

# stock we have or that is already coming
df["Stock_Position"] = df["Current_Stock"] + df["On_Order"]
df["Stock_After_Lead_Time"] = df["Stock_Position"] - df["Lead_Time_Demand"]

# safety stock should still be untouched after the lead time
df["Buffer_vs_Safety"] = df["Stock_After_Lead_Time"] - df["Safety_Stock"]

df["Below_Reorder_Point"] = (df["Stock_Position"] < df["Reorder_Point"]).astype(int)


def risk_category(row):
    if row["Stock_After_Lead_Time"] < 0:
        return "High Risk"          # will run out even with the on-order stock
    if row["Buffer_vs_Safety"] < 0 or row["Below_Reorder_Point"] == 1:
        return "Medium Risk"        # safety stock gets used up or below reorder point
    if row["Days_of_Cover"] > overstock_days:
        return "Overstock"
    return "Healthy"


df["Risk_Category"] = df.apply(risk_category, axis=1)

# a simple suggestion: enough to cover lead time demand + safety stock
df["Suggested_Order_Qty"] = np.ceil(
    np.maximum(df["Lead_Time_Demand"] + df["Safety_Stock"] - df["Stock_Position"], 0))

print("\nrisk categories:")
print(df["Risk_Category"].value_counts())

print("\nhighest risk SKUs:")
cols = ["SKU", "Current_Stock", "On_Order", "Lead_Time_Days", "Avg_Daily_Demand",
        "Days_of_Cover", "Stock_After_Lead_Time", "Risk_Category", "Suggested_Order_Qty"]
order = {"High Risk": 0, "Medium Risk": 1, "Overstock": 2, "Healthy": 3}
print(df.sort_values("Risk_Category", key=lambda s: s.map(order))[cols].head(10).round(1).to_string(index=False))

num_cols = df.select_dtypes("number").columns
df[num_cols] = df[num_cols].round(2)
df.to_csv(outputs / "inventory_analysis.csv", index=False)
print("\nsaved to outputs/inventory_analysis.csv (", len(df), "rows )")