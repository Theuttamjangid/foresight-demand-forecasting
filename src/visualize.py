# visualize.py
# Charts for the model results (forecast and inventory risk). Run after forecast.py
# and inventory_analysis.py. Charts go to the reports folder.

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

base = Path(__file__).resolve().parent.parent
data = base / "data"
outputs = base / "outputs"
reports = base / "reports"
reports.mkdir(exist_ok=True)

weekly = pd.read_csv(data / "weekly_features.csv", parse_dates=["Week_Start"])
forecast = pd.read_csv(outputs / "forecasts.csv", parse_dates=["Week_Start"])
inventory = pd.read_csv(outputs / "inventory_analysis.csv")

# 1. total weekly demand: history + forecast
history = weekly.groupby("Week_Start")["Units_Sold"].sum()
future = forecast.groupby("Week_Start")["Predicted_Units"].sum()

plt.figure(figsize=(11, 5))
plt.plot(history.index, history.values, label="Actual")
plt.plot(future.index, future.values, label="Forecast", marker="o", color="orange")
plt.title("Total Weekly Demand: Actual and Forecast")
plt.xlabel("Week")
plt.ylabel("Units")
plt.legend()
plt.tight_layout()
plt.savefig(reports / "forecast_total.png", dpi=120)
plt.close()

# 2. forecast for the 4 biggest SKUs
top_skus = weekly.groupby("SKU")["Units_Sold"].sum().nlargest(4).index

fig, axes = plt.subplots(2, 2, figsize=(12, 7), sharex=True)
for ax, sku in zip(axes.flatten(), top_skus):
    h = weekly[weekly["SKU"] == sku].tail(26)
    f = forecast[forecast["SKU"] == sku]
    ax.plot(h["Week_Start"], h["Units_Sold"], label="Actual")
    ax.plot(f["Week_Start"], f["Predicted_Units"], label="Forecast", color="orange", marker="o")
    ax.set_title(sku)
    ax.tick_params(axis="x", rotation=30)
axes[0][0].legend()
fig.suptitle("Forecast for the Top 4 SKUs (last 26 weeks + next 8)")
plt.tight_layout()
plt.savefig(reports / "forecast_top_skus.png", dpi=120)
plt.close()

# 3. inventory risk counts
counts = inventory["Risk_Category"].value_counts()
plt.figure(figsize=(6, 4))
plt.bar(counts.index, counts.values)
plt.title("SKUs by Inventory Risk Category")
plt.ylabel("Number of SKUs")
plt.xticks(rotation=20)
plt.tight_layout()
plt.savefig(reports / "inventory_risk.png", dpi=120)
plt.close()

print("charts saved in reports folder")