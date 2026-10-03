# build_dataset.py
# Joins sales, calendar and SKU master into one analysis-ready table.
# (Inventory is not joined because it is a monthly snapshot, not a daily record.)

import pandas as pd
from pathlib import Path

base = Path(__file__).resolve().parent.parent
data = base / "data"

sales = pd.read_csv(data / "sales_clean.csv", parse_dates=["Date"])
calendar = pd.read_csv(data / "calendar_clean.csv", parse_dates=["Date"])
sku_master = pd.read_csv(data / "sku_master_clean.csv", parse_dates=["Launch_Date"])

# how="left" keeps every sales row even if a match is missing
dataset = sales.merge(calendar, on="Date", how="left")
dataset = dataset.merge(sku_master, on="SKU", how="left")

# sanity checks - the merge should not add or lose rows
assert len(dataset) == len(sales), "row count changed after merge"
print("missing after merge:", dataset[["Category", "season"]].isnull().sum().sum())

dataset.to_csv(data / "analysis_ready.csv", index=False)

print("rows:", len(dataset), "| columns:", len(dataset.columns))
print(list(dataset.columns))
print("saved to data/analysis_ready.csv")