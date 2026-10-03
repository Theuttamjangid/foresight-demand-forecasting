"""
Step 2: Load -> Clean -> Filter -> Save.
Raw files in data/ are never modified; cleaned copies are saved alongside.

Run from the project root:
    python src/clean_all_data.py
"""

import pandas as pd
from pathlib import Path

# --------------------------------------------------
# 1. PATHS
# --------------------------------------------------

project_folder = Path(__file__).resolve().parent.parent
data_folder = project_folder / "data"

# --------------------------------------------------
# 2. LOAD
# --------------------------------------------------

sales = pd.read_csv(data_folder / "sales_daily.csv")
sku_master = pd.read_csv(data_folder / "sku_master.csv")
calendar = pd.read_csv(data_folder / "calendar.csv")
inventory = pd.read_csv(data_folder / "inventory_snapshots.csv")

print("LOADED")
print("Sales:", sales.shape, "| SKU master:", sku_master.shape,
      "| Calendar:", calendar.shape, "| Inventory:", inventory.shape)

# --------------------------------------------------
# 3. CLEAN: dates -> datetime
# --------------------------------------------------
# errors="coerce" turns unreadable dates into NaT instead of crashing.

sales["Date"] = pd.to_datetime(sales["Date"], errors="coerce")
calendar["date"] = pd.to_datetime(calendar["date"], errors="coerce")
inventory["Snapshot_Date"] = pd.to_datetime(inventory["Snapshot_Date"], errors="coerce")
sku_master["Launch_Date"] = pd.to_datetime(sku_master["Launch_Date"], errors="coerce")

# Same key name in every table makes merging easier later
calendar = calendar.rename(columns={"date": "Date"})

# --------------------------------------------------
# 4. CLEAN: duplicates
# --------------------------------------------------

for name, df in [("sales", sales), ("sku_master", sku_master),
                 ("calendar", calendar), ("inventory", inventory)]:
    print(f"Duplicate rows in {name}:", df.duplicated().sum())

sales = sales.drop_duplicates()
sku_master = sku_master.drop_duplicates()
calendar = calendar.drop_duplicates()
inventory = inventory.drop_duplicates()

# --------------------------------------------------
# 5. CLEAN: missing values
# --------------------------------------------------
# Drop sales rows only if an ESSENTIAL field is missing.
# (Currently none are, but the check keeps the script safe for new data.)

before = len(sales)
sales = sales.dropna(subset=["Date", "SKU", "Units_Sold", "Revenue"])
print("Sales rows dropped for missing essentials:", before - len(sales))

# Calendar: NaN in holiday / promotion_event means "none that day", not an error.
calendar["holiday"] = calendar["holiday"].fillna("None")
calendar["promotion_event"] = calendar["promotion_event"].fillna("None")

# --------------------------------------------------
# 6. CLEAN: text columns (remove stray spaces, consistent case)
# --------------------------------------------------

sales["SKU"] = sales["SKU"].str.strip().str.upper()
sku_master["SKU"] = sku_master["SKU"].str.strip().str.upper()
inventory["SKU"] = inventory["SKU"].str.strip().str.upper()

# --------------------------------------------------
# 7. CLEAN: sanity checks on values (flag, do not silently delete)
# --------------------------------------------------

print("Negative Units_Sold rows:", (sales["Units_Sold"] < 0).sum())
print("Zero Units_Sold rows (kept, valid zero demand):", (sales["Units_Sold"] == 0).sum())

negative_margin_skus = sku_master[sku_master["Gross_Margin_Per_Unit"] < 0]["SKU"].tolist()
print("SKUs with negative margin (kept, flagged):", len(negative_margin_skus))

# --------------------------------------------------
# 8. FILTER: keep only SKUs that exist in sales
# --------------------------------------------------
# Inventory has 200 SKUs, but only 50 have sales (so only those can be forecast).

valid_skus = set(sales["SKU"].unique())

inventory = inventory[inventory["SKU"].isin(valid_skus)].copy()
sku_master = sku_master[sku_master["SKU"].isin(valid_skus)].copy()

print("Inventory rows kept after SKU filter:", len(inventory))

# --------------------------------------------------
# 9. SORT + SAVE (raw files stay untouched)
# --------------------------------------------------

sales = sales.sort_values(["SKU", "Date"]).reset_index(drop=True)
inventory = inventory.sort_values(["SKU", "Snapshot_Date"]).reset_index(drop=True)

sales.to_csv(data_folder / "sales_clean.csv", index=False)
sku_master.to_csv(data_folder / "sku_master_clean.csv", index=False)
calendar.to_csv(data_folder / "calendar_clean.csv", index=False)
inventory.to_csv(data_folder / "inventory_clean.csv", index=False)

print("\nCLEANING COMPLETED")
print("sales_clean.csv       :", sales.shape)
print("sku_master_clean.csv  :", sku_master.shape)
print("calendar_clean.csv    :", calendar.shape)
print("inventory_clean.csv   :", inventory.shape)