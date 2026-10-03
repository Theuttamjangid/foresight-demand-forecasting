# inspect_data.py
# First look at the four raw files. Nothing is changed or saved here.

import pandas as pd
from pathlib import Path

base = Path(__file__).resolve().parent.parent
data = base / "data"

files = {
    "calendar": "calendar.csv",
    "inventory": "inventory_snapshots.csv",
    "sales": "sales_daily.csv",
    "sku_master": "sku_master.csv",
}

tables = {}
for name, fname in files.items():
    df = pd.read_csv(data / fname)
    tables[name] = df

    print("\n----", name, "----")
    print("rows:", len(df), "| columns:", len(df.columns))
    print(list(df.columns))
    print("missing values:")
    print(df.isnull().sum())
    print("duplicate rows:", df.duplicated().sum())

# sales summary
sales = tables["sales"]
sales["Date"] = pd.to_datetime(sales["Date"])

print("\n---- sales summary ----")
print("total units:", sales["Units_Sold"].sum())
print("total revenue:", round(sales["Revenue"].sum(), 2))
print("skus:", sales["SKU"].nunique())
print("dates:", sales["Date"].min().date(), "to", sales["Date"].max().date(), "(", sales["Date"].nunique(), "days )")
print("negative units:", (sales["Units_Sold"] < 0).sum())
print("zero units:", (sales["Units_Sold"] == 0).sum())

# do the SKUs match across files?
s = set(sales["SKU"])
m = set(tables["sku_master"]["SKU"])
i = set(tables["inventory"]["SKU"])

print("\n---- sku check ----")
print("sales:", len(s), "| master:", len(m), "| inventory:", len(i))
print("in sales but not in master:", len(s - m))
print("in inventory but not in sales:", len(i - s))