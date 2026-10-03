"""
Step 3: Exploratory Data Analysis (EDA).
Reads the cleaned files, prints key statistics and saves charts to reports/.

Run from the project root (after clean_all_data.py):
    python src/eda.py
"""

import pandas as pd
import matplotlib
matplotlib.use("Agg")  # save charts to files, no pop-up windows
import matplotlib.pyplot as plt
from pathlib import Path

# --------------------------------------------------
# PATHS
# --------------------------------------------------

project_folder = Path(__file__).resolve().parent.parent
data_folder = project_folder / "data"
reports_folder = project_folder / "reports"
reports_folder.mkdir(exist_ok=True)

# --------------------------------------------------
# LOAD + MERGE (in memory, for analysis only)
# --------------------------------------------------

sales = pd.read_csv(data_folder / "sales_clean.csv", parse_dates=["Date"])
sku = pd.read_csv(data_folder / "sku_master_clean.csv")
calendar = pd.read_csv(data_folder / "calendar_clean.csv", parse_dates=["Date"])

df = (
    sales
    .merge(sku[["SKU", "Category", "Subcategory", "Cost_Price", "Gross_Margin_Per_Unit"]],
           on="SKU", how="left")
    .merge(calendar[["Date", "is_holiday", "holiday", "season"]],
           on="Date", how="left")
)

df["Month"] = df["Date"].dt.to_period("M")
df["Day_Name"] = df["Date"].dt.day_name()
df["Gross_Profit"] = df["Units_Sold"] * df["Gross_Margin_Per_Unit"]

print("Merged dataset:", df.shape)


def save_chart(name):
    plt.tight_layout()
    plt.savefig(reports_folder / name, dpi=120)
    plt.close()


# --------------------------------------------------
# 1. BASIC STATISTICS
# --------------------------------------------------

print("\n--- BASIC STATISTICS ---")
print(df[["Units_Sold", "Revenue", "Price"]].describe().round(2))

# --------------------------------------------------
# 2. MONTHLY REVENUE TREND (seasonality)
# --------------------------------------------------

monthly = df.groupby("Month")[["Revenue", "Units_Sold"]].sum()

print("\n--- MONTHLY REVENUE ---")
print(monthly.round(0))

plt.figure(figsize=(10, 5))
plt.plot(monthly.index.astype(str), monthly["Revenue"], marker="o")
plt.title("Monthly Revenue Trend")
plt.xlabel("Month")
plt.ylabel("Revenue")
plt.xticks(rotation=45)
save_chart("monthly_revenue_trend.png")

# --------------------------------------------------
# 3. WEEKLY UNITS TREND
# --------------------------------------------------

weekly = df.set_index("Date")["Units_Sold"].resample("W").sum()

plt.figure(figsize=(10, 4))
plt.plot(weekly.index, weekly.values)
plt.title("Weekly Units Sold (all SKUs)")
plt.xlabel("Week")
plt.ylabel("Units")
save_chart("weekly_units_trend.png")

# --------------------------------------------------
# 4. TOP AND BOTTOM SKUs
# --------------------------------------------------

sku_units = df.groupby("SKU")["Units_Sold"].sum().sort_values(ascending=False)

print("\n--- TOP 10 SKUs BY UNITS ---")
print(sku_units.head(10))
print("\n--- BOTTOM 10 SKUs BY UNITS ---")
print(sku_units.tail(10))

plt.figure(figsize=(10, 5))
plt.bar(sku_units.head(10).index, sku_units.head(10).values)
plt.title("Top 10 SKUs by Units Sold")
plt.xlabel("SKU")
plt.ylabel("Units Sold")
plt.xticks(rotation=45)
save_chart("top_10_skus.png")

# --------------------------------------------------
# 5. DAY-OF-WEEK PATTERN
# --------------------------------------------------

order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
dow = df.groupby("Day_Name")["Units_Sold"].mean().reindex(order)

print("\n--- AVG UNITS PER SKU-DAY BY WEEKDAY ---")
print(dow.round(2))

plt.figure(figsize=(8, 4))
plt.bar(dow.index, dow.values)
plt.title("Average Units per SKU per Day, by Weekday")
plt.xticks(rotation=30)
plt.ylabel("Avg units")
save_chart("weekday_pattern.png")

# --------------------------------------------------
# 6. PROMOTION EFFECT (average per row, not total)
# --------------------------------------------------

promo = df.groupby("Promotion").agg(
    Rows=("Units_Sold", "size"),
    Avg_Units=("Units_Sold", "mean"),
    Avg_Revenue=("Revenue", "mean"),
)
promo["Uplift_%"] = (promo["Avg_Units"] / promo.loc[0, "Avg_Units"] - 1) * 100

print("\n--- PROMOTION EFFECT ---")
print(promo.round(2))

plt.figure(figsize=(6, 4))
plt.bar(["No promotion", "Promotion"], promo["Avg_Units"].values)
plt.title("Average Units per SKU-Day: Promotion vs Normal")
plt.ylabel("Avg units")
save_chart("promotion_effect.png")

# --------------------------------------------------
# 7. HOLIDAY EFFECT
# --------------------------------------------------

holiday = df.groupby("is_holiday")["Units_Sold"].mean()

print("\n--- HOLIDAY EFFECT (avg units per SKU-day) ---")
print(holiday.round(2))

# --------------------------------------------------
# 8. CATEGORY ANALYSIS
# --------------------------------------------------

category = df.groupby("Category").agg(
    Units=("Units_Sold", "sum"),
    Revenue=("Revenue", "sum"),
    Gross_Profit=("Gross_Profit", "sum"),
).sort_values("Revenue", ascending=False)

print("\n--- CATEGORY SUMMARY ---")
print(category.round(0))

category["Revenue"].plot(kind="bar", figsize=(8, 4), title="Revenue by Category")
plt.ylabel("Revenue")
plt.xticks(rotation=30)
save_chart("category_revenue.png")

# --------------------------------------------------
# 9. NEGATIVE-MARGIN SKUs
# --------------------------------------------------

neg = sku[sku["Gross_Margin_Per_Unit"] < 0]
neg_profit = df[df["SKU"].isin(neg["SKU"])]["Gross_Profit"].sum()

print("\n--- NEGATIVE-MARGIN SKUs ---")
print("Count:", len(neg))
print("Total gross profit from these SKUs:", round(neg_profit, 0))

# --------------------------------------------------
# 10. DEMAND VARIABILITY + DISTRIBUTION
# --------------------------------------------------

variability = df.groupby("SKU")["Units_Sold"].agg(["mean", "std"])
variability["CV"] = variability["std"] / variability["mean"]

print("\n--- MOST VARIABLE SKUs (high CV = harder to forecast) ---")
print(variability.sort_values("CV", ascending=False).head(5).round(2))

plt.figure(figsize=(8, 4))
plt.hist(df["Units_Sold"], bins=40)
plt.title("Distribution of Daily Units Sold")
plt.xlabel("Units sold")
plt.ylabel("Frequency")
save_chart("units_distribution.png")

# --------------------------------------------------
# 11. OUTLIERS (IQR rule; flagged only, not removed)
# --------------------------------------------------

q1, q3 = df["Units_Sold"].quantile([0.25, 0.75])
upper = q3 + 1.5 * (q3 - q1)
outliers = (df["Units_Sold"] > upper).sum()

print("\n--- OUTLIERS ---")
print(f"Upper limit (IQR rule): {upper:.1f} units")
print(f"Rows above limit: {outliers} ({outliers / len(df) * 100:.2f}%) - kept, not removed")

print("\nEDA completed. Charts saved in:", reports_folder)
