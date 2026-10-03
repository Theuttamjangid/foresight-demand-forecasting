# Data Quality Report

## 1. Purpose

This report records the checks, cleaning decisions and preparation steps applied to the four source datasets before analysis. Raw files were never modified; cleaned copies were saved separately (`*_clean.csv`).

## 2. Source datasets

The project uses four main datasets:

1. Sales Daily
2. SKU Master
3. Calendar
4. Inventory Snapshots

The original raw CSV files were kept unchanged.
Cleaned versions were created separately.


## 3. Checks performed

- Row counts, column names and data types for every file.
- Missing values and duplicate rows.
- Date columns converted to datetime (no invalid dates found).
- Negative and zero values in `Units_Sold`.
- SKU consistency across sales, SKU master and inventory.
- Margin check: `Selling_Price - Cost_Price` equals `Gross_Margin_Per_Unit` for all SKUs.
- Revenue check: `Revenue = Units_Sold x Price` in every sales row.

## 4. Findings and decisions

### 4.1 Sales
- No negative `Units_Sold`. 202 rows have zero units. These are valid zero-demand days and were **kept**.
- `Revenue` is calculated directly from `Units_Sold`, so it is not used as a model feature (it would leak the target).

### 4.2 SKU master
- 16 of 50 SKUs have a negative gross margin. They were kept and flagged; the EDA shows their combined gross profit is negative.

### 4.3 Calendar
- `holiday` and `promotion_event` are empty on normal days. These are not data errors, so they were filled with `No_Holiday` and `No_Event`.
- The calendar contains 4 holidays per year (Republic Day, Independence Day, Diwali, Christmas) and 75 days of "Seasonal Promotion" across the two years.

### 4.4 Inventory
- The file contains 200 SKUs, but only 50 have sales history. The 150 others cannot be forecast, so inventory was filtered to the 50 sales SKUs (1,200 rows, 24 snapshots each).

## 5. Outputs

| File | Description |
|---|---|
| `sales_clean.csv` | 36,550 rows |
| `sku_master_clean.csv` | 50 rows |
| `calendar_clean.csv` | 731 rows |
| `inventory_clean.csv` | 1,200 rows (50 SKUs) |
| `analysis_ready.csv` | Sales + calendar + SKU master: 36,550 rows, 23 columns |

Inventory is not merged into the analysis-ready table because it is a monthly snapshot, not a daily record.

## 6. Reproducibility

```
python src/inspect_data.py
python src/clean_all_data.py
python src/build_dataset.py
```

## 7. Conclusion

The data is complete and consistent. The main points for later stages are the 150 inventory SKUs without sales, the 16 negative-margin SKUs, and the fact that `Revenue` must stay out of the model features.