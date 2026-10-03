# Feature Engineering Notes

## Objective

Create weekly, SKU-level features for demand forecasting. Weekly data was chosen over daily data because inventory decisions work on lead times of several days and weekly totals are less noisy.

## Daily to weekly

- Weeks run Monday to Sunday (`Week_Start` is the Monday).
- For each SKU and week: total `Units_Sold`, number of promotion days (`Promo_Days`) and number of holiday days (`Holiday_Days`).
- The last week of the data (29-31 Dec 2025) has only 3 days. Incomplete weeks were dropped, because they would look like a false drop in demand.
- Result: 104 complete weeks for each of 50 SKUs = 5,200 rows.

## Features

| Feature | Description |
|---|---|
| Lag_1, Lag_2, Lag_4 | Units sold 1, 2 and 4 weeks earlier (same SKU) |
| Rolling_4 | Average of the previous 4 weeks (current week excluded) |
| Week_of_Year | ISO week number, captures yearly seasonality |
| Month | Month of the week start |
| Promo_Days | Days with a promotion in that week |
| Holiday_Days | Holiday days in that week |
| SKU_Code | Numeric SKU id, treated as a categorical feature by the model |
| Lag_52 | Units sold in the same week last year. Used only for the baseline, not as a model feature |

## Avoiding leakage

- All lag and rolling features use `shift()` inside each SKU, so a row only sees past weeks.
- `Revenue`, `Price` and gross profit are not used, because they are derived from `Units_Sold`.
- `Year` is not used: with two years of data it cannot generalise to future dates.

## Missing values

Lag and rolling features are empty at the start of each SKU's history. They were **not** replaced with zero, because zero would mean real zero demand. Rows with missing features are excluded from training (5,000 usable rows).

## Future weeks

Promotions and holidays for future weeks are not known. The forecast copies `Promo_Days` and `Holiday_Days` from the same week of the previous year, since promotions repeat at the same time each year in this data.

## Output

`data/weekly_features.csv` (5,200 rows, 13 columns). Reproduce with `python src/feature_engineering.py`.