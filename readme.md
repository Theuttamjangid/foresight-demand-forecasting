# FORESIGHT: Demand and Inventory Intelligence

Weekly demand forecasting for 50 SKUs, with inventory risk analysis and a Streamlit dashboard.

## What the project does

1. Cleans four raw datasets (sales, SKU master, calendar, inventory snapshots).
2. Explores the data (seasonality, promotion effect, category performance).
3. Converts daily sales into weekly data and builds lag, rolling and calendar features.
4. Trains a gradient boosting model and compares it with a seasonal naive baseline.
5. Forecasts the next 8 weeks for every SKU.
6. Combines the forecast with inventory data to flag stock-out and overstock risk.
7. Shows everything in a Streamlit app.

## Results

| Check | Model WAPE | Baseline WAPE (same week last year) |
|---|---:|---:|
| One step ahead, last 12 weeks | 8.35% | 11.51% |
| Recursive 8-week backtest | 8.67% | 11.87% |

The recursive backtest is the honest number, because it uses the model's own predictions as lags, exactly as in real forecasting.

## Folder structure

```
FORESIGHT_PROJECT/
├── data/            raw CSVs + cleaned files (not tracked by git)
├── models/          demand_model.pkl, feature_list.pkl
├── outputs/         forecasts.csv, inventory_analysis.csv, backtest_results.csv
├── reports/         charts and written reports
├── src/             all pipeline scripts
├── app.py           Streamlit dashboard
├── requirements.txt
└── readme.md
```

## Setup

Python 3.11 is recommended.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Place the four raw files in `data/`: `sales_daily.csv`, `sku_master.csv`, `calendar.csv`, `inventory_snapshots.csv`.

## How to run

Run the scripts from the project root, in this order:

```powershell
python src/inspect_data.py          # look at the raw data
python src/clean_all_data.py        # clean and save *_clean.csv
python src/build_dataset.py         # merged analysis table
python src/eda.py                   # statistics and charts
python src/feature_engineering.py   # weekly features
python src/baseline.py              # seasonal naive benchmark
python src/train_model.py           # train and save the model
python src/evaluate_model.py        # recursive backtest
python src/forecast.py              # next 8 weeks
python src/inventory_analysis.py    # stock risk
python src/visualize.py             # forecast and risk charts
streamlit run app.py                # dashboard
```

## Assumptions and limitations

- Future promotions and holidays are unknown, so the forecast copies the same week of last year (promotions repeat at the same time each year in this data). Change `future_events` in `src/forecast.py` to `"none"` to assume no promotions.
- The inventory file has 200 SKUs but only 50 have sales history. Only these 50 are analysed.
- The latest inventory snapshot (1 Dec 2025) is used as the current position, while the forecast starts on 29 Dec 2025.
- 16 SKUs have a negative gross margin (selling price below cost). They are kept in the analysis and flagged in the report.
- The data covers only two years, so yearly seasonality is learned from a single repeat.