# run_all.py
# Runs the whole pipeline in order, from raw data to the forecast and inventory analysis.
# Stops at the first script that fails, so you can see exactly where the problem is.
#
# Usage (from the project root, with the virtual environment active):
#     python run_all.py          -> run the pipeline
#     python run_all.py --app    -> run the pipeline, then open the dashboard

import subprocess
import sys
import time
from pathlib import Path

base = Path(__file__).resolve().parent
data = base / "data"

raw_files = ["sales_daily.csv", "sku_master.csv", "calendar.csv", "inventory_snapshots.csv"]

steps = [
    "inspect_data.py",
    "clean_all_data.py",
    "build_dataset.py",
    "eda.py",
    "feature_engineering.py",
    "baseline.py",
    "train_model.py",
    "evaluate_model.py",
    "forecast.py",
    "inventory_analysis.py",
    "visualize.py",
]

# 1. check the raw data is in place
missing = [f for f in raw_files if not (data / f).exists()]
if missing:
    print("These raw files are missing from the data folder:")
    for f in missing:
        print("  -", f)
    print("\nPut them in:", data)
    sys.exit(1)

print("Raw data found. Starting the pipeline.\n")

# 2. run every script with the same Python that runs this file (the venv one)
start = time.time()

for i, name in enumerate(steps, start=1):
    print("=" * 60)
    print(f"[{i}/{len(steps)}] {name}")
    print("=" * 60)

    result = subprocess.run([sys.executable, str(base / "src" / name)], cwd=base)

    if result.returncode != 0:
        print(f"\nStopped: {name} failed. Fix the error above and run again.")
        sys.exit(1)

    print()

print("=" * 60)
print(f"Pipeline finished in {time.time() - start:.0f} seconds.")
print("Outputs   : outputs/forecasts.csv, outputs/inventory_analysis.csv, outputs/backtest_results.csv")
print("Model     : models/demand_model.pkl, models/feature_list.pkl")
print("Charts    : reports/*.png")
print("=" * 60)

# 3. optionally open the dashboard
if "--app" in sys.argv:
    print("\nOpening the dashboard (press Ctrl+C in this terminal to stop it)...")
    subprocess.run([sys.executable, "-m", "streamlit", "run", str(base / "app.py")], cwd=base)
else:
    print("\nTo open the dashboard run:  python -m streamlit run app.py")
    