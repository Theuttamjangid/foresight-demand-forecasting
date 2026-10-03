# app.py
# Streamlit dashboard for the FORESIGHT project.
# Run from the project root:  streamlit run app.py
# (run the scripts in src/ first so the data and output files exist)

import numpy as np
import pandas as pd
import streamlit as st
from pathlib import Path

base = Path(__file__).resolve().parent
data = base / "data"
outputs = base / "outputs"

st.set_page_config(page_title="FORESIGHT", layout="wide")


@st.cache_data
def load_data():
    weekly = pd.read_csv(data / "weekly_features.csv", parse_dates=["Week_Start"])
    forecast = pd.read_csv(outputs / "forecasts.csv", parse_dates=["Week_Start"])
    inventory = pd.read_csv(outputs / "inventory_analysis.csv")
    sku = pd.read_csv(data / "sku_master_clean.csv")
    backtest = pd.read_csv(outputs / "backtest_results.csv", parse_dates=["Week_Start"])
    return weekly, forecast, inventory, sku, backtest


try:
    weekly, forecast, inventory, sku, backtest = load_data()
except FileNotFoundError as e:
    st.error(f"A file is missing: {e.filename}")
    st.info("Run the scripts in src/ in order (see readme.md) and then reload this page.")
    st.stop()


def wape(actual, predicted):
    return np.abs(actual - predicted).sum() / actual.sum() * 100


st.title("FORESIGHT: Demand and Inventory Intelligence")
st.caption("Weekly demand forecast for 50 SKUs and stock risk based on the forecast")

tab1, tab2, tab3 = st.tabs(["Overview", "SKU forecast", "Inventory risk"])

# ---------------- overview ----------------
with tab1:
    n_weeks = forecast["Week_Start"].nunique()
    high = (inventory["Risk_Category"] == "High Risk").sum()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("SKUs", forecast["SKU"].nunique())
    c2.metric(f"Forecast units ({n_weeks} weeks)", f"{forecast['Predicted_Units'].sum():,.0f}")
    c3.metric("High risk SKUs", int(high))
    c4.metric("Model WAPE (backtest)", f"{wape(backtest['Units_Sold'], backtest['Predicted']):.1f}%",
              delta=f"baseline {wape(backtest['Units_Sold'], backtest['Lag_52']):.1f}%",
              delta_color="off")

    st.subheader("Total weekly demand")
    actual = weekly.groupby("Week_Start")["Units_Sold"].sum().tail(52).rename("Actual")
    predicted = forecast.groupby("Week_Start")["Predicted_Units"].sum().rename("Forecast")
    st.line_chart(pd.concat([actual, predicted], axis=1))

    st.caption("Future promotions and holidays are assumed to repeat the same week as last year.")

# ---------------- sku forecast ----------------
with tab2:
    sku_list = sorted(forecast["SKU"].unique())
    chosen = st.selectbox("Choose a SKU", sku_list)

    info = sku[sku["SKU"] == chosen]
    if not info.empty:
        r = info.iloc[0]
        st.write(f"**{r['Product_Name']}** | {r['Category']} / {r['Subcategory']} | "
                 f"selling price {r['Selling_Price']:,.0f} | margin per unit {r['Gross_Margin_Per_Unit']:,.0f}")

    hist = (weekly[weekly["SKU"] == chosen].set_index("Week_Start")["Units_Sold"]
            .tail(52).rename("Actual"))
    fc = (forecast[forecast["SKU"] == chosen].set_index("Week_Start")["Predicted_Units"]
          .rename("Forecast"))
    st.line_chart(pd.concat([hist, fc], axis=1))

    st.dataframe(forecast[forecast["SKU"] == chosen][["Week_Start", "Predicted_Units", "Promo_Days_Assumed"]]
                 .reset_index(drop=True))

# ---------------- inventory risk ----------------
with tab3:
    colours = {"High Risk": "#f8b4b4", "Medium Risk": "#fde3a7",
               "Overstock": "#b9d7f2", "Healthy": "#c6ecc6"}

    left, right = st.columns([1, 3])
    with left:
        categories = st.multiselect("Risk category", list(colours),
                                    default=["High Risk", "Medium Risk"])
        st.bar_chart(inventory["Risk_Category"].value_counts())
    with right:
        shown = inventory[inventory["Risk_Category"].isin(categories)]
        cols = ["SKU", "Current_Stock", "On_Order", "Lead_Time_Days", "Avg_Daily_Demand",
                "Days_of_Cover", "Stock_After_Lead_Time", "Safety_Stock",
                "Suggested_Order_Qty", "Risk_Category"]
        styled = shown[cols].style.apply(
            lambda row: [f"background-color: {colours[row['Risk_Category']]}"] * len(row), axis=1)
        st.dataframe(styled, hide_index=True)
        st.download_button("Download this table", shown[cols].to_csv(index=False),
                           file_name="inventory_risk.csv", mime="text/csv")

    st.caption(f"Inventory snapshot date: {inventory['Snapshot_Date'].iloc[0]}. "
               "Suggested order = lead time demand + safety stock - (stock + on order).")