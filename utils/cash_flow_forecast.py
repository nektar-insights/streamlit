import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
from datetime import datetime, timedelta

alt.data_transformers.disable_max_rows()

# ---------- helper ----------
def _sanitize(df, date_cols=None, num_cols=None):
    if df is None:
        return df
    if date_cols:
        for c in date_cols:
            if c in df.columns:
                df[c] = pd.to_datetime(df[c], errors="coerce")
    if num_cols:
        for c in num_cols:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce")
    return df
# ----------------------------

def create_cash_flow_forecast(deals_df, closed_won_df, qbo_df=None):
    """Simplified cash-flow forecast:
       - Inflows from QBO historical weekly/monthly average
       - OpEx is user-entered
       - Charts only use columns present in forecast_df
    """
    # 1) Sanitize
    closed_won_df = _sanitize(closed_won_df, ["date_created"], ["amount"])
    qbo_df        = _sanitize(qbo_df,        ["txn_date"],     ["total_amount"])

    st.header("Capital Deployment Forecast")
    st.markdown("---")

    # 2) Compute inflow rates (historical averages)
    has_qbo_data = qbo_df is not None and not qbo_df.empty and "txn_date" in qbo_df.columns
    weekly_inflow_rate = monthly_inflow_rate = 0.0
    if has_qbo_data:
        cash_tx = qbo_df[qbo_df["transaction_type"].isin(
            ["Payment", "Deposit", "Receipt", "Invoice", "Bill", "Expense"]
        )].copy()
        cash_tx = cash_tx.dropna(subset=["txn_date"])
        customer_payments = cash_tx[
            cash_tx["transaction_type"].isin(["Payment", "Receipt"])
            & ~cash_tx["customer_name"].isin(["CSL", "VEEM"])
        ].copy()

        if not cash_tx.empty:
            min_d = cash_tx["txn_date"].min()
            max_d = cash_tx["txn_date"].max()
            total_days   = (max_d - min_d).days + 1
            total_weeks  = total_days / 7
            total_months = total_days / 30.44

            total_inflows = customer_payments["total_amount"].abs().sum()
            weekly_inflow_rate  = total_inflows / total_weeks  if total_weeks  > 0 else 0.0
            monthly_inflow_rate = total_inflows / total_months if total_months > 0 else 0.0

    # 3) Compute deployment baseline from deals (historical average)
    weekly_deployment_rate = monthly_deployment_rate = 0.0
    if closed_won_df is not None and not closed_won_df.empty and "date_created" in closed_won_df.columns:
        deal_min = closed_won_df["date_created"].min()
        deal_max = closed_won_df["date_created"].max()
        d_days   = (deal_max - deal_min).days + 1
        d_weeks  = d_days / 7
        d_months = d_days / 30.44
        total_deployed = closed_won_df["amount"].sum()
        weekly_deployment_rate  = total_deployed / d_weeks   if d_weeks   > 0 else 0.0
        monthly_deployment_rate = total_deployed / d_months  if d_months  > 0 else 0.0

    # Debug counts
    with st.expander("🔍 Debug – inputs"):
        st.write("rows →", {"closed_won_df": len(closed_won_df) if closed_won_df is not None else 0,
                            "qbo_df": len(qbo_df) if qbo_df is not None else 0})
        st.write("weekly_inflow_rate:", round(weekly_inflow_rate, 2),
                 "monthly_inflow_rate:", round(monthly_inflow_rate, 2))
        st.write("weekly_deployment_rate:", round(weekly_deployment_rate, 2),
                 "monthly_deployment_rate:", round(monthly_deployment_rate, 2))

    # 4) Simple config (always available)
    col1, col2, col3 = st.columns(3)
    with col1:
        starting_cash = st.number_input("Starting Cash", min_value=0, value=500_000, step=50_000, format="%d")
        forecast_period = st.selectbox("Forecast Period", ["Weekly", "Monthly"])
        forecast_horizon = st.slider(
            "Forecast Horizon",
            min_value=4 if forecast_period == "Weekly" else 3,
            max_value=52 if forecast_period == "Weekly" else 24,
            value=26 if forecast_period == "Weekly" else 12
        )

    # Use historical rates as defaults; user can override if desired
    base_dep = weekly_deployment_rate  if forecast_period == "Weekly" else monthly_deployment_rate
    base_inf = weekly_inflow_rate      if forecast_period == "Weekly" else monthly_inflow_rate

    with col2:
        deployment_rate = st.number_input(
            f"Deployment per { 'week' if forecast_period=='Weekly' else 'month' }",
            min_value=0, value=int(base_dep), step=10_000, format="%d"
        )
        inflow_rate = st.number_input(
            f"Inflows per { 'week' if forecast_period=='Weekly' else 'month' }",
            min_value=0, value=int(base_inf), step=10_000, format="%d"
        )

    with col3:
        # OpEx is user-entered (no historical calc)
        opex_rate = st.number_input(
            f"OpEx per { 'week' if forecast_period=='Weekly' else 'month' }",
            min_value=0, value=0, step=5_000, format="%d",
            help="Enter your estimated operating expenses for the chosen period"
        )

    # 5) Build forecast_df with only the fields we plot
    if forecast_period == "Weekly":
        dates = pd.date_range(start=datetime.now(), periods=forecast_horizon + 1, freq="W")
        unit = "week"
    else:
        dates = pd.date_range(start=datetime.now(), periods=forecast_horizon + 1, freq="M")
        unit = "month"

    current_cash = float(starting_cash)
    rows = []
    for i, dt in enumerate(dates[1:], start=1):
        period_inflows    = float(inflow_rate)
        period_deployment = float(deployment_rate)
        period_opex       = float(opex_rate)
        period_net        = period_inflows - period_deployment - period_opex
        current_cash     += period_net

        rows.append({
            "Date": dt,
            "Period": i,
            "Cash Position": current_cash,
            "Inflows": period_inflows,
            "Deployment": period_deployment,
            "OpEx": period_opex,
            "Net Flow": period_net,
        })

    forecast_df = pd.DataFrame(rows)

    with st.expander("🧮 Forecast DF preview"):
        st.dataframe(forecast_df.head())
        st.write("rows:", len(forecast_df))

    # 6) Charts – only use columns that exist in forecast_df
    date_fmt = "%b %d" if forecast_period == "Weekly" else "%b %Y"

    cash_line = (
        alt.Chart(forecast_df)
        .mark_line(strokeWidth=3)
        .encode(
            x=alt.X("Date:T", title=f"Date ({unit})", axis=alt.Axis(format=date_fmt, labelAngle=-45)),
            y=alt.Y("Cash Position:Q", title="Cash Position ($)", axis=alt.Axis(format="$,.0f")),
            tooltip=[alt.Tooltip("Date:T", format="%b %d, %Y"),
                     alt.Tooltip("Cash Position:Q", format="$,.0f"),
                     alt.Tooltip("Net Flow:Q", format="$+,.0f")]
        )
        .properties(height=380, title="Projected Cash Position")
    )
    st.altair_chart(cash_line, use_container_width=True)

    st.subheader("Cash Flow Components")
    flow_long = (
        forecast_df[["Date", "Inflows", "Deployment", "OpEx"]]
        .melt(id_vars=["Date"], var_name="Component", value_name="Amount")
    )
    flow_chart = (
        alt.Chart(flow_long)
        .mark_bar()
        .encode(
            x=alt.X("Date:T", title=f"Date ({unit})", axis=alt.Axis(format=date_fmt, labelAngle=-45)),
            y=alt.Y("Amount:Q", title="Cash Flow ($)", axis=alt.Axis(format="$,.0f")),
            color="Component:N",
            tooltip=[alt.Tooltip("Date:T", format="%b %d, %Y"),
                     alt.Tooltip("Component:N"),
                     alt.Tooltip("Amount:Q", format="$+,.0f")]
        )
        .properties(height=300, title="Inflows vs Deployment vs OpEx")
    )
    st.altair_chart(flow_chart, use_container_width=True)

    # 7) Summary table
    st.subheader("Detailed Cash Flow Summary")
    summary_df = forecast_df.copy()
    st.dataframe(
        summary_df.assign(Date=summary_df["Date"].dt.strftime(date_fmt)),
        use_container_width=True,
        column_config={
            "Date": st.column_config.TextColumn("Period", width="small"),
            "Cash Position": st.column_config.NumberColumn("Cash Position", format="$%.0f"),
            "Deployment": st.column_config.NumberColumn("Deployment", format="$%.0f"),
            "Inflows": st.column_config.NumberColumn("Inflows", format="$%.0f"),
            "OpEx": st.column_config.NumberColumn("OpEx", format="$%.0f"),
            "Net Flow": st.column_config.NumberColumn("Net Flow", format="$%+.0f"),
        },
        hide_index=True,
    )
