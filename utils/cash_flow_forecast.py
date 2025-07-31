import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
from datetime import datetime, timedelta

def create_cash_flow_forecast(deals_df, closed_won_df, qbo_df=None):
    """
    Create an integrated cash flow forecast that includes:
    - Capital deployment (outflows for new deals)
    - Loan repayments (inflows from QBO)
    - Operating expenses (other outflows from QBO)
    """
    st.header("Capital Deployment Forecast")
    st.markdown("---")

    # —————————————————————————————
    # 1) QBO DATA PREP
    # —————————————————————————————
    has_qbo = qbo_df is not None and not qbo_df.empty and "txn_date" in qbo_df.columns
    if has_qbo:
        qbo_df["txn_date"] = pd.to_datetime(qbo_df["txn_date"], errors="coerce")
        qbo_df["total_amount"] = pd.to_numeric(qbo_df.get("total_amount", 0), errors="coerce")
        ct = qbo_df[qbo_df["transaction_type"].isin(
            ["Payment","Deposit","Receipt","Invoice","Bill","Expense"]
        )].dropna(subset=["txn_date"]).copy()
        ct["cash_impact"] = np.where(
            ct["transaction_type"].isin(["Payment","Deposit","Receipt"]),
            ct["total_amount"].abs(),
            -ct["total_amount"].abs()
        )
        cust_pay = ct[
            ct["transaction_type"].isin(["Payment","Receipt"])
            & ~ct["customer_name"].isin(["CSL","VEEM"])
        ]
        opex   = ct[ct["transaction_type"].isin(["Bill","Expense"])]

        if not ct.empty:
            days  = (ct["txn_date"].max() - ct["txn_date"].min()).days + 1
            wks   = days/7
            mths  = days/30.44
            total_in = cust_pay["total_amount"].abs().sum()
            total_op = opex["total_amount"].abs().sum()
            weekly_inflow_rate   = total_in/wks  if wks else 0
            monthly_inflow_rate  = total_in/mths if mths else 0
            weekly_opex_rate     = total_op/wks  if wks else 0
            monthly_opex_rate    = total_op/mths if mths else 0
        else:
            weekly_inflow_rate = monthly_inflow_rate = 0
            weekly_opex_rate   = monthly_opex_rate   = 0
            has_qbo = False
    else:
        weekly_inflow_rate = monthly_inflow_rate = 0
        weekly_opex_rate   = monthly_opex_rate   = 0

    # —————————————————————————————
    # 2) HISTORICAL DEPLOYMENT METRICS
    # —————————————————————————————
    if not closed_won_df.empty and "date_created" in closed_won_df.columns:
        st.subheader("Historical Cash Flow Analysis")
        dmin, dmax = closed_won_df["date_created"].min(), closed_won_df["date_created"].max()
        ddays = (dmax - dmin).days + 1
        dwks  = ddays/7
        dmths = ddays/30.44
        total_deployed = closed_won_df["amount"].sum()
        deal_count     = len(closed_won_df)
        weekly_deployment_rate  = total_deployed/dwks  if dwks else 0
        monthly_deployment_rate = total_deployed/dmths if dmths else 0
        avg_deal_size  = closed_won_df["amount"].mean()
        median_deal_size = closed_won_df["amount"].median()
        deals_per_week  = deal_count/dwks  if dwks else 0
        deals_per_month = deal_count/dmths if dmths else 0

        # display metrics (unchanged)…
        # [ your existing st.metric() blocks go here ]
        st.markdown("---")
    else:
        st.warning("No historical deal data available for forecasting.")
        return

    # —————————————————————————————
    # 3) FORECAST CONFIGURATION
    # —————————————————————————————
    col1, col2, col3 = st.columns(3)
    with col1:
        starting_cash = st.number_input(
            "Current Cash Position" if has_qbo else "Available Capital",
            min_value=0, value=500_000, step=100_000, format="%d"
        )
        forecast_period = st.selectbox("Forecast Period", ["Weekly","Monthly"])
        if forecast_period=="Weekly":
            forecast_horizon = st.slider("Horizon (weeks)", 4, 52, 26)
        else:
            forecast_horizon = st.slider("Horizon (months)",3,24,12)

    with col2:
        deployment_method = st.selectbox(
            "Deployment Rate", 
            ["Historical Average","Conservative (75%)","Aggressive (125%)","Custom"]
        )
        if deployment_method=="Custom":
            custom_dep = st.number_input(
                "Custom Deployment",min_value=0,
                value=int(weekly_deployment_rate if forecast_period=="Weekly" else monthly_deployment_rate),
                step=10_000, format="%d"
            )
        if has_qbo:
            inflow_method = st.selectbox(
                "Repayment Rate", 
                ["Historical Average","Conservative (75%)","Optimistic (125%)","Custom"]
            )
            if inflow_method=="Custom":
                custom_in = st.number_input(
                    "Custom Inflow",min_value=0,
                    value=int(weekly_inflow_rate if forecast_period=="Weekly" else monthly_inflow_rate),
                    step=10_000, format="%d"
                )
        else:
            deal_size_method = st.selectbox(
                "Deal Size Assumption", ["Historical Average","Historical Median","Custom Amount"]
            )
            if deal_size_method=="Custom Amount":
                forecast_deal_size = st.number_input(
                    "Custom Deal Size",min_value=1_000,
                    value=int(avg_deal_size),step=1_000,format="%d"
                )
            else:
                forecast_deal_size = avg_deal_size if deal_size_method=="Historical Average" else median_deal_size

    with col3:
        if has_qbo:
            opex_method = st.selectbox(
                "OpEx Rate", ["Historical Average","Reduced (75%)","Increased (125%)","Custom"]
            )
            if opex_method=="Custom":
                custom_opex = st.number_input(
                    "Custom OpEx",min_value=0,
                    value=int(weekly_opex_rate if forecast_period=="Weekly" else monthly_opex_rate),
                    step=5_000, format="%d"
                )
            min_cash_threshold = st.number_input(
                "Minimum Cash Reserve",min_value=0,value=100_000,step=50_000,format="%d"
            )

    # pick base rates
    if forecast_period=="Weekly":
        base_dep, base_in, base_op = weekly_deployment_rate, weekly_inflow_rate, weekly_opex_rate
        time_unit="week"
    else:
        base_dep, base_in, base_op = monthly_deployment_rate, monthly_inflow_rate, monthly_opex_rate
        time_unit="month"

    # adjust by method
    def adjust(rate, method, custom):
        if method=="Historical Average": return rate
        if "75%" in method: return rate*0.75
        if "125%" in method: return rate*1.25
        return custom

    deployment_rate = adjust(base_dep, deployment_method, custom_dep if 'custom_dep' in locals() else 0)
    inflow_rate    = adjust(base_in,  inflow_method   , custom_in  if 'custom_in' in locals()  else 0) if has_qbo else 0
    opex_rate      = adjust(base_op,  opex_method     , custom_opex if 'custom_opex' in locals()else 0) if has_qbo else 0
    net_flow_per_period = inflow_rate - deployment_rate - opex_rate

    st.markdown("---")

    # —————————————————————————————
    # 4) SIMULATE FORECAST
    # —————————————————————————————
    freq = 'W' if forecast_period=="Weekly" else 'M'
    dates = pd.date_range(start=datetime.now(), periods=forecast_horizon, freq=freq)

    forecast_rows = [{
        "Date": datetime.now(),
        "Period": 0,
        "Cash Position": starting_cash,
        "Deployment": 0,
        "Inflows": 0,
        "OpEx": 0,
        "Net Flow": 0
    }]
    current_cash = starting_cash

    for i, d in enumerate(dates, start=1):
        dep  = deployment_rate
        # throttle below reserve
        if has_qbo and (current_cash - dep - opex_rate) < min_cash_threshold:
            dep = max(0, current_cash - min_cash_threshold - opex_rate)
        infl = inflow_rate if has_qbo else 0
        opx  = opex_rate    if has_qbo else 0
        net  = infl - dep - opx
        current_cash += net

        forecast_rows.append({
            "Date": d,
            "Period": i,
            "Cash Position": current_cash,
            "Deployment": dep,
            "Inflows": infl,
            "OpEx": opx,
            "Net Flow": net
        })

    forecast_df = pd.DataFrame(forecast_rows)
    forecast_df["Date"] = pd.to_datetime(forecast_df["Date"])
    date_fmt = "%b %d" if forecast_period=="Weekly" else "%b %Y"

    # —————————————————————————————
    # 5) PLOT CHARTS
    # —————————————————————————————
    # 5a) Projected Cash Position
    st.subheader("Cash Flow Projection")
    cash_line = (
        alt.Chart(forecast_df)
           .mark_line(strokeWidth=3)
           .encode(
               x=alt.X("Date:T", axis=alt.Axis(format=date_fmt, labelAngle=-45)),
               y=alt.Y("Cash Position:Q", axis=alt.Axis(format="$,.0f")),
               tooltip=[
                   alt.Tooltip("Date:T", format="%b %d, %Y"),
                   alt.Tooltip("Cash Position:Q", format="$,.0f"),
                   alt.Tooltip("Net Flow:Q", format="$+,.0f")
               ]
           )
    )
    threshold = (
        alt.Chart(pd.DataFrame({"y":[min_cash_threshold]}))
           .mark_rule(color="red", strokeDash=[5,5], strokeWidth=2)
           .encode(y="y:Q", tooltip=alt.Tooltip("y:Q", format="$,.0f"))
    )
    st.altair_chart(
        (cash_line + threshold)
          .properties(title="Projected Cash Position", height=400),
        use_container_width=True
    )

    # 5b) Components
    st.subheader("Cash Flow Components")
    flow = forecast_df.loc[forecast_df.Period>0, ["Date","Inflows","Deployment","OpEx"]].copy()
    flow["Deployment"] *= -1
    flow["OpEx"]       *= -1
    long = flow.melt(
        id_vars=["Date"], 
        value_vars=["Inflows","Deployment","OpEx"],
        var_name="Component", value_name="Amount"
    )
    flow_chart = (
        alt.Chart(long)
           .mark_bar()
           .encode(
               x=alt.X("Date:T", axis=alt.Axis(format=date_fmt, labelAngle=-45)),
               y=alt.Y("Amount:Q", stack="zero", axis=alt.Axis(format="$,.0f")),
               color=alt.Color("Component:N", legend=alt.Legend(title="Component")),
               tooltip=[
                   alt.Tooltip("Date:T", format="%b %d, %Y"),
                   alt.Tooltip("Component:N"),
                   alt.Tooltip("Amount:Q", format="$+,.0f")
               ]
           )
           .properties(title="Cash Flow Components by Period", height=300)
    )
    st.altair_chart(flow_chart, use_container_width=True)

    # —————————————————————————————
    # 6) DETAILED TABLE & WARNINGS
    # —————————————————————————————
    st.subheader("Detailed Cash Flow Summary")
    summary = forecast_df.loc[forecast_df.Period>0, ["Date","Cash Position","Deployment","Inflows","OpEx","Net Flow"]].copy()
    summary["Date"] = summary["Date"].dt.strftime(date_fmt)
    st.dataframe(summary, use_container_width=True)
    if forecast_df["Cash Position"].iloc[-1] < min_cash_threshold:
        st.error(f"⚠️ Cash falls below reserve (${min_cash_threshold:,.0f})")
    if net_flow_per_period < 0:
        burn = abs(net_flow_per_period) * (4.33 if forecast_period=="Weekly" else 1)
        st.warning(f"📉 Burning ${burn:,.0f} per month")
