# cash_flow_forecast.py
import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
from datetime import datetime, timedelta
import json
from utils.imports import get_supabase_client


def store_forecast_snapshot(forecast_df, params):
    """
    Store a forecast snapshot to Supabase for historical tracking.

    Args:
        forecast_df: DataFrame with forecast data (Date, Forecast/Value, etc.)
        params: dict with forecast parameters:
            - forecast_period: 'weekly' or 'monthly'
            - forecast_horizon: int
            - starting_cash: float
            - deployment_rate: float
            - inflow_rate: float
            - opex_rate: float
            - growth_rate: float (decimal, e.g., 0.02 for 2%)

    Returns:
        bool: True if successful, False otherwise
    """
    try:
        supabase = get_supabase_client()

        # Build period forecasts JSON
        period_forecasts = []
        for _, row in forecast_df.iterrows():
            if "Forecast" in forecast_df.columns:
                amount = row.get("Forecast", row.get("Value", 0))
            else:
                amount = row.get("Value", 0)

            period_date = row.get("Date", row.get("Month"))
            if pd.notna(period_date):
                period_forecasts.append({
                    "period_date": period_date.strftime("%Y-%m-%d") if hasattr(period_date, "strftime") else str(period_date),
                    "forecast_amount": float(amount) if pd.notna(amount) else 0,
                    "actual_amount": None  # To be filled when actuals come in
                })

        # Calculate aggregate metrics
        total_forecast = sum(p["forecast_amount"] for p in period_forecasts)
        ending_cash = params.get("starting_cash", 0) + total_forecast - (
            params.get("deployment_rate", 0) * len(period_forecasts)
        ) - (params.get("opex_rate", 0) * len(period_forecasts))

        # Insert into Supabase
        data = {
            "forecast_date": datetime.now().strftime("%Y-%m-%d"),
            "forecast_period": params.get("forecast_period", "monthly"),
            "forecast_horizon": params.get("forecast_horizon", 6),
            "starting_cash": params.get("starting_cash", 0),
            "deployment_rate": params.get("deployment_rate", 0),
            "inflow_rate": params.get("inflow_rate", 0),
            "opex_rate": params.get("opex_rate", 0),
            "growth_rate": params.get("growth_rate", 0),
            "total_forecast_inflows": total_forecast,
            "total_forecast_deployment": params.get("deployment_rate", 0) * len(period_forecasts),
            "ending_cash_forecast": ending_cash,
            "period_forecasts": json.dumps(period_forecasts)
        }

        response = supabase.table("cash_flow_forecast_history").insert(data).execute()

        if response.data:
            return True
        return False

    except Exception as e:
        print(f"Error storing forecast snapshot: {e}")
        return False


def get_forecast_history(days=90, limit=50):
    """
    Retrieve historical forecast snapshots from Supabase.

    Args:
        days: Number of days of history to retrieve
        limit: Maximum number of forecasts to return

    Returns:
        pd.DataFrame: Historical forecasts with their parameters and accuracy
    """
    try:
        supabase = get_supabase_client()

        cutoff_date = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")

        response = (
            supabase.table("cash_flow_forecast_history")
            .select("*")
            .gte("forecast_date", cutoff_date)
            .order("created_at", desc=True)
            .limit(limit)
            .execute()
        )

        if response.data:
            df = pd.DataFrame(response.data)
            df["created_at"] = pd.to_datetime(df["created_at"])
            df["forecast_date"] = pd.to_datetime(df["forecast_date"])
            return df

        return pd.DataFrame()

    except Exception as e:
        print(f"Error retrieving forecast history: {e}")
        return pd.DataFrame()


def update_forecast_with_actuals(forecast_id, qbo_df):
    """
    Update a historical forecast with actual cash flow data.

    Args:
        forecast_id: UUID of the forecast record to update
        qbo_df: DataFrame with actual QBO payment data

    Returns:
        float: Forecast accuracy percentage (0-100)
    """
    try:
        supabase = get_supabase_client()

        # Get the forecast record
        response = supabase.table("cash_flow_forecast_history").select("*").eq("id", forecast_id).execute()

        if not response.data:
            return None

        forecast = response.data[0]
        period_forecasts = json.loads(forecast["period_forecasts"]) if isinstance(forecast["period_forecasts"], str) else forecast["period_forecasts"]

        # Calculate actuals for each period
        if "txn_date" in qbo_df.columns:
            qbo_df = qbo_df.copy()
            qbo_df["txn_date"] = pd.to_datetime(qbo_df["txn_date"], errors="coerce")
            qbo_df["total_amount"] = pd.to_numeric(qbo_df["total_amount"], errors="coerce").abs()

            # Filter to payments only
            payment_types = ["Payment", "Receipt"]
            if "transaction_type" in qbo_df.columns:
                payments = qbo_df[qbo_df["transaction_type"].isin(payment_types)]
            else:
                payments = qbo_df

            # Calculate actuals for each forecasted period
            total_forecast = 0
            total_actual = 0
            updated_forecasts = []

            for period in period_forecasts:
                period_date = pd.to_datetime(period["period_date"])
                forecast_amount = period["forecast_amount"]
                total_forecast += forecast_amount

                # Get actual for this period (same month or week)
                if forecast["forecast_period"] == "weekly":
                    period_start = period_date - timedelta(days=period_date.weekday())
                    period_end = period_start + timedelta(days=6)
                else:
                    period_start = period_date.replace(day=1)
                    next_month = period_start + timedelta(days=32)
                    period_end = next_month.replace(day=1) - timedelta(days=1)

                actual = payments[
                    (payments["txn_date"] >= period_start) &
                    (payments["txn_date"] <= period_end)
                ]["total_amount"].sum()

                total_actual += actual
                period["actual_amount"] = float(actual)
                updated_forecasts.append(period)

            # Calculate accuracy (how close were we?)
            if total_forecast > 0:
                accuracy = max(0, 100 - abs(total_forecast - total_actual) / total_forecast * 100)
            else:
                accuracy = 0 if total_actual > 0 else 100

            # Update the record
            supabase.table("cash_flow_forecast_history").update({
                "period_forecasts": json.dumps(updated_forecasts),
                "actuals_updated_at": datetime.now().isoformat(),
                "forecast_accuracy_pct": accuracy
            }).eq("id", forecast_id).execute()

            return accuracy

        return None

    except Exception as e:
        print(f"Error updating forecast with actuals: {e}")
        return None


def render_forecast_history_tracking(qbo_df=None):
    """
    Render the forecast history tracking UI component.
    Shows historical forecasts and compares them to actuals.
    """
    st.subheader("📊 Forecast History & Accuracy Tracking")

    # Load historical forecasts
    history_df = get_forecast_history(days=180, limit=20)

    if history_df.empty:
        st.info("No forecast history available yet. Save a forecast to start tracking accuracy over time.")
        st.markdown("""
        **How it works:**
        1. When you generate a forecast, click "Save Forecast Snapshot" to store it
        2. As time passes and actuals come in, the system compares forecasts to reality
        3. Track your forecast accuracy to improve future predictions
        """)
        return

    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Total Forecasts", len(history_df))

    with col2:
        forecasts_with_accuracy = history_df[history_df["forecast_accuracy_pct"].notna()]
        if not forecasts_with_accuracy.empty:
            avg_accuracy = forecasts_with_accuracy["forecast_accuracy_pct"].mean()
            st.metric("Avg Accuracy", f"{avg_accuracy:.1f}%")
        else:
            st.metric("Avg Accuracy", "N/A")

    with col3:
        recent = history_df.iloc[0] if len(history_df) > 0 else None
        if recent is not None:
            st.metric("Latest Forecast", recent["forecast_date"].strftime("%b %d, %Y"))
        else:
            st.metric("Latest Forecast", "None")

    with col4:
        if recent is not None and pd.notna(recent.get("forecast_accuracy_pct")):
            st.metric("Latest Accuracy", f"{recent['forecast_accuracy_pct']:.1f}%")
        else:
            st.metric("Latest Accuracy", "Pending")

    # Accuracy trend chart
    forecasts_with_accuracy = history_df[history_df["forecast_accuracy_pct"].notna()].copy()

    if not forecasts_with_accuracy.empty:
        st.markdown("### Forecast Accuracy Trend")

        chart_data = forecasts_with_accuracy[["forecast_date", "forecast_accuracy_pct"]].copy()
        chart_data = chart_data.sort_values("forecast_date")

        accuracy_chart = alt.Chart(chart_data).mark_line(point=True, color="#34a853").encode(
            x=alt.X("forecast_date:T", title="Forecast Date"),
            y=alt.Y("forecast_accuracy_pct:Q", title="Accuracy %", scale=alt.Scale(domain=[0, 100])),
            tooltip=[
                alt.Tooltip("forecast_date:T", title="Date", format="%b %d, %Y"),
                alt.Tooltip("forecast_accuracy_pct:Q", title="Accuracy", format=".1f")
            ]
        ).properties(
            width=700,
            height=300,
            title="Forecast Accuracy Over Time"
        )

        # Add target line at 80%
        target_line = alt.Chart(pd.DataFrame({"y": [80]})).mark_rule(
            color="orange",
            strokeDash=[5, 5]
        ).encode(y="y:Q")

        st.altair_chart(accuracy_chart + target_line, width="stretch")

    # Forecast history table
    st.markdown("### Forecast History")

    display_df = history_df[[
        "forecast_date", "forecast_period", "forecast_horizon",
        "total_forecast_inflows", "ending_cash_forecast", "forecast_accuracy_pct"
    ]].copy()

    display_df.columns = ["Date", "Period", "Horizon", "Forecast Inflows", "Ending Cash", "Accuracy %"]
    display_df["Date"] = display_df["Date"].dt.strftime("%Y-%m-%d")

    st.dataframe(
        display_df,
        column_config={
            "Date": st.column_config.TextColumn("Forecast Date"),
            "Period": st.column_config.TextColumn("Period"),
            "Horizon": st.column_config.NumberColumn("Horizon"),
            "Forecast Inflows": st.column_config.NumberColumn("Forecast Inflows", format="$%.0f"),
            "Ending Cash": st.column_config.NumberColumn("Ending Cash", format="$%.0f"),
            "Accuracy %": st.column_config.NumberColumn("Accuracy", format="%.1f%%"),
        },
        hide_index=True,
        width="stretch"
    )

    # Update actuals button
    if qbo_df is not None and not qbo_df.empty:
        st.markdown("---")
        if st.button("🔄 Update Forecasts with Actuals", help="Compare past forecasts to actual cash flow"):
            with st.spinner("Updating forecasts with actual data..."):
                updated_count = 0
                for _, row in history_df.iterrows():
                    if pd.isna(row.get("forecast_accuracy_pct")):
                        accuracy = update_forecast_with_actuals(row["id"], qbo_df)
                        if accuracy is not None:
                            updated_count += 1

                if updated_count > 0:
                    st.success(f"Updated {updated_count} forecasts with actual data!")
                    st.rerun()
                else:
                    st.info("No forecasts needed updating.")

def create_cash_flow_forecast(deals_df, closed_won_df, qbo_df=None):
    """
    Create a cash flow forecast that includes:
    - Capital deployment (outflows for new deals)
    - Loan repayments (inflows from QBO)
    - Operating expenses (user input)
    """
    
    st.header("Capital Deployment Forecast")
    st.markdown("---")
    
    # Initialize rates
    weekly_inflow_rate = monthly_inflow_rate = 0
    has_qbo_data = False
    
    # Process QBO data if available
    if qbo_df is not None and not qbo_df.empty and "txn_date" in qbo_df.columns:
        # Create explicit copy to avoid SettingWithCopyWarning
        qbo_df = qbo_df.copy()
        
        # Ensure data types
        qbo_df["txn_date"] = pd.to_datetime(qbo_df["txn_date"], errors="coerce")
        qbo_df["total_amount"] = pd.to_numeric(qbo_df["total_amount"], errors="coerce")
        
        # Filter for customer payments only (inflows)
        customer_payments = qbo_df[
            (qbo_df["transaction_type"].isin(["Payment", "Receipt"])) &
            (~qbo_df["customer_name"].isin(["CSL", "VEEM"])) &
            (qbo_df["txn_date"].notna())
        ].copy()
        
        if len(customer_payments) > 0:
            has_qbo_data = True
            min_date = customer_payments["txn_date"].min()
            max_date = customer_payments["txn_date"].max()
            total_days = (max_date - min_date).days + 1
            
            if total_days > 0:
                total_inflows = customer_payments["total_amount"].abs().sum()
                weekly_inflow_rate = total_inflows / (total_days / 7)
                monthly_inflow_rate = total_inflows / (total_days / 30.44)
    
    # Process deals data
    if not closed_won_df.empty and "date_created" in closed_won_df.columns:
        # Create explicit copy to avoid SettingWithCopyWarning
        closed_won_df = closed_won_df.copy()
        
        # Ensure data types
        closed_won_df["date_created"] = pd.to_datetime(closed_won_df["date_created"], errors="coerce")
        closed_won_df["amount"] = pd.to_numeric(closed_won_df["amount"], errors="coerce")
        
        # Filter valid data
        closed_won_df = closed_won_df[closed_won_df["date_created"].notna()]
        
        if len(closed_won_df) > 0:
            # Calculate deployment metrics
            deal_min_date = closed_won_df["date_created"].min()
            deal_max_date = closed_won_df["date_created"].max()
            deal_total_days = (deal_max_date - deal_min_date).days + 1
            
            total_deployed = closed_won_df["amount"].sum()
            deal_count = len(closed_won_df)
            
            # Rates
            weekly_deployment_rate = total_deployed / (deal_total_days / 7) if deal_total_days > 0 else 0
            monthly_deployment_rate = total_deployed / (deal_total_days / 30.44) if deal_total_days > 0 else 0
            
            # Deal metrics
            avg_deal_size = closed_won_df["amount"].mean()
            median_deal_size = closed_won_df["amount"].median()
            deals_per_week = deal_count / (deal_total_days / 7) if deal_total_days > 0 else 0
            deals_per_month = deal_count / (deal_total_days / 30.44) if deal_total_days > 0 else 0
            
            # Display historical metrics
            st.subheader("Historical Analysis")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.write("**Capital Deployment**")
                st.metric("Weekly Deployment", f"${weekly_deployment_rate:,.0f}")
                st.metric("Monthly Deployment", f"${monthly_deployment_rate:,.0f}")
                st.metric("Avg Deal Size", f"${avg_deal_size:,.0f}")
            
            with col2:
                st.write("**Loan Repayments**")
                st.metric("Weekly Inflows", f"${weekly_inflow_rate:,.0f}")
                st.metric("Monthly Inflows", f"${monthly_inflow_rate:,.0f}")
                if has_qbo_data:
                    st.success("✓ QBO data available")
                else:
                    st.info("No QBO payment data")
            
            with col3:
                st.write("**Deal Activity**")
                st.metric("Deals per Week", f"{deals_per_week:.1f}")
                st.metric("Deals per Month", f"{deals_per_month:.1f}")
                st.metric("Total Deals", f"{deal_count}")
            
            st.markdown("---")
            
            # Forecast Configuration
            st.subheader("Forecast Configuration")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.write("**Starting Position**")
                starting_cash = st.number_input(
                    "Current Cash Position",
                    min_value=0,
                    value=500000,
                    step=100000,
                    format="%d",
                    help="Your current available cash"
                )
                
                min_cash_threshold = st.number_input(
                    "Minimum Cash Reserve",
                    min_value=0,
                    value=100000,
                    step=50000,
                    format="%d",
                    help="Minimum cash to maintain"
                )
                
                forecast_period = st.selectbox(
                    "Forecast Period",
                    ["Weekly", "Monthly"],
                    help="Time period for forecasting"
                )
            
            with col2:
                st.write("**Deployment Assumptions**")
                
                # Deployment rate
                deployment_method = st.selectbox(
                    "Deployment Rate",
                    ["Historical Average", "Conservative (75%)", "Aggressive (125%)", "Deal-Based", "Custom Amount"],
                    help="How to calculate deployment rate"
                )
                
                if deployment_method == "Deal-Based":
                    # Key lever: Number of deals
                    st.write("**Deal Participation Levers**")
                    if forecast_period == "Weekly":
                        target_deals_per_period = st.number_input(
                            "Target Deals per Week",
                            min_value=0,
                            value=round(deals_per_week),
                            step=1,
                            format="%d",
                            help="How many deals to participate in per week"
                        )
                    else:
                        target_deals_per_period = st.number_input(
                            "Target Deals per Month",
                            min_value=0,
                            value=round(deals_per_month),
                            step=1,
                            format="%d",
                            help="How many deals to participate in per month"
                        )
                    
                    # Key lever: Average participation amount - round up to nearest 500
                    rounded_avg_deal = int(np.ceil(avg_deal_size / 500) * 500)
                    avg_participation = st.number_input(
                        "Avg Participation per Deal",
                        min_value=0,
                        value=rounded_avg_deal,
                        step=5000,
                        format="%d",
                        help="Average amount to invest per deal (rounded to nearest $500)"
                    )
                    
                elif deployment_method == "Custom Amount":
                    if forecast_period == "Weekly":
                        custom_deployment = st.number_input(
                            "Custom Weekly Deployment",
                            min_value=0,
                            value=int(weekly_deployment_rate),
                            step=10000,
                            format="%d",
                            help="Enter exact weekly deployment amount"
                        )
                    else:
                        custom_deployment = st.number_input(
                            "Custom Monthly Deployment",
                            min_value=0,
                            value=int(monthly_deployment_rate),
                            step=50000,
                            format="%d",
                            help="Enter exact monthly deployment amount"
                        )
                
                # Inflow assumptions
                if has_qbo_data:
                    inflow_method = st.selectbox(
                        "Repayment Rate",
                        ["Historical Average", "Conservative (75%)", "Optimistic (125%)", "Custom"],
                        help="Expected loan repayments"
                    )
                    
                    if inflow_method == "Custom":
                        if forecast_period == "Weekly":
                            custom_inflow = st.number_input(
                                "Custom Weekly Inflows",
                                min_value=0,
                                value=int(weekly_inflow_rate),
                                step=10000,
                                format="%d"
                            )
                        else:
                            custom_inflow = st.number_input(
                                "Custom Monthly Inflows",
                                min_value=0,
                                value=int(monthly_inflow_rate),
                                step=50000,
                                format="%d"
                            )
                    
                    # Growth rate for inflows
                    st.write("**Inflow Growth Assumptions**")
                    monthly_growth_rate = st.slider(
                        "Monthly Inflow Growth Rate",
                        min_value=-10.0,
                        max_value=10.0,
                        value=2.0,
                        step=0.5,
                        format="%.1f%%",
                        help="Expected monthly growth in loan repayments (compound)"
                    ) / 100  # Convert percentage to decimal
                else:
                    st.info("Enable QBO integration for repayment forecasting")
            
            with col3:
                st.write("**Operating Expenses**")
                
                # Manual OpEx input with better defaults
                if forecast_period == "Weekly":
                    opex_input = st.number_input(
                        "Weekly Operating Expenses",
                        min_value=0,
                        value=2500,  # ~$10k/month
                        step=500,
                        format="%d",
                        help="Your average weekly operating costs"
                    )
                else:
                    opex_input = st.number_input(
                        "Monthly Operating Expenses",
                        min_value=0,
                        value=10000,  # $10k/month as requested
                        step=1000,
                        format="%d",
                        help="Your average monthly operating costs"
                    )
                
                # Forecast horizon - limited to 6 months
                if forecast_period == "Weekly":
                    forecast_horizon = st.slider(
                        "Forecast Horizon (weeks)",
                        min_value=4,
                        max_value=26,  # 6 months
                        value=13  # 3 months default
                    )
                else:
                    forecast_horizon = st.slider(
                        "Forecast Horizon (months)",
                        min_value=1,
                        max_value=6,
                        value=3
                    )
            
            # Calculate rates based on selections
            if forecast_period == "Weekly":
                base_deployment = weekly_deployment_rate
                base_inflow = weekly_inflow_rate
                time_unit = "week"
            else:
                base_deployment = monthly_deployment_rate
                base_inflow = monthly_inflow_rate
                time_unit = "month"
            
            # Adjust deployment rate based on method
            if deployment_method == "Historical Average":
                deployment_rate = base_deployment
            elif deployment_method == "Conservative (75%)":
                deployment_rate = base_deployment * 0.75
            elif deployment_method == "Aggressive (125%)":
                deployment_rate = base_deployment * 1.25
            elif deployment_method == "Deal-Based":
                # Calculate from deals × participation
                deployment_rate = target_deals_per_period * avg_participation
                st.info(f"Deployment rate: {target_deals_per_period:d} deals × ${avg_participation:,.0f} = ${deployment_rate:,.0f} per {time_unit}")
            else:  # Custom Amount
                deployment_rate = custom_deployment
                st.info(f"Custom deployment rate: ${deployment_rate:,.0f} per {time_unit}")
            
            # Adjust inflow rate
            if has_qbo_data:
                if inflow_method == "Historical Average":
                    inflow_rate = base_inflow
                elif inflow_method == "Conservative (75%)":
                    inflow_rate = base_inflow * 0.75
                elif inflow_method == "Optimistic (125%)":
                    inflow_rate = base_inflow * 1.25
                else:
                    inflow_rate = custom_inflow
                    
                # Get growth rate (default to 0 if not defined)
                if 'monthly_growth_rate' not in locals():
                    monthly_growth_rate = 0
            else:
                inflow_rate = 0
                monthly_growth_rate = 0
            
            # Use manual OpEx input
            opex_rate = opex_input
            
            # Calculate net flow
            net_flow_per_period = inflow_rate - deployment_rate - opex_rate
            
            # Display forecast results
            st.markdown("---")
            st.subheader("Forecast Results")
            st.info("📊 **Note**: The summary metrics below are calculated using starting flow rates for simplicity. The actual forecast chart and detailed table include the {:.1f}% monthly growth in loan repayments, which extends your runway and improves cash position over time.".format(monthly_growth_rate * 100))
            
            # Show parameters
            if has_qbo_data and monthly_growth_rate != 0:
                st.info(f"""
                **Forecast Parameters:**
                - Capital Deployment: ${deployment_rate:,.0f} per {time_unit}
                - Loan Repayments: ${inflow_rate:,.0f} per {time_unit} (starting)
                - Inflow Growth Rate: {monthly_growth_rate*100:.1f}% monthly
                - Operating Expenses: ${opex_rate:,.0f} per {time_unit}
                - **Net Cash Flow: ${net_flow_per_period:,.0f} per {time_unit}** (starting)
                """)
            else:
                st.info(f"""
                **Forecast Parameters:**
                - Capital Deployment: ${deployment_rate:,.0f} per {time_unit}
                - Loan Repayments: ${inflow_rate:,.0f} per {time_unit}
                - Operating Expenses: ${opex_rate:,.0f} per {time_unit}
                - **Net Cash Flow: ${net_flow_per_period:,.0f} per {time_unit}**
                """)
            
            # Add warning if net flow is very negative
            if net_flow_per_period < -50000:
                st.error(f"""
                ⚠️ **Critical Cash Flow Warning**
                
                Your net cash flow is extremely negative (${net_flow_per_period:,.0f} per {time_unit}).
                
                **Current rates:**
                - Deploying: ${deployment_rate:,.0f}
                - Receiving: ${inflow_rate:,.0f}
                - OpEx: ${opex_rate:,.0f}
                
                The forecast will automatically reduce deployment to preserve minimum cash.
                Consider adjusting your deployment strategy or operating expenses.
                """)
            
            # Show what's actually happening
            if deployment_method == "Deal-Based":
                st.success(f"""
                **Deal-Based Deployment:**
                - {target_deals_per_period:d} deals × ${avg_participation:,.0f} = ${deployment_rate:,.0f} per {time_unit}
                """)
            elif deployment_method == "Custom Amount":
                st.success(f"""
                **Custom Deployment:**
                - Deployment rate: ${deployment_rate:,.0f} per {time_unit}
                """)
            else:
                st.success(f"""
                **{deployment_method} Deployment:**
                - Historical rate: ${base_deployment:,.0f}
                - Adjusted rate: ${deployment_rate:,.0f}
                """)
            
            # Calculate runway ending date
            runway_ending_date = None
            if net_flow_per_period < 0:
                usable_cash = starting_cash - min_cash_threshold
                if usable_cash > 0:
                    runway_periods = usable_cash / abs(net_flow_per_period)
                    if forecast_period == "Weekly":
                        runway_ending_date = datetime.now() + timedelta(weeks=runway_periods)
                    else:
                        runway_ending_date = datetime.now() + timedelta(days=runway_periods * 30.44)
            
            # Key metrics in 2x2 format
            col1, col2 = st.columns(2)
            
            with col1:
                if net_flow_per_period < 0:
                    usable_cash = starting_cash - min_cash_threshold
                    if usable_cash > 0:
                        runway = usable_cash / abs(net_flow_per_period)
                        st.metric(
                            "Cash Runway",
                            f"{runway:.1f} {time_unit}s",
                            help="Time until minimum reserve"
                        )
                    else:
                        st.metric("Cash Runway", "Below minimum")
                else:
                    st.metric(
                        "Cash Flow",
                        "Positive",
                        delta=f"+${net_flow_per_period:,.0f}/{time_unit}"
                    )
                
                ending_cash = starting_cash + (net_flow_per_period * forecast_horizon)
                st.metric(
                    f"Cash in {forecast_horizon} {time_unit}s",
                    f"${ending_cash:,.0f}" if ending_cash >= 0 else f"-${abs(ending_cash):,.0f}",
                    delta=f"{ending_cash - starting_cash:+,.0f}"
                )
            
            with col2:
                if runway_ending_date:
                    st.metric(
                        "Runway Ends",
                        runway_ending_date.strftime("%b %d, %Y"),
                        help="Date when cash reaches minimum"
                    )
                else:
                    st.metric("Runway", "Indefinite", help="Positive cash flow")
                
                breakeven_deployment = inflow_rate - opex_rate
                st.metric(
                    "Break-even Deploy",
                    f"${max(0, breakeven_deployment):,.0f}",
                    help=f"Max deployment for neutral flow"
                )
            
            # Generate forecast data
            st.markdown("---")
            st.subheader("Cash Flow Projection")
            
            # Create forecast periods - include starting point
            if forecast_period == "Weekly":
                dates = pd.date_range(start=datetime.now(), periods=forecast_horizon + 1, freq='W')
            else:
                dates = pd.date_range(start=datetime.now(), periods=forecast_horizon + 1, freq='M')
            
            forecast_data = []
            current_cash = starting_cash
            
            # Add starting point
            forecast_data.append({
                "Date": datetime.now(),
                "Starting Cash": starting_cash,
                "Deployment": 0,
                "Inflows": 0,
                "OpEx": 0,
                "Net Flow": 0,
                "Ending Cash": starting_cash
            })
            
            # Add forecast periods
            for i, date in enumerate(dates[1:], 1):
                period_starting_cash = current_cash
                
                # Calculate flows for this period
                period_deployment = deployment_rate
                
                # Apply growth to inflows
                if forecast_period == "Weekly":
                    # Convert monthly growth to weekly
                    weekly_growth_rate = (1 + monthly_growth_rate) ** (1/4.33) - 1
                    period_inflows = inflow_rate * ((1 + weekly_growth_rate) ** (i - 1))
                else:
                    # Apply monthly growth directly
                    period_inflows = inflow_rate * ((1 + monthly_growth_rate) ** (i - 1))
                
                period_opex = opex_rate
                
                # Calculate net flow
                period_net_flow = period_inflows - period_deployment - period_opex
                
                # Calculate ending cash for this period
                period_ending_cash = period_starting_cash + period_net_flow
                
                # Store the period data
                forecast_data.append({
                    "Date": date,
                    "Starting Cash": period_starting_cash,
                    "Deployment": period_deployment,
                    "Inflows": period_inflows,
                    "OpEx": period_opex,
                    "Net Flow": period_net_flow,
                    "Ending Cash": period_ending_cash
                })
                
                # Update current cash for next period
                current_cash = period_ending_cash
            
            forecast_df = pd.DataFrame(forecast_data)
            
            # Define date format before using it
            date_format = "%b %d" if forecast_period == "Weekly" else "%b %Y"
            
            # Show calculation breakdown in expander
            with st.expander("View Cash Flow Details"):
                tab1, tab2 = st.tabs(["Calculation Breakdown", "Detailed Summary"])
                
                with tab1:
                    st.write("**First 3 Periods Breakdown:**")
                    for i in range(min(3, len(forecast_df)-1)):
                        row = forecast_df.iloc[i+1]
                        
                        # Calculate growth factor for this period
                        if forecast_period == "Weekly":
                            weekly_growth_rate = (1 + monthly_growth_rate) ** (1/4.33) - 1
                            growth_factor = (1 + weekly_growth_rate) ** i
                        else:
                            growth_factor = (1 + monthly_growth_rate) ** i
                        
                        st.write(f"""
                        Period {i+1} ({row['Date'].strftime('%b %d, %Y')}):
                        - Starting Cash: ${row['Starting Cash']:,.0f}
                        - Inflows: +${row['Inflows']:,.0f} (Growth factor: {growth_factor:.3f})
                        - Deployment: -${row['Deployment']:,.0f}
                        - OpEx: -${row['OpEx']:,.0f}
                        - Net Flow: ${row['Net Flow']:,.0f}
                        - **Ending Cash: ${row['Ending Cash']:,.0f}**
                        """)
                    
                    # Show inflow progression
                    if has_qbo_data and monthly_growth_rate != 0:
                        st.write("\n**Inflow Growth Progression:**")
                        st.write(f"- Base inflow rate: ${inflow_rate:,.0f}")
                        st.write(f"- Monthly growth rate: {monthly_growth_rate*100:.1f}%")
                        if forecast_period == "Weekly":
                            weekly_rate = (1 + monthly_growth_rate) ** (1/4.33) - 1
                            st.write(f"- Weekly growth rate: {weekly_rate*100:.1f}%")
                        
                        # Show first 5 periods
                        for i in range(min(5, len(forecast_df)-1)):
                            row = forecast_df.iloc[i+1]
                            st.write(f"- Period {i+1}: ${row['Inflows']:,.0f}")
                
                with tab2:
                    summary_df = forecast_df.iloc[1:].copy()  # Skip first row (starting point)
                    summary_df["Date"] = summary_df["Date"].dt.strftime(date_format)
                    
                    st.dataframe(
                        summary_df[["Date", "Starting Cash", "Inflows", "Deployment", "OpEx", "Net Flow", "Ending Cash"]],
                        width="stretch",
                        column_config={
                            "Date": st.column_config.TextColumn("Period"),
                            "Starting Cash": st.column_config.NumberColumn("Starting Cash", format="$%.0f"),
                            "Inflows": st.column_config.NumberColumn("Inflows", format="$%.0f"),
                            "Deployment": st.column_config.NumberColumn("Deployment", format="$%.0f"),
                            "OpEx": st.column_config.NumberColumn("OpEx", format="$%.0f"),
                            "Net Flow": st.column_config.NumberColumn("Net Flow", format="$%+.0f"),
                            "Ending Cash": st.column_config.NumberColumn("Ending Cash", format="$%.0f")
                        },
                        hide_index=True
                    )
            
            # Cash position chart
            
            # Create the chart with ending cash positions
            y_min = min(forecast_df["Ending Cash"].min(), min_cash_threshold) - 50000
            y_max = forecast_df["Ending Cash"].max() + 50000
            
            # Ensure we can see variation
            if y_max - y_min < 100000:
                y_range_center = (y_max + y_min) / 2
                y_min = y_range_center - 100000
                y_max = y_range_center + 100000
            
            # Use the EXACT working test chart format
            test_data = forecast_df[['Date', 'Ending Cash']].copy()
            cash_chart = alt.Chart(test_data).mark_line(point=True).encode(
                x='Date:T',
                y='Ending Cash:Q'
            ).properties(
                width=700,
                height=400,
                title="Projected Cash Position Over Time"
            )
            
            # Create reserve line data
            reserve_data = pd.DataFrame({
                'Reserve': [min_cash_threshold]
            })
            
            # Add simple reserve line
            reserve_line = alt.Chart(reserve_data).mark_rule(
                color='red',
                strokeDash=[5, 5]
            ).encode(
                y='Reserve:Q'
            )
            
            # Layer them together
            combined_chart = cash_chart + reserve_line
            
            st.altair_chart(combined_chart, width="stretch")
            
            # Warnings
            if forecast_df["Ending Cash"].min() < min_cash_threshold:
                periods_below = len(forecast_df[forecast_df["Ending Cash"] < min_cash_threshold])
                st.error(f"⚠️ Warning: Cash will fall below minimum reserve of ${min_cash_threshold:,.0f} in {periods_below} periods")
            
            if forecast_df["Ending Cash"].min() < 0:
                first_negative = forecast_df[forecast_df["Ending Cash"] < 0].iloc[0]
                st.error(f"💸 Cash goes negative on {first_negative['Date'].strftime('%b %d, %Y')}")
            
            if net_flow_per_period < 0:
                monthly_burn = net_flow_per_period * (4.33 if forecast_period == "Weekly" else 1)
                st.warning(f"📉 Negative cash flow: Burning ${abs(monthly_burn):,.0f} per month")

            # ===== SAVE FORECAST SNAPSHOT =====
            st.markdown("---")
            st.subheader("💾 Save Forecast for Tracking")

            col1, col2 = st.columns([2, 1])

            with col1:
                st.markdown("""
                Save this forecast to track its accuracy over time.
                When actuals come in, you can compare your predictions to reality.
                """)

            with col2:
                if st.button("📊 Save Forecast Snapshot", type="primary"):
                    # Build forecast params
                    forecast_params = {
                        "forecast_period": forecast_period.lower(),
                        "forecast_horizon": forecast_horizon,
                        "starting_cash": starting_cash,
                        "deployment_rate": deployment_rate,
                        "inflow_rate": inflow_rate,
                        "opex_rate": opex_rate,
                        "growth_rate": monthly_growth_rate if 'monthly_growth_rate' in locals() else 0
                    }

                    # Use the forecast_df that was already created
                    if store_forecast_snapshot(forecast_df, forecast_params):
                        st.success("✅ Forecast saved! Track accuracy in the History section below.")
                    else:
                        st.error("Failed to save forecast. Please try again.")

            # ===== FORECAST HISTORY TRACKING =====
            st.markdown("---")
            render_forecast_history_tracking(qbo_df)

        else:
            st.error("No valid deal data available for forecasting")

    else:
        st.error("No deal data available for forecasting")
