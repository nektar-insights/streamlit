import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
from datetime import datetime, timedelta
alt.data_transformers.disable_max_rows()

# -------------------------------------------------
# Helper lives at module level (above the main func)
# -------------------------------------------------
def _sanitize(df, date_cols=None, num_cols=None):
    """Force dates → datetime64 and numbers → numeric."""
    if date_cols:
        for c in date_cols:
            if c in df.columns:
                df[c] = pd.to_datetime(df[c], errors="coerce")
    if num_cols:
        for c in num_cols:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def create_cash_flow_forecast(deals_df, closed_won_df, qbo_df=None):
    """
    Create an integrated cash-flow forecast that includes:
    - Capital deployment (outflows for new deals)
    - Loan repayments (inflows from QBO)
    - Operating expenses (other outflows from QBO)
    """

    # ---------- 1) Sanitize inputs ----------
    closed_won_df = _sanitize(
        closed_won_df,
        date_cols=["date_created"],
        num_cols=["amount"]
    )

    if qbo_df is not None:
        qbo_df = _sanitize(
            qbo_df,
            date_cols=["txn_date"],
            num_cols=["total_amount"]
        )

    # (historical metrics, forecast config, charts, etc.)
    
    st.header("Capital Deployment Forecast")
    st.markdown("---")
    
    # Check if we have QBO data for integrated forecast
    has_qbo_data = qbo_df is not None and not qbo_df.empty and "txn_date" in qbo_df.columns
    
    # Prepare QBO data if available
    if has_qbo_data:
        # Ensure txn_date is datetime
        qbo_df["txn_date"] = pd.to_datetime(qbo_df["txn_date"], errors="coerce")
        
        # Ensure total_amount is numeric
        if "total_amount" in qbo_df.columns:
            qbo_df["total_amount"] = pd.to_numeric(qbo_df["total_amount"], errors="coerce")
        # Filter for relevant cash transactions
        cash_transactions = qbo_df[qbo_df["transaction_type"].isin([
            "Payment", "Deposit", "Receipt", "Invoice", "Bill", "Expense"
        ])].copy()
        
        # Drop any rows with invalid dates
        cash_transactions = cash_transactions.dropna(subset=["txn_date"])
        
        # Categorize cash flows
        cash_transactions["cash_impact"] = np.where(
            cash_transactions["transaction_type"].isin(["Payment", "Deposit", "Receipt"]),
            cash_transactions["total_amount"].abs(),  # Inflows
            -cash_transactions["total_amount"].abs()  # Outflows
        )
        
        # Separate customer payments (loan repayments) from other transactions
        customer_payments = cash_transactions[
            (cash_transactions["transaction_type"].isin(["Payment", "Receipt"])) &
            (~cash_transactions["customer_name"].isin(["CSL", "VEEM"]))
        ].copy()
        
        # Operating expenses (bills, expenses, non-customer payments)
        operating_expenses = cash_transactions[
            cash_transactions["transaction_type"].isin(["Bill", "Expense"])
        ].copy()
        
        # Calculate historical metrics only if we have valid data
        if len(cash_transactions) > 0 and cash_transactions["txn_date"].notna().any():
            min_date = cash_transactions["txn_date"].min()
            max_date = cash_transactions["txn_date"].max()
            total_days = (max_date - min_date).days + 1
            total_weeks = total_days / 7
            total_months = total_days / 30.44
            
            # Inflow metrics (loan repayments)
            total_inflows = customer_payments["total_amount"].abs().sum()
            weekly_inflow_rate = total_inflows / total_weeks if total_weeks > 0 else 0
            monthly_inflow_rate = total_inflows / total_months if total_months > 0 else 0
            
            # Operating expense metrics
            total_opex = operating_expenses["total_amount"].abs().sum()
            weekly_opex_rate = total_opex / total_weeks if total_weeks > 0 else 0
            monthly_opex_rate = total_opex / total_months if total_months > 0 else 0
        else:
            # No valid QBO data
            weekly_inflow_rate = monthly_inflow_rate = 0
            weekly_opex_rate = monthly_opex_rate = 0
            has_qbo_data = False
        
    else:
        # Default to zero if no QBO data
        weekly_inflow_rate = monthly_inflow_rate = 0
        weekly_opex_rate = monthly_opex_rate = 0
    
    # Historical deployment metrics (from deals)
    if not closed_won_df.empty and "date_created" in closed_won_df.columns:
        # Historical analysis period
        st.subheader("Historical Cash Flow Analysis")
        
        # Date range for analysis
        deal_min_date = closed_won_df["date_created"].min()
        deal_max_date = closed_won_df["date_created"].max()
        deal_total_days = (deal_max_date - deal_min_date).days + 1
        deal_total_weeks = deal_total_days / 7
        deal_total_months = deal_total_days / 30.44
        
        # Calculate historical deployment metrics
        total_deployed = closed_won_df["amount"].sum()
        deal_count = len(closed_won_df)
        
        # Deployment rates
        weekly_deployment_rate = total_deployed / deal_total_weeks if deal_total_weeks > 0 else 0
        monthly_deployment_rate = total_deployed / deal_total_months if deal_total_months > 0 else 0
        
        # Deal metrics
        avg_deal_size = closed_won_df["amount"].mean()
        median_deal_size = closed_won_df["amount"].median()
        deals_per_week = deal_count / deal_total_weeks if deal_total_weeks > 0 else 0
        deals_per_month = deal_count / deal_total_months if deal_total_months > 0 else 0
        
        # Display comprehensive historical metrics
        if has_qbo_data:
            # Show integrated metrics
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.write("**Capital Deployment (Outflows)**")
                st.metric("Weekly Deployment", f"${weekly_deployment_rate:,.0f}")
                st.metric("Monthly Deployment", f"${monthly_deployment_rate:,.0f}")
                st.metric("Avg Deal Size", f"${avg_deal_size:,.0f}")
            
            with col2:
                st.write("**Loan Repayments (Inflows)**")
                st.metric("Weekly Inflows", f"${weekly_inflow_rate:,.0f}")
                st.metric("Monthly Inflows", f"${monthly_inflow_rate:,.0f}")
                net_weekly = weekly_inflow_rate - weekly_deployment_rate - weekly_opex_rate
                st.metric("Net Weekly Flow", f"${net_weekly:,.0f}", 
                         delta="Positive" if net_weekly > 0 else "Negative")
            
            with col3:
                st.write("**Operating Expenses**")
                st.metric("Weekly OpEx", f"${weekly_opex_rate:,.0f}")
                st.metric("Monthly OpEx", f"${monthly_opex_rate:,.0f}")
                net_monthly = monthly_inflow_rate - monthly_deployment_rate - monthly_opex_rate
                st.metric("Net Monthly Flow", f"${net_monthly:,.0f}",
                         delta="Positive" if net_monthly > 0 else "Negative")
        else:
            # Show simple deployment metrics only
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Avg Weekly Deployment", f"${weekly_deployment_rate:,.0f}")
                st.metric("Avg Deal Size", f"${avg_deal_size:,.0f}")
            with col2:
                st.metric("Avg Monthly Deployment", f"${monthly_deployment_rate:,.0f}")
                st.metric("Median Deal Size", f"${median_deal_size:,.0f}")
            with col3:
                st.metric("Deals per Week", f"{deals_per_week:.1f}")
                st.metric("Total Deals", f"{deal_count}")
            with col4:
                st.metric("Deals per Month", f"{deals_per_month:.1f}")
                st.metric("Total Deployed", f"${total_deployed:,.0f}")
        
        st.markdown("---")
        
        # Forecast Configuration Section
        if has_qbo_data:
            st.subheader("Integrated Forecast Configuration")
        else:
            st.subheader("Forecast Configuration")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.write("**Starting Position**")
            # Starting cash position
            if has_qbo_data:
                starting_cash = st.number_input(
                    "Current Cash Position",
                    min_value=0,
                    value=500000,
                    step=100000,
                    format="%d",
                    help="Enter your current available cash balance"
                )
            else:
                starting_cash = st.number_input(
                    "Available Capital (Dry Powder)",
                    min_value=0,
                    value=500000,
                    step=100000,
                    format="%d",
                    help="Enter the amount of capital available for deployment"
                )
            
            # Forecast period
            forecast_period = st.selectbox(
                "Forecast Period",
                ["Weekly", "Monthly"],
                help="Choose the time period for forecasting"
            )
            
            # Forecast horizon
            if has_qbo_data:
                if forecast_period == "Weekly":
                    forecast_horizon = st.slider(
                        "Forecast Horizon (weeks)",
                        min_value=4,
                        max_value=52,
                        value=26,
                        help="How many weeks to forecast"
                    )
                else:
                    forecast_horizon = st.slider(
                        "Forecast Horizon (months)",
                        min_value=3,
                        max_value=24,
                        value=12,
                        help="How many months to forecast"
                    )
        
        with col2:
            st.write("**Deployment Assumptions**")
            # Deployment rate method
            deployment_method = st.selectbox(
                "Deployment Rate",
                ["Historical Average", "Conservative (75%)", "Aggressive (125%)", "Custom"],
                help="Select deployment rate assumption"
            )
            
            if deployment_method == "Custom":
                if forecast_period == "Weekly":
                    custom_deployment = st.number_input(
                        "Custom Weekly Deployment",
                        min_value=0,
                        value=int(weekly_deployment_rate),
                        step=10000,
                        format="%d"
                    )
                else:
                    custom_deployment = st.number_input(
                        "Custom Monthly Deployment",
                        min_value=0,
                        value=int(monthly_deployment_rate),
                        step=50000,
                        format="%d"
                    )
            
            if has_qbo_data:
                # Inflow assumptions
                inflow_method = st.selectbox(
                    "Repayment Rate",
                    ["Historical Average", "Conservative (75%)", "Optimistic (125%)", "Custom"],
                    help="Select expected repayment rate"
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
            else:
                # Deal size assumption for simple forecast
                deal_size_method = st.selectbox(
                    "Deal Size Assumption",
                    ["Historical Average", "Historical Median", "Custom Amount"],
                    help="Select how to estimate future deal sizes"
                )
                
                if deal_size_method == "Custom Amount":
                    custom_deal_size = st.number_input(
                        "Custom Average Deal Size",
                        min_value=1000,
                        value=int(avg_deal_size),
                        step=1000,
                        format="%d"
                    )
                    forecast_deal_size = custom_deal_size
                else:
                    forecast_deal_size = avg_deal_size if deal_size_method == "Historical Average" else median_deal_size
        
        with col3:
            if has_qbo_data:
                st.write("**Operating Expense Assumptions**")
                # OpEx assumptions
                opex_method = st.selectbox(
                    "Operating Expenses",
                    ["Historical Average", "Reduced (75%)", "Increased (125%)", "Custom"],
                    help="Select operating expense assumption"
                )
                
                if opex_method == "Custom":
                    if forecast_period == "Weekly":
                        custom_opex = st.number_input(
                            "Custom Weekly OpEx",
                            min_value=0,
                            value=int(weekly_opex_rate),
                            step=5000,
                            format="%d"
                        )
                    else:
                        custom_opex = st.number_input(
                            "Custom Monthly OpEx",
                            min_value=0,
                            value=int(monthly_opex_rate),
                            step=20000,
                            format="%d"
                        )
                
                # Minimum cash threshold
                min_cash_threshold = st.number_input(
                    "Minimum Cash Reserve",
                    min_value=0,
                    value=100000,
                    step=50000,
                    format="%d",
                    help="Minimum cash to maintain"
                )
        
        # Calculate rates based on selections
        if forecast_period == "Weekly":
            base_deployment = weekly_deployment_rate
            base_inflow = weekly_inflow_rate
            base_opex = weekly_opex_rate
            time_unit = "week"
            deals_per_period = deals_per_week
        else:
            base_deployment = monthly_deployment_rate
            base_inflow = monthly_inflow_rate
            base_opex = monthly_opex_rate
            time_unit = "month"
            deals_per_period = deals_per_month
        
        # Adjust deployment rate
        if deployment_method == "Historical Average":
            deployment_rate = base_deployment
        elif deployment_method == "Conservative (75%)":
            deployment_rate = base_deployment * 0.75
        elif deployment_method == "Aggressive (125%)":
            deployment_rate = base_deployment * 1.25
        else:  # Custom
            deployment_rate = custom_deployment
        
        if has_qbo_data:
            # Adjust inflow rate
            if inflow_method == "Historical Average":
                inflow_rate = base_inflow
            elif inflow_method == "Conservative (75%)":
                inflow_rate = base_inflow * 0.75
            elif inflow_method == "Optimistic (125%)":
                inflow_rate = base_inflow * 1.25
            else:  # Custom
                inflow_rate = custom_inflow
            
            # Adjust opex rate
            if opex_method == "Historical Average":
                opex_rate = base_opex
            elif opex_method == "Reduced (75%)":
                opex_rate = base_opex * 0.75
            elif opex_method == "Increased (125%)":
                opex_rate = base_opex * 1.25
            else:  # Custom
                opex_rate = custom_opex
            
            # Calculate net flow
            net_flow_per_period = inflow_rate - deployment_rate - opex_rate
        else:
            # Simple forecast without QBO data
            inflow_rate = 0
            opex_rate = 0
            net_flow_per_period = -deployment_rate
        
        # Display forecast results
        st.markdown("---")
        if has_qbo_data:
            st.subheader("Integrated Forecast Results")
            
            # Show selected parameters
            st.info(f"""
            **Selected Parameters:**
            - Capital Deployment: ${deployment_rate:,.0f} per {time_unit}
            - Expected Inflows: ${inflow_rate:,.0f} per {time_unit}
            - Operating Expenses: ${opex_rate:,.0f} per {time_unit}
            - **Net Cash Flow: ${net_flow_per_period:,.0f} per {time_unit}**
            - Starting Cash: ${starting_cash:,.0f}
            - Minimum Reserve: ${min_cash_threshold:,.0f}
            """)
            
            # Key metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                if net_flow_per_period < 0:
                    # Calculate runway
                    usable_cash = starting_cash - min_cash_threshold
                    if usable_cash > 0:
                        runway = usable_cash / abs(net_flow_per_period)
                        st.metric(
                            "Cash Runway",
                            f"{runway:.1f} {time_unit}s",
                            help="Time until minimum reserve is reached"
                        )
                    else:
                        st.metric("Cash Runway", "Already below minimum")
                else:
                    st.metric(
                        "Cash Flow Status",
                        "Positive",
                        delta=f"+${net_flow_per_period:,.0f}/{time_unit}"
                    )
            
            with col2:
                # Deals possible with available capital
                if avg_deal_size > 0:
                    deals_possible = (starting_cash - min_cash_threshold) / avg_deal_size
                    st.metric(
                        "Deals Possible",
                        f"{max(0, deals_possible):.0f}",
                        help="Based on average deal size and min reserve"
                    )
            
            with col3:
                # Cash position after forecast period
                ending_cash = starting_cash + (net_flow_per_period * forecast_horizon)
                st.metric(
                    f"Cash in {forecast_horizon} {time_unit}s",
                    f"${max(0, ending_cash):,.0f}",
                    delta=f"{ending_cash - starting_cash:+,.0f}"
                )
            
            with col4:
                # Break-even deployment rate
                breakeven_deployment = inflow_rate - opex_rate
                st.metric(
                    "Break-even Deployment",
                    f"${max(0, breakeven_deployment):,.0f}",
                    help=f"Max deployment for neutral cash flow per {time_unit}"
                )
        else:
            # Simple forecast results
            st.subheader("Forecast Results")
            
            if deployment_rate > 0:
                periods_until_depleted = starting_cash / deployment_rate
                deals_remaining = (starting_cash / forecast_deal_size) if forecast_deal_size > 0 else 0
                
                # Show selected parameters
                st.info(f"""
                **Selected Parameters:**
                - Deployment Rate: ${deployment_rate:,.0f} per {time_unit}
                - Deal Size: ${forecast_deal_size:,.0f}
                - Available Capital: ${starting_cash:,.0f}
                """)
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric(
                        f"Capital Lasts",
                        f"{periods_until_depleted:.1f} {time_unit}s",
                        help=f"Based on ${deployment_rate:,.0f} per {time_unit}"
                    )
                with col2:
                    st.metric(
                        "Estimated Deals",
                        f"{deals_remaining:.0f}",
                        help=f"Based on ${forecast_deal_size:,.0f} avg deal size"
                    )
                with col3:
                    depletion_date = datetime.now() + timedelta(
                        weeks=periods_until_depleted if forecast_period == "Weekly" else periods_until_depleted * 4.33
                    )
                    st.metric(
                        "Depletion Date",
                        depletion_date.strftime("%b %d, %Y"),
                        help="Estimated date when capital will be fully deployed"
                    )
        
        # Generate forecast visualization
        st.markdown("---")
        st.subheader("Cash Flow Projection")
        
        # Create forecast data
        if has_qbo_data:
            # Integrated forecast with QBO data
            if forecast_period == "Weekly":
                dates = pd.date_range(start=datetime.now(), periods=forecast_horizon + 1, freq='W')
            else:
                dates = pd.date_range(start=datetime.now(), periods=forecast_horizon + 1, freq='M')
            
            forecast_data = []
            current_cash = starting_cash
            
            for i, date in enumerate(dates[1:], 1):
                # Calculate flows for this period
                period_deployment = deployment_rate
                period_inflows = inflow_rate
                period_opex = opex_rate
                
                # Adjust deployment if cash is getting low
                if current_cash - period_deployment - period_opex < min_cash_threshold:
                    period_deployment = max(0, current_cash - min_cash_threshold - period_opex)
                
                # Calculate net flow
                period_net_flow = period_inflows - period_deployment - period_opex
                current_cash = current_cash + period_net_flow
                
                forecast_data.append({
                    "Date": date,
                    "Period": i,
                    "Cash Position": current_cash,
                    "Deployment": period_deployment,
                    "Inflows": period_inflows,
                    "OpEx": period_opex,
                    "Net Flow": period_net_flow,
                    "Above Minimum": current_cash > min_cash_threshold
                })
            
            forecast_df = pd.DataFrame(forecast_data)

            with st.expander("🧮 Forecast DF preview"):
                st.dataframe(forecast_df.head())
                st.write("rows:", len(forecast_df))
        
            # Cash position chart
            date_format = "%b %d" if forecast_period == "Weekly" else "%b %Y"
            
            # Create base chart
            base = alt.Chart(forecast_df).encode(
                x=alt.X("Date:T", 
                       title=f"Date ({forecast_period})",
                       axis=alt.Axis(format=date_format, labelAngle=-45))
            )
            
            # Cash position line
            cash_line = base.mark_line(strokeWidth=3, color="#2E8B57").encode(
                y=alt.Y("Cash Position:Q", 
                       title="Cash Position ($)", 
                       axis=alt.Axis(format="$,.0f")),
                tooltip=[
                    alt.Tooltip("Date:T", format="%b %d, %Y"),
                    alt.Tooltip("Cash Position:Q", format="$,.0f"),
                    alt.Tooltip("Net Flow:Q", format="$+,.0f")
                ]
            )
            
            # Minimum threshold line
            threshold_line = alt.Chart(pd.DataFrame({
                "Minimum Reserve": [min_cash_threshold]
            })).mark_rule(color="red", strokeDash=[5, 5], strokeWidth=2).encode(
                y="Minimum Reserve:Q",
                tooltip=alt.Tooltip("Minimum Reserve:Q", format="$,.0f")
            )

            st.subheader("🐞 Quick chart smoke-test")
            test_chart = (
                alt.Chart(forecast_df.head(10))
                  .mark_point(size=60)
                  .encode(
                      x="Date:T",
                      y="Cash Position:Q",
                      tooltip=["Date:T", "Cash Position:Q"]
                  )
            )
            st.altair_chart(test_chart, use_container_width=True)

            # Combine charts
            combined_chart = (cash_line + threshold_line).properties(
                height=400,
                title="Projected Cash Position"
            )
            
            st.altair_chart(combined_chart, use_container_width=True)
            
            # Flow components chart
            st.subheader("Cash Flow Components")
            
            # Prepare data for stacked bar chart
            flow_data = forecast_df[["Date", "Inflows", "Deployment", "OpEx"]].copy()
            flow_data["Deployment"] = -flow_data["Deployment"]  # Make negative for visualization
            flow_data["OpEx"] = -flow_data["OpEx"]  # Make negative for visualization
            
            flow_long = flow_data.melt(id_vars=["Date"], 
                                      value_vars=["Inflows", "Deployment", "OpEx"],
                                      var_name="Component", 
                                      value_name="Amount")
            
            flow_chart = alt.Chart(flow_long).mark_bar().encode(
                x=alt.X("Date:T", 
                       title=f"Date ({forecast_period})",
                       axis=alt.Axis(format=date_format, labelAngle=-45)),
                y=alt.Y("Amount:Q", 
                       title="Cash Flow ($)", 
                       axis=alt.Axis(format="$,.0f")),
                color=alt.Color("Component:N", 
                              scale=alt.Scale(
                                  domain=["Inflows", "Deployment", "OpEx"],
                                  range=["#27AE60", "#E74C3C", "#F39C12"]
                              )),
                tooltip=[
                    alt.Tooltip("Date:T", format="%b %d, %Y"),
                    alt.Tooltip("Component:N"),
                    alt.Tooltip("Amount:Q", format="$+,.0f")
                ]
            ).properties(
                height=300,
                title="Cash Flow Components by Period"
            )
            
            st.altair_chart(flow_chart, use_container_width=True)
        else:
            # Simple forecast visualization
            if forecast_period == "Weekly":
                periods = int(min(periods_until_depleted + 1, 52))  # Cap at 1 year
                dates = pd.date_range(start=datetime.now(), periods=periods, freq='W')
            else:
                periods = int(min(periods_until_depleted + 1, 12))  # Cap at 1 year
                dates = pd.date_range(start=datetime.now(), periods=periods, freq='M')
            
            forecast_data = []
            remaining_capital = starting_cash
            cumulative_deals = 0
            
            for i, date in enumerate(dates):
                if remaining_capital > 0:
                    deployed_this_period = min(deployment_rate, remaining_capital)
                    remaining_capital -= deployed_this_period
                    deals_this_period = deployed_this_period / forecast_deal_size if forecast_deal_size > 0 else 0
                    cumulative_deals += deals_this_period
                else:
                    deployed_this_period = 0
                    deals_this_period = 0
                
                forecast_data.append({
                    "Date": date,
                    "Period": i + 1,
                    "Remaining Capital": remaining_capital,
                    "Deployed This Period": deployed_this_period,
                    "Cumulative Deployed": starting_cash - remaining_capital,
                    "Deals This Period": deals_this_period,
                    "Cumulative Deals": cumulative_deals
                })
            
            forecast_df = pd.DataFrame(forecast_data)
            
            # Capital depletion chart with proper date formatting
            date_format = "%b %d" if forecast_period == "Weekly" else "%b %Y"
            
            capital_chart = alt.Chart(forecast_df).transform_fold(
                ["Remaining Capital", "Cumulative Deployed"],
                as_=["Metric", "Value"]
            ).mark_line(strokeWidth=3).encode(
                x=alt.X("Date:T", 
                       title=f"Date ({forecast_period})",
                       axis=alt.Axis(format=date_format, labelAngle=-45)),
                y=alt.Y("Value:Q", title="Capital ($)", axis=alt.Axis(format="$,.0f")),
                color=alt.Color("Metric:N", 
                              scale=alt.Scale(domain=["Remaining Capital", "Cumulative Deployed"],
                                            range=["#e74c3c", "#27ae60"])),
                tooltip=[
                    alt.Tooltip("Date:T", format="%b %d, %Y"),
                    alt.Tooltip("Metric:N"),
                    alt.Tooltip("Value:Q", format="$,.0f")
                ]
            ).properties(
                height=400,
                title="Capital Deployment Forecast"
            )
            
            st.altair_chart(capital_chart, use_container_width=True)
        
        # Summary table
        st.subheader("Detailed Cash Flow Summary")
        
        if has_qbo_data:
            # Create summary table for integrated forecast
            summary_df = forecast_df[["Date", "Cash Position", "Deployment", "Inflows", "OpEx", "Net Flow"]].copy()
            summary_df["Date"] = summary_df["Date"].dt.strftime(date_format)
            
            # Display with formatting
            st.dataframe(
                summary_df,
                use_container_width=True,
                column_config={
                    "Date": st.column_config.TextColumn("Period", width="small"),
                    "Cash Position": st.column_config.NumberColumn("Cash Position", format="$%.0f"),
                    "Deployment": st.column_config.NumberColumn("Deployment", format="$%.0f"),
                    "Inflows": st.column_config.NumberColumn("Inflows", format="$%.0f"),
                    "OpEx": st.column_config.NumberColumn("OpEx", format="$%.0f"),
                    "Net Flow": st.column_config.NumberColumn("Net Flow", format="$%+.0f")
                },
                hide_index=True
            )
            
            # Warnings
            if ending_cash < min_cash_threshold:
                st.error(f"⚠️ Warning: Cash position will fall below minimum reserve of ${min_cash_threshold:,.0f}")
            
            if net_flow_per_period < 0:
                monthly_burn = net_flow_per_period * (4.33 if forecast_period == "Weekly" else 1)
                st.warning(f"📉 Negative cash flow: Burning ${abs(monthly_burn):,.0f} per month")
        else:
            # Simple summary table
            summary_df = forecast_df[["Date", "Deployed This Period", "Deals This Period", "Remaining Capital"]].copy()
            summary_df["Date"] = summary_df["Date"].dt.strftime(date_format)
            summary_df = summary_df[summary_df["Deployed This Period"] > 0]  # Only show periods with deployment
            
            # Display as formatted table
            st.dataframe(
                summary_df,
                use_container_width=True,
                column_config={
                    "Date": st.column_config.TextColumn("Period", width="small"),
                    "Deployed This Period": st.column_config.NumberColumn("Capital Deployed", format="$%.0f", width="medium"),
                    "Deals This Period": st.column_config.NumberColumn("Deals", format="%.1f", width="small"),
                    "Remaining Capital": st.column_config.NumberColumn("Remaining", format="$%.0f", width="medium")
                },
                hide_index=True
            )
        
        # Scenario Analysis
        st.markdown("---")
        st.subheader("Scenario Analysis")
        
        # Allow user to adjust parameters
        col1, col2 = st.columns(2)
        
        with col1:
            participation_rate_change = st.slider(
                "Adjust Participation Rate",
                min_value=-50,
                max_value=50,
                value=0,
                step=10,
                format="%d%%",
                help="Adjust the participation rate up or down"
            )
        
        with col2:
            deal_size_change = st.slider(
                "Adjust Average Deal Size",
                min_value=-50,
                max_value=50,
                value=0,
                step=10,
                format="%d%%",
                help="Adjust the average deal size up or down"
            )
        
        # Recalculate with adjustments
        adjusted_deployment_rate = deployment_rate * (1 + participation_rate_change / 100)
        adjusted_deal_size = forecast_deal_size * (1 + deal_size_change / 100) if not has_qbo_data else avg_deal_size * (1 + deal_size_change / 100)
        
        # Show adjusted parameters
        st.info(f"""
        **Adjusted Parameters:**
        - Deployment Rate: ${deployment_rate:,.0f} → ${adjusted_deployment_rate:,.0f} per {time_unit} ({participation_rate_change:+d}%)
        - Deal Size: ${forecast_deal_size if not has_qbo_data else avg_deal_size:,.0f} → ${adjusted_deal_size:,.0f} ({deal_size_change:+d}%)
        """)
        
        if has_qbo_data:
            # Integrated scenario analysis
            adjusted_net_flow = inflow_rate - adjusted_deployment_rate - opex_rate
            
            col1, col2, col3 = st.columns(3)
            with col1:
                if adjusted_net_flow < 0 and net_flow_per_period < 0:
                    usable_cash = starting_cash - min_cash_threshold
                    if usable_cash > 0:
                        adjusted_runway = usable_cash / abs(adjusted_net_flow)
                        original_runway = usable_cash / abs(net_flow_per_period)
                        delta_runway = adjusted_runway - original_runway
                        st.metric(
                            "Adjusted Runway",
                            f"{adjusted_runway:.1f} {time_unit}s",
                            delta=f"{delta_runway:+.1f}"
                        )
                elif adjusted_net_flow >= 0 and net_flow_per_period < 0:
                    st.metric(
                        "Adjusted Status",
                        "Cash Positive",
                        delta="Improved"
                    )
                elif adjusted_net_flow < 0 and net_flow_per_period >= 0:
                    usable_cash = starting_cash - min_cash_threshold
                    if usable_cash > 0:
                        adjusted_runway = usable_cash / abs(adjusted_net_flow)
                        st.metric(
                            "Adjusted Status",
                            f"{adjusted_runway:.1f} {time_unit}s runway",
                            delta="Negative"
                        )
            
            with col2:
                adjusted_deals = (starting_cash - min_cash_threshold) / adjusted_deal_size if adjusted_deal_size > 0 else 0
                original_deals = (starting_cash - min_cash_threshold) / avg_deal_size if avg_deal_size > 0 else 0
                delta_deals = adjusted_deals - original_deals
                st.metric(
                    "Adjusted Deal Count",
                    f"{adjusted_deals:.0f}",
                    delta=f"{delta_deals:+.0f}"
                )
            
            with col3:
                adjusted_ending = starting_cash + (adjusted_net_flow * forecast_horizon)
                delta_ending = adjusted_ending - ending_cash
                st.metric(
                    f"Adjusted Cash ({forecast_horizon} {time_unit}s)",
                    f"${max(0, adjusted_ending):,.0f}",
                    delta=f"{delta_ending:+,.0f}"
                )
        else:
            # Simple scenario analysis
            if adjusted_deployment_rate > 0:
                adjusted_periods = starting_cash / adjusted_deployment_rate
                adjusted_deals = starting_cash / adjusted_deal_size if adjusted_deal_size > 0 else 0
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    delta_periods = adjusted_periods - periods_until_depleted
                    st.metric(
                        "Adjusted Duration",
                        f"{adjusted_periods:.1f} {time_unit}s",
                        delta=f"{delta_periods:+.1f}"
                    )
                with col2:
                    delta_deals = adjusted_deals - deals_remaining
                    st.metric(
                        "Adjusted Deal Count",
                        f"{adjusted_deals:.0f}",
                        delta=f"{delta_deals:+.0f}"
                    )
                with col3:
                    adjusted_date = datetime.now() + timedelta(
                        weeks=adjusted_periods if forecast_period == "Weekly" else adjusted_periods * 4.33
                    )
                    delta_days = (adjusted_date - depletion_date).days
                    st.metric(
                        "Adjusted Depletion",
                        adjusted_date.strftime("%b %d, %Y"),
                        delta=f"{delta_days:+d} days"
                    )
        
    else:
        st.warning("No historical deal data available for forecasting.")
