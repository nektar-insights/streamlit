# cash_flow_forecast.py
import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
from datetime import datetime, timedelta

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
            
            # Key metrics
            col1, col2, col3, col4 = st.columns(4)
            
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
            
            with col2:
                if runway_ending_date:
                    st.metric(
                        "Runway Ends",
                        runway_ending_date.strftime("%b %d, %Y"),
                        help="Date when cash reaches minimum"
                    )
                else:
                    st.metric("Runway", "Indefinite", help="Positive cash flow")
            
            with col3:
                ending_cash = starting_cash + (net_flow_per_period * forecast_horizon)
                st.metric(
                    f"Cash in {forecast_horizon} {time_unit}s",
                    f"${ending_cash:,.0f}" if ending_cash >= 0 else f"-${abs(ending_cash):,.0f}",
                    delta=f"{ending_cash - starting_cash:+,.0f}"
                )
            
            with col4:
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
                            st.write(f"- Weekly growth rate: {weekly_rate*100:.2f}%")
                        
                        # Show first 5 periods
                        for i in range(min(5, len(forecast_df)-1)):
                            row = forecast_df.iloc[i+1]
                            st.write(f"- Period {i+1}: ${row['Inflows']:,.0f}")
                
                with tab2:
                    summary_df = forecast_df.iloc[1:].copy()  # Skip first row (starting point)
                    summary_df["Date"] = summary_df["Date"].dt.strftime(date_format)
                    
                    st.dataframe(
                        summary_df[["Date", "Starting Cash", "Inflows", "Deployment", "OpEx", "Net Flow", "Ending Cash"]],
                        use_container_width=True,
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
            
            st.altair_chart(combined_chart, use_container_width=True)
            
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
        
        else:
            st.error("No valid deal data available for forecasting")
    
    else:
        st.error("No deal data available for forecasting")
