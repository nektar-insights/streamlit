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
                    ["Historical Average", "Conservative (75%)", "Aggressive (125%)", "Custom"],
                    help="How much capital to deploy"
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
                else:
                    st.info("Enable QBO integration for repayment forecasting")
            
            with col3:
                st.write("**Operating Expenses**")
                
                # Manual OpEx input
                if forecast_period == "Weekly":
                    opex_input = st.number_input(
                        "Weekly Operating Expenses",
                        min_value=0,
                        value=25000,
                        step=5000,
                        format="%d",
                        help="Your average weekly operating costs"
                    )
                else:
                    opex_input = st.number_input(
                        "Monthly Operating Expenses",
                        min_value=0,
                        value=100000,
                        step=10000,
                        format="%d",
                        help="Your average monthly operating costs"
                    )
                
                # Forecast horizon
                if forecast_period == "Weekly":
                    forecast_horizon = st.slider(
                        "Forecast Horizon (weeks)",
                        min_value=4,
                        max_value=52,
                        value=26
                    )
                else:
                    forecast_horizon = st.slider(
                        "Forecast Horizon (months)",
                        min_value=3,
                        max_value=24,
                        value=12
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
            
            # Adjust deployment rate
            if deployment_method == "Historical Average":
                deployment_rate = base_deployment
            elif deployment_method == "Conservative (75%)":
                deployment_rate = base_deployment * 0.75
            elif deployment_method == "Aggressive (125%)":
                deployment_rate = base_deployment * 1.25
            else:
                deployment_rate = custom_deployment
            
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
            else:
                inflow_rate = 0
            
            # Use manual OpEx input
            opex_rate = opex_input
            
            # Calculate net flow
            net_flow_per_period = inflow_rate - deployment_rate - opex_rate
            
            # Display forecast results
            st.markdown("---")
            st.subheader("Forecast Results")
            
            # Show parameters
            st.info(f"""
            **Forecast Parameters:**
            - Capital Deployment: ${deployment_rate:,.0f} per {time_unit}
            - Loan Repayments: ${inflow_rate:,.0f} per {time_unit}
            - Operating Expenses: ${opex_rate:,.0f} per {time_unit}
            - **Net Cash Flow: ${net_flow_per_period:,.0f} per {time_unit}**
            """)
            
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
                if avg_deal_size > 0:
                    deals_possible = (starting_cash - min_cash_threshold) / avg_deal_size
                    st.metric(
                        "Deals Possible",
                        f"{max(0, deals_possible):.0f}",
                        help="Based on avg deal size"
                    )
            
            with col3:
                ending_cash = starting_cash + (net_flow_per_period * forecast_horizon)
                st.metric(
                    f"Cash in {forecast_horizon} {time_unit}s",
                    f"${max(0, ending_cash):,.0f}",
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
            
            # Create forecast periods
            if forecast_period == "Weekly":
                dates = pd.date_range(start=datetime.now(), periods=forecast_horizon, freq='W')
            else:
                dates = pd.date_range(start=datetime.now(), periods=forecast_horizon, freq='M')
            
            forecast_data = []
            current_cash = starting_cash
            
            for i, date in enumerate(dates):
                # Calculate flows
                period_deployment = deployment_rate
                period_inflows = inflow_rate
                period_opex = opex_rate
                
                # Constrain deployment if low on cash
                if current_cash - period_deployment - period_opex < min_cash_threshold:
                    period_deployment = max(0, current_cash - min_cash_threshold - period_opex)
                
                # Net flow and new position
                period_net_flow = period_inflows - period_deployment - period_opex
                current_cash = current_cash + period_net_flow
                
                forecast_data.append({
                    "Date": date,
                    "Cash Position": current_cash,
                    "Deployment": period_deployment,
                    "Inflows": period_inflows,
                    "OpEx": period_opex,
                    "Net Flow": period_net_flow
                })
            
            forecast_df = pd.DataFrame(forecast_data)
            
            # Cash position chart
            date_format = "%b %d" if forecast_period == "Weekly" else "%b %Y"
            
            # Create the chart
            cash_line = alt.Chart(forecast_df).mark_line(
                strokeWidth=3,
                color="#2E8B57"
            ).encode(
                x=alt.X("Date:T", 
                       title="Date",
                       axis=alt.Axis(format=date_format, labelAngle=-45)),
                y=alt.Y("Cash Position:Q", 
                       title="Cash Position ($)",
                       axis=alt.Axis(format="$,.0f")),
                tooltip=[
                    alt.Tooltip("Date:T", format="%b %d, %Y"),
                    alt.Tooltip("Cash Position:Q", format="$,.0f"),
                    alt.Tooltip("Net Flow:Q", format="$+,.0f")
                ]
            )
            
            # Add minimum threshold line
            threshold_line = alt.Chart(
                pd.DataFrame({"y": [min_cash_threshold]})
            ).mark_rule(
                color="red",
                strokeDash=[5, 5],
                strokeWidth=2
            ).encode(
                y="y:Q"
            )
            
            combined_chart = (cash_line + threshold_line).properties(
                height=400,
                title="Projected Cash Position"
            )
            
            st.altair_chart(combined_chart, use_container_width=True)
            
            # Cash flow components
            st.subheader("Cash Flow Components")
            
            # Prepare data for visualization
            components_data = []
            for _, row in forecast_df.iterrows():
                components_data.extend([
                    {"Date": row["Date"], "Type": "Inflows", "Amount": row["Inflows"]},
                    {"Date": row["Date"], "Type": "Deployment", "Amount": -row["Deployment"]},
                    {"Date": row["Date"], "Type": "OpEx", "Amount": -row["OpEx"]}
                ])
            
            components_df = pd.DataFrame(components_data)
            
            flow_chart = alt.Chart(components_df).mark_bar().encode(
                x=alt.X("Date:T",
                       title="Date",
                       axis=alt.Axis(format=date_format, labelAngle=-45)),
                y=alt.Y("Amount:Q",
                       title="Cash Flow ($)",
                       axis=alt.Axis(format="$,.0f")),
                color=alt.Color("Type:N",
                              scale=alt.Scale(
                                  domain=["Inflows", "Deployment", "OpEx"],
                                  range=["#27AE60", "#E74C3C", "#F39C12"]
                              )),
                tooltip=[
                    alt.Tooltip("Date:T", format="%b %d, %Y"),
                    alt.Tooltip("Type:N"),
                    alt.Tooltip("Amount:Q", format="$+,.0f")
                ]
            ).properties(
                height=300,
                title="Cash Flow Components by Period"
            )
            
            st.altair_chart(flow_chart, use_container_width=True)
            
            # Summary table
            st.subheader("Detailed Cash Flow Summary")
            
            summary_df = forecast_df.copy()
            summary_df["Date"] = summary_df["Date"].dt.strftime(date_format)
            
            st.dataframe(
                summary_df,
                use_container_width=True,
                column_config={
                    "Date": st.column_config.TextColumn("Period"),
                    "Cash Position": st.column_config.NumberColumn("Cash Position", format="$%.0f"),
                    "Deployment": st.column_config.NumberColumn("Deployment", format="$%.0f"),
                    "Inflows": st.column_config.NumberColumn("Inflows", format="$%.0f"),
                    "OpEx": st.column_config.NumberColumn("OpEx", format="$%.0f"),
                    "Net Flow": st.column_config.NumberColumn("Net Flow", format="$%+.0f")
                },
                hide_index=True
            )
            
            # Warnings
            if forecast_df["Cash Position"].min() < min_cash_threshold:
                st.error(f"⚠️ Warning: Cash will fall below minimum reserve of ${min_cash_threshold:,.0f}")
            
            if net_flow_per_period < 0:
                monthly_burn = net_flow_per_period * (4.33 if forecast_period == "Weekly" else 1)
                st.warning(f"📉 Negative cash flow: Burning ${abs(monthly_burn):,.0f} per month")
            
            # Scenario Analysis
            st.markdown("---")
            st.subheader("Scenario Analysis")
            
            col1, col2 = st.columns(2)
            
            with col1:
                deployment_change = st.slider(
                    "Adjust Deployment Rate",
                    min_value=-50,
                    max_value=50,
                    value=0,
                    step=10,
                    format="%d%%"
                )
            
            with col2:
                inflow_change = st.slider(
                    "Adjust Repayment Rate",
                    min_value=-50,
                    max_value=50,
                    value=0,
                    step=10,
                    format="%d%%"
                )
            
            # Calculate adjusted values
            adjusted_deployment = deployment_rate * (1 + deployment_change / 100)
            adjusted_inflows = inflow_rate * (1 + inflow_change / 100)
            adjusted_net_flow = adjusted_inflows - adjusted_deployment - opex_rate
            
            # Show adjusted parameters
            st.info(f"""
            **Adjusted Scenario:**
            - Deployment: ${deployment_rate:,.0f} → ${adjusted_deployment:,.0f} ({deployment_change:+d}%)
            - Inflows: ${inflow_rate:,.0f} → ${adjusted_inflows:,.0f} ({inflow_change:+d}%)
            - Net Flow: ${net_flow_per_period:,.0f} → ${adjusted_net_flow:,.0f} per {time_unit}
            """)
            
            # Adjusted metrics
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if adjusted_net_flow < 0:
                    usable_cash = starting_cash - min_cash_threshold
                    if usable_cash > 0:
                        adjusted_runway = usable_cash / abs(adjusted_net_flow)
                        if net_flow_per_period < 0:
                            original_runway = usable_cash / abs(net_flow_per_period)
                            delta = adjusted_runway - original_runway
                        else:
                            delta = -adjusted_runway
                        st.metric(
                            "Adjusted Runway",
                            f"{adjusted_runway:.1f} {time_unit}s",
                            delta=f"{delta:+.1f}"
                        )
                else:
                    st.metric("Adjusted Status", "Cash Positive", delta="✓")
            
            with col2:
                adjusted_ending = starting_cash + (adjusted_net_flow * forecast_horizon)
                delta_ending = adjusted_ending - ending_cash
                st.metric(
                    f"Adjusted Ending Cash",
                    f"${max(0, adjusted_ending):,.0f}",
                    delta=f"{delta_ending:+,.0f}"
                )
            
            with col3:
                adjusted_breakeven = adjusted_inflows - opex_rate
                delta_breakeven = adjusted_breakeven - breakeven_deployment
                st.metric(
                    "Adjusted Break-even",
                    f"${max(0, adjusted_breakeven):,.0f}",
                    delta=f"{delta_breakeven:+,.0f}"
                )
        
        else:
            st.error("No valid deal data available for forecasting")
    
    else:
        st.error("No deal data available for forecasting")
