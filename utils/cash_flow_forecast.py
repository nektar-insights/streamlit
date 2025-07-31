# cash_flow_forecast.py
import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
from datetime import datetime, timedelta

def create_cash_flow_forecast(deals_df, closed_won_df):
    """
    Create a cash flow forecast module for the dashboard
    
    Args:
        deals_df: DataFrame with all deals data
        closed_won_df: DataFrame with only closed/won deals
    """
    
    st.header("Capital Deployment Forecast")
    st.markdown("---")
    
    # Calculate historical metrics from your existing data
    if not closed_won_df.empty and "date_created" in closed_won_df.columns:
        # Historical analysis period
        st.subheader("Historical Capital Deployment Analysis")
        
        # Date range for analysis
        min_date = closed_won_df["date_created"].min()
        max_date = closed_won_df["date_created"].max()
        total_days = (max_date - min_date).days + 1
        total_weeks = total_days / 7
        total_months = total_days / 30.44
        
        # Calculate historical deployment metrics
        total_deployed = closed_won_df["amount"].sum()
        deal_count = len(closed_won_df)
        
        # Deployment rates
        daily_deployment_rate = total_deployed / total_days if total_days > 0 else 0
        weekly_deployment_rate = total_deployed / total_weeks if total_weeks > 0 else 0
        monthly_deployment_rate = total_deployed / total_months if total_months > 0 else 0
        
        # Deal frequency
        deals_per_week = deal_count / total_weeks if total_weeks > 0 else 0
        deals_per_month = deal_count / total_months if total_months > 0 else 0
        
        # Average deal size
        avg_deal_size = closed_won_df["amount"].mean()
        median_deal_size = closed_won_df["amount"].median()
        
        # Display historical metrics
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
        st.subheader("Forecast Configuration")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Dry powder input
            dry_powder = st.number_input(
                "Available Capital (Dry Powder)",
                min_value=0,
                value=1000000,
                step=100000,
                format="%d",
                help="Enter the amount of capital available for deployment"
            )
            
            # Forecast method selection
            forecast_method = st.selectbox(
                "Deployment Rate Method",
                ["Historical Average", "Median", "Conservative (75% of avg)", "Aggressive (125% of avg)", "Custom"],
                help="Select how to estimate future deployment rate"
            )
            
            # Forecast period
            forecast_period = st.selectbox(
                "Forecast Period",
                ["Weekly", "Monthly"],
                help="Choose the time period for forecasting"
            )
        
        with col2:
            # Deal size assumption
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
            
            # Custom deployment rate if selected
            if forecast_method == "Custom":
                if forecast_period == "Weekly":
                    custom_rate = st.number_input(
                        "Custom Weekly Deployment Rate",
                        min_value=0,
                        value=int(weekly_deployment_rate),
                        step=10000,
                        format="%d"
                    )
                else:
                    custom_rate = st.number_input(
                        "Custom Monthly Deployment Rate",
                        min_value=0,
                        value=int(monthly_deployment_rate),
                        step=50000,
                        format="%d"
                    )
        
        # Calculate deployment rate based on selection
        if forecast_period == "Weekly":
            base_rate = weekly_deployment_rate
            time_unit = "week"
            deals_per_period = deals_per_week
        else:
            base_rate = monthly_deployment_rate
            time_unit = "month"
            deals_per_period = deals_per_month
        
        # Adjust rate based on method
        if forecast_method == "Historical Average":
            deployment_rate = base_rate
        elif forecast_method == "Median":
            # Use median deal size to calculate rate
            deployment_rate = median_deal_size * deals_per_period
        elif forecast_method == "Conservative (75% of avg)":
            deployment_rate = base_rate * 0.75
        elif forecast_method == "Aggressive (125% of avg)":
            deployment_rate = base_rate * 1.25
        else:  # Custom
            deployment_rate = custom_rate
        
        # Calculate forecast
        if deployment_rate > 0:
            periods_until_depleted = dry_powder / deployment_rate
            deals_remaining = (dry_powder / forecast_deal_size) if forecast_deal_size > 0 else 0
            
            # Display forecast results
            st.markdown("---")
            st.subheader("Forecast Results")
            
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
            
            # Create forecast visualization
            st.markdown("---")
            st.subheader("Capital Deployment Visualization")
            
            # Generate forecast data
            if forecast_period == "Weekly":
                periods = int(min(periods_until_depleted + 1, 52))  # Cap at 1 year
                dates = pd.date_range(start=datetime.now(), periods=periods, freq='W')
            else:
                periods = int(min(periods_until_depleted + 1, 12))  # Cap at 1 year
                dates = pd.date_range(start=datetime.now(), periods=periods, freq='M')
            
            forecast_data = []
            remaining_capital = dry_powder
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
                    "Cumulative Deployed": dry_powder - remaining_capital,
                    "Deals This Period": deals_this_period,
                    "Cumulative Deals": cumulative_deals
                })
            
            forecast_df = pd.DataFrame(forecast_data)
            
            # Capital depletion chart
            capital_chart = alt.Chart(forecast_df).transform_fold(
                ["Remaining Capital", "Cumulative Deployed"],
                as_=["Metric", "Value"]
            ).mark_line(strokeWidth=3).encode(
                x=alt.X("Date:T", title="Date"),
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
            
            # Deals forecast chart
            deals_chart = alt.Chart(forecast_df).mark_bar(
                color="#3498db",
                opacity=0.7
            ).encode(
                x=alt.X("Date:T", title="Date"),
                y=alt.Y("Deals This Period:Q", title="Number of Deals"),
                tooltip=[
                    alt.Tooltip("Date:T", format="%b %d, %Y"),
                    alt.Tooltip("Deals This Period:Q", format=".1f"),
                    alt.Tooltip("Cumulative Deals:Q", format=".0f")
                ]
            ).properties(
                height=300,
                title="Projected Deal Flow"
            )
            
            st.altair_chart(deals_chart, use_container_width=True)
            
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
            adjusted_deal_size = forecast_deal_size * (1 + deal_size_change / 100)
            
            if adjusted_deployment_rate > 0:
                adjusted_periods = dry_powder / adjusted_deployment_rate
                adjusted_deals = dry_powder / adjusted_deal_size if adjusted_deal_size > 0 else 0
                
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
            st.warning("Deployment rate is zero. Please check your data or adjust the forecast parameters.")
    
    else:
        st.warning("No historical data available for forecasting. Please ensure you have closed/won deals with valid dates.")
