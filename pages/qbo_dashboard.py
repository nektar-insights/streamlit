# pages/qbo_dashboard_enhanced.py
from utils.imports import *
from utils.qbo_data_loader import load_qbo_data, load_deals, load_mca_deals
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# -------------------------
# Setup: Supabase Connection & Load Data
# -------------------------
supabase = get_supabase_client()

# Load data using centralized functions
df, gl_df = load_qbo_data()
deals_df = load_deals()
mca_deals_df = load_mca_deals()

# -------------------------
# Enhanced Data Preprocessing
# -------------------------
def preprocess_financial_data(dataframe):
    """Enhanced preprocessing for financial transaction data"""
    if dataframe.empty:
        return dataframe
    
    # Convert numeric columns
    numeric_cols = ["total_amount", "balance", "amount"]
    for col in numeric_cols:
        if col in dataframe.columns:
            dataframe[col] = pd.to_numeric(dataframe[col], errors="coerce")
    
    # Convert date columns
    date_cols = ["date", "txn_date", "due_date", "created_date"]
    for col in date_cols:
        if col in dataframe.columns:
            dataframe[col] = pd.to_datetime(dataframe[col], errors="coerce")
    
    # Create derived columns for analysis
    if "txn_date" in dataframe.columns:
        dataframe["year_month"] = dataframe["txn_date"].dt.to_period("M")
        dataframe["week"] = dataframe["txn_date"].dt.isocalendar().week
        dataframe["day_of_week"] = dataframe["txn_date"].dt.day_name()
        dataframe["days_since_txn"] = (pd.Timestamp.now() - dataframe["txn_date"]).dt.days
    
    return dataframe

# Apply enhanced preprocessing
df = preprocess_financial_data(df)
gl_df = preprocess_financial_data(gl_df)

st.title("🏦 Enhanced MCA Dashboard - Risk, Cash Flow & Forecasting")
st.markdown("---")

# -------------------------
# 1. RISK ANALYSIS SECTION
# -------------------------
st.header("🚨 Risk Analysis")

# Transaction-based risk metrics
if not df.empty:
    # Focus on transaction data for risk analysis
    risk_df = df[df["transaction_type"].isin(["Invoice", "Payment"])].copy()
    risk_df = risk_df[~risk_df["customer_name"].isin(["CSL", "VEEM"])]
    
    if not risk_df.empty:
        # Calculate comprehensive risk metrics
        risk_df["total_amount"] = risk_df["total_amount"].abs()
        
        # Customer-level risk analysis
        customer_risk = risk_df.groupby("customer_name").agg({
            "total_amount": ["sum", "count", "std"],
            "txn_date": ["min", "max"]
        }).reset_index()
        
        customer_risk.columns = ["customer_name", "total_volume", "txn_count", "amount_volatility", "first_txn", "last_txn"]
        customer_risk["days_active"] = (customer_risk["last_txn"] - customer_risk["first_txn"]).dt.days
        customer_risk["avg_txn_size"] = customer_risk["total_volume"] / customer_risk["txn_count"]
        customer_risk["volatility_ratio"] = customer_risk["amount_volatility"] / customer_risk["avg_txn_size"]
        
        # Calculate payment behavior
        payment_behavior = risk_df.pivot_table(
            index="customer_name",
            columns="transaction_type", 
            values="total_amount",
            aggfunc="sum",
            fill_value=0
        ).reset_index()
        
        if "Invoice" in payment_behavior.columns and "Payment" in payment_behavior.columns:
            payment_behavior["payment_ratio"] = np.where(
                payment_behavior["Invoice"] > 0,
                payment_behavior["Payment"] / payment_behavior["Invoice"],
                0
            )
            payment_behavior["outstanding_balance"] = payment_behavior["Invoice"] - payment_behavior["Payment"]
            
            # Risk scoring
            payment_behavior["risk_score"] = (
                (1 - payment_behavior["payment_ratio"]) * 0.4 +  # 40% weight on payment ratio
                (payment_behavior["outstanding_balance"] / payment_behavior["Invoice"].max()) * 0.3 +  # 30% weight on relative balance
                np.where(payment_behavior["payment_ratio"] < 0.5, 0.3, 0)  # 30% penalty for low payment ratio
            )
            
            # Risk categories
            payment_behavior["risk_category"] = pd.cut(
                payment_behavior["risk_score"],
                bins=[0, 0.2, 0.5, 1.0],
                labels=["🟢 Low Risk", "🟡 Medium Risk", "🔴 High Risk"]
            )
            
            # Display risk dashboard
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                high_risk_count = (payment_behavior["risk_category"] == "🔴 High Risk").sum()
                st.metric("High Risk Customers", high_risk_count)
            with col2:
                avg_payment_ratio = payment_behavior["payment_ratio"].mean()
                st.metric("Avg Payment Ratio", f"{avg_payment_ratio:.1%}")
            with col3:
                total_outstanding = payment_behavior["outstanding_balance"].sum()
                st.metric("Total Outstanding", f"${total_outstanding:,.0f}")
            with col4:
                avg_risk_score = payment_behavior["risk_score"].mean()
                st.metric("Avg Risk Score", f"{avg_risk_score:.2f}")
            
            # Risk visualization
            risk_chart = alt.Chart(payment_behavior.head(20)).mark_circle(size=100).encode(
                x=alt.X("payment_ratio:Q", title="Payment Ratio", axis=alt.Axis(format=".1%")),
                y=alt.Y("outstanding_balance:Q", title="Outstanding Balance ($)", axis=alt.Axis(format="$,.0f")),
                color=alt.Color("risk_category:N", title="Risk Category"),
                size=alt.Size("Invoice:Q", title="Invoice Volume"),
                tooltip=["customer_name", "payment_ratio:Q", "outstanding_balance:Q", "risk_category"]
            ).properties(
                width=700, height=400,
                title="Customer Risk Profile: Payment Ratio vs Outstanding Balance"
            )
            
            st.altair_chart(risk_chart, use_container_width=True)

# -------------------------
# 2. ENHANCED CASH FLOW ANALYSIS
# -------------------------
st.header("💰 Cash Flow Analysis")

if not df.empty and "txn_date" in df.columns:
    # Transaction-based cash flow analysis
    cash_flow_df = df.copy()
    
    # Categorize transactions for cash flow
    cash_flow_df["cash_impact"] = np.where(
        cash_flow_df["transaction_type"].isin(["Payment", "Deposit", "Receipt"]),
        cash_flow_df["total_amount"].abs(),  # Positive cash flow
        np.where(
            cash_flow_df["transaction_type"].isin(["Invoice", "Bill", "Expense"]),
            -cash_flow_df["total_amount"].abs(),  # Negative cash flow
            0
        )
    )
    
    # Daily cash flow analysis
    daily_cash_flow = cash_flow_df.groupby(cash_flow_df["txn_date"].dt.date).agg({
        "cash_impact": "sum",
        "total_amount": "count"
    }).reset_index()
    daily_cash_flow.columns = ["date", "net_cash_flow", "transaction_count"]
    daily_cash_flow["cumulative_cash_flow"] = daily_cash_flow["net_cash_flow"].cumsum()
    
    # Cash flow trend chart
    daily_cash_flow["date_str"] = daily_cash_flow["date"].astype(str)
    
    base = alt.Chart(daily_cash_flow).add_selection(
        alt.selection_interval(bind="scales")
    )
    
    cash_flow_chart = base.mark_line(color="steelblue", strokeWidth=2).encode(
        x=alt.X("date:T", title="Date"),
        y=alt.Y("net_cash_flow:Q", title="Daily Net Cash Flow ($)", axis=alt.Axis(format="$,.0f")),
        tooltip=["date:T", "net_cash_flow:Q", "transaction_count:Q"]
    ).properties(
        width=700, height=300,
        title="Daily Net Cash Flow Trend"
    )
    
    cumulative_chart = base.mark_line(color="orange", strokeWidth=2).encode(
        x=alt.X("date:T", title="Date"),
        y=alt.Y("cumulative_cash_flow:Q", title="Cumulative Cash Flow ($)", axis=alt.Axis(format="$,.0f")),
        tooltip=["date:T", "cumulative_cash_flow:Q"]
    ).properties(
        width=700, height=300,
        title="Cumulative Cash Flow"
    )
    
    st.altair_chart(cash_flow_chart, use_container_width=True)
    st.altair_chart(cumulative_chart, use_container_width=True)
    
    # Cash flow statistics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        avg_daily_flow = daily_cash_flow["net_cash_flow"].mean()
        st.metric("Avg Daily Cash Flow", f"${avg_daily_flow:,.0f}")
    with col2:
        cash_flow_volatility = daily_cash_flow["net_cash_flow"].std()
        st.metric("Cash Flow Volatility", f"${cash_flow_volatility:,.0f}")
    with col3:
        positive_days = (daily_cash_flow["net_cash_flow"] > 0).sum()
        total_days = len(daily_cash_flow)
        st.metric("Positive Cash Flow Days", f"{positive_days}/{total_days}")
    with col4:
        current_position = daily_cash_flow["cumulative_cash_flow"].iloc[-1]
        st.metric("Current Cash Position", f"${current_position:,.0f}")

# -------------------------
# 3. ADVANCED FORECASTING
# -------------------------
st.header("🔮 Cash Flow Forecasting")

if not df.empty and len(daily_cash_flow) >= 30:
    # Prepare data for forecasting
    forecast_data = daily_cash_flow.copy()
    forecast_data["date"] = pd.to_datetime(forecast_data["date"])
    forecast_data = forecast_data.sort_values("date")
    
    # Simple moving average forecast
    window_size = min(30, len(forecast_data) // 2)
    forecast_data["ma_30"] = forecast_data["net_cash_flow"].rolling(window=window_size).mean()
    
    # Seasonal decomposition for better forecasting
    forecast_data["day_of_week"] = forecast_data["date"].dt.day_name()
    forecast_data["week_of_month"] = forecast_data["date"].dt.day // 7 + 1
    
    # Weekly pattern analysis
    weekly_pattern = forecast_data.groupby("day_of_week")["net_cash_flow"].mean()
    
    # Generate forecast for next 30 days
    last_date = forecast_data["date"].max()
    forecast_dates = pd.date_range(start=last_date + timedelta(days=1), periods=30, freq="D")
    
    # Simple forecast using historical average and weekly patterns
    forecast_values = []
    for date in forecast_dates:
        day_name = date.day_name()
        base_forecast = forecast_data["net_cash_flow"].tail(30).mean()
        seasonal_adjustment = weekly_pattern.get(day_name, 0) - weekly_pattern.mean()
        forecast_values.append(base_forecast + seasonal_adjustment)
    
    forecast_df = pd.DataFrame({
        "date": forecast_dates,
        "forecasted_cash_flow": forecast_values,
        "type": "Forecast"
    })
    
    # Combine historical and forecast data for visualization
    historical_viz = forecast_data[["date", "net_cash_flow"]].copy()
    historical_viz["type"] = "Historical"
    historical_viz = historical_viz.rename(columns={"net_cash_flow": "forecasted_cash_flow"})
    
    combined_forecast = pd.concat([
        historical_viz.tail(60),  # Last 60 days of historical data
        forecast_df
    ], ignore_index=True)
    
    # Forecast visualization
    forecast_chart = alt.Chart(combined_forecast).mark_line().encode(
        x=alt.X("date:T", title="Date"),
        y=alt.Y("forecasted_cash_flow:Q", title="Cash Flow ($)", axis=alt.Axis(format="$,.0f")),
        color=alt.Color("type:N", title="Data Type", scale=alt.Scale(range=["steelblue", "orange"])),
        strokeDash=alt.StrokeDash("type:N", scale=alt.Scale(range=[[1,0], [5,5]])),
        tooltip=["date:T", "forecasted_cash_flow:Q", "type:N"]
    ).properties(
        width=700, height=400,
        title="30-Day Cash Flow Forecast"
    )
    
    st.altair_chart(forecast_chart, use_container_width=True)
    
    # Forecast summary metrics
    forecast_sum = sum(forecast_values)
    forecast_avg = np.mean(forecast_values)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("30-Day Forecast Total", f"${forecast_sum:,.0f}")
    with col2:
        st.metric("Avg Daily Forecast", f"${forecast_avg:,.0f}")
    with col3:
        confidence_interval = np.std(forecast_data["net_cash_flow"].tail(30)) * 1.96
        st.metric("Forecast Confidence (±)", f"${confidence_interval:,.0f}")

# -------------------------
# 4. THREE ADDITIONAL ADVANCED VISUALS
# -------------------------

st.header("📊 Advanced Analytics")

# VISUAL 1: Transaction Velocity and Frequency Analysis
st.subheader("1. Transaction Velocity & Frequency Analysis")
if not df.empty:
    # Calculate transaction velocity (transactions per customer per time period)
    velocity_df = df.groupby(["customer_name", df["txn_date"].dt.to_period("M")]).agg({
        "total_amount": ["sum", "count"],
        "txn_date": ["min", "max"]
    }).reset_index()
    
    velocity_df.columns = ["customer_name", "month", "total_amount", "txn_count", "first_txn", "last_txn"]
    velocity_df["velocity_score"] = velocity_df["txn_count"] * (velocity_df["total_amount"] / velocity_df["txn_count"])
    
    # Top customers by transaction velocity
    top_velocity = velocity_df.groupby("customer_name")["velocity_score"].mean().sort_values(ascending=False).head(15)
    
    velocity_chart = alt.Chart(top_velocity.reset_index()).mark_bar().encode(
        x=alt.X("velocity_score:Q", title="Velocity Score"),
        y=alt.Y("customer_name:N", sort="-x", title="Customer"),
        color=alt.Color("velocity_score:Q", scale=alt.Scale(scheme="viridis"), legend=None),
        tooltip=["customer_name", "velocity_score:Q"]
    ).properties(
        width=700, height=400,
        title="Top 15 Customers by Transaction Velocity"
    )
    
    st.altair_chart(velocity_chart, use_container_width=True)

# VISUAL 2: Payment Timing Analysis (Critical for MCA risk assessment)
st.subheader("2. Payment Timing Analysis")
if not df.empty:
    payment_timing = df[df["transaction_type"] == "Payment"].copy()
    if not payment_timing.empty:
        payment_timing["hour"] = payment_timing["txn_date"].dt.hour
        payment_timing["day_of_week"] = payment_timing["txn_date"].dt.day_name()
        
        # Heatmap data
        timing_heatmap = payment_timing.groupby(["day_of_week", "hour"]).size().reset_index(name="payment_count")
        
        # Create heatmap
        heatmap = alt.Chart(timing_heatmap).mark_rect().encode(
            x=alt.X("hour:O", title="Hour of Day"),
            y=alt.Y("day_of_week:O", title="Day of Week", sort=["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]),
            color=alt.Color("payment_count:Q", scale=alt.Scale(scheme="blues"), title="Payment Count"),
            tooltip=["day_of_week", "hour:O", "payment_count"]
        ).properties(
            width=700, height=300,
            title="Payment Timing Heatmap - When Do Customers Pay?"
        )
        
        st.altair_chart(heatmap, use_container_width=True)

# VISUAL 3: Customer Lifecycle and Cohort Analysis
st.subheader("3. Customer Lifecycle & Cohort Analysis")
if not df.empty:
    # Calculate customer lifecycle metrics
    lifecycle_df = df.groupby("customer_name").agg({
        "txn_date": ["min", "max", "count"],
        "total_amount": ["sum", "mean"]
    }).reset_index()
    
    lifecycle_df.columns = ["customer_name", "first_transaction", "last_transaction", "total_transactions", "total_value", "avg_transaction"]
    lifecycle_df["customer_lifespan"] = (lifecycle_df["last_transaction"] - lifecycle_df["first_transaction"]).dt.days
    lifecycle_df["customer_age_months"] = (pd.Timestamp.now() - lifecycle_df["first_transaction"]).dt.days / 30
    
    # Create cohort analysis
    lifecycle_df["cohort_month"] = lifecycle_df["first_transaction"].dt.to_period("M")
    lifecycle_df["value_per_month"] = np.where(
        lifecycle_df["customer_age_months"] > 0,
        lifecycle_df["total_value"] / lifecycle_df["customer_age_months"],
        lifecycle_df["total_value"]
    )
    
    # Scatter plot: Customer Age vs Value
    lifecycle_scatter = alt.Chart(lifecycle_df).mark_circle(size=60).encode(
        x=alt.X("customer_age_months:Q", title="Customer Age (Months)"),
        y=alt.Y("total_value:Q", title="Total Customer Value ($)", axis=alt.Axis(format="$,.0f")),
        size=alt.Size("total_transactions:Q", title="Transaction Count"),
        color=alt.Color("value_per_month:Q", scale=alt.Scale(scheme="viridis"), title="Value/Month"),
        tooltip=["customer_name", "customer_age_months:Q", "total_value:Q", "total_transactions:Q"]
    ).properties(
        width=700, height=400,
        title="Customer Lifecycle Analysis: Age vs Total Value"
    )
    
    st.altair_chart(lifecycle_scatter, use_container_width=True)
    
    # Cohort summary
    cohort_summary = lifecycle_df.groupby("cohort_month").agg({
        "customer_name": "count",
        "total_value": "mean",
        "customer_age_months": "mean"
    }).reset_index()
    cohort_summary.columns = ["cohort_month", "customer_count", "avg_value", "avg_age"]
    
    st.subheader("Cohort Summary")
    st.dataframe(cohort_summary.head(10), use_container_width=True)

# -------------------------
# EXECUTIVE SUMMARY
# -------------------------
st.header("📋 Executive Summary")

if not df.empty:
    # Key metrics summary
    total_customers = df["customer_name"].nunique()
    total_transactions = len(df)
    total_volume = df["total_amount"].sum()
    avg_transaction_size = df["total_amount"].mean()
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Customers", f"{total_customers:,}")
    with col2:
        st.metric("Total Transactions", f"{total_transactions:,}")
    with col3:
        st.metric("Total Volume", f"${total_volume:,.0f}")
    with col4:
        st.metric("Avg Transaction", f"${avg_transaction_size:,.0f}")
    
    # Risk summary
    if 'payment_behavior' in locals():
        high_risk_pct = (payment_behavior["risk_category"] == "🔴 High Risk").mean() * 100
        st.warning(f"⚠️ {high_risk_pct:.1f}% of customers are classified as High Risk")
    
    # Cash flow summary
    if 'daily_cash_flow' in locals():
        recent_trend = daily_cash_flow["net_cash_flow"].tail(7).mean()
        if recent_trend > 0:
            st.success(f"✅ Positive 7-day average cash flow: ${recent_trend:,.0f}")
        else:
            st.error(f"❌ Negative 7-day average cash flow: ${recent_trend:,.0f}")

st.markdown("---")
st.markdown("*Dashboard last updated: " + pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S") + "*")
