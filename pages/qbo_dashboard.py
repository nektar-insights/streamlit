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
gl_df = preprocess_financial_data(gl_df)  # Keep GL data available for future use

st.title("QBO Dashboard")
st.markdown("---")

# -------------------------
# EXECUTIVE SUMMARY
# -------------------------
st.header("Executive Summary")

if not df.empty:
    # Data source indicator
    st.info("Data Source: Transaction Data (qbo_invoice_payments table) | GL Data available for future analysis")
    
    # Data range indicator
    if "txn_date" in df.columns:
        date_range = f"{df['txn_date'].min().strftime('%Y-%m-%d')} to {df['txn_date'].max().strftime('%Y-%m-%d')}"
        st.info(f"Date Range: {date_range}")
    
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

st.markdown("---")

st.title("MCA Dashboard")
st.markdown("---")

# -------------------------
# 1. RISK ANALYSIS SECTION
# -------------------------
st.header("Risk Analysis")

# Transaction-based risk metrics
if df.empty:
    st.warning("No transaction data available for risk analysis")
else:
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
                labels=["Low Risk", "Medium Risk", "High Risk"]
            )
            
            # Display risk dashboard with explanations
            st.markdown("Risk Scoring Methodology: Risk Score = 40% Payment Ratio + 30% Relative Balance + 30% Low Payment Penalty")
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                high_risk_count = (payment_behavior["risk_category"] == "High Risk").sum()
                st.metric("High Risk Customers", high_risk_count, 
                         help="Customers with risk score > 0.5 (poor payment behavior)")
            with col2:
                avg_payment_ratio = payment_behavior["payment_ratio"].mean()
                st.metric("Avg Payment Ratio", f"{avg_payment_ratio:.1%}",
                         help="Average of (Total Payments / Total Invoices) across all customers")
            with col3:
                total_outstanding = payment_behavior["outstanding_balance"].sum()
                st.metric("Total Outstanding", f"${total_outstanding:,.0f}",
                         help="Sum of all unpaid invoice amounts across customers")
            with col4:
                avg_risk_score = payment_behavior["risk_score"].mean()
                st.metric("Avg Risk Score", f"{avg_risk_score:.2f}",
                         help="Average risk score (0=lowest risk, 1=highest risk)")
            
            # Risk visualization with light colors
            risk_chart = alt.Chart(payment_behavior.head(20)).mark_circle(size=100, stroke='black', strokeWidth=1).encode(
                x=alt.X("payment_ratio:Q", title="Payment Ratio", axis=alt.Axis(format=".1%")),
                y=alt.Y("outstanding_balance:Q", title="Outstanding Balance ($)", axis=alt.Axis(format="$,.0f")),
                color=alt.Color("risk_category:N", title="Risk Category", 
                              scale=alt.Scale(range=["lightgreen", "orange", "lightcoral"])),
                size=alt.Size("Invoice:Q", title="Invoice Volume", scale=alt.Scale(range=[50, 400])),
                tooltip=["customer_name", "payment_ratio:Q", "outstanding_balance:Q", "risk_category"]
            ).properties(
                width=700, height=400,
                title="Customer Risk Profile: Payment Ratio vs Outstanding Balance"
            )
            
            st.altair_chart(risk_chart, use_container_width=True)

# -------------------------
# 2. ENHANCED CASH FLOW ANALYSIS
# -------------------------
st.header("Cash Flow Analysis")

if df.empty or "txn_date" not in df.columns:
    st.warning("No transaction data with dates available for cash flow analysis")
else:
    
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
        st.metric("Avg Daily Cash Flow", f"${avg_daily_flow:,.0f}",
                 help="Average daily net cash flow (inflows minus outflows)")
    with col2:
        cash_flow_volatility = daily_cash_flow["net_cash_flow"].std()
        st.metric("Cash Flow Volatility", f"${cash_flow_volatility:,.0f}",
                 help="Standard deviation of daily cash flows - higher values indicate more unpredictable cash flow")
    with col3:
        positive_days = (daily_cash_flow["net_cash_flow"] > 0).sum()
        total_days = len(daily_cash_flow)
        st.metric("Positive Cash Flow Days", f"{positive_days}/{total_days}",
                 help="Number of days with positive net cash flow out of total days")
    with col4:
        current_position = daily_cash_flow["cumulative_cash_flow"].iloc[-1]
        st.metric("Current Cash Position", f"${current_position:,.0f}",
                 help="Cumulative cash flow from all transactions in the period")

# -------------------------
# 3. ADVANCED FORECASTING
# -------------------------
st.header("Cash Flow Forecasting")

if df.empty or "txn_date" not in df.columns or len(daily_cash_flow) < 30:
    st.warning("Insufficient transaction data for forecasting (need at least 30 days)")
else:
    st.info("Forecasting Method: 30-day moving average with seasonal adjustments based on day-of-week patterns")
    
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
        st.metric("30-Day Forecast Total", f"${forecast_sum:,.0f}",
                 help="Sum of projected daily cash flows for the next 30 days")
    with col2:
        st.metric("Avg Daily Forecast", f"${forecast_avg:,.0f}",
                 help="Average projected daily cash flow over the next 30 days")
    with col3:
        confidence_interval = np.std(forecast_data["net_cash_flow"].tail(30)) * 1.96
        st.metric("Forecast Confidence (±)", f"${confidence_interval:,.0f}",
                 help="95% confidence interval - actual values likely within ± this amount of forecast")

# -------------------------
# 4. THREE ADDITIONAL ADVANCED VISUALS
# -------------------------

st.header("Advanced Analytics")

# VISUAL 1: Transaction Velocity and Frequency Analysis
st.subheader("1. Transaction Velocity & Frequency Analysis")
if df.empty:
    st.warning("No transaction data available")
else:
    st.markdown("Velocity Score: Transaction Count × Average Transaction Size - measures customer engagement intensity")
    
    # Calculate transaction velocity (transactions per customer per time period)
    velocity_df = df.groupby(["customer_name", df["txn_date"].dt.to_period("M")]).agg({
        "total_amount": ["sum", "count"],
        "txn_date": ["min", "max"]
    }).reset_index()
    
    velocity_df.columns = ["customer_name", "month", "total_amount", "txn_count", "first_txn", "last_txn"]
    velocity_df["velocity_score"] = velocity_df["txn_count"] * (velocity_df["total_amount"] / velocity_df["txn_count"])
    
    # Top customers by transaction velocity
    top_velocity = velocity_df.groupby("customer_name")["velocity_score"].mean().sort_values(ascending=False).head(15)
    
    velocity_chart = alt.Chart(top_velocity.reset_index()).mark_bar(color='lightblue', stroke='darkblue', strokeWidth=1).encode(
        x=alt.X("velocity_score:Q", title="Velocity Score"),
        y=alt.Y("customer_name:N", sort="-x", title="Customer"),
        tooltip=["customer_name", "velocity_score:Q"]
    ).properties(
        width=700, height=400,
        title="Top 15 Customers by Transaction Velocity"
    )
    
    st.altair_chart(velocity_chart, use_container_width=True)

# VISUAL 2: Customer Geographic Distribution Analysis
st.subheader("2. Customer Volume Distribution")
if df.empty:
    st.warning("No transaction data available")
else:
    st.markdown("Purpose: Shows transaction volume distribution across customers to identify concentration risk")
    
    # Customer transaction volume analysis
    customer_volume = df.groupby("customer_name").agg({
        "total_amount": ["sum", "count", "mean"],
        "txn_date": ["min", "max"]
    }).reset_index()
    
    customer_volume.columns = ["customer_name", "total_volume", "transaction_count", "avg_transaction", "first_txn", "last_txn"]
    customer_volume["volume_percentage"] = (customer_volume["total_volume"] / customer_volume["total_volume"].sum()) * 100
    customer_volume = customer_volume.sort_values("total_volume", ascending=False)
    
    # Concentration analysis
    top_10_concentration = customer_volume.head(10)["volume_percentage"].sum()
    top_20_concentration = customer_volume.head(20)["volume_percentage"].sum()
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Top 10 Customer Concentration", f"{top_10_concentration:.1f}%",
                 help="Percentage of total volume from top 10 customers - high concentration indicates risk")
    with col2:
        st.metric("Top 20 Customer Concentration", f"{top_20_concentration:.1f}%",
                 help="Percentage of total volume from top 20 customers")
    with col3:
        total_customers = len(customer_volume)
        st.metric("Total Active Customers", total_customers,
                 help="Number of customers with transactions in the period")
    
    # Volume distribution chart
    top_customers = customer_volume.head(20)
    volume_chart = alt.Chart(top_customers).mark_bar(color='steelblue', stroke='darkblue', strokeWidth=1).encode(
        x=alt.X("total_volume:Q", title="Total Volume ($)", axis=alt.Axis(format="$,.0f")),
        y=alt.Y("customer_name:N", sort="-x", title="Customer"),
        tooltip=["customer_name", "total_volume:Q", "transaction_count:Q", "volume_percentage:Q"]
    ).properties(
        width=700, height=500,
        title="Top 20 Customers by Transaction Volume"
    )
    
    st.altair_chart(volume_chart, use_container_width=True)

# VISUAL 3: Cohort Performance Analysis
st.subheader("3. Cohort Performance Analysis")
if df.empty:
    st.warning("No transaction data available")
else:
    st.markdown("Cohort Performance: Compares customer acquisition cohorts against benchmark metrics to identify trends")
    
    # Calculate customer lifecycle metrics
    lifecycle_df = df.groupby("customer_name").agg({
        "txn_date": ["min", "max", "count"],
        "total_amount": ["sum", "mean"]
    }).reset_index()
    
    lifecycle_df.columns = ["customer_name", "first_transaction", "last_transaction", "total_transactions", "total_value", "avg_transaction"]
    lifecycle_df["customer_lifespan"] = (lifecycle_df["last_transaction"] - lifecycle_df["first_transaction"]).dt.days
    lifecycle_df["customer_age_months"] = (pd.Timestamp.now() - lifecycle_df["first_transaction"]).dt.days / 30
    
    # Create cohort analysis with quarterly cohorts for better grouping
    lifecycle_df["cohort_quarter"] = lifecycle_df["first_transaction"].dt.to_period("Q")
    lifecycle_df["value_per_month"] = np.where(
        lifecycle_df["customer_age_months"] > 0,
        lifecycle_df["total_value"] / lifecycle_df["customer_age_months"],
        lifecycle_df["total_value"]
    )
    
    # Cohort performance metrics
    cohort_performance = lifecycle_df.groupby("cohort_quarter").agg({
        "customer_name": "count",
        "total_value": ["mean", "median", "sum"],
        "value_per_month": ["mean", "median"],
        "customer_age_months": "mean",
        "total_transactions": "mean"
    }).reset_index()
    
    # Flatten column names
    cohort_performance.columns = [
        "cohort_quarter", "customer_count", "avg_total_value", "median_total_value", 
        "cohort_total_value", "avg_value_per_month", "median_value_per_month", 
        "avg_age_months", "avg_transactions"
    ]
    
    # Calculate benchmarks (overall averages)
    benchmark_avg_value = lifecycle_df["total_value"].mean()
    benchmark_value_per_month = lifecycle_df["value_per_month"].mean()
    benchmark_transactions = lifecycle_df["total_transactions"].mean()
    
    # Performance vs benchmark
    cohort_performance["value_vs_benchmark"] = (cohort_performance["avg_total_value"] / benchmark_avg_value - 1) * 100
    cohort_performance["efficiency_vs_benchmark"] = (cohort_performance["avg_value_per_month"] / benchmark_value_per_month - 1) * 100
    cohort_performance["activity_vs_benchmark"] = (cohort_performance["avg_transactions"] / benchmark_transactions - 1) * 100
    
    # Display benchmark metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Benchmark Avg Customer Value", f"${benchmark_avg_value:,.0f}",
                 help="Average total value per customer across all cohorts")
    with col2:
        st.metric("Benchmark Value per Month", f"${benchmark_value_per_month:,.0f}",
                 help="Average monthly value generation per customer")
    with col3:
        st.metric("Benchmark Transactions", f"{benchmark_transactions:.1f}",
                 help="Average number of transactions per customer")
    
    # Cohort performance chart - Value vs Benchmark
    cohort_performance["cohort_str"] = cohort_performance["cohort_quarter"].astype(str)
    
    performance_chart = alt.Chart(cohort_performance).mark_circle(size=150, stroke='black', strokeWidth=1).encode(
        x=alt.X("cohort_str:N", title="Acquisition Cohort (Quarter)", axis=alt.Axis(labelAngle=-45)),
        y=alt.Y("value_vs_benchmark:Q", title="Performance vs Benchmark (%)", 
                axis=alt.Axis(format=".1f"), scale=alt.Scale(zero=False)),
        color=alt.Color("value_vs_benchmark:Q", 
                       scale=alt.Scale(scheme="redyellowgreen", domain=[-50, 50]), 
                       title="Performance %"),
        size=alt.Size("customer_count:Q", title="Customers in Cohort", scale=alt.Scale(range=[100, 500])),
        tooltip=[
            "cohort_str:N", 
            "customer_count:Q", 
            "value_vs_benchmark:Q", 
            "avg_total_value:Q",
            "avg_value_per_month:Q"
        ]
    ).properties(
        width=700, height=400,
        title="Cohort Performance: Customer Value vs Overall Benchmark"
    )
    
    st.altair_chart(performance_chart, use_container_width=True)
    
    # Detailed cohort table
    st.subheader("Detailed Cohort Metrics")
    display_cohorts = cohort_performance.copy()
    display_cohorts["cohort_quarter"] = display_cohorts["cohort_quarter"].astype(str)
    display_cohorts["avg_total_value"] = display_cohorts["avg_total_value"].apply(lambda x: f"${x:,.0f}")
    display_cohorts["avg_value_per_month"] = display_cohorts["avg_value_per_month"].apply(lambda x: f"${x:,.0f}")
    display_cohorts["value_vs_benchmark"] = display_cohorts["value_vs_benchmark"].apply(lambda x: f"{x:+.1f}%")
    display_cohorts["efficiency_vs_benchmark"] = display_cohorts["efficiency_vs_benchmark"].apply(lambda x: f"{x:+.1f}%")
    
    st.dataframe(
        display_cohorts[[
            "cohort_quarter", "customer_count", "avg_total_value", 
            "avg_value_per_month", "value_vs_benchmark", "efficiency_vs_benchmark"
        ]].rename(columns={
            "cohort_quarter": "Cohort",
            "customer_count": "Customers",
            "avg_total_value": "Avg Total Value",
            "avg_value_per_month": "Avg Monthly Value",
            "value_vs_benchmark": "Value vs Benchmark",
            "efficiency_vs_benchmark": "Efficiency vs Benchmark"
        }),
        use_container_width=True
    )

# -------------------------
# SUMMARY & ALERTS
# -------------------------
st.header("Summary & Alerts")

if df.empty:
    st.error("No transaction data available for analysis")
else:
    # Risk summary
    if 'payment_behavior' in locals():
        high_risk_pct = (payment_behavior["risk_category"] == "High Risk").mean() * 100
        st.warning(f"WARNING: {high_risk_pct:.1f}% of customers are classified as High Risk")
    
    # Cash flow summary
    if 'daily_cash_flow' in locals():
        recent_trend = daily_cash_flow["net_cash_flow"].tail(7).mean()
        if recent_trend > 0:
            st.success(f"POSITIVE: Positive 7-day average cash flow: ${recent_trend:,.0f}")
        else:
            st.error(f"NEGATIVE: Negative 7-day average cash flow: ${recent_trend:,.0f}")

st.markdown("---")
st.markdown("*Dashboard last updated: " + pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S") + "*")
st.markdown("*GL Data loaded and available for future analysis modules*")
