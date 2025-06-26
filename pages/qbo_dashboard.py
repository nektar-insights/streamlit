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
    numeric_cols = ["total_amount", "balance", "amount", "debit", "credit"]
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
        dataframe["quarter"] = dataframe["txn_date"].dt.quarter
        dataframe["year"] = dataframe["txn_date"].dt.year
    
    return dataframe

# Apply enhanced preprocessing
df = preprocess_financial_data(df)
gl_df = preprocess_financial_data(gl_df)

st.title("MCA Dashboard")
st.markdown("---")

# -------------------------
# DATA SOURCE OVERVIEW
# -------------------------
st.header("Data Overview")

col1, col2 = st.columns(2)
with col1:
    st.subheader("Transaction Data")
    if not df.empty:
        st.metric("Records", f"{len(df):,}")
        st.metric("Date Range", f"{df['txn_date'].min().strftime('%Y-%m-%d')} to {df['txn_date'].max().strftime('%Y-%m-%d')}")
        st.metric("Unique Customers", f"{df['customer_name'].nunique():,}")
    else:
        st.warning("No transaction data available")

with col2:
    st.subheader("General Ledger Data")
    if not gl_df.empty:
        st.metric("Records", f"{len(gl_df):,}")
        if 'txn_date' in gl_df.columns:
            st.metric("Date Range", f"{gl_df['txn_date'].min().strftime('%Y-%m-%d')} to {gl_df['txn_date'].max().strftime('%Y-%m-%d')}")
        if 'account_name' in gl_df.columns:
            st.metric("Unique Accounts", f"{gl_df['account_name'].nunique():,}")
    else:
        st.warning("No general ledger data available")

# -------------------------
# EXECUTIVE SUMMARY
# -------------------------
st.header("Executive Summary")

if not df.empty or not gl_df.empty:
    # Use the most comprehensive dataset available
    primary_data = gl_df if not gl_df.empty else df
    
    col1, col2, col3, col4 = st.columns(4)
    
    if not df.empty:
        with col1:
            total_customers = df["customer_name"].nunique()
            st.metric("Total Customers", f"{total_customers:,}")
        with col2:
            total_transactions = len(df)
            st.metric("Customer Transactions", f"{total_transactions:,}")
    
    if not gl_df.empty:
        with col3:
            total_gl_entries = len(gl_df)
            st.metric("GL Entries", f"{total_gl_entries:,}")
        with col4:
            if 'amount' in gl_df.columns:
                total_gl_volume = gl_df['amount'].abs().sum()
                st.metric("Total GL Volume", f"${total_gl_volume:,.0f}")

st.markdown("---")

# -------------------------
# 1. ENHANCED RISK ANALYSIS USING BOTH DATASETS
# -------------------------
st.header("Risk Analysis")

# CUSTOMER RISK FROM TRANSACTION DATA
if not df.empty:
    st.subheader("Customer Payment Behavior Risk")
    st.info("Data Source: Transaction Data for customer-specific risk assessment")
    
    risk_df = df[df["transaction_type"].isin(["Invoice", "Payment"])].copy()
    risk_df = risk_df[~risk_df["customer_name"].isin(["CSL", "VEEM"])]
    
    if not risk_df.empty:
        risk_df["total_amount"] = risk_df["total_amount"].abs()
        
        # Payment behavior analysis
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
            
            # Enhanced risk scoring with GL insights
            payment_behavior["risk_score"] = (
                (1 - payment_behavior["payment_ratio"]) * 0.4 +
                (payment_behavior["outstanding_balance"] / payment_behavior["Invoice"].max()) * 0.3 +
                np.where(payment_behavior["payment_ratio"] < 0.5, 0.3, 0)
            )
            
            payment_behavior["risk_category"] = pd.cut(
                payment_behavior["risk_score"],
                bins=[0, 0.2, 0.5, 1.0],
                labels=["Low Risk", "Medium Risk", "High Risk"]
            )
            
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

# FINANCIAL HEALTH FROM GENERAL LEDGER
if not gl_df.empty:
    st.subheader("Overall Financial Health Analysis")
    st.info("Data Source: General Ledger for comprehensive financial health assessment")
    
    # Account-based analysis
    if 'account_name' in gl_df.columns and 'amount' in gl_df.columns:
        # Categorize accounts for financial health
        gl_df['account_category'] = 'Other'
        
        # Revenue accounts
        revenue_keywords = ['Revenue', 'Income', 'Sales', 'Service']
        revenue_mask = gl_df['account_name'].str.contains('|'.join(revenue_keywords), case=False, na=False)
        gl_df.loc[revenue_mask, 'account_category'] = 'Revenue'
        
        # Expense accounts
        expense_keywords = ['Expense', 'Cost', 'Payroll', 'Rent', 'Utilities', 'Marketing']
        expense_mask = gl_df['account_name'].str.contains('|'.join(expense_keywords), case=False, na=False)
        gl_df.loc[expense_mask, 'account_category'] = 'Expense'
        
        # Asset accounts
        asset_keywords = ['Cash', 'Bank', 'Receivable', 'Asset', 'Equipment']
        asset_mask = gl_df['account_name'].str.contains('|'.join(asset_keywords), case=False, na=False)
        gl_df.loc[asset_mask, 'account_category'] = 'Asset'
        
        # Liability accounts
        liability_keywords = ['Payable', 'Liability', 'Loan', 'Credit', 'Debt']
        liability_mask = gl_df['account_name'].str.contains('|'.join(liability_keywords), case=False, na=False)
        gl_df.loc[liability_mask, 'account_category'] = 'Liability'
        
        # Financial health metrics
        monthly_summary = gl_df.groupby(['year_month', 'account_category'])['amount'].sum().reset_index()
        monthly_pivot = monthly_summary.pivot(index='year_month', columns='account_category', values='amount').fillna(0)
        
        if 'Revenue' in monthly_pivot.columns and 'Expense' in monthly_pivot.columns:
            monthly_pivot['Net_Income'] = monthly_pivot['Revenue'] - monthly_pivot['Expense'].abs()
            monthly_pivot['Profit_Margin'] = np.where(
                monthly_pivot['Revenue'] != 0,
                monthly_pivot['Net_Income'] / monthly_pivot['Revenue'],
                0
            )
            
            # Current financial health
            latest_month = monthly_pivot.index.max()
            current_metrics = monthly_pivot.loc[latest_month]
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Current Month Revenue", f"${current_metrics['Revenue']:,.0f}",
                         help="Total revenue for the most recent month in GL")
            with col2:
                st.metric("Current Month Expenses", f"${current_metrics['Expense']:,.0f}",
                         help="Total expenses for the most recent month in GL")
            with col3:
                st.metric("Current Month Net Income", f"${current_metrics['Net_Income']:,.0f}",
                         help="Revenue minus expenses for current month")
            with col4:
                st.metric("Current Profit Margin", f"{current_metrics['Profit_Margin']:.1%}",
                         help="Net income as percentage of revenue")
            
            # Financial health trend
            monthly_pivot_reset = monthly_pivot.reset_index()
            monthly_pivot_reset['period'] = monthly_pivot_reset['year_month'].astype(str)
            
            # Revenue vs Expense trend
            revenue_expense_chart = alt.Chart(monthly_pivot_reset).transform_fold(
                ['Revenue', 'Expense'],
                as_=['Metric', 'Amount']
            ).mark_line(point=True, strokeWidth=3).encode(
                x=alt.X('period:N', title='Month'),
                y=alt.Y('Amount:Q', title='Amount ($)', axis=alt.Axis(format='$,.0f')),
                color=alt.Color('Metric:N', scale=alt.Scale(range=['lightgreen', 'lightcoral'])),
                tooltip=['period:N', 'Metric:N', 'Amount:Q']
            ).properties(
                width=700, height=300,
                title='Monthly Revenue vs Expenses Trend'
            )
            
            st.altair_chart(revenue_expense_chart, use_container_width=True)

# -------------------------
# 2. COMPREHENSIVE CASH FLOW ANALYSIS
# -------------------------
st.header("Cash Flow Analysis")

# OPERATIONAL CASH FLOW FROM GENERAL LEDGER
if not gl_df.empty and 'txn_date' in gl_df.columns:
    st.subheader("Operational Cash Flow (General Ledger)")
    st.info("Data Source: General Ledger for comprehensive cash flow analysis including all accounts")
    
    # Cash flow analysis from GL
    gl_cash_flow = gl_df.copy()
    
    # Categorize cash flow impact
    gl_cash_flow['cash_flow_impact'] = 0
    
    # Cash inflows (positive)
    if 'account_category' in gl_cash_flow.columns:
        inflow_mask = (gl_cash_flow['account_category'] == 'Revenue') | \
                     (gl_cash_flow['account_name'].str.contains('Cash|Bank', case=False, na=False) & (gl_cash_flow['amount'] > 0))
        gl_cash_flow.loc[inflow_mask, 'cash_flow_impact'] = gl_cash_flow.loc[inflow_mask, 'amount'].abs()
        
        # Cash outflows (negative)
        outflow_mask = (gl_cash_flow['account_category'] == 'Expense') | \
                      (gl_cash_flow['account_name'].str.contains('Cash|Bank', case=False, na=False) & (gl_cash_flow['amount'] < 0))
        gl_cash_flow.loc[outflow_mask, 'cash_flow_impact'] = -gl_cash_flow.loc[outflow_mask, 'amount'].abs()
    
    # Daily cash flow from GL
    daily_gl_cash_flow = gl_cash_flow.groupby(gl_cash_flow['txn_date'].dt.date).agg({
        'cash_flow_impact': 'sum',
        'amount': 'count'
    }).reset_index()
    daily_gl_cash_flow.columns = ['date', 'net_cash_flow', 'entry_count']
    daily_gl_cash_flow['cumulative_cash_flow'] = daily_gl_cash_flow['net_cash_flow'].cumsum()
    
    # Cash flow visualizations
    daily_gl_cash_flow['date_dt'] = pd.to_datetime(daily_gl_cash_flow['date'])
    
    # Daily cash flow chart
    daily_cash_chart = alt.Chart(daily_gl_cash_flow).mark_line(color='steelblue', strokeWidth=2).encode(
        x=alt.X('date_dt:T', title='Date'),
        y=alt.Y('net_cash_flow:Q', title='Daily Net Cash Flow ($)', axis=alt.Axis(format='$,.0f')),
        tooltip=['date:T', 'net_cash_flow:Q', 'entry_count:Q']
    ).properties(
        width=700, height=300,
        title='Daily Net Cash Flow from General Ledger'
    )
    
    st.altair_chart(daily_cash_chart, use_container_width=True)
    
    # Cash flow statistics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        avg_daily_flow = daily_gl_cash_flow['net_cash_flow'].mean()
        st.metric("Avg Daily Cash Flow", f"${avg_daily_flow:,.0f}",
                 help="Average daily net cash flow from all GL entries")
    with col2:
        cash_flow_volatility = daily_gl_cash_flow['net_cash_flow'].std()
        st.metric("Cash Flow Volatility", f"${cash_flow_volatility:,.0f}",
                 help="Standard deviation of daily cash flows from GL")
    with col3:
        positive_days = (daily_gl_cash_flow['net_cash_flow'] > 0).sum()
        total_days = len(daily_gl_cash_flow)
        st.metric("Positive Cash Flow Days", f"{positive_days}/{total_days}",
                 help="Number of days with positive net cash flow")
    with col4:
        current_position = daily_gl_cash_flow['cumulative_cash_flow'].iloc[-1]
        st.metric("Cumulative Cash Position", f"${current_position:,.0f}",
                 help="Total cumulative cash flow from GL data")

# CUSTOMER CASH FLOW FROM TRANSACTION DATA
if not df.empty:
    st.subheader("Customer Transaction Cash Flow")
    st.info("Data Source: Transaction Data for customer-specific cash flow patterns")
    
    # Customer cash flow analysis
    customer_cash_flow = df.copy()
    customer_cash_flow['cash_impact'] = np.where(
        customer_cash_flow['transaction_type'].isin(['Payment', 'Deposit']),
        customer_cash_flow['total_amount'].abs(),
        np.where(
            customer_cash_flow['transaction_type'].isin(['Invoice', 'Bill']),
            -customer_cash_flow['total_amount'].abs(),
            0
        )
    )
    
    # Monthly customer cash flow
    monthly_customer_flow = customer_cash_flow.groupby(
        customer_cash_flow['txn_date'].dt.to_period('M')
    )['cash_impact'].sum().reset_index()
    monthly_customer_flow['period'] = monthly_customer_flow['txn_date'].astype(str)
    
    customer_flow_chart = alt.Chart(monthly_customer_flow).mark_bar(color='lightblue', stroke='darkblue', strokeWidth=1).encode(
        x=alt.X('period:N', title='Month'),
        y=alt.Y('cash_impact:Q', title='Monthly Customer Cash Flow ($)', axis=alt.Axis(format='$,.0f')),
        tooltip=['period:N', 'cash_impact:Q']
    ).properties(
        width=700, height=300,
        title='Monthly Customer Transaction Cash Flow'
    )
    
    st.altair_chart(customer_flow_chart, use_container_width=True)

# -------------------------
# 3. ADVANCED GL-BASED INSIGHTS
# -------------------------
st.header("Advanced General Ledger Insights")

if not gl_df.empty:
    # ACCOUNT ANALYSIS
    st.subheader("1. Account Activity Analysis")
    st.markdown("Purpose: Identifies most active accounts and unusual account activity patterns")
    
    if 'account_name' in gl_df.columns:
        account_activity = gl_df.groupby('account_name').agg({
            'amount': ['count', 'sum', 'mean', 'std'],
            'txn_date': ['min', 'max']
        }).reset_index()
        
        account_activity.columns = ['account_name', 'transaction_count', 'total_amount', 'avg_amount', 'amount_std', 'first_txn', 'last_txn']
        account_activity['activity_score'] = account_activity['transaction_count'] * account_activity['avg_amount'].abs()
        account_activity = account_activity.sort_values('activity_score', ascending=False).head(20)
        
        activity_chart = alt.Chart(account_activity).mark_circle(stroke='black', strokeWidth=1).encode(
            x=alt.X('transaction_count:Q', title='Transaction Count'),
            y=alt.Y('total_amount:Q', title='Total Amount ($)', axis=alt.Axis(format='$,.0f')),
            size=alt.Size('activity_score:Q', title='Activity Score', scale=alt.Scale(range=[50, 400])),
            color=alt.Color('avg_amount:Q', scale=alt.Scale(scheme='lightgreyteal'), title='Avg Amount'),
            tooltip=['account_name', 'transaction_count:Q', 'total_amount:Q', 'avg_amount:Q']
        ).properties(
            width=700, height=400,
            title='Account Activity: Transaction Count vs Total Amount (Top 20 Accounts)'
        )
        
        st.altair_chart(activity_chart, use_container_width=True)
    
    # SEASONAL PATTERNS
    st.subheader("2. Seasonal Business Patterns")
    st.markdown("Purpose: Identifies seasonal trends in business activity for better forecasting")
    
    if 'txn_date' in gl_df.columns:
        seasonal_data = gl_df.copy()
        seasonal_data['month'] = seasonal_data['txn_date'].dt.month
        seasonal_data['quarter'] = seasonal_data['txn_date'].dt.quarter
        
        monthly_activity = seasonal_data.groupby('month').agg({
            'amount': ['count', 'sum'],
            'txn_date': 'count'
        }).reset_index()
        monthly_activity.columns = ['month', 'entry_count', 'total_amount', 'transaction_days']
        monthly_activity['month_name'] = pd.to_datetime(monthly_activity['month'], format='%m').dt.strftime('%B')
        
        seasonal_chart = alt.Chart(monthly_activity).mark_bar(color='lightcoral', stroke='darkred', strokeWidth=1).encode(
            x=alt.X('month_name:N', title='Month', sort=['January', 'February', 'March', 'April', 'May', 'June', 
                                                        'July', 'August', 'September', 'October', 'November', 'December']),
            y=alt.Y('total_amount:Q', title='Total Amount ($)', axis=alt.Axis(format='$,.0f')),
            tooltip=['month_name', 'total_amount:Q', 'entry_count:Q']
        ).properties(
            width=700, height=300,
            title='Seasonal Activity Pattern - Total Amount by Month'
        )
        
        st.altair_chart(seasonal_chart, use_container_width=True)
    
    # TRANSACTION TIMING ANALYSIS
    st.subheader("3. Business Transaction Timing")
    st.markdown("Purpose: Shows when business transactions typically occur - useful for cash flow timing")
    
    if 'txn_date' in gl_df.columns:
        timing_data = gl_df.copy()
        timing_data['hour'] = timing_data['txn_date'].dt.hour
        timing_data['day_of_week'] = timing_data['txn_date'].dt.day_name()
        
        timing_heatmap = timing_data.groupby(['day_of_week', 'hour']).size().reset_index(name='entry_count')
        
        heatmap = alt.Chart(timing_heatmap).mark_rect(stroke='white', strokeWidth=2).encode(
            x=alt.X('hour:O', title='Hour of Day'),
            y=alt.Y('day_of_week:O', title='Day of Week', sort=['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']),
            color=alt.Color('entry_count:Q', scale=alt.Scale(scheme='lightgreyteal'), title='Entry Count'),
            tooltip=['day_of_week', 'hour:O', 'entry_count']
        ).properties(
            width=700, height=300,
            title='Transaction Timing Heatmap - When Do Business Transactions Occur?'
        )
        
        st.altair_chart(heatmap, use_container_width=True)

# -------------------------
# 4. INTEGRATED FORECASTING
# -------------------------
st.header("Integrated Forecasting")

# Use GL data for more comprehensive forecasting
if not gl_df.empty and 'daily_gl_cash_flow' in locals() and len(daily_gl_cash_flow) >= 30:
    st.info("Forecasting Method: Combined GL and transaction data with trend analysis and seasonal adjustments")
    
    forecast_data = daily_gl_cash_flow.copy()
    forecast_data['date'] = pd.to_datetime(forecast_data['date'])
    forecast_data = forecast_data.sort_values('date')
    
    # Enhanced forecasting with trend analysis
    window_size = min(30, len(forecast_data) // 2)
    forecast_data['ma_30'] = forecast_data['net_cash_flow'].rolling(window=window_size).mean()
    forecast_data['trend'] = forecast_data['net_cash_flow'].rolling(window=7).mean()
    
    # Seasonal patterns
    forecast_data['day_of_week'] = forecast_data['date'].dt.day_name()
    forecast_data['week_of_month'] = forecast_data['date'].dt.day // 7 + 1
    
    weekly_pattern = forecast_data.groupby('day_of_week')['net_cash_flow'].mean()
    
    # Generate 60-day forecast
    last_date = forecast_data['date'].max()
    forecast_dates = pd.date_range(start=last_date + timedelta(days=1), periods=60, freq='D')
    
    # Enhanced forecast calculation
    recent_trend = forecast_data['trend'].tail(7).mean()
    base_forecast = forecast_data['net_cash_flow'].tail(30).mean()
    
    forecast_values = []
    for i, date in enumerate(forecast_dates):
        day_name = date.day_name()
        seasonal_adjustment = weekly_pattern.get(day_name, 0) - weekly_pattern.mean()
        trend_adjustment = recent_trend * 0.1  # Small trend component
        daily_forecast = base_forecast + seasonal_adjustment + trend_adjustment
        forecast_values.append(daily_forecast)
    
    forecast_df = pd.DataFrame({
        'date': forecast_dates,
        'forecasted_cash_flow': forecast_values,
        'type': 'Forecast'
    })
    
    # Combine for visualization
    historical_viz = forecast_data[['date', 'net_cash_flow']].copy()
    historical_viz['type'] = 'Historical'
    historical_viz = historical_viz.rename(columns={'net_cash_flow': 'forecasted_cash_flow'})
    
    combined_forecast = pd.concat([
        historical_viz.tail(90),
        forecast_df
    ], ignore_index=True)
    
    forecast_chart = alt.Chart(combined_forecast).mark_line().encode(
        x=alt.X('date:T', title='Date'),
        y=alt.Y('forecasted_cash_flow:Q', title='Cash Flow ($)', axis=alt.Axis(format='$,.0f')),
        color=alt.Color('type:N', title='Data Type', scale=alt.Scale(range=['steelblue', 'orange'])),
        strokeDash=alt.StrokeDash('type:N', scale=alt.Scale(range=[[1,0], [5,5]])),
        tooltip=['date:T', 'forecasted_cash_flow:Q', 'type:N']
    ).properties(
        width=700, height=400,
        title='60-Day Cash Flow Forecast (GL-Based)'
    )
    
    st.altair_chart(forecast_chart, use_container_width=True)
    
    # Forecast metrics
    forecast_30_day = sum(forecast_values[:30])
    forecast_60_day = sum(forecast_values)
    forecast_avg = np.mean(forecast_values)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("30-Day Forecast", f"${forecast_30_day:,.0f}",
                 help="Projected cash flow for next 30 days based on GL trends")
    with col2:
        st.metric("60-Day Forecast", f"${forecast_60_day:,.0f}",
                 help="Projected cash flow for next 60 days")
    with col3:
        confidence_interval = np.std(forecast_data['net_cash_flow'].tail(30)) * 1.96
        st.metric("Forecast Confidence (±)", f"${confidence_interval:,.0f}",
                 help="95% confidence interval for forecast accuracy")

# -------------------------
# 5. SUMMARY & ALERTS
# -------------------------
st.header("Summary & Alerts")

# Generate intelligent alerts based on both datasets
alerts = []

if not df.empty and 'payment_behavior' in locals():
    high_risk_pct = (payment_behavior["risk_category"] == "High Risk").mean() * 100
    if high_risk_pct > 20:
        alerts.append(f"HIGH ALERT: {high_risk_pct:.1f}% of customers are High Risk (>20% threshold)")
    elif high_risk_pct > 10:
        alerts.append(f"MEDIUM ALERT: {high_risk_pct:.1f}% of customers are High Risk")

if not gl_df.empty and 'daily_gl_cash_flow' in locals():
    recent_trend = daily_gl_cash_flow['net_cash_flow'].tail(7).mean()
    if recent_trend < -10000:
        alerts.append(f"CASH FLOW ALERT: Negative 7-day average: ${recent_trend:,.0f}")
    elif recent_trend > 0:
        alerts.append(f"POSITIVE: 7-day average cash flow: ${recent_trend:,.0f}")

if not gl_df.empty and 'monthly_pivot' in locals():
    if 'Profit_Margin' in monthly_pivot.columns:
        latest_margin = monthly_pivot['Profit_Margin'].iloc[-1]
        if latest_margin < 0.05:
            alerts.append(f"PROFITABILITY ALERT: Current profit margin only {latest_margin:.1%}")

# Display alerts
for alert in alerts:
    if "HIGH ALERT" in alert or "CASH FLOW ALERT" in alert:
        st.error(alert)
    elif "MEDIUM ALERT" in alert or "PROFITABILITY ALERT" in alert:
        st.warning(alert)
    else:
        st.success(alert)

if not alerts:
    st.info("No significant alerts at this time")

st.markdown("---")
st.markdown("Dashboard last updated: " + pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"))
