# pages/qbo_dashboard.py

from utils.imports import *
from utils.config import (
    inject_global_styles,
    inject_logo,
    get_supabase_client,
    PRIMARY_COLOR,
    COLOR_PALETTE,
    PLATFORM_FEE_RATE,
)

from utils.qbo_data_loader import load_qbo_data, load_deals, load_mca_deals
from utils.loan_tape_loader import (
    load_loan_tape_data,
    load_unified_loan_customer_data,
    get_customer_payment_summary,
    get_data_diagnostics,
)
import numpy as np
from datetime import datetime, timedelta
import warnings

warnings.filterwarnings('ignore')

# -------------------------
# Page config & branding
# -------------------------
st.set_page_config(
    page_title="CSL Capital | QBO Dashboard",
    layout="wide",
)
inject_global_styles()
inject_logo()

# -------------------------
# Setup: Supabase Connection & Load Data
# -------------------------
supabase = get_supabase_client()

# Load data using centralized functions
df, gl_df = load_qbo_data()
deals_df = load_deals()
mca_deals_df = load_mca_deals()

# Load loan tape data - both individual and unified
loan_tape_df = load_loan_tape_data()
unified_data_df = load_unified_loan_customer_data()

# Load diagnostics data
diagnostics = get_data_diagnostics()

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

st.title("QBO Dashboard")
st.markdown("---")

# -------------------------
# UNIFIED LOAN & CUSTOMER ANALYSIS
# -------------------------
st.header("Unified Loan & Customer Performance")

# Add diagnostic section
with st.expander("Data Diagnostics - Click to investigate the join"):
    st.subheader("Data Join Analysis")
    
    if diagnostics:
        # Overall Data Summary - 2x4 layout
        st.write("**Overall Data Summary:**")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Total QBO Payments", f"${diagnostics.get('total_qbo_amount', 0):,.0f}")
            st.metric("Total Deals", f"{diagnostics.get('raw_deals_count', 0):,}")
        with col2:
            st.metric("QBO Transactions", f"{diagnostics.get('raw_qbo_count', 0):,}")
            st.metric("Closed Won Deals", f"{diagnostics.get('closed_won_deals', 0):,}")
        
        # Transaction Type Analysis
        st.write("**Transaction Type Breakdown:**")
        if "transaction_types" in diagnostics and diagnostics["transaction_types"]:
            try:
                txn_data = []
                for txn_type, data in diagnostics["transaction_types"].items():
                    if isinstance(data, dict):
                        total_amount = data.get("total_amount", 0)
                        count = data.get("count", data.get("transaction_id", 0))
                    else:
                        total_amount = 0
                        count = 0
                    
                    txn_data.append({
                        "Transaction Type": txn_type,
                        "Total Amount": total_amount,
                        "Count": count
                    })
                
                txn_df = pd.DataFrame(txn_data)
                st.dataframe(
                    txn_df,
                    use_container_width=True,
                    column_config={
                        "Total Amount": st.column_config.NumberColumn("Total Amount", format="$%.0f"),
                        "Count": st.column_config.NumberColumn("Count")
                    }
                )
            except Exception as e:
                st.error(f"Error processing transaction types: {e}")
        else:
            st.write("No transaction type data available")
        
        # Data limitation alerts
        if diagnostics.get("raw_qbo_count", 0) == 1000:
            st.error("**DATA LIMITATION ALERT**: Only showing first 1,000 QBO transactions due to Supabase query limits.")
            st.info("**Solution**: Consider upgrading Supabase plan or implementing proper pagination.")
        
        # Payment Type Filtering Impact - 2x2 layout
        st.write("**Impact of Payment Type Filtering:**")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Before Filtering", f"${diagnostics.get('total_qbo_amount', 0):,.0f}")
        with col2:
            filtering_loss = diagnostics.get('total_qbo_amount', 0) - diagnostics.get('payment_types_amount', 0)
            st.metric("After Filtering (Payment/Deposit/Receipt)", 
                     f"${diagnostics.get('payment_types_amount', 0):,.0f}",
                     delta=f"-${filtering_loss:,.0f}")
        
        if filtering_loss > 0:
            percentage_lost = (filtering_loss / diagnostics.get('total_qbo_amount', 1)) * 100
            st.warning(f"**Missing ${filtering_loss:,.0f}** ({percentage_lost:.1f}%) due to filtering out non-payment transactions")
        
        # Loan ID Attribution Analysis - 2x2 layout
        st.write("**Loan ID Attribution:**")
        col1, col2 = st.columns(2)
        with col1:
            with_loan_id = diagnostics.get('qbo_with_loan_id', {})
            st.metric("Payments WITH Loan ID", 
                     f"${with_loan_id.get('amount', 0):,.0f}",
                     help=f"{with_loan_id.get('count', 0)} transactions")
        with col2:
            without_loan_id = diagnostics.get('qbo_without_loan_id', {})
            st.metric("Payments WITHOUT Loan ID", 
                     f"${without_loan_id.get('amount', 0):,.0f}",
                     help=f"{without_loan_id.get('count', 0)} transactions")
        
        # Loan ID Matching Analysis - 2x4 layout
        st.write("**Loan ID Matching:**")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Unique Deal Loan IDs", diagnostics.get('unique_deal_loan_ids', 0))
            st.metric("Overlapping Loan IDs", diagnostics.get('overlapping_loan_ids', 0))
        with col2:
            st.metric("Unique QBO Loan IDs", diagnostics.get('unique_qbo_loan_ids', 0))
            # Calculate overlap rate
            overlap_rate = 0
            if diagnostics.get('unique_deal_loan_ids', 0) > 0:
                overlap_rate = (diagnostics.get('overlapping_loan_ids', 0) / diagnostics.get('unique_deal_loan_ids', 0)) * 100
            st.metric("Overlap Rate", f"{overlap_rate:.1f}%")
        
        if overlap_rate < 80:
            st.warning(f"Only {overlap_rate:.1f}% of deal loan IDs have matching QBO payments")
        else:
            st.success(f"{overlap_rate:.1f}% of deal loan IDs have matching QBO payments")
        
        # Top Customers Analysis
        st.write("**Top 10 Customers by Payment Amount:**")
        if "top_customers" in diagnostics:
            top_customers_data = [
                {"Customer": customer, "Total Payments": amount}
                for customer, amount in diagnostics["top_customers"].items()
            ]
            top_customers_df = pd.DataFrame(top_customers_data)
            st.dataframe(
                top_customers_df,
                use_container_width=True,
                column_config={
                    "Total Payments": st.column_config.NumberColumn("Total Payments", format="$%.0f")
                }
            )
    else:
        st.error("Unable to load diagnostic data")

# Add tabs for different views
tab1, tab2, tab3 = st.tabs(["Unified Analysis", "Loan Tape Only", "Customer Summary"])

with tab1:
    st.subheader("Comprehensive Loan & Customer Analysis")
    
    if unified_data_df.empty:
        st.warning("No unified data available. This could be due to:")
        st.markdown("- No closed/won deals in HubSpot")
        st.markdown("- No matching data between deals and QBO payments")
        st.markdown("- Data inconsistency between deal names and customer names")
    else:
        # Calculate unified summary metrics
        total_unified_loans = len(unified_data_df)
        total_participation = unified_data_df.get("Participation Amount", pd.Series([0])).sum()
        total_expected_return = unified_data_df.get("Expected Return", pd.Series([0])).sum()
        total_rtr_amount = unified_data_df.get("RTR Amount", pd.Series([0])).sum()
        avg_rtr_percentage = unified_data_df.get("RTR %", pd.Series([0])).mean()
        loans_with_payments = (unified_data_df.get("RTR Amount", pd.Series([0])) > 0).sum()
        portfolio_rtr = (total_rtr_amount / total_participation) * 100 if total_participation > 0 else 0
        total_unattributed = unified_data_df.get("Unattributed Amount", pd.Series([0])).sum()
        
        # Display unified summary metrics in 2x4 layout
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Total Loans", f"{total_unified_loans:,}")
            st.metric("Expected Return", f"${total_expected_return:,.0f}")
            st.metric("Avg RTR %", f"{avg_rtr_percentage:.1f}%")
            st.metric("Unattributed Payments", f"${total_unattributed:,.0f}")
        with col2:
            st.metric("Total Participation", f"${total_participation:,.0f}")
            st.metric("Actual RTR", f"${total_rtr_amount:,.0f}")
            st.metric("Loans with Payments", f"{loans_with_payments}/{total_unified_loans}")
            st.metric("Portfolio RTR %", f"{portfolio_rtr:.1f}%")
        
        # Display unified table
        st.subheader("Unified Loan & Customer Performance Table")
        
        st.dataframe(
            unified_data_df,
            use_container_width=True,
            column_config={
                "Loan ID": st.column_config.TextColumn("Loan ID", width="small"),
                "Deal Name": st.column_config.TextColumn("Deal Name", width="medium"),
                "QBO Customer": st.column_config.TextColumn("QBO Customer", width="medium"),
                "Factor Rate": st.column_config.NumberColumn("Factor Rate", width="small", format="%.3f"),
                "Participation Amount": st.column_config.NumberColumn("Participation", width="medium", format="$%.0f"),
                "Expected Return": st.column_config.NumberColumn("Expected Return", width="medium", format="$%.0f"),
                "RTR Amount": st.column_config.NumberColumn("RTR Amount", width="medium", format="$%.0f"),
                "RTR %": st.column_config.NumberColumn("RTR %", width="small", format="%.1f%%"),
                "Total Customer Payments": st.column_config.NumberColumn("Customer Total", width="medium", format="$%.0f"),
                "Attribution %": st.column_config.NumberColumn("Attribution %", width="small", format="%.1f%%"),
                "Unattributed Amount": st.column_config.NumberColumn("Unattributed", width="medium", format="$%.0f"),
                "TIB": st.column_config.NumberColumn("TIB", width="small", format="%.0f"),
                "FICO": st.column_config.NumberColumn("FICO", width="small", format="%.0f"),
                "Days Since Last Payment": st.column_config.NumberColumn("Days Since Last Payment", width="small", format="%.0f"),
                "Loan Payment Count": st.column_config.NumberColumn("Loan Payments", width="small"),
                "Total Customer Payment Count": st.column_config.NumberColumn("Customer Payments", width="small"),
                # “Customer Active Loans” and “Unattributed Count” have been moved out
            }
        )
        
        # Download unified data
        unified_csv = unified_data_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="Download Unified Analysis (CSV)",
            data=unified_csv,
            file_name=f"unified_loan_customer_analysis_{pd.Timestamp.now().strftime('%Y%m%d')}.csv",
            mime="text/csv",
            key="download_unified_analysis"
        )

with tab2:
    st.subheader("Traditional Loan Tape")
    
    if loan_tape_df.empty:
        st.warning("No loan tape data available.")
    else:
        # Calculate loan tape summary metrics
        total_loans = len(loan_tape_df)
        total_participation = loan_tape_df.get("Total Participation", pd.Series([0])).sum()
        total_expected_return = loan_tape_df.get("Total Return", pd.Series([0])).sum()
        total_rtr_amount = loan_tape_df.get("RTR Amount", pd.Series([0])).sum()
        avg_rtr_percentage = loan_tape_df.get("RTR %", pd.Series([0])).mean()
        loans_with_payments = (loan_tape_df.get("RTR Amount", pd.Series([0])) > 0).sum()
        portfolio_rtr = (total_rtr_amount / total_participation) * 100 if total_participation > 0 else 0
        realized_vs_expected = (total_rtr_amount / total_expected_return) * 100 if total_expected_return > 0 else 0
        
        # Display summary metrics in 2x4 layout
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Total Loans", f"{total_loans:,}")
            st.metric("Expected Return", f"${total_expected_return:,.0f}")
            st.metric("Avg RTR %", f"{avg_rtr_percentage:.1f}%")
            st.metric("Realized vs Expected", f"{realized_vs_expected:.1f}%")
        with col2:
            st.metric("Total Participation", f"${total_participation:,.0f}")
            st.metric("Actual RTR", f"${total_rtr_amount:,.0f}")
            st.metric("Loans with Payments", f"{loans_with_payments}/{total_loans}")
            st.metric("Portfolio RTR %", f"{portfolio_rtr:.1f}%")
        
        # Display loan tape table
        st.subheader("Loan Performance Details")
        
        st.dataframe(
            loan_tape_df,
            use_container_width=True,
            column_config={
                "Loan ID": st.column_config.TextColumn("Loan ID", width="small"),
                "Customer": st.column_config.TextColumn("Customer", width="medium"),
                "Factor Rate": st.column_config.NumberColumn("Factor Rate", width="small", format="%.3f"),
                "Total Participation": st.column_config.NumberColumn("Total Participation", width="medium", format="$%.0f"),
                "Total Return": st.column_config.NumberColumn("Total Return", width="medium", format="$%.0f"),
                "RTR Amount": st.column_config.NumberColumn("RTR Amount", width="medium", format="$%.0f"),
                "RTR %": st.column_config.NumberColumn("RTR %", width="small", format="%.1f%%"),
                "Payment Count": st.column_config.NumberColumn("Payment Count", width="small"),
                "TIB": st.column_config.NumberColumn("TIB", width="small", format="%.0f"),
                "FICO": st.column_config.NumberColumn("FICO", width="small", format="%.0f"),
                "Partner Source": st.column_config.TextColumn("Partner Source", width="medium"),
                "Date Created": st.column_config.DateColumn("Date Created", width="medium")
            }
        )
        
        # Download loan tape
        loan_tape_csv = loan_tape_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="Download Loan Tape (CSV)",
            data=loan_tape_csv,
            file_name=f"loan_tape_{pd.Timestamp.now().strftime('%Y%m%d')}.csv",
            mime="text/csv",
            key="download_loan_tape"
        )

with tab3:
    st.subheader("Customer Payment Summary")
    
    # Load customer summary using the updated function that handles all data
    customer_summary_df = get_customer_payment_summary()
    
    if not customer_summary_df.empty:
        st.dataframe(
            customer_summary_df,
            use_container_width=True,
            column_config={
                "Customer": st.column_config.TextColumn("Customer", width="medium"),
                "Total Payments": st.column_config.NumberColumn("Total Payments", width="medium", format="$%.0f"),
                "Payment Count": st.column_config.NumberColumn("Payment Count", width="small"),
                "Unique Loans": st.column_config.NumberColumn("Unique Loans", width="small"),
                "Unattributed Amount": st.column_config.NumberColumn("Unattributed Amount", width="medium", format="$%.0f"),
                "Unattributed Count": st.column_config.NumberColumn("Unattributed Count", width="small")
            }
        )
        
        # Summary of unattributed payments
        total_unattributed = customer_summary_df["Unattributed Amount"].sum()
        customers_with_unattributed = (customer_summary_df["Unattributed Amount"] > 0).sum()
        
        if total_unattributed > 0:
            st.warning(f"${total_unattributed:,.0f} in unattributed payments across {customers_with_unattributed} customers")
        
        # Download customer summary
        customer_csv = customer_summary_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="Download Customer Summary (CSV)",
            data=customer_csv,
            file_name=f"customer_payment_summary_{pd.Timestamp.now().strftime('%Y%m%d')}.csv",
            mime="text/csv",
            key="download_customer_summary"
        )
    else:
        st.info("No customer payment data available")

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
# RISK ANALYSIS SECTION
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
# ENHANCED CASH FLOW ANALYSIS
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
# ADVANCED FORECASTING
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
# ADVANCED ANALYTICS
# -------------------------

# VISUAL 3: Cohort Performance Analysis
st.subheader("Cohort Performance Analysis")
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
    
    # Unified data alerts
    if not unified_data_df.empty:
        total_unattributed = unified_data_df.get("Unattributed Amount", pd.Series([0])).sum()
        if total_unattributed > 0:
            st.warning(f"ATTENTION: ${total_unattributed:,.0f} in unattributed payments require loan ID assignment")
        
        avg_attribution = unified_data_df.get("Attribution %", pd.Series([100])).mean()
        if avg_attribution < 90:
            st.warning(f"ATTRIBUTION: Average attribution rate is {avg_attribution:.1f}% - consider improving loan ID tracking")
    
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
