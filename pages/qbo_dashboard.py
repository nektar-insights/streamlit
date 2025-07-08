# pages/qbo_dashboard.py
"""
Updated QBO dashboard using centralized data loader
"""

from utils.imports import *
from utils.data_loader import load_qbo_data, load_deals, load_combined_mca_deals, get_data_diagnostics, clear_data_cache
from utils.loan_tape_loader import load_loan_tape_data, load_unified_loan_customer_data, get_customer_payment_summary
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# -------------------------
# Load data using centralized functions
# -------------------------
df, gl_df = load_qbo_data()
deals_df = load_deals()
mca_deals_df = load_combined_mca_deals()

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
    
    # Note: Most preprocessing is now handled by centralized data loader
    # Only add QBO-specific transformations here
    
    dataframe = dataframe.copy()
    
    # Create derived columns for analysis (if not already created by centralized loader)
    if "txn_date" in dataframe.columns and "year_month" not in dataframe.columns:
        dataframe["year_month"] = dataframe["txn_date"].dt.to_period("M")
        dataframe["week"] = dataframe["txn_date"].dt.isocalendar().week
        dataframe["day_of_week"] = dataframe["txn_date"].dt.day_name()
        dataframe["days_since_txn"] = (pd.Timestamp.now() - dataframe["txn_date"]).dt.days
    
    return dataframe

# Apply additional preprocessing if needed
df = preprocess_financial_data(df)
gl_df = preprocess_financial_data(gl_df)

st.title("QBO Dashboard")
st.markdown("---")

# -------------------------
# UNIFIED LOAN & CUSTOMER ANALYSIS
# -------------------------
st.header("🎯 Unified Loan & Customer Performance")

# Add diagnostic section
with st.expander("🔍 Data Diagnostics - Click to investigate the join"):
    st.subheader("Data Join Analysis")
    
    if diagnostics and "error" not in diagnostics:
        # Overall Data Summary
        st.write("**📊 Overall Data Summary:**")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total QBO Payments", f"${diagnostics.get('total_qbo_amount', 0):,.0f}")
        with col2:
            st.metric("QBO Transactions", f"{diagnostics.get('raw_qbo_count', 0):,}")
        with col3:
            st.metric("Total Deals", f"{diagnostics.get('raw_deals_count', 0):,}")
        with col4:
            st.metric("Closed Won Deals", f"{diagnostics.get('closed_won_deals', 0):,}")
        
        # Transaction Type Analysis
        st.write("**💳 Transaction Type Breakdown:**")
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
        
        # Data quality alerts
        if diagnostics.get("raw_qbo_count", 0) == 1000:
            st.error("⚠️ **DATA LIMITATION ALERT**: Only showing first 1,000 QBO transactions due to Supabase query limits.")
            st.info("💡 **Solution**: The centralized data loader now uses pagination to fetch all records.")
        
        # Loan ID Attribution Analysis
        st.write("**🔗 Loan ID Attribution:**")
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
        
        # Top Customers Analysis
        st.write("**🔝 Top 10 Customers by Payment Amount:**")
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
tab1, tab2, tab3 = st.tabs(["📊 Unified Analysis", "🏦 Loan Tape Only", "👥 Customer Summary"])

with tab1:
    st.subheader("Comprehensive Loan & Customer Analysis")
    
    if unified_data_df.empty:
        st.warning("No unified data available. This could be due to:")
        st.markdown("- No closed/won deals in HubSpot")
        st.markdown("- No matching data between deals and QBO payments")
        st.markdown("- Data inconsistency between deal names and customer names")
    else:
        # Unified summary metrics
        total_unified_loans = len(unified_data_df)
        total_participation = unified_data_df.get("Participation Amount", pd.Series([0])).sum()
        total_expected_return = unified_data_df.get("Expected Return", pd.Series([0])).sum()
        total_rtr_amount = unified_data_df.get("RTR Amount", pd.Series([0])).sum()
        avg_rtr_percentage = unified_data_df.get("RTR %", pd.Series([0])).mean()
        
        # Display unified summary metrics
        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            st.metric("Total Loans", f"{total_unified_loans:,}")
        with col2:
            st.metric("Total Participation", f"${total_participation:,.0f}")
        with col3:
            st.metric("Expected Return", f"${total_expected_return:,.0f}")
        with col4:
            st.metric("Actual RTR", f"${total_rtr_amount:,.0f}")
        with col5:
            st.metric("Avg RTR %", f"{avg_rtr_percentage:.1f}%")
        
        # Additional unified metrics
        col6, col7, col8 = st.columns(3)
        with col6:
            loans_with_payments = (unified_data_df.get("RTR Amount", pd.Series([0])) > 0).sum()
            st.metric("Loans with Payments", f"{loans_with_payments}/{total_unified_loans}")
        with col7:
            portfolio_rtr = (total_rtr_amount / total_participation) * 100 if total_participation > 0 else 0
            st.metric("Portfolio RTR %", f"{portfolio_rtr:.1f}%")
        with col8:
            total_unattributed = unified_data_df.get("Unattributed Amount", pd.Series([0])).sum()
            st.metric("Unattributed Payments", f"${total_unattributed:,.0f}")
        
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
                "Customer Active Loans": st.column_config.NumberColumn("Active Loans", width="small"),
                "Unattributed Count": st.column_config.NumberColumn("Unattributed Count", width="small")
            }
        )
        
        # Download unified data
        unified_csv = unified_data_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="📥 Download Unified Analysis (CSV)",
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
        # Loan tape summary metrics
        total_loans = len(loan_tape_df)
        total_participation = loan_tape_df.get("Total Participation", pd.Series([0])).sum()
        total_expected_return = loan_tape_df.get("Total Return", pd.Series([0])).sum()
        total_rtr_amount = loan_tape_df.get("RTR Amount", pd.Series([0])).sum()
        avg_rtr_percentage = loan_tape_df.get("RTR %", pd.Series([0])).mean()
        loans_with_payments = (loan_tape_df.get("RTR Amount", pd.Series([0])) > 0).sum()
        
        # Display summary metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Loans", f"{total_loans:,}")
        with col2:
            st.metric("Total Participation", f"${total_participation:,.0f}")
        with col3:
            st.metric("Actual RTR", f"${total_rtr_amount:,.0f}")
        with col4:
            st.metric("Loans with Payments", f"{loans_with_payments}/{total_loans}")
        
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
            label="📥 Download Loan Tape (CSV)",
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
            st.warning(f"⚠️ ${total_unattributed:,.0f} in unattributed payments across {customers_with_unattributed} customers")
        
        # Download customer summary
        customer_csv = customer_summary_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="📥 Download Customer Summary (CSV)",
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
    st.info("Data Source: Transaction Data (qbo_invoice_payments table) | Using centralized data loader with pagination")
    
    # Data range indicator
    if "txn_date" in df.columns:
        date_range = f"{df['txn_date'].min().strftime('%Y-%m-%d')} to {df['txn_date'].max().strftime('%Y-%m-%d')}"
        st.info(f"Date Range: {date_range}")
    
    # Key metrics summary
    total_customers = df["customer_name"].nunique() if "customer_name" in df.columns else 0
    total_transactions = len(df)
    total_volume = df["total_amount"].sum() if "total_amount" in df.columns else 0
    avg_transaction_size = df["total_amount"].mean() if "total_amount" in df.columns else 0
    
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

# -------------------------
# RISK ANALYSIS SECTION
# -------------------------
st.header("Risk Analysis")

# Transaction-based risk metrics
if df.empty:
    st.warning("No transaction data available for risk analysis")
else:
    # Focus on transaction data for risk analysis
    if "transaction_type" in df.columns and "customer_name" in df.columns:
        risk_df = df[df["transaction_type"].isin(["Invoice", "Payment"])].copy()
        risk_df = risk_df[~risk_df["customer_name"].isin(["CSL", "VEEM"])]
        
        if not risk_df.empty and "total_amount" in risk_df.columns:
            # Calculate comprehensive risk metrics
            risk_df["total_amount"] = risk_df["total_amount"].abs()
            
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
                
                # Display risk dashboard
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
                
                # Risk visualization
                if len(payment_behavior) > 0:
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
# CASH FLOW ANALYSIS
# -------------------------
st.header("Cash Flow Analysis")

if df.empty or "txn_date" not in df.columns:
    st.warning("No transaction data with dates available for cash flow analysis")
else:
    # Transaction-based cash flow analysis
    cash_flow_df = df.copy()
    
    if "transaction_type" in cash_flow_df.columns and "total_amount" in cash_flow_df.columns:
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
        
        if len(daily_cash_flow) > 0:
            # Cash flow trend chart
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
            
            st.altair_chart(cash_flow_chart, use_container_width=True)
            
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
# CACHE MANAGEMENT
# -------------------------
st.header("🔧 Data Management")

st.subheader("Cache Management")
st.info("💡 Use these buttons to refresh cached data. The centralized data loader automatically handles pagination to fetch all records.")

col1, col2, col3, col4 = st.columns(4)

with col1:
    if st.button("🔄 Refresh QBO Data", help="Clears cache for QBO data (load_qbo_data function)"):
        clear_data_cache('qbo')
        st.success("QBO data cache cleared!")
        
with col2:
    if st.button("🔄 Refresh Deal Data", help="Clears cache for deal data (load_deals function)"):
        clear_data_cache('deals')
        st.success("Deal data cache cleared!")
        
with col3:
    if st.button("🔄 Refresh MCA Data", help="Clears cache for MCA data (load_combined_mca_deals function)"):
        clear_data_cache('combined_mca')
        st.success("MCA data cache cleared!")

with col4:
    if st.button("🔄 Refresh All Caches", type="primary", help="Clears all cached data"):
        clear_data_cache()
        st.success("All data caches cleared!")

# -------------------------
# SUMMARY & ALERTS
# -------------------------
st.header("Summary & Alerts")

if df.empty:
    st.error("No transaction data available for analysis")
else:
    # Risk summary
    if 'payment_behavior' in locals() and not payment_behavior.empty:
        high_risk_pct = (payment_behavior["risk_category"] == "High Risk").mean() * 100
        if high_risk_pct > 0:
            st.warning(f"WARNING: {high_risk_pct:.1f}% of customers are classified as High Risk")
    
    # Unified data alerts
    if not unified_data_df.empty:
        total_unattributed = unified_data_df.get("Unattributed Amount", pd.Series([0])).sum()
        if total_unattributed > 0:
            st.warning(f"ATTENTION: ${total_unattributed:,.0f} in unattributed payments require loan ID assignment")
        
        avg_attribution = unified_data_df.get("Attribution %", pd.Series([100])).mean()
        if avg_attribution < 90:
            st.warning(f"ATTRIBUTION: Average attribution rate is {avg_attribution:.1f}% - consider improving loan ID tracking")

st.markdown("---")
st.markdown("*Dashboard last updated: " + pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S") + "*")
st.markdown("*Using centralized data loader with automatic pagination for complete data retrieval*")
