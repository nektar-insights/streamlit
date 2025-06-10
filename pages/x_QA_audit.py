# pages/comprehensive_audit.py
from utils.imports import *
import numpy as np

# ----------------------------
# Supabase connection
# ----------------------------
supabase = get_supabase_client()

# ----------------------------
# Load and prepare data
# ----------------------------
@st.cache_data(ttl=3600)
def load_deals():
    res = supabase.table("deals").select("*").execute()
    return pd.DataFrame(res.data)

@st.cache_data(ttl=3600)
def load_qbo_data():
    df_txn = pd.DataFrame(supabase.table("qbo_transactions").select("*").execute().data)
    df_gl = pd.DataFrame(supabase.table("qbo_general_ledger").select("*").execute().data)
    return df_txn, df_gl

def preprocess_data(dataframe):
    """Clean and preprocess dataframe"""
    df_clean = dataframe.copy()
    
    # Handle numeric columns
    numeric_cols = ['amount', 'debit', 'credit', 'balance']
    for col in numeric_cols:
        if col in df_clean.columns:
            df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
    
    # Handle date columns
    date_cols = ['date', 'txn_date', 'date_created']
    for col in date_cols:
        if col in df_clean.columns:
            df_clean[col] = pd.to_datetime(df_clean[col], errors='coerce')
    
    return df_clean

# Load all data
deals_df = load_deals()
qbo_txn_df, qbo_gl_df = load_qbo_data()

# Preprocess all datasets
deals_df = preprocess_data(deals_df)
qbo_txn_df = preprocess_data(qbo_txn_df)
qbo_gl_df = preprocess_data(qbo_gl_df)

# ----------------------------
# Page setup
# ----------------------------
st.title("🔍 Comprehensive Data Audit Dashboard")
st.markdown("Complete quality assurance checks for deal data integrity and QBO financial analysis")

# ----------------------------
# Executive Summary
# ----------------------------
st.header("📊 Executive Summary")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Total Deals", len(deals_df))
    won_deals_count = len(deals_df[deals_df["is_closed_won"] == True]) if "is_closed_won" in deals_df.columns else 0
    st.metric("Won Deals", won_deals_count)

with col2:
    st.metric("QBO Transactions", len(qbo_txn_df))
    st.metric("QBO GL Entries", len(qbo_gl_df))

with col3:
    # Missing loan IDs calculation
    if "is_closed_won" in deals_df.columns and "loan_id" in deals_df.columns:
        won_deals = deals_df[deals_df["is_closed_won"] == True]
        missing_loan_ids = won_deals[
            (won_deals["loan_id"].isna()) | 
            (won_deals["loan_id"] == "") |
            (won_deals["loan_id"].astype(str).str.strip() == "")
        ]
        missing_pct = (len(missing_loan_ids) / len(won_deals) * 100) if len(won_deals) > 0 else 0
        st.metric("Missing Loan IDs", len(missing_loan_ids))
        st.metric("Missing Rate", f"{missing_pct:.1f}%")
    else:
        st.metric("Missing Loan IDs", "N/A")
        st.metric("Missing Rate", "N/A")

with col4:
    # Data freshness
    if len(deals_df) > 0 and 'date_created' in deals_df.columns:
        latest_deal = deals_df["date_created"].max()
        days_since_last = (pd.Timestamp.now() - latest_deal).days
        st.metric("Days Since Last Deal", days_since_last)
    else:
        st.metric("Days Since Last Deal", "N/A")
    
    # QBO data freshness
    qbo_latest = None
    if len(qbo_gl_df) > 0 and 'txn_date' in qbo_gl_df.columns:
        qbo_latest = qbo_gl_df["txn_date"].max()
    elif len(qbo_txn_df) > 0 and 'date' in qbo_txn_df.columns:
        qbo_latest = qbo_txn_df["date"].max()
    
    if qbo_latest:
        qbo_days_since = (pd.Timestamp.now() - qbo_latest).days
        st.metric("Days Since Last QBO Entry", qbo_days_since)
    else:
        st.metric("Days Since Last QBO Entry", "N/A")

# ----------------------------
# DEAL DATA AUDIT SECTION
# ----------------------------
st.header("💼 Deal Data Audit")

# Debug Info Expander
with st.expander("🔍 Deal Data Debug Info", expanded=False):
    st.write("**Data source:** `deals` table from Supabase")
    st.write(f"**Total rows:** {len(deals_df)}")
    st.write("**Available columns:**")
    st.dataframe(pd.DataFrame({"Column Name": deals_df.columns, "Data Type": deals_df.dtypes.astype(str)}), use_container_width=True)
    
    st.write("**Sample data (first 3 rows):**")
    if len(deals_df) > 0:
        st.dataframe(deals_df.head(3), use_container_width=True)
    else:
        st.error("⚠️ No data available in deals table!")
    
    # Check for loan_id field specifically
    if "loan_id" in deals_df.columns:
        null_loan_ids = deals_df["loan_id"].isna().sum()
        empty_loan_ids = (deals_df["loan_id"] == "").sum() if deals_df["loan_id"].dtype == 'object' else 0
        st.write(f"**Loan ID field status:** Found! Null values: {null_loan_ids}, Empty values: {empty_loan_ids}")
    else:
        st.error("⚠️ No 'loan_id' column found!")

# QA Check 1: Missing Loan IDs in Won Deals
st.subheader("1. Missing Loan IDs in Won Deals")

# Check if required columns exist
if "is_closed_won" not in deals_df.columns:
    st.error("⚠️ Column 'is_closed_won' not found in deals table!")
elif "loan_id" not in deals_df.columns:
    st.error("⚠️ Column 'loan_id' not found in deals table!")
else:
    # Filter for won deals
    won_deals = deals_df[deals_df["is_closed_won"] == True].copy()
    
    # Check for missing loan IDs
    missing_loan_ids = won_deals[
        (won_deals["loan_id"].isna()) | 
        (won_deals["loan_id"] == "") |
        (won_deals["loan_id"].astype(str).str.strip() == "")
    ].copy()
    
    # Display status
    if len(missing_loan_ids) == 0:
        st.success("✅ All won deals have loan IDs assigned!")
    else:
        st.warning(f"⚠️ Found {len(missing_loan_ids)} won deals missing loan IDs")
        
        # Display detailed table of missing loan IDs
        st.subheader("Deals Missing Loan IDs")
        
        # Select relevant columns for display
        possible_deal_name_fields = ["deal_name", "name", "company_name", "business_name", "dba", "client_name"]
        deal_name_field = None
        
        for field in possible_deal_name_fields:
            if field in missing_loan_ids.columns:
                deal_name_field = field
                break
        
        # Base columns always included
        display_columns = ["id", "date_created", "partner_source", "amount", 
                          "total_funded_amount", "factor_rate", "loan_term", "loan_id"]
        
        # Add deal name field if found
        if deal_name_field:
            display_columns.insert(1, deal_name_field)
            st.info(f"✅ Using '{deal_name_field}' for deal names")
        else:
            st.warning(f"⚠️ No deal name field found. Available fields: {list(missing_loan_ids.columns)}")
        
        # Filter to only include existing columns
        available_columns = [col for col in display_columns if col in missing_loan_ids.columns]
        display_df = missing_loan_ids[available_columns].copy()
        
        # Format the display
        if "amount" in display_df.columns:
            display_df["amount"] = display_df["amount"].apply(lambda x: f"${x:,.0f}" if pd.notna(x) else "N/A")
        if "total_funded_amount" in display_df.columns:
            display_df["total_funded_amount"] = display_df["total_funded_amount"].apply(lambda x: f"${x:,.0f}" if pd.notna(x) else "N/A")
        if "factor_rate" in display_df.columns:
            display_df["factor_rate"] = display_df["factor_rate"].apply(lambda x: f"{x:.2f}" if pd.notna(x) else "N/A")
        if "date_created" in display_df.columns:
            display_df["date_created"] = display_df["date_created"].dt.strftime("%Y-%m-%d")
        
        # Rename columns for better display
        column_rename = {
            "id": "Deal ID",
            "date_created": "Date Created", 
            "partner_source": "Partner Source",
            "amount": "Participation Amount",
            "total_funded_amount": "Total Funded",
            "factor_rate": "Factor Rate",
            "loan_term": "Term (months)",
            "loan_id": "Loan ID"
        }
        
        if deal_name_field:
            column_rename[deal_name_field] = "Deal Name"
        
        display_df = display_df.rename(columns=column_rename)
        st.dataframe(display_df, use_container_width=True, hide_index=True)
        
        # Download option
        csv_data = missing_loan_ids[available_columns].to_csv(index=False).encode("utf-8")
        st.download_button(
            label="📥 Download Missing Loan IDs as CSV",
            data=csv_data,
            file_name=f"missing_loan_ids_{pd.Timestamp.now().strftime('%Y%m%d')}.csv",
            mime="text/csv"
        )

# Additional Deal QA Checks
st.subheader("2. Additional Deal Quality Checks")

col1, col2 = st.columns(2)

with col1:
    # Duplicate check
    st.write("**Duplicate Check**")
    if "loan_id" in deals_df.columns:
        duplicate_loan_ids = deals_df[deals_df["loan_id"].notna() & (deals_df["loan_id"] != "")]["loan_id"].duplicated().sum()
        st.metric("Duplicate Loan IDs", duplicate_loan_ids)
        if duplicate_loan_ids == 0:
            st.success("✅ No duplicate loan IDs found")
        else:
            st.error(f"❌ Found {duplicate_loan_ids} duplicate loan IDs")

with col2:
    # Recent activity
    st.write("**Recent Activity (30 days)**")
    recent_cutoff = pd.Timestamp.now() - pd.Timedelta(days=30)
    if 'date_created' in deals_df.columns:
        recent_deals = deals_df[deals_df["date_created"] >= recent_cutoff]
        recent_won = recent_deals[recent_deals["is_closed_won"] == True] if "is_closed_won" in deals_df.columns else pd.DataFrame()
        
        st.metric("Recent Deals", len(recent_deals))
        st.metric("Recent Won Deals", len(recent_won))
        
        recent_close_rate = (len(recent_won) / len(recent_deals) * 100) if len(recent_deals) > 0 else 0
        st.metric("Recent Close Rate", f"{recent_close_rate:.1f}%")

# Missing critical fields in won deals
if "is_closed_won" in deals_df.columns:
    won_deals = deals_df[deals_df["is_closed_won"] == True]
    critical_fields = ["amount", "factor_rate", "loan_term", "commission"]
    existing_critical_fields = [field for field in critical_fields if field in won_deals.columns]
    
    if existing_critical_fields:
        st.write("**Missing Critical Fields in Won Deals**")
        missing_critical_data = []
        for field in existing_critical_fields:
            missing_count = won_deals[field].isna().sum()
            missing_critical_data.append({
                "Field": field.replace("_", " ").title(),
                "Missing Count": missing_count,
                "Missing %": f"{(missing_count / len(won_deals) * 100):.1f}%" if len(won_deals) > 0 else "0.0%"
            })
        
        critical_df = pd.DataFrame(missing_critical_data)
        st.dataframe(critical_df, use_container_width=True, hide_index=True)

# ----------------------------
# QBO DATA ANALYSIS SECTION
# ----------------------------
st.header("💰 QBO Financial Data Analysis")

# QBO Data Quality Overview
col1, col2 = st.columns(2)

with col1:
    st.subheader("Transaction Data Quality")
    if len(qbo_txn_df) > 0:
        st.metric("Total Records", len(qbo_txn_df))
        if 'date' in qbo_txn_df.columns:
            date_range = f"{qbo_txn_df['date'].min().strftime('%Y-%m-%d') if qbo_txn_df['date'].min() else 'N/A'} to {qbo_txn_df['date'].max().strftime('%Y-%m-%d') if qbo_txn_df['date'].max() else 'N/A'}"
            st.metric("Date Range", date_range)
        st.metric("Null Values", qbo_txn_df.isnull().sum().sum())
    else:
        st.warning("No QBO transaction data available")

with col2:
    st.subheader("General Ledger Quality")
    if len(qbo_gl_df) > 0:
        st.metric("Total Records", len(qbo_gl_df))
        if 'txn_date' in qbo_gl_df.columns:
            date_range = f"{qbo_gl_df['txn_date'].min().strftime('%Y-%m-%d') if qbo_gl_df['txn_date'].min() else 'N/A'} to {qbo_gl_df['txn_date'].max().strftime('%Y-%m-%d') if qbo_gl_df['txn_date'].max() else 'N/A'}"
            st.metric("Date Range", date_range)
        st.metric("Null Values", qbo_gl_df.isnull().sum().sum())
    else:
        st.warning("No QBO general ledger data available")

# QBO Analysis Tabs
tab1, tab2, tab3, tab4 = st.tabs(["General Ledger Analysis", "Transaction Analysis", "Loan Performance", "Data Quality Issues"])

with tab1:
    st.subheader("General Ledger Breakdown")
    
    if not qbo_gl_df.empty:
        # Group by transaction type
        if 'txn_type' in qbo_gl_df.columns and 'amount' in qbo_gl_df.columns:
            gl_by_type = qbo_gl_df.groupby('txn_type')['amount'].agg(['sum', 'count', 'mean']).round(2)
            gl_by_type.columns = ['Total Amount', 'Transaction Count', 'Average Amount']
            gl_by_type = gl_by_type.sort_values('Total Amount', ascending=False)
            
            st.write("**By Transaction Type:**")
            st.dataframe(gl_by_type.style.format({
                'Total Amount': '${:,.2f}',
                'Average Amount': '${:,.2f}'
            }), use_container_width=True)
            
            # Visualization
            chart_data = gl_by_type.reset_index()
            fig_type = alt.Chart(chart_data).mark_bar().encode(
                x=alt.X('txn_type:N', title='Transaction Type', axis=alt.Axis(labelAngle=-45)),
                y=alt.Y('Total Amount:Q', title='Total Amount ($)', axis=alt.Axis(format='$,.0f')),
                tooltip=['txn_type:N', alt.Tooltip('Total Amount:Q', format='$,.2f')]
            ).properties(
                width=600,
                height=400,
                title='General Ledger: Total Amount by Transaction Type'
            )
            st.altair_chart(fig_type, use_container_width=True)
        
        # Group by name
        if 'name' in qbo_gl_df.columns:
            gl_by_name = qbo_gl_df.groupby('name')['amount'].agg(['sum', 'count', 'mean']).round(2)
            gl_by_name.columns = ['Total Amount', 'Transaction Count', 'Average Amount']
            gl_by_name = gl_by_name.sort_values('Total Amount', ascending=False).head(20)
            
            st.write("**Top 20 by Name:**")
            st.dataframe(gl_by_name.style.format({
                'Total Amount': '${:,.2f}',
                'Average Amount': '${:,.2f}'
            }), use_container_width=True)
    else:
        st.info("No general ledger data available for analysis")

with tab2:
    st.subheader("Transaction Report Analysis")
    
    if not qbo_txn_df.empty:
        # Group by transaction type
        if 'transaction_type' in qbo_txn_df.columns and 'amount' in qbo_txn_df.columns:
            txn_by_type = qbo_txn_df.groupby('transaction_type')['amount'].agg(['sum', 'count', 'mean']).round(2)
            txn_by_type.columns = ['Total Amount', 'Transaction Count', 'Average Amount']
            txn_by_type = txn_by_type.sort_values('Total Amount', ascending=False)
            
            st.write("**By Transaction Type:**")
            st.dataframe(txn_by_type.style.format({
                'Total Amount': '${:,.2f}',
                'Average Amount': '${:,.2f}'
            }), use_container_width=True)
        
        # Group by account
        if 'account' in qbo_txn_df.columns:
            txn_by_account = qbo_txn_df.groupby('account')['amount'].agg(['sum', 'count', 'mean']).round(2)
            txn_by_account.columns = ['Total Amount', 'Transaction Count', 'Average Amount']
            txn_by_account = txn_by_account.sort_values('Total Amount', ascending=False).head(15)
            
            st.write("**Top 15 by Account:**")
            st.dataframe(txn_by_account.style.format({
                'Total Amount': '${:,.2f}',
                'Average Amount': '${:,.2f}'
            }), use_container_width=True)
    else:
        st.info("No transaction data available for analysis")

with tab3:
    st.subheader("Enhanced Loan Performance Analysis")
    
    # Use General Ledger for more reliable analysis if available
    analysis_df = qbo_gl_df if not qbo_gl_df.empty else qbo_txn_df
    
    if not analysis_df.empty:
        # Filter for loan-related transactions
        loan_transactions = analysis_df[
            (analysis_df.get('txn_type', analysis_df.get('transaction_type', '')).isin(['Invoice', 'Payment', 'Bill', 'Credit Memo'])) |
            (analysis_df.get('account', '').str.contains('Loan|Receivable|Interest', case=False, na=False))
        ].copy()
        
        if not loan_transactions.empty:
            # Enhanced pivot analysis
            amount_col = 'amount' if 'amount' in loan_transactions.columns else 'credit'
            type_col = 'txn_type' if 'txn_type' in loan_transactions.columns else 'transaction_type'
            
            if amount_col in loan_transactions.columns and type_col in loan_transactions.columns:
                loan_transactions[amount_col] = loan_transactions[amount_col].abs()
                
                pivot_enhanced = loan_transactions.pivot_table(
                    index="name",
                    columns=type_col,
                    values=amount_col,
                    aggfunc="sum",
                    fill_value=0
                ).reset_index()
                
                # Calculate enhanced metrics
                pivot_enhanced["total_invoiced"] = pivot_enhanced.get("Invoice", 0)
                pivot_enhanced["total_payments"] = pivot_enhanced.get("Payment", 0)
                pivot_enhanced["outstanding_balance"] = pivot_enhanced["total_invoiced"] - pivot_enhanced["total_payments"]
                pivot_enhanced["payment_ratio"] = np.where(
                    pivot_enhanced["total_invoiced"] > 0,
                    pivot_enhanced["total_payments"] / pivot_enhanced["total_invoiced"],
                    0
                )
                pivot_enhanced["risk_score"] = np.where(
                    pivot_enhanced["payment_ratio"] < 0.5, "High Risk",
                    np.where(pivot_enhanced["payment_ratio"] < 0.8, "Medium Risk", "Low Risk")
                )
                
                # Display enhanced analysis
                display_cols = ["name", "total_invoiced", "total_payments", "outstanding_balance", "payment_ratio", "risk_score"]
                existing_cols = [col for col in display_cols if col in pivot_enhanced.columns]
                
                if existing_cols:
                    pivot_display = pivot_enhanced[existing_cols].copy()
                    pivot_display = pivot_display.sort_values("outstanding_balance", ascending=False)
                    
                    # Format currency columns
                    currency_cols = ["total_invoiced", "total_payments", "outstanding_balance"]
                    for col in currency_cols:
                        if col in pivot_display.columns:
                            pivot_display[col] = pivot_display[col].apply(lambda x: f"${x:,.2f}")
                    
                    if "payment_ratio" in pivot_display.columns:
                        pivot_display["payment_ratio"] = pivot_display["payment_ratio"].apply(lambda x: f"{x:.1%}")
                    
                    st.dataframe(pivot_display, use_container_width=True)
                else:
                    st.info("Unable to perform loan performance analysis - missing required columns")
            else:
                st.info("Unable to perform loan performance analysis - missing amount or transaction type columns")
        else:
            st.info("No loan-related transactions found")
    else:
        st.info("No QBO data available for loan performance analysis")
    
    # Original analysis for comparison (using transaction data)
    if not qbo_txn_df.empty and 'transaction_type' in qbo_txn_df.columns and 'name' in qbo_txn_df.columns:
        st.subheader("Original Analysis (Transaction Report)")
        
        filtered_df = qbo_txn_df[qbo_txn_df["transaction_type"].isin(["Invoice", "Payment"])].copy()
        filtered_df = filtered_df[~filtered_df["name"].isin(["CSL", "VEEM"])]
        
        if 'amount' in filtered_df.columns:
            filtered_df["amount"] = filtered_df["amount"].abs()
            
            pivot = filtered_df.pivot_table(
                index="name",
                columns="transaction_type",
                values="amount",
                aggfunc="sum",
                fill_value=0
            ).reset_index()
            
            if "Invoice" in pivot.columns:
                pivot["balance"] = pivot.get("Invoice", 0) - pivot.get("Payment", 0)
                pivot["balance_ratio"] = pivot["balance"] / pivot["Invoice"]
                pivot["indicator"] = pivot["balance_ratio"].apply(
                    lambda x: "🔴" if x >= 0.25 else ("🟡" if x >= 0.10 else "🟢")
                )
                
                pivot_display = pivot.copy()
                pivot_display["Invoice"] = pivot_display["Invoice"].map("${:,.2f}".format)
                pivot_display["Payment"] = pivot_display["Payment"].map("${:,.2f}".format)
                pivot_display["balance"] = pivot_display["balance"].map("${:,.2f}".format)
                pivot_display["Deal Name"] = pivot_display["name"]
                pivot_display["Balance (with Risk)"] = pivot_display["indicator"] + " " + pivot_display["balance"]
                
                st.dataframe(
                    pivot_display[["Deal Name", "Invoice", "Payment", "Balance (with Risk)"]]
                    .sort_values("Balance (with Risk)", ascending=False),
                    use_container_width=True
                )

with tab4:
    st.subheader("Data Quality Issues")
    
    # NA Count Analysis
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Transaction Report - NA Counts:**")
        if not qbo_txn_df.empty:
            txn_na_counts = qbo_txn_df.isnull().sum().reset_index()
            txn_na_counts.columns = ['Column', 'NA Count']
            txn_na_counts['Total Records'] = len(qbo_txn_df)
            txn_na_counts['NA Percentage'] = (txn_na_counts['NA Count'] / len(qbo_txn_df) * 100).round(2)
            txn_na_counts = txn_na_counts.sort_values('NA Count', ascending=False)
            
            st.dataframe(txn_na_counts.style.format({
                'NA Percentage': '{:.2f}%'
            }), use_container_width=True)
        else:
            st.write("No transaction data available")
    
    with col2:
        st.write("**General Ledger - NA Counts:**")
        if not qbo_gl_df.empty:
            gl_na_counts = qbo_gl_df.isnull().sum().reset_index()
            gl_na_counts.columns = ['Column', 'NA Count']
            gl_na_counts['Total Records'] = len(qbo_gl_df)
            gl_na_counts['NA Percentage'] = (gl_na_counts['NA Count'] / len(qbo_gl_df) * 100).round(2)
            gl_na_counts = gl_na_counts.sort_values('NA Count', ascending=False)
            
            st.dataframe(gl_na_counts.style.format({
                'NA Percentage': '{:.2f}%'
            }), use_container_width=True)
        else:
            st.write("No general ledger data available")
    
    # Additional quality issues
    st.write("### ⚠️ Additional Data Quality Issues")
    
    issues = []
    
    # Date issues
    if 'date' in qbo_txn_df.columns:
        null_dates_txn = qbo_txn_df['date'].isnull().sum()
        if null_dates_txn > 0:
            issues.append(f"Transaction Report: {null_dates_txn} records with null dates")
    
    if 'txn_date' in qbo_gl_df.columns:
        null_dates_gl = qbo_gl_df['txn_date'].isnull().sum()
        if null_dates_gl > 0:
            issues.append(f"General Ledger: {null_dates_gl} records with null dates")
    
    # Amount issues
    if 'amount' in qbo_txn_df.columns:
        null_amounts_txn = qbo_txn_df['amount'].isnull().sum()
        zero_amounts_txn = (qbo_txn_df['amount'] == 0).sum()
        if null_amounts_txn > 0:
            issues.append(f"Transaction Report: {null_amounts_txn} records with null amounts")
        if zero_amounts_txn > 0:
            issues.append(f"Transaction Report: {zero_amounts_txn} records with zero amounts")
    
    if 'amount' in qbo_gl_df.columns:
        null_amounts_gl = qbo_gl_df['amount'].isnull().sum()
        zero_amounts_gl = (qbo_gl_df['amount'] == 0).sum()
        if null_amounts_gl > 0:
            issues.append(f"General Ledger: {null_amounts_gl} records with null amounts")
        if zero_amounts_gl > 0:
            issues.append(f"General Ledger: {zero_amounts_gl} records with zero amounts")
    
    if issues:
        for issue in issues:
            st.write(f"• {issue}")
    else:
        st.success("✅ No major data quality issues detected")

# ----------------------------
# Cache clearing functions
# ----------------------------
def clear_pipeline_cache():
    """Clear cache for pipeline dashboard"""
    if hasattr(st.session_state, '_cache'):
        for key in list(st.session_state._cache.keys()):
            if 'load_deals' in key:
                del st.session_state._cache[key]
    st.cache_data.clear()

def clear_mca_cache():
    """Clear cache for MCA dashboard"""
    if hasattr(st.session_state, '_cache'):
        for key in list(st.session_state._cache.keys()):
            if 'load_mca_deals' in key or 'combine_deals' in key:
                del st.session_state._cache[key]
    st.cache_data.clear()

def clear_qbo_cache():
    """Clear cache for QBO dashboard"""
    if hasattr(st.session_state, '_cache'):
        for key in list(st.session_state._cache.keys()):
            if 'load_qbo_data' in key:
                del st.session_state._cache[key]
    st.cache_data.clear()

# ----------------------------
# Data Management & Cache Refresh
# ----------------------------
st.header("🔧 Data Management")

st.subheader("Cache Management")
st.info("💡 Use these buttons to refresh cached data across different dashboards. After clicking, navigate to the respective dashboard to see updated data.")

col1, col2, col3 = st.columns(3)

with col1:
    if st.button("🔄 Refresh Pipeline Data", help="Clears cache for main pipeline dashboard (load_deals function)"):
        clear_pipeline_cache()
        st.success("✅ Pipeline dashboard cache cleared!")
        
with col2:
    if st.button("🔄 Refresh MCA Data", help="Clears cache for MCA dashboard (load_mca_deals & combine_deals functions)"):
        clear_mca_cache()
        st.success("✅ MCA dashboard cache cleared!")
        
with col3:
    if st.button("🔄 Refresh QBO Data", help="Clears cache for QBO dashboard (load_qbo_data function)"):
        clear_qbo_cache()
        st.success("✅ QBO dashboard cache cleared!")

if st.button("🔄 Refresh All Data Caches", type="primary", help="Clears all cached data across the entire application"):
    st.cache_data.clear()
    st.success("✅ All data caches cleared! Navigate to other pages to see fresh data.")

# Show cache status
st.subheader("Cache Status")
cache_info = []

# Check if we have any cached functions
if hasattr(st.session_state, '_cache'):
    cache_count = len(st.session_state._cache)
    cache_info.append(f"📊 Session cache entries: {cache_count}")
else:
    cache_info.append("📊 No session cache detected")

# Display cache info
for info in cache_info:
    st.text(info)

# Add timestamp of last refresh
if "last_cache_clear" not in st.session_state:
    st.session_state.last_cache_clear = "Never"

if st.session_state.get("last_cache_clear") != "Never":
    st.text(f"🕒 Last cache clear: {st.session_state.last_cache_clear}")

# Update timestamp when any cache is cleared
if st.button("📝 Mark Cache Clear Time"):
    st.session_state.last_cache_clear = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
    st.success(f"✅ Cache clear time marked: {st.session_state.last_cache_clear}")

# ----------------------------
# Cross-Dataset Analysis
# ----------------------------
st.header("🔗 Cross-Dataset Analysis")

st.subheader("Data Comparison Overview")
col1, col2 = st.columns(2)

with col1:
    st.write("**Dataset Comparison:**")
    comparison_data = {
        'Metric': ['Total Records', 'Unique Names', 'Date Range (Days)', 'Total Amount'],
        'Deal Data': [
            len(deals_df),
            deals_df['name'].nunique() if 'name' in deals_df.columns else 0,
            (deals_df['date_created'].max() - deals_df['date_created'].min()).days if 'date_created' in deals_df.columns and deals_df['date_created'].notna().any() else 0,
            deals_df['amount'].sum() if 'amount' in deals_df.columns else 0
        ],
        'QBO Transactions': [
            len(qbo_txn_df),
            qbo_txn_df['name'].nunique() if 'name' in qbo_txn_df.columns else 0,
            (qbo_txn_df['date'].max() - qbo_txn_df['date'].min()).days if 'date' in qbo_txn_df.columns and qbo_txn_df['date'].notna().any() else 0,
            qbo_txn_df['amount'].sum() if 'amount' in qbo_txn_df.columns else 0
        ],
        'QBO General Ledger': [
            len(qbo_gl_df),
            qbo_gl_df['name'].nunique() if 'name' in qbo_gl_df.columns else 0,
            (qbo_gl_df['txn_date'].max() - qbo_gl_df['txn_date'].min()).days if 'txn_date' in qbo_gl_df.columns and qbo_gl_df['txn_date'].notna().any() else 0,
            qbo_gl_df['amount'].sum() if 'amount' in qbo_gl_df.columns else 0
        ]
    }
    comparison_df = pd.DataFrame(comparison_data)
    st.dataframe(comparison_df, use_container_width=True)

with col2:
    # Common names analysis
    if 'name' in deals_df.columns and 'name' in qbo_txn_df.columns:
        deal_names = set(deals_df['name'].dropna().unique()) if 'name' in deals_df.columns else set()
        qbo_txn_names = set(qbo_txn_df['name'].dropna().unique()) if 'name' in qbo_txn_df.columns else set()
        qbo_gl_names = set(qbo_gl_df['name'].dropna().unique()) if 'name' in qbo_gl_df.columns else set()
        
        common_deal_qbo = deal_names.intersection(qbo_txn_names)
        common_all = deal_names.intersection(qbo_txn_names).intersection(qbo_gl_names)
        
        st.write("**Name Overlap Analysis:**")
        st.metric("Deal & QBO Transaction Names", len(common_deal_qbo))
        st.metric("Common Across All Datasets", len(common_all))
        st.metric("Deal Names Only", len(deal_names - qbo_txn_names - qbo_gl_names))

# ----------------------------
# System Health Summary
# ----------------------------
st.header("🏥 System Health Summary")

health_status = []

# Deal data health
if len(deals_df) > 0:
    if "is_closed_won" in deals_df.columns and "loan_id" in deals_df.columns:
        won_deals = deals_df[deals_df["is_closed_won"] == True]
        missing_loan_ids = won_deals[
            (won_deals["loan_id"].isna()) | 
            (won_deals["loan_id"] == "") |
            (won_deals["loan_id"].astype(str).str.strip() == "")
        ]
        missing_pct = (len(missing_loan_ids) / len(won_deals) * 100) if len(won_deals) > 0 else 0
        
        if missing_pct == 0:
            health_status.append("✅ Deal Data: All won deals have loan IDs")
        elif missing_pct < 5:
            health_status.append(f"🟡 Deal Data: {missing_pct:.1f}% missing loan IDs (minor issue)")
        else:
            health_status.append(f"🔴 Deal Data: {missing_pct:.1f}% missing loan IDs (requires attention)")
    else:
        health_status.append("🟡 Deal Data: Missing required columns for loan ID validation")
else:
    health_status.append("🔴 Deal Data: No deal data available")

# QBO data health
qbo_health_issues = 0
if len(qbo_txn_df) == 0:
    qbo_health_issues += 1
if len(qbo_gl_df) == 0:
    qbo_health_issues += 1

if qbo_health_issues == 0:
    health_status.append("✅ QBO Data: All datasets available and populated")
elif qbo_health_issues == 1:
    health_status.append("🟡 QBO Data: One dataset missing or empty")
else:
    health_status.append("🔴 QBO Data: Multiple datasets missing or empty")

# Data freshness health
freshness_issues = 0
if len(deals_df) > 0 and 'date_created' in deals_df.columns:
    latest_deal = deals_df["date_created"].max()
    days_since_last = (pd.Timestamp.now() - latest_deal).days
    if days_since_last > 7:
        freshness_issues += 1

if len(qbo_gl_df) > 0 and 'txn_date' in qbo_gl_df.columns:
    qbo_latest = qbo_gl_df["txn_date"].max()
    qbo_days_since = (pd.Timestamp.now() - qbo_latest).days
    if qbo_days_since > 7:
        freshness_issues += 1

if freshness_issues == 0:
    health_status.append("✅ Data Freshness: All data appears current")
elif freshness_issues == 1:
    health_status.append("🟡 Data Freshness: One dataset may be stale")
else:
    health_status.append("🔴 Data Freshness: Multiple datasets appear stale")

# Display health status
st.subheader("Overall System Health")
for status in health_status:
    st.write(status)

# Health score calculation
green_count = sum(1 for status in health_status if status.startswith("✅"))
yellow_count = sum(1 for status in health_status if status.startswith("🟡"))
red_count = sum(1 for status in health_status if status.startswith("🔴"))

total_checks = len(health_status)
health_score = (green_count * 100 + yellow_count * 50) / total_checks if total_checks > 0 else 0

if health_score >= 90:
    st.success(f"🎉 System Health Score: {health_score:.0f}% - Excellent!")
elif health_score >= 70:
    st.warning(f"⚠️ System Health Score: {health_score:.0f}% - Good, minor issues detected")
else:
    st.error(f"🚨 System Health Score: {health_score:.0f}% - Attention required")

# ----------------------------
# Export and Reporting
# ----------------------------
st.header("📋 Export and Reporting")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Generate Summary Report")
    if st.button("📄 Generate Health Report", type="primary"):
        # Create summary report
        report_data = {
            "Report Generated": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
            "Deal Data Records": len(deals_df),
            "QBO Transaction Records": len(qbo_txn_df),
            "QBO GL Records": len(qbo_gl_df),
            "System Health Score": f"{health_score:.0f}%",
            "Health Status": health_status
        }
        
        # Convert to DataFrame for easy export
        report_df = pd.DataFrame(list(report_data.items()), columns=['Metric', 'Value'])
        csv_data = report_df.to_csv(index=False).encode("utf-8")
        
        st.download_button(
            label="📥 Download Health Report CSV",
            data=csv_data,
            file_name=f"system_health_report_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )

with col2:
    st.subheader("Data Export Options")
    
    # Export deals with missing loan IDs
    if "is_closed_won" in deals_df.columns and "loan_id" in deals_df.columns:
        won_deals = deals_df[deals_df["is_closed_won"] == True]
        missing_loan_ids = won_deals[
            (won_deals["loan_id"].isna()) | 
            (won_deals["loan_id"] == "") |
            (won_deals["loan_id"].astype(str).str.strip() == "")
        ]
        
        if len(missing_loan_ids) > 0:
            csv_data = missing_loan_ids.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="📥 Export Missing Loan IDs",
                data=csv_data,
                file_name=f"missing_loan_ids_{pd.Timestamp.now().strftime('%Y%m%d')}.csv",
                mime="text/csv"
            )
    
    # Export QBO data quality issues
    if not qbo_txn_df.empty or not qbo_gl_df.empty:
        if st.button("📊 Export QBO Quality Report"):
            quality_data = []
            
            # Add transaction data quality info
            if not qbo_txn_df.empty:
                quality_data.append({
                    "Dataset": "QBO Transactions",
                    "Total Records": len(qbo_txn_df),
                    "Null Values": qbo_txn_df.isnull().sum().sum(),
                    "Date Range": f"{qbo_txn_df['date'].min() if 'date' in qbo_txn_df.columns else 'N/A'} to {qbo_txn_df['date'].max() if 'date' in qbo_txn_df.columns else 'N/A'}"
                })
            
            # Add GL data quality info
            if not qbo_gl_df.empty:
                quality_data.append({
                    "Dataset": "QBO General Ledger",
                    "Total Records": len(qbo_gl_df),
                    "Null Values": qbo_gl_df.isnull().sum().sum(),
                    "Date Range": f"{qbo_gl_df['txn_date'].min() if 'txn_date' in qbo_gl_df.columns else 'N/A'} to {qbo_gl_df['txn_date'].max() if 'txn_date' in qbo_gl_df.columns else 'N/A'}"
                })
            
            quality_df = pd.DataFrame(quality_data)
            csv_data = quality_df.to_csv(index=False).encode("utf-8")
            
            st.download_button(
                label="📥 Download QBO Quality Report",
                data=csv_data,
                file_name=f"qbo_quality_report_{pd.Timestamp.now().strftime('%Y%m%d')}.csv",
                mime="text/csv"
            )

# ----------------------------
# Footer with Last Updated
# ----------------------------
st.divider()
st.caption(f"Dashboard last updated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')} | Data sources: Supabase (deals, qbo_transactions, qbo_general_ledger)")
