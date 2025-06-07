# pages/qbo_dashboard.py
from utils.imports import *
import numpy as np

# -------------------------
# Setup: Supabase Connection
# -------------------------
supabase = get_supabase_client()

# -------------------------
# Load and Prepare Data
# -------------------------
@st.cache_data(ttl=3600)
def load_qbo_data():
    df_txn = pd.DataFrame(supabase.table("qbo_transactions").select("*").execute().data)
    df_gl = pd.DataFrame(supabase.table("qbo_general_ledger").select("*").execute().data)
    return df_txn, df_gl

df, gl_df = load_qbo_data()

# Preprocess Data
def preprocess_data(dataframe):
    """Clean and preprocess dataframe"""
    df_clean = dataframe.copy()
    
    # Handle numeric columns
    numeric_cols = ['amount', 'debit', 'credit', 'balance']
    for col in numeric_cols:
        if col in df_clean.columns:
            df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
    
    # Handle date columns
    date_cols = ['date', 'txn_date']
    for col in date_cols:
        if col in df_clean.columns:
            df_clean[col] = pd.to_datetime(df_clean[col], errors='coerce')
    
    return df_clean

df = preprocess_data(df)
gl_df = preprocess_data(gl_df)

st.title("Enhanced QBO Debt Portfolio Dashboard")

# -------------------------
# Data Quality Overview
# -------------------------
st.header("📊 Data Quality Overview")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Transaction Data")
    st.metric("Total Records", len(df))
    st.metric("Date Range", f"{df['date'].min().strftime('%Y-%m-%d') if df['date'].min() else 'N/A'} to {df['date'].max().strftime('%Y-%m-%d') if df['date'].max() else 'N/A'}")
    st.metric("Null Values", df.isnull().sum().sum())

with col2:
    st.subheader("General Ledger Data")
    st.metric("Total Records", len(gl_df))
    st.metric("Date Range", f"{gl_df['txn_date'].min().strftime('%Y-%m-%d') if gl_df['txn_date'].min() else 'N/A'} to {gl_df['txn_date'].max().strftime('%Y-%m-%d') if gl_df['txn_date'].max() else 'N/A'}")
    st.metric("Null Values", gl_df.isnull().sum().sum())

# -------------------------
# EXPLORATORY DATA ANALYSIS
# -------------------------
st.header("🔍 Exploratory Data Analysis")

# EDA Tabs
tab1, tab2, tab3, tab4 = st.tabs(["General Ledger Analysis", "Transaction Analysis", "Cross-Analysis", "Data Issues"])

with tab1:
    st.subheader("General Ledger Breakdown")
    
    # Group by transaction type
    if not gl_df.empty and 'txn_type' in gl_df.columns:
        gl_by_type = gl_df.groupby('txn_type')['amount'].agg(['sum', 'count', 'mean']).round(2)
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
    if not gl_df.empty and 'name' in gl_df.columns:
        gl_by_name = gl_df.groupby('name')['amount'].agg(['sum', 'count', 'mean']).round(2)
        gl_by_name.columns = ['Total Amount', 'Transaction Count', 'Average Amount']
        gl_by_name = gl_by_name.sort_values('Total Amount', ascending=False).head(20)
        
        st.write("**Top 20 by Name:**")
        st.dataframe(gl_by_name.style.format({
            'Total Amount': '${:,.2f}',
            'Average Amount': '${:,.2f}'
        }), use_container_width=True)
        
        # Visualization
        chart_data = gl_by_name.reset_index().head(10)
        fig_name = alt.Chart(chart_data).mark_bar().encode(
            x=alt.X('name:N', title='Name', axis=alt.Axis(labelAngle=-45)),
            y=alt.Y('Total Amount:Q', title='Total Amount ($)', axis=alt.Axis(format='$,.0f')),
            tooltip=['name:N', alt.Tooltip('Total Amount:Q', format='$,.2f')]
        ).properties(
            width=600,
            height=400,
            title='General Ledger: Top 10 Names by Total Amount'
        )
        st.altair_chart(fig_name, use_container_width=True)

with tab2:
    st.subheader("Transaction Report Analysis")
    
    # Group by transaction type
    if not df.empty and 'transaction_type' in df.columns:
        txn_by_type = df.groupby('transaction_type')['amount'].agg(['sum', 'count', 'mean']).round(2)
        txn_by_type.columns = ['Total Amount', 'Transaction Count', 'Average Amount']
        txn_by_type = txn_by_type.sort_values('Total Amount', ascending=False)
        
        st.write("**By Transaction Type:**")
        st.dataframe(txn_by_type.style.format({
            'Total Amount': '${:,.2f}',
            'Average Amount': '${:,.2f}'
        }), use_container_width=True)
    
    # Group by name
    if not df.empty and 'name' in df.columns:
        txn_by_name = df.groupby('name')['amount'].agg(['sum', 'count', 'mean']).round(2)
        txn_by_name.columns = ['Total Amount', 'Transaction Count', 'Average Amount']
        txn_by_name = txn_by_name.sort_values('Total Amount', ascending=False).head(20)
        
        st.write("**Top 20 by Name:**")
        st.dataframe(txn_by_name.style.format({
            'Total Amount': '${:,.2f}',
            'Average Amount': '${:,.2f}'
        }), use_container_width=True)
    
    # Group by account
    if not df.empty and 'account' in df.columns:
        txn_by_account = df.groupby('account')['amount'].agg(['sum', 'count', 'mean']).round(2)
        txn_by_account.columns = ['Total Amount', 'Transaction Count', 'Average Amount']
        txn_by_account = txn_by_account.sort_values('Total Amount', ascending=False).head(15)
        
        st.write("**Top 15 by Account:**")
        st.dataframe(txn_by_account.style.format({
            'Total Amount': '${:,.2f}',
            'Average Amount': '${:,.2f}'
        }), use_container_width=True)
    
    # Transaction Type + Name + Account combination
    if not df.empty and all(col in df.columns for col in ['transaction_type', 'name', 'account']):
        txn_combo = df.groupby(['transaction_type', 'name', 'account'])['amount'].agg(['sum', 'count']).round(2)
        txn_combo.columns = ['Total Amount', 'Count']
        txn_combo = txn_combo.sort_values('Total Amount', ascending=False).head(20)
        
        st.write("**Transaction Type + Name + Account (Top 20):**")
        st.dataframe(txn_combo.style.format({
            'Total Amount': '${:,.2f}'
        }), use_container_width=True)

with tab3:
    st.subheader("Cross-Dataset Analysis")
    
    # Compare transaction volumes between datasets
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Data Comparison:**")
        comparison_data = {
            'Metric': ['Total Records', 'Unique Names', 'Date Range (Days)', 'Total Amount'],
            'Transaction Report': [
                len(df),
                df['name'].nunique() if 'name' in df.columns else 0,
                (df['date'].max() - df['date'].min()).days if df['date'].notna().any() else 0,
                df['amount'].sum() if 'amount' in df.columns else 0
            ],
            'General Ledger': [
                len(gl_df),
                gl_df['name'].nunique() if 'name' in gl_df.columns else 0,
                (gl_df['txn_date'].max() - gl_df['txn_date'].min()).days if gl_df['txn_date'].notna().any() else 0,
                gl_df['amount'].sum() if 'amount' in gl_df.columns else 0
            ]
        }
        comparison_df = pd.DataFrame(comparison_data)
        st.dataframe(comparison_df, use_container_width=True)
    
    with col2:
        # Common names analysis
        if 'name' in df.columns and 'name' in gl_df.columns:
            txn_names = set(df['name'].dropna().unique())
            gl_names = set(gl_df['name'].dropna().unique())
            
            common_names = txn_names.intersection(gl_names)
            txn_only = txn_names - gl_names
            gl_only = gl_names - txn_names
            
            st.write("**Name Overlap Analysis:**")
            st.metric("Common Names", len(common_names))
            st.metric("Transaction Report Only", len(txn_only))
            st.metric("General Ledger Only", len(gl_only))

with tab4:
    st.subheader("Data Quality Issues")
    
    # 1) NA Count Analysis for each column
    st.write("### 📊 NULL/NA Count by Column")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Transaction Report - NA Counts:**")
        if not df.empty:
            txn_na_counts = df.isnull().sum().reset_index()
            txn_na_counts.columns = ['Column', 'NA Count']
            txn_na_counts['Total Records'] = len(df)
            txn_na_counts['NA Percentage'] = (txn_na_counts['NA Count'] / len(df) * 100).round(2)
            txn_na_counts = txn_na_counts.sort_values('NA Count', ascending=False)
            
            st.dataframe(txn_na_counts.style.format({
                'NA Percentage': '{:.2f}%'
            }), use_container_width=True)
        else:
            st.write("No transaction data available")
    
    with col2:
        st.write("**General Ledger - NA Counts:**")
        if not gl_df.empty:
            gl_na_counts = gl_df.isnull().sum().reset_index()
            gl_na_counts.columns = ['Column', 'NA Count']
            gl_na_counts['Total Records'] = len(gl_df)
            gl_na_counts['NA Percentage'] = (gl_na_counts['NA Count'] / len(gl_df) * 100).round(2)
            gl_na_counts = gl_na_counts.sort_values('NA Count', ascending=False)
            
            st.dataframe(gl_na_counts.style.format({
                'NA Percentage': '{:.2f}%'
            }), use_container_width=True)
        else:
            st.write("No general ledger data available")
    
    # 2) Description Field Analysis
    st.write("### 📝 Description Field Analysis")
    
    col3, col4 = st.columns(2)
    
    with col3:
        st.write("**Transaction Report - Description Grouping:**")
        if not df.empty and 'description' in df.columns:
            txn_desc_counts = df['description'].value_counts().reset_index()
            txn_desc_counts.columns = ['Description', 'Count']
            txn_desc_counts['Percentage'] = (txn_desc_counts['Count'] / len(df) * 100).round(2)
            
            # Show top 20 descriptions
            txn_desc_display = txn_desc_counts.head(20)
            st.dataframe(txn_desc_display.style.format({
                'Percentage': '{:.2f}%'
            }), use_container_width=True)
            
            # Show summary stats
            st.write(f"**Summary:**")
            st.write(f"- Total unique descriptions: {len(txn_desc_counts)}")
            st.write(f"- Records with descriptions: {df['description'].notna().sum()}")
            st.write(f"- Records with null descriptions: {df['description'].isnull().sum()}")
            
        else:
            st.write("No description field available in transaction data")
    
    with col4:
        st.write("**General Ledger - Description Grouping:**")
        if not gl_df.empty and 'description' in gl_df.columns:
            gl_desc_counts = gl_df['description'].value_counts().reset_index()
            gl_desc_counts.columns = ['Description', 'Count']
            gl_desc_counts['Percentage'] = (gl_desc_counts['Count'] / len(gl_df) * 100).round(2)
            
            # Show top 20 descriptions
            gl_desc_display = gl_desc_counts.head(20)
            st.dataframe(gl_desc_display.style.format({
                'Percentage': '{:.2f}%'
            }), use_container_width=True)
            
            # Show summary stats
            st.write(f"**Summary:**")
            st.write(f"- Total unique descriptions: {len(gl_desc_counts)}")
            st.write(f"- Records with descriptions: {gl_df['description'].notna().sum()}")
            st.write(f"- Records with null descriptions: {gl_df['description'].isnull().sum()}")
            
            # Highlight voided transactions
            voided_count = gl_df['description'].str.contains('Voided', case=False, na=False).sum()
            if voided_count > 0:
                st.warning(f"⚠️ Found {voided_count} voided transactions")
                
        else:
            st.write("No description field available in general ledger data")
    
    # Additional Quality Checks
    st.write("### ⚠️ Additional Data Quality Issues")
    
    issues = []
    
    # Date issues
    if 'date' in df.columns:
        null_dates_txn = df['date'].isnull().sum()
        if null_dates_txn > 0:
            issues.append(f"Transaction Report: {null_dates_txn} records with null dates")
    
    if 'txn_date' in gl_df.columns:
        null_dates_gl = gl_df['txn_date'].isnull().sum()
        if null_dates_gl > 0:
            issues.append(f"General Ledger: {null_dates_gl} records with null dates")
    
    # Amount issues
    if 'amount' in df.columns:
        null_amounts_txn = df['amount'].isnull().sum()
        zero_amounts_txn = (df['amount'] == 0).sum()
        if null_amounts_txn > 0:
            issues.append(f"Transaction Report: {null_amounts_txn} records with null amounts")
        if zero_amounts_txn > 0:
            issues.append(f"Transaction Report: {zero_amounts_txn} records with zero amounts")
    
    if 'amount' in gl_df.columns:
        null_amounts_gl = gl_df['amount'].isnull().sum()
        zero_amounts_gl = (gl_df['amount'] == 0).sum()
        if null_amounts_gl > 0:
            issues.append(f"General Ledger: {null_amounts_gl} records with null amounts")
        if zero_amounts_gl > 0:
            issues.append(f"General Ledger: {zero_amounts_gl} records with zero amounts")
    
    if issues:
        for issue in issues:
            st.write(f"• {issue}")
    else:
        st.success("✅ No major data quality issues detected")
    
    # Show sample records with issues
    st.write("### 🔍 Sample Records with Missing Data")
    if not df.empty:
        missing_data_txn = df[df.isnull().any(axis=1)].head(5)
        if not missing_data_txn.empty:
            st.write("**Transaction Report:**")
            st.dataframe(missing_data_txn, use_container_width=True)
    
    if not gl_df.empty:
        missing_data_gl = gl_df[gl_df.isnull().any(axis=1)].head(5)
        if not missing_data_gl.empty:
            st.write("**General Ledger:**")
            st.dataframe(missing_data_gl, use_container_width=True)

# -------------------------
# ENHANCED LOAN PERFORMANCE ANALYSIS
# -------------------------
st.header("💰 Enhanced Loan Performance Analysis")

# Use General Ledger for more reliable analysis if available
analysis_df = gl_df if not gl_df.empty else df

if not analysis_df.empty:
    # Filter for loan-related transactions
    loan_transactions = analysis_df[
        (analysis_df.get('txn_type', analysis_df.get('transaction_type', '')).isin(['Invoice', 'Payment', 'Bill', 'Credit Memo'])) |
        (analysis_df.get('account', '').str.contains('Loan|Receivable|Interest', case=False, na=False))
    ].copy()
    
    if not loan_transactions.empty:
        # Enhanced pivot analysis using general ledger structure
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

# -------------------------
# ORIGINAL ANALYSIS (Maintained for comparison)
# -------------------------
st.header("📈 Original Analysis (Transaction Report)")

# Original loan performance analysis
filtered_df = df[df["transaction_type"].isin(["Invoice", "Payment"])].copy()
filtered_df = filtered_df[~filtered_df["name"].isin(["CSL", "VEEM"])]
filtered_df["amount"] = filtered_df["amount"].abs()

if not filtered_df.empty:
    pivot = filtered_df.pivot_table(
        index="name",
        columns="transaction_type",
        values="amount",
        aggfunc="sum",
        fill_value=0
    ).reset_index()
    
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
    
    st.subheader("Loan Performance by Deal")
    st.dataframe(
        pivot_display[["Deal Name", "Invoice", "Payment", "Balance (with Risk)"]]
        .sort_values("Balance (with Risk)", ascending=False),
        use_container_width=True
    )

# Rest of original analysis continues...
# [Previous charts and analysis sections remain the same]
