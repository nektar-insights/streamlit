# pages/audit.py
from utils.imports import *
import numpy as np

# ----------------------------
# DETAILED DATA QUALITY ISSUES
# ----------------------------
st.header("⚠️ Detailed Data Quality Issues")

# Create tabs for different quality checks
quality_tab1, quality_tab2, quality_tab3 = st.tabs(["Missing Data Analysis", "Duplicate Analysis", "Data Consistency"])

with quality_tab1:
    st.subheader("Missing Data Analysis")
    
    # Pipeline deals missing data
    if not df.empty:
        st.write("**Pipeline Deals - Missing Data:**")
        pipeline_missing = df.isnull().sum().reset_index()
        pipeline_missing.columns = ["Column", "Missing Count"]
        pipeline_missing["Total Records"] = len(df)
        pipeline_missing["Missing %"] = (pipeline_missing["Missing Count"] / len(df) * 100).round(2)
        pipeline_missing = pipeline_missing[pipeline_missing["Missing Count"] > 0].sort_values("Missing Count", ascending=False)
        
        if not pipeline_missing.empty:
            st.dataframe(pipeline_missing.style.format({"Missing %": "{:.2f}%"}), use_container_width=True)
        else:
            st.success("✅ No missing data in pipeline deals")
    
    # QBO missing data
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**QBO Transactions - Missing Data:**")
        if not qbo_txn.empty:
            qbo_txn_missing = qbo_txn.isnull().sum().reset_index()
            qbo_txn_missing.columns = ["Column", "Missing Count"]
            qbo_txn_missing["Total Records"] = len(qbo_txn)
            qbo_txn_missing["Missing %"] = (qbo_txn_missing["Missing Count"] / len(qbo_txn) * 100).round(2)
            qbo_txn_missing = qbo_txn_missing[qbo_txn_missing["Missing Count"] > 0].sort_values("Missing Count", ascending=False)
            
            if not qbo_txn_missing.empty:
                st.dataframe(qbo_txn_missing.style.format({"Missing %": "{:.2f}%"}), use_container_width=True)
            else:
                st.success("✅ No missing data in QBO transactions")
        else:
            st.info("No QBO transaction data available")
    
    with col2:
        st.write("**QBO General Ledger - Missing Data:**")
        if not qbo_gl.empty:
            qbo_gl_missing = qbo_gl.isnull().sum().reset_index()
            qbo_gl_missing.columns = ["Column", "Missing Count"]
            qbo_gl_missing["Total Records"] = len(qbo_gl)
            qbo_gl_missing["Missing %"] = (qbo_gl_missing["Missing Count"] / len(qbo_gl) * 100).round(2)
            qbo_gl_missing = qbo_gl_missing[qbo_gl_missing["Missing Count"] > 0].sort_values("Missing Count", ascending=False)
            
            if not qbo_gl_missing.empty:
                st.dataframe(qbo_gl_missing.style.format({"Missing %": "{:.2f}%"}), use_container_width=True)
            else:
                st.success("✅ No missing data in QBO general ledger")
        else:
            st.info("No QBO general ledger data available")

with quality_tab2:
    st.subheader("Duplicate Analysis")
    
    # Check for duplicates in each dataset
    datasets_for_dup_check = [
        ("Pipeline Deals", df, "id"),
        ("QBO Transactions", qbo_txn, "id" if "id" in qbo_txn.columns else None),
        ("QBO General Ledger", qbo_gl, "id" if "id" in qbo_gl.columns else None),
        ("MCA Deals", mca_df, "deal_number")
    ]
    
    for dataset_name, dataset, id_col in datasets_for_dup_check:
        if not dataset.empty and id_col and id_col in dataset.columns:
            duplicates = dataset[dataset[id_col].duplicated(keep=False)]
            duplicate_count = len(duplicates)
            
            col1, col2 = st.columns([3, 1])
            with col1:
                st.write(f"**{dataset_name}:**")
            with col2:
                if duplicate_count > 0:
                    st.error(f"🔴 {duplicate_count} duplicates")
                else:
                    st.success("✅ No duplicates")
            
            if duplicate_count > 0:
                st.write(f"Duplicate {id_col} values:")
                duplicate_ids = duplicates[id_col].unique()
                st.write(duplicate_ids[:10])  # Show first 10
                if len(duplicate_ids) > 10:
                    st.write(f"... and {len(duplicate_ids) - 10} more")

with quality_tab3:
    st.subheader("Data Consistency Checks")
    
    # Date consistency checks
    st.write("**Date Range Consistency:**")
    date_info = []
    
    if not df.empty and "date_created" in df.columns:
        df_dates = df["date_created"].dropna()
        if len(df_dates) > 0:
            date_info.append({
                "Dataset": "Pipeline Deals",
                "Date Field": "date_created",
                "Min Date": df_dates.min().strftime("%Y-%m-%d"),
                "Max Date": df_dates.max().strftime("%Y-%m-%d"),
                "Records with Dates": len(df_dates)
            })
    
    if not qbo_txn.empty and "date" in qbo_txn.columns:
        qbo_txn_dates = qbo_txn["date"].dropna()
        if len(qbo_txn_dates) > 0:
            date_info.append({
                "Dataset": "QBO Transactions",
                "Date Field": "date",
                "Min Date": qbo_txn_dates.min().strftime("%Y-%m-%d"),
                "Max Date": qbo_txn_dates.max().strftime("%Y-%m-%d"),
                "Records with Dates": len(qbo_txn_dates)
            })
    
    if not qbo_gl.empty and "txn_date" in qbo_gl.columns:
        qbo_gl_dates = qbo_gl["txn_date"].dropna()
        if len(qbo_gl_dates) > 0:
            date_info.append({
                "Dataset": "QBO General Ledger",
                "Date Field": "txn_date",
                "Min Date": qbo_gl_dates.min().strftime("%Y-%m-%d"),
                "Max Date": qbo_gl_dates.max().strftime("%Y-%m-%d"),
                "Records with Dates": len(qbo_gl_dates)
            })
    
    if date_info:
        date_df = pd.DataFrame(date_info)
        st.dataframe(date_df, use_container_width=True)
    
    # Amount consistency checks
    st.write("**Amount Field Consistency:**")
    amount_info = []
    
    for dataset_name, dataset, amount_field in [
        ("Pipeline Deals", df, "amount"),
        ("QBO Transactions", qbo_txn, "amount"),
        ("QBO General Ledger", qbo_gl, "amount"),
        ("MCA Deals", mca_df, "amount_hubspot")
    ]:
        if not dataset.empty and amount_field in dataset.columns:
            amounts = dataset[amount_field].dropna()
            if len(amounts) > 0:
                amount_info.append({
                    "Dataset": dataset_name,
                    "Amount Field": amount_field,
                    "Min Amount": f"${amounts.min():,.2f}",
                    "Max Amount": f"${amounts.max():,.2f}",
                    "Zero/Negative": len(amounts[amounts <= 0]),
                    "Records with Amounts": len(amounts)
                })
    
    if amount_info:
        amount_df = pd.DataFrame(amount_info)
        st.dataframe(amount_df, use_container_width=True)

# ----------------------------
# EXPLORATORY DATA ANALYSIS
# ----------------------------
st.header("🔍 Exploratory Data Analysis")

# EDA Tabs for all datasets
tab1, tab2, tab3, tab4, tab5 = st.tabs(["Pipeline Deals", "QBO General Ledger", "QBO Transactions", "MCA Deals", "Cross-Dataset Analysis"])

with tab1:
    st.subheader("Pipeline Deals Analysis")
    
    if not df.empty:
        # Basic stats
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Records", len(df))
        with col2:
            st.metric("Won Deals", len(df[df.get("is_closed_won", False) == True]))
        with col3:
            st.metric("Unique Partners", df["partner_source"].nunique() if "partner_source" in df.columns else 0)
        
        # Group by partner source
        if "partner_source" in df.columns:
            partner_analysis = df.groupby("partner_source").agg({
                "id": "count",
                "amount": ["sum", "mean"] if "amount" in df.columns else "count",
                "is_closed_won": "sum" if "is_closed_won" in df.columns else "count"
            }).round(2)
            
            # Flatten column names
            partner_analysis.columns = ['_'.join(col).strip() if isinstance(col, tuple) else col for col in partner_analysis.columns]
            partner_analysis = partner_analysis.rename(columns={
                "id_count": "Total Deals",
                "amount_sum": "Total Amount",
                "amount_mean": "Avg Amount",
                "is_closed_won_sum": "Won Deals"
            })
            
            st.write("**By Partner Source:**")
            st.dataframe(partner_analysis, use_container_width=True)
        
        # Status analysis
        if "is_closed_won" in df.columns:
            status_counts = df["is_closed_won"].value_counts()
            st.write("**Deal Status:**")
            st.dataframe(pd.DataFrame({"Status": ["Won", "Not Won"], "Count": [status_counts.get(True, 0), status_counts.get(False, 0)]}))
        
        # Null analysis
        st.write("**Data Quality - Null Values:**")
        null_analysis = df.isnull().sum().reset_index()
        null_analysis.columns = ["Column", "Null Count"]
        null_analysis["Null %"] = (null_analysis["Null Count"] / len(df) * 100).round(2)
        st.dataframe(null_analysis[null_analysis["Null Count"] > 0], use_container_width=True)

with tab2:
    st.subheader("QBO General Ledger Analysis")
    
    if not qbo_gl.empty:
        # Basic stats
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Records", len(qbo_gl))
        with col2:
            st.metric("Date Range", f"{qbo_gl['txn_date'].min().strftime('%Y-%m-%d') if qbo_gl['txn_date'].min() else 'N/A'} to {qbo_gl['txn_date'].max().strftime('%Y-%m-%d') if qbo_gl['txn_date'].max() else 'N/A'}")
        with col3:
            st.metric("Total Amount", f"${qbo_gl['amount'].sum():,.2f}" if "amount" in qbo_gl.columns else "N/A")
        
        # Group by transaction type
        if "txn_type" in qbo_gl.columns:
            gl_by_type = qbo_gl.groupby("txn_type")["amount"].agg(["sum", "count", "mean"]).round(2)
            gl_by_type.columns = ["Total Amount", "Transaction Count", "Average Amount"]
            gl_by_type = gl_by_type.sort_values("Total Amount", ascending=False)
            
            st.write("**By Transaction Type:**")
            st.dataframe(gl_by_type.style.format({
                "Total Amount": "${:,.2f}",
                "Average Amount": "${:,.2f}"
            }), use_container_width=True)
            
            # Visualization
            chart_data = gl_by_type.reset_index()
            fig_type = alt.Chart(chart_data).mark_bar().encode(
                x=alt.X("txn_type:N", title="Transaction Type", axis=alt.Axis(labelAngle=-45)),
                y=alt.Y("Total Amount:Q", title="Total Amount ($)", axis=alt.Axis(format="$,.0f")),
                tooltip=["txn_type:N", alt.Tooltip("Total Amount:Q", format="$,.2f")]
            ).properties(
                width=600,
                height=400,
                title="General Ledger: Total Amount by Transaction Type"
            )
            st.altair_chart(fig_type, use_container_width=True)
        
        # Group by name (top 20)
        if "name" in qbo_gl.columns:
            gl_by_name = qbo_gl.groupby("name")["amount"].agg(["sum", "count", "mean"]).round(2)
            gl_by_name.columns = ["Total Amount", "Transaction Count", "Average Amount"]
            gl_by_name = gl_by_name.sort_values("Total Amount", ascending=False).head(20)
            
            st.write("**Top 20 by Name:**")
            st.dataframe(gl_by_name.style.format({
                "Total Amount": "${:,.2f}",
                "Average Amount": "${:,.2f}"
            }), use_container_width=True)

with tab3:
    st.subheader("QBO Transaction Report Analysis")
    
    if not qbo_txn.empty:
        # Basic stats
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Records", len(qbo_txn))
        with col2:
            st.metric("Date Range", f"{qbo_txn['date'].min().strftime('%Y-%m-%d') if qbo_txn['date'].min() else 'N/A'} to {qbo_txn['date'].max().strftime('%Y-%m-%d') if qbo_txn['date'].max() else 'N/A'}")
        with col3:
            st.metric("Total Amount", f"${qbo_txn['amount'].sum():,.2f}" if "amount" in qbo_txn.columns else "N/A")
        
        # Group by transaction type
        if "transaction_type" in qbo_txn.columns:
            txn_by_type = qbo_txn.groupby("transaction_type")["amount"].agg(["sum", "count", "mean"]).round(2)
            txn_by_type.columns = ["Total Amount", "Transaction Count", "Average Amount"]
            txn_by_type = txn_by_type.sort_values("Total Amount", ascending=False)
            
            st.write("**By Transaction Type:**")
            st.dataframe(txn_by_type.style.format({
                "Total Amount": "${:,.2f}",
                "Average Amount": "${:,.2f}"
            }), use_container_width=True)
        
        # Group by name (top 20)
        if "name" in qbo_txn.columns:
            txn_by_name = qbo_txn.groupby("name")["amount"].agg(["sum", "count", "mean"]).round(2)
            txn_by_name.columns = ["Total Amount", "Transaction Count", "Average Amount"]
            txn_by_name = txn_by_name.sort_values("Total Amount", ascending=False).head(20)
            
            st.write("**Top 20 by Name:**")
            st.dataframe(txn_by_name.style.format({
                "Total Amount": "${:,.2f}",
                "Average Amount": "${:,.2f}"
            }), use_container_width=True)

with tab4:
    st.subheader("MCA Deals Analysis")
    
    if not mca_df.empty:
        # Basic stats
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Records", len(mca_df))
        with col2:
            st.metric("Unique Deal Numbers", mca_df["deal_number"].nunique() if "deal_number" in mca_df.columns else 0)
        with col3:
            total_amount = mca_df["amount_hubspot"].sum() if "amount_hubspot" in mca_df.columns else 0
            st.metric("Total Amount", f"${total_amount:,.2f}")
        
        # Status analysis
        if "status_category" in mca_df.columns:
            status_counts = mca_df["status_category"].value_counts()
            st.write("**By Status Category:**")
            st.dataframe(pd.DataFrame({"Status": status_counts.index, "Count": status_counts.values}))
        
        # Null analysis
        st.write("**Data Quality - Null Values:**")
        null_analysis = mca_df.isnull().sum().reset_index()
        null_analysis.columns = ["Column", "Null Count"]
        null_analysis["Null %"] = (null_analysis["Null Count"] / len(mca_df) * 100).round(2)
        st.dataframe(null_analysis[null_analysis["Null Count"] > 0], use_container_width=True)

with tab5:
    st.subheader("Cross-Dataset Analysis")
    
    # Dataset comparison
    datasets = {
        "Pipeline Deals": df,
        "QBO Transactions": qbo_txn,
        "QBO General Ledger": qbo_gl,
        "MCA Deals": mca_df
    }
    
    comparison_data = []
    for name, dataset in datasets.items():
        if not dataset.empty:
            comparison_data.append({
                "Dataset": name,
                "Records": len(dataset),
                "Columns": len(dataset.columns),
                "Null Values": dataset.isnull().sum().sum(),
                "Memory Usage (MB)": round(dataset.memory_usage(deep=True).sum() / 1024 / 1024, 2)
            })
    
    if comparison_data:
        comparison_df = pd.DataFrame(comparison_data)
        st.write("**Dataset Comparison:**")
        st.dataframe(comparison_df, use_container_width=True)
    
    # Common name analysis between datasets
    st.write("**Name/Deal Overlap Analysis:**")
    
    # Get names from each dataset
    pipeline_names = set(df["deal_name"].dropna().unique()) if "deal_name" in df.columns else set()
    qbo_txn_names = set(qbo_txn["name"].dropna().unique()) if "name" in qbo_txn.columns else set()
    qbo_gl_names = set(qbo_gl["name"].dropna().unique()) if "name" in qbo_gl.columns else set()
    mca_names = set(mca_df["dba"].dropna().unique()) if "dba" in mca_df.columns else set()
    
    overlap_data = []
    if pipeline_names:
        overlap_data.append({"Dataset": "Pipeline Deals", "Unique Names": len(pipeline_names)})
    if qbo_txn_names:
        overlap_data.append({"Dataset": "QBO Transactions", "Unique Names": len(qbo_txn_names)})
    if qbo_gl_names:
        overlap_data.append({"Dataset": "QBO General Ledger", "Unique Names": len(qbo_gl_names)})
    if mca_names:
        overlap_data.append({"Dataset": "MCA Deals", "Unique Names": len(mca_names)})
    
    if overlap_data:
        overlap_df = pd.DataFrame(overlap_data)
        st.dataframe(overlap_df, use_container_width=True)
    
    # Show overlaps between specific datasets
    if qbo_txn_names and qbo_gl_names:
        common_qbo = qbo_txn_names.intersection(qbo_gl_names)
        st.write(f"**QBO Common Names:** {len(common_qbo)} names appear in both transaction report and general ledger")

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

@st.cache_data(ttl=3600)
def load_mca_deals():
    res = supabase.table("mca_deals").select("*").execute()
    return pd.DataFrame(res.data)

# ----------------------------
# Data preprocessing helper
# ----------------------------
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

# ----------------------------
# Page setup
# ----------------------------
st.title("🔍 Data Audit Dashboard")
st.markdown("Quality assurance checks for deal data integrity")

# Load all data
df = load_deals()
qbo_txn, qbo_gl = load_qbo_data()
mca_df = load_mca_deals()

# Preprocess QBO data
qbo_txn = preprocess_data(qbo_txn)
qbo_gl = preprocess_data(qbo_gl)

# Debug: Check what columns are available
st.subheader("🔍 Debug Info")
with st.expander("Available Columns in Dataset", expanded=True):
    st.write("**Data source:** `deals` table from Supabase")
    st.write(f"**Total rows:** {len(df)}")
    st.write("**Available columns:**")
    st.dataframe(pd.DataFrame({"Column Name": df.columns, "Data Type": df.dtypes.astype(str)}), use_container_width=True)
    
    st.write("**Sample data (first 3 rows):**")
    if len(df) > 0:
        st.dataframe(df.head(3), use_container_width=True)
    else:
        st.error("⚠️ No data available in deals table!")
    
    # Check for deals with is_closed_won = True
    won_deals_count = len(df[df["is_closed_won"] == True]) if "is_closed_won" in df.columns else 0
    st.write(f"**Won deals count:** {won_deals_count}")
    
    # Check for loan_id field specifically
    if "loan_id" in df.columns:
        null_loan_ids = df["loan_id"].isna().sum()
        empty_loan_ids = (df["loan_id"] == "").sum() if df["loan_id"].dtype == 'object' else 0
        st.write(f"**Loan ID field status:** Found! Null values: {null_loan_ids}, Empty values: {empty_loan_ids}")
    else:
        st.error("⚠️ No 'loan_id' column found!")

# Convert date column
df["date_created"] = pd.to_datetime(df["date_created"], errors="coerce")

# ----------------------------
# QA Check 1: Missing Loan IDs in Won Deals
# ----------------------------
st.header("1. Missing Loan IDs in Won Deals")

# Check if required columns exist
if "is_closed_won" not in df.columns:
    st.error("⚠️ Column 'is_closed_won' not found in deals table!")
    st.stop()

if "loan_id" not in df.columns:
    st.error("⚠️ Column 'loan_id' not found in deals table!")
    st.stop()

# Filter for won deals
won_deals = df[df["is_closed_won"] == True].copy()

st.write(f"**Debug:** Found {len(won_deals)} won deals out of {len(df)} total deals")

# Check for missing loan IDs (null, empty, or NaN)
missing_loan_ids = won_deals[
    (won_deals["loan_id"].isna()) | 
    (won_deals["loan_id"] == "") |
    (won_deals["loan_id"].astype(str).str.strip() == "")
].copy()

st.write(f"**Debug:** Found {len(missing_loan_ids)} deals missing loan IDs")

# Display summary metrics
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Total Won Deals", len(won_deals))
    
with col2:
    st.metric("Missing Loan IDs", len(missing_loan_ids))
    
with col3:
    missing_pct = (len(missing_loan_ids) / len(won_deals) * 100) if len(won_deals) > 0 else 0
    st.metric("Missing Rate", f"{missing_pct:.1f}%")

# Show status
if len(missing_loan_ids) == 0:
    st.success("✅ All won deals have loan IDs assigned!")
else:
    st.warning(f"⚠️ Found {len(missing_loan_ids)} won deals missing loan IDs")

# Display detailed table of missing loan IDs
if len(missing_loan_ids) > 0:
    st.subheader("Deals Missing Loan IDs")
    
    # Select relevant columns for display - try different possible deal name fields
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
        display_columns.insert(1, deal_name_field)  # Insert after id
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
    
    # Rename columns for better display - handle dynamic deal name field
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
    
    # Add the deal name field to rename mapping if it exists
    if deal_name_field:
        column_rename[deal_name_field] = "Deal Name"
    
    display_df = display_df.rename(columns=column_rename)
    
    st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True
    )
    
    # Download option for missing loan IDs
    csv_data = missing_loan_ids[available_columns].to_csv(index=False).encode("utf-8")
    st.download_button(
        label="📥 Download Missing Loan IDs as CSV",
        data=csv_data,
        file_name=f"missing_loan_ids_{pd.Timestamp.now().strftime('%Y%m%d')}.csv",
        mime="text/csv"
    )

# ----------------------------
# Additional QA Checks Section
# ----------------------------
st.header("2. Additional Data Quality Checks")

# Check for duplicates
st.subheader("Duplicate Check")
duplicate_loan_ids = df[df["loan_id"].notna() & (df["loan_id"] != "")]["loan_id"].duplicated().sum()
col1, col2 = st.columns(2)
with col1:
    st.metric("Duplicate Loan IDs", duplicate_loan_ids)
with col2:
    if duplicate_loan_ids == 0:
        st.success("✅ No duplicate loan IDs found")
    else:
        st.error(f"❌ Found {duplicate_loan_ids} duplicate loan IDs")

# Check for missing critical fields in won deals
st.subheader("Missing Critical Fields in Won Deals")
critical_fields = ["amount", "factor_rate", "loan_term", "commission"]
existing_critical_fields = [field for field in critical_fields if field in won_deals.columns]

if existing_critical_fields:
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
# Recent Activity Summary
# ----------------------------
st.header("3. Recent Activity Summary")

# Last 30 days activity
recent_cutoff = pd.Timestamp.now() - pd.Timedelta(days=30)
recent_deals = df[df["date_created"] >= recent_cutoff]
recent_won = recent_deals[recent_deals["is_closed_won"] == True]

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Recent Deals (30d)", len(recent_deals))
with col2:
    st.metric("Recent Won Deals", len(recent_won))
with col3:
    recent_missing_ids = recent_won[
        (recent_won["loan_id"].isna()) | 
        (recent_won["loan_id"] == "") |
        (recent_won["loan_id"].astype(str).str.strip() == "")
    ]
    st.metric("Recent Missing IDs", len(recent_missing_ids))
with col4:
    recent_close_rate = (len(recent_won) / len(recent_deals) * 100) if len(recent_deals) > 0 else 0
    st.metric("Recent Close Rate", f"{recent_close_rate:.1f}%")

# ----------------------------
# Data Freshness Check
# ----------------------------
st.header("4. Data Freshness")
if len(df) > 0:
    latest_deal = df["date_created"].max()
    days_since_last = (pd.Timestamp.now() - latest_deal).days
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Latest Deal Date", latest_deal.strftime("%Y-%m-%d"))
    with col2:
        st.metric("Days Since Last Deal", days_since_last)
        
    if days_since_last > 7:
        st.warning(f"⚠️ It's been {days_since_last} days since the last deal was recorded")
    else:
        st.success("✅ Data appears current")

# ----------------------------
# Cache clearing functions for specific dashboards
# ----------------------------
def clear_pipeline_cache():
    """Clear cache for pipeline dashboard"""
    # Target the specific cache function from streamlit_app.py
    if hasattr(st.session_state, '_cache'):
        for key in list(st.session_state._cache.keys()):
            if 'load_deals' in key:
                del st.session_state._cache[key]
    st.cache_data.clear()

def clear_mca_cache():
    """Clear cache for MCA dashboard"""
    # Target the specific cache function from mca_dashboard.py
    if hasattr(st.session_state, '_cache'):
        for key in list(st.session_state._cache.keys()):
            if 'load_mca_deals' in key or 'combine_deals' in key:
                del st.session_state._cache[key]
    st.cache_data.clear()

def clear_qbo_cache():
    """Clear cache for QBO dashboard"""
    # Target the specific cache function from qbo_dashboard.py
    if hasattr(st.session_state, '_cache'):
        for key in list(st.session_state._cache.keys()):
            if 'load_qbo_data' in key:
                del st.session_state._cache[key]
    st.cache_data.clear()

# ----------------------------
# Data Management & Cache Refresh
# ----------------------------
st.header("5. Data Management")

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
