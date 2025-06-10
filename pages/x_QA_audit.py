# pages/audit.py
from utils.imports import *

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

# ----------------------------
# Page setup
# ----------------------------
st.title("🔍 Data Audit Dashboard")
st.markdown("Quality assurance checks for deal data integrity")

# Load data
df = load_deals()

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
