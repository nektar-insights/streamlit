# pages/comprehensive_audit.py
from utils.imports import *
import numpy as np
from scripts.combine_hubspot_mca import combine_deals

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
    df_txn = pd.DataFrame(supabase.table("qbo_invoice_payments").select("*").execute().data)
    df_gl = pd.DataFrame(supabase.table("qbo_general_ledger").select("*").execute().data)
    return df_txn, df_gl

@st.cache_data(ttl=3600)
def load_mca_deals():
    """Load MCA deals from Supabase"""
    res = supabase.table("mca_deals").select("*").execute()
    return pd.DataFrame(res.data)

@st.cache_data(ttl=3600)
def load_combined_mca_deals():
    """Load combined MCA deals using the combine_deals function"""
    return combine_deals()

def preprocess_data(dataframe):
    """Clean and preprocess dataframe"""
    df_clean = dataframe.copy()
    
    # Handle numeric columns (updated for new schema)
    numeric_cols = ['total_amount', 'balance', 'debit', 'credit', 'amount', 'purchase_price', 'receivables_amount', 
                   'current_balance', 'past_due_amount', 'principal_amount', 'rtr_balance', 
                   'amount_hubspot', 'total_funded_amount']
    for col in numeric_cols:
        if col in df_clean.columns:
            df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
    
    # Handle date columns (updated for new schema)
    date_cols = ['txn_date', 'due_date', 'date', 'date_created', 'funding_date', 'created_time', 'last_updated_time']
    for col in date_cols:
        if col in df_clean.columns:
            df_clean[col] = pd.to_datetime(df_clean[col], errors='coerce')
    
    return df_clean

# Load all data
deals_df = load_deals()
qbo_txn_df, qbo_gl_df = load_qbo_data()
mca_deals_raw = load_mca_deals()
mca_deals_combined = load_combined_mca_deals()

# Preprocess all datasets
deals_df = preprocess_data(deals_df)
qbo_txn_df = preprocess_data(qbo_txn_df)
qbo_gl_df = preprocess_data(qbo_gl_df)
mca_deals_raw = preprocess_data(mca_deals_raw)
mca_deals_combined = preprocess_data(mca_deals_combined)

# ----------------------------
# Page setup
# ----------------------------
st.title("Comprehensive Data Audit Dashboard")
st.markdown("Complete quality assurance checks for deal data integrity, MCA deals, and QBO financial analysis")

# ----------------------------
# Executive Summary
# ----------------------------
st.header("Executive Summary")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Total Deals", len(deals_df))
    won_deals_count = len(deals_df[deals_df["is_closed_won"] == True]) if "is_closed_won" in deals_df.columns else 0
    st.metric("Won Deals", won_deals_count)

with col2:
    st.metric("QBO Invoice/Payments", len(qbo_txn_df))
    st.metric("QBO GL Entries", len(qbo_gl_df))

with col3:
    st.metric("MCA Deals (Raw)", len(mca_deals_raw))
    st.metric("MCA Deals (Combined)", len(mca_deals_combined))

with col4:
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

# Data freshness indicators
st.subheader("Data Freshness")
col1, col2, col3 = st.columns(3)

with col1:
    if len(deals_df) > 0 and 'date_created' in deals_df.columns:
        latest_deal = deals_df["date_created"].max()
        days_since_last = (pd.Timestamp.now() - latest_deal).days
        st.metric("Days Since Last Deal", days_since_last)
    else:
        st.metric("Days Since Last Deal", "N/A")

with col2:
    if len(mca_deals_raw) > 0 and 'funding_date' in mca_deals_raw.columns:
        latest_mca = mca_deals_raw["funding_date"].max()
        mca_days_since = (pd.Timestamp.now() - latest_mca).days
        st.metric("Days Since Last MCA Deal", mca_days_since)
    else:
        st.metric("Days Since Last MCA Deal", "N/A")

with col3:
    # QBO data freshness
    qbo_latest = None
    if len(qbo_gl_df) > 0 and 'txn_date' in qbo_gl_df.columns:
        qbo_latest = qbo_gl_df["txn_date"].max()
    elif len(qbo_txn_df) > 0 and 'txn_date' in qbo_txn_df.columns:
        qbo_latest = qbo_txn_df["txn_date"].max()
    
    if qbo_latest:
        qbo_days_since = (pd.Timestamp.now() - qbo_latest).days
        st.metric("Days Since Last QBO Entry", qbo_days_since)
    else:
        st.metric("Days Since Last QBO Entry", "N/A")

# ----------------------------
# MCA DEALS AUDIT SECTION
# ----------------------------
st.header("🏦 MCA Deals Audit")

# MCA Data Quality Overview
col1, col2 = st.columns(2)

with col1:
    st.subheader("Raw MCA Data Quality")
    if len(mca_deals_raw) > 0:
        st.metric("Total Records", len(mca_deals_raw))
        if 'funding_date' in mca_deals_raw.columns:
            funding_dates = mca_deals_raw['funding_date'].dropna()
            if len(funding_dates) > 0:
                date_range = f"{funding_dates.min().strftime('%m/%d/%y')} to {funding_dates.max().strftime('%m/%d/%y')}"
                st.metric("Funding Date Range", date_range)
        st.metric("Null Values", mca_deals_raw.isnull().sum().sum())
        
        # Check for duplicate deal numbers
        if 'deal_number' in mca_deals_raw.columns:
            duplicate_deals = mca_deals_raw['deal_number'].duplicated().sum()
            st.metric("Duplicate Deal Numbers", duplicate_deals)
    else:
        st.warning("No MCA deals data available")

with col2:
    st.subheader("Combined MCA Data Quality")
    if len(mca_deals_combined) > 0:
        st.metric("Total Records", len(mca_deals_combined))
        if 'funding_date' in mca_deals_combined.columns:
            funding_dates = mca_deals_combined['funding_date'].dropna()
            if len(funding_dates) > 0:
                date_range = f"{funding_dates.min().strftime('%m/%d/%y')} to {funding_dates.max().strftime('%m/%d/%y')}"
                st.metric("Funding Date Range", date_range)
        st.metric("Null Values", mca_deals_combined.isnull().sum().sum())
        
        # Status category distribution
        if 'status_category' in mca_deals_combined.columns:
            status_counts = mca_deals_combined['status_category'].value_counts()
            st.write("**Status Distribution:**")
            for status, count in status_counts.items():
                st.write(f"• {status}: {count}")
    else:
        st.warning("No combined MCA deals data available")

# MCA Debug Info Expander
with st.expander("MCA Data Debug Info", expanded=False):
    st.write("**Raw MCA Data source:** `mca_deals` table from Supabase")
    st.write(f"**Total raw rows:** {len(mca_deals_raw)}")
    if len(mca_deals_raw) > 0:
        st.write("**Available columns in raw data:**")
        st.dataframe(pd.DataFrame({"Column Name": mca_deals_raw.columns, "Data Type": mca_deals_raw.dtypes.astype(str)}), use_container_width=True)
        
        st.write("**Sample raw data (first 3 rows):**")
        st.dataframe(mca_deals_raw.head(3), use_container_width=True)
    
    st.write("---")
    st.write("**Combined MCA Data source:** `combine_deals()` function")
    st.write(f"**Total combined rows:** {len(mca_deals_combined)}")
    if len(mca_deals_combined) > 0:
        st.write("**Available columns in combined data:**")
        st.dataframe(pd.DataFrame({"Column Name": mca_deals_combined.columns, "Data Type": mca_deals_combined.dtypes.astype(str)}), use_container_width=True)
        
        st.write("**Sample combined data (first 3 rows):**")
        st.dataframe(mca_deals_combined.head(3), use_container_width=True)

# MCA QA Checks
st.subheader("MCA Data Quality Checks")

# QA Check 1: Missing Critical Fields
st.write("### 1. Missing Critical Fields in MCA Deals")

critical_mca_fields = ['deal_number', 'dba', 'funding_date', 'purchase_price', 'total_funded_amount', 'status_category']
missing_critical_mca = []

for field in critical_mca_fields:
    if field in mca_deals_combined.columns:
        missing_count = mca_deals_combined[field].isna().sum()
        missing_critical_mca.append({
            "Field": field.replace("_", " ").title(),
            "Missing Count": missing_count,
            "Missing %": f"{(missing_count / len(mca_deals_combined) * 100):.1f}%" if len(mca_deals_combined) > 0 else "0.0%"
        })
    else:
        missing_critical_mca.append({
            "Field": field.replace("_", " ").title(),
            "Missing Count": "FIELD NOT FOUND",
            "Missing %": "N/A"
        })

critical_mca_df = pd.DataFrame(missing_critical_mca)
st.dataframe(critical_mca_df, use_container_width=True, hide_index=True)

# QA Check 2: Financial Data Validation
st.write("### 2. Financial Data Validation")

col1, col2 = st.columns(2)

with col1:
    st.write("**Purchase Price vs Total Funded Analysis**")
    if 'purchase_price' in mca_deals_combined.columns and 'total_funded_amount' in mca_deals_combined.columns:
        # Check for mismatches between purchase price and total funded
        valid_comparison = mca_deals_combined[
            (mca_deals_combined['purchase_price'].notna()) & 
            (mca_deals_combined['total_funded_amount'].notna()) &
            (mca_deals_combined['purchase_price'] > 0) &
            (mca_deals_combined['total_funded_amount'] > 0)
        ].copy()
        
        if len(valid_comparison) > 0:
            valid_comparison['price_difference'] = abs(valid_comparison['purchase_price'] - valid_comparison['total_funded_amount'])
            valid_comparison['price_diff_pct'] = valid_comparison['price_difference'] / valid_comparison['total_funded_amount']
            
            significant_differences = valid_comparison[valid_comparison['price_diff_pct'] > 0.05]  # > 5% difference
            
            st.metric("Deals with Price Mismatches (>5%)", len(significant_differences))
            st.metric("Total Deals Comparable", len(valid_comparison))
            
            if len(significant_differences) > 0:
                st.write("**Deals with Significant Price Differences:**")
                price_diff_display = significant_differences[['deal_number', 'dba', 'purchase_price', 'total_funded_amount', 'price_diff_pct']].copy()
                price_diff_display['price_diff_pct'] = price_diff_display['price_diff_pct'].apply(lambda x: f"{x:.1%}")
                price_diff_display['purchase_price'] = price_diff_display['purchase_price'].apply(lambda x: f"${x:,.0f}")
                price_diff_display['total_funded_amount'] = price_diff_display['total_funded_amount'].apply(lambda x: f"${x:,.0f}")
                
                st.dataframe(price_diff_display.rename(columns={
                    'deal_number': 'Deal Number',
                    'dba': 'Business Name',
                    'purchase_price': 'Purchase Price',
                    'total_funded_amount': 'Total Funded',
                    'price_diff_pct': 'Difference %'
                }), use_container_width=True, hide_index=True)
        else:
            st.info("No valid price comparisons available")
    else:
        st.warning("Purchase price or total funded amount fields not available")

with col2:
    st.write("**Balance Validation**")
    if 'current_balance' in mca_deals_combined.columns and 'past_due_amount' in mca_deals_combined.columns:
        balance_issues = mca_deals_combined[
            (mca_deals_combined['past_due_amount'] > mca_deals_combined['current_balance']) &
            (mca_deals_combined['past_due_amount'].notna()) &
            (mca_deals_combined['current_balance'].notna()) &
            (mca_deals_combined['current_balance'] > 0)
        ]
        
        st.metric("Past Due > Current Balance", len(balance_issues))
        
        # Negative balance check
        negative_balances = mca_deals_combined[
            (mca_deals_combined['current_balance'] < 0) &
            (mca_deals_combined['current_balance'].notna())
        ]
        st.metric("Negative Current Balances", len(negative_balances))
        
        # Zero balances but not matured
        if 'status_category' in mca_deals_combined.columns:
            zero_balance_not_matured = mca_deals_combined[
                (mca_deals_combined['current_balance'] == 0) &
                (mca_deals_combined['status_category'] != 'Matured') &
                (mca_deals_combined['current_balance'].notna())
            ]
            st.metric("Zero Balance (Not Matured)", len(zero_balance_not_matured))
    else:
        st.warning("Balance fields not available for validation")

# QA Check 3: Status Category Analysis
st.write("### 3. Status Category Analysis")

if 'status_category' in mca_deals_combined.columns:
    status_analysis = mca_deals_combined['status_category'].value_counts()
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.write("**Status Distribution:**")
        status_df = pd.DataFrame({
            'Status': status_analysis.index,
            'Count': status_analysis.values,
            'Percentage': (status_analysis.values / len(mca_deals_combined) * 100).round(1)
        })
        st.dataframe(status_df, use_container_width=True, hide_index=True)
    
    with col2:
        # Check for unusual status patterns
        st.write("**Status Validation:**")
        
        # Matured deals with balance
        if 'current_balance' in mca_deals_combined.columns:
            matured_with_balance = mca_deals_combined[
                (mca_deals_combined['status_category'] == 'Matured') &
                (mca_deals_combined['current_balance'] > 0) &
                (mca_deals_combined['current_balance'].notna())
            ]
            st.metric("Matured with Balance > 0", len(matured_with_balance))
        
        # Current deals with past due
        if 'past_due_amount' in mca_deals_combined.columns:
            current_with_past_due = mca_deals_combined[
                (mca_deals_combined['status_category'] == 'Current') &
                (mca_deals_combined['past_due_amount'] > 0) &
                (mca_deals_combined['past_due_amount'].notna())
            ]
            st.metric("Current with Past Due", len(current_with_past_due))

# QA Check 4: Missing Deals Analysis (Raw vs Combined)
st.write("### 4. Missing Deals Analysis (Raw vs Combined)")

if len(mca_deals_raw) > 0 and len(mca_deals_combined) > 0:
    # Compare deal numbers between raw and combined datasets
    if 'deal_number' in mca_deals_raw.columns and 'deal_number' in mca_deals_combined.columns:
        raw_deal_numbers = set(mca_deals_raw['deal_number'].dropna().astype(str))
        combined_deal_numbers = set(mca_deals_combined['deal_number'].dropna().astype(str))
        
        # Deals in raw but not in combined
        missing_from_combined = raw_deal_numbers - combined_deal_numbers
        
        # Deals in combined but not in raw (shouldn't happen, but good to check)
        extra_in_combined = combined_deal_numbers - raw_deal_numbers
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Deals Missing from Combined Dataset:**")
            st.metric("Deals in Raw but Not Combined", len(missing_from_combined))
            st.metric("Total Raw Deals", len(raw_deal_numbers))
            
            if len(missing_from_combined) > 0:
                missing_pct = (len(missing_from_combined) / len(raw_deal_numbers)) * 100
                st.metric("Missing Percentage", f"{missing_pct:.1f}%")
                
                # Show details of missing deals
                missing_deals_df = mca_deals_raw[
                    mca_deals_raw['deal_number'].astype(str).isin(missing_from_combined)
                ].copy()
                
                if len(missing_deals_df) > 0:
                    st.write("**Details of Missing Deals:**")
                    
                    # Select relevant columns for display
                    display_cols = ['deal_number', 'dba', 'funding_date', 'status_category', 
                                  'purchase_price', 'total_funded_amount']
                    available_cols = [col for col in display_cols if col in missing_deals_df.columns]
                    
                    missing_display = missing_deals_df[available_cols].copy()
                    
                    # Format for display
                    if 'funding_date' in missing_display.columns:
                        missing_display['funding_date'] = pd.to_datetime(missing_display['funding_date']).dt.strftime('%Y-%m-%d')
                    
                    for col in ['purchase_price', 'total_funded_amount']:
                        if col in missing_display.columns:
                            missing_display[col] = missing_display[col].apply(
                                lambda x: f"${x:,.0f}" if pd.notna(x) else "N/A"
                            )
                    
                    # Rename columns for display
                    column_rename = {
                        'deal_number': 'Deal Number',
                        'dba': 'Business Name',
                        'funding_date': 'Funding Date',
                        'status_category': 'Status',
                        'purchase_price': 'Purchase Price',
                        'total_funded_amount': 'Total Funded'
                    }
                    
                    missing_display = missing_display.rename(columns=column_rename)
                    st.dataframe(missing_display, use_container_width=True, hide_index=True)
                    
                    # Analyze patterns in missing deals
                    st.write("**Analysis of Missing Deals:**")
                    
                    # Status analysis
                    if 'status_category' in missing_deals_df.columns:
                        status_counts = missing_deals_df['status_category'].value_counts()
                        st.write("*Status distribution of missing deals:*")
                        for status, count in status_counts.items():
                            st.write(f"• {status}: {count}")
                    
                    # Date analysis
                    if 'funding_date' in missing_deals_df.columns:
                        funding_dates = pd.to_datetime(missing_deals_df['funding_date'], errors='coerce')
                        if funding_dates.notna().any():
                            earliest = funding_dates.min()
                            latest = funding_dates.max()
                            st.write(f"*Funding date range of missing deals:* {earliest.strftime('%Y-%m-%d')} to {latest.strftime('%Y-%m-%d')}")
                    
                    # Amount analysis
                    if 'purchase_price' in missing_deals_df.columns:
                        purchase_prices = pd.to_numeric(missing_deals_df['purchase_price'], errors='coerce')
                        if purchase_prices.notna().any():
                            avg_amount = purchase_prices.mean()
                            total_amount = purchase_prices.sum()
                            st.write(f"*Average deal size:* ${avg_amount:,.0f}")
                            st.write(f"*Total missing volume:* ${total_amount:,.0f}")
                    
                    # Download option for missing deals
                    csv_data = missing_deals_df[available_cols].to_csv(index=False).encode("utf-8")
                    st.download_button(
                        label="📥 Download Missing Deals List",
                        data=csv_data,
                        file_name=f"missing_deals_raw_vs_combined_{pd.Timestamp.now().strftime('%Y%m%d')}.csv",
                        mime="text/csv"
                    )
            else:
                st.success("✅ All raw deals found in combined dataset")
        
        with col2:
            st.write("**Data Processing Validation:**")
            st.metric("Deals in Combined but Not Raw", len(extra_in_combined))
            
            if len(extra_in_combined) > 0:
                st.warning(f"⚠️ Found {len(extra_in_combined)} deals in combined that aren't in raw data")
                st.write("*This could indicate data processing issues*")
                
                # Show extra deals
                extra_deals_df = mca_deals_combined[
                    mca_deals_combined['deal_number'].astype(str).isin(extra_in_combined)
                ]
                
                if len(extra_deals_df) > 0:
                    st.write("**Extra Deals in Combined:**")
                    extra_display_cols = ['deal_number', 'dba', 'funding_date']
                    extra_available_cols = [col for col in extra_display_cols if col in extra_deals_df.columns]
                    st.dataframe(extra_deals_df[extra_available_cols], use_container_width=True, hide_index=True)
            else:
                st.success("✅ No unexpected deals in combined dataset")
            
            # Processing efficiency metrics
            processing_efficiency = (len(combined_deal_numbers) / len(raw_deal_numbers)) * 100 if len(raw_deal_numbers) > 0 else 0
            st.metric("Processing Efficiency", f"{processing_efficiency:.1f}%")
            
            # Recommendations based on missing deals
            if len(missing_from_combined) > 0:
                st.write("**🔍 Investigation Recommendations:**")
                
                # Check if missing deals have common characteristics
                if 'status_category' in missing_deals_df.columns:
                    status_counts = missing_deals_df['status_category'].value_counts()
                    most_common_status = status_counts.index[0] if len(status_counts) > 0 else "Unknown"
                    
                    if status_counts.iloc[0] / len(missing_deals_df) > 0.7:  # If >70% have same status
                        st.write(f"• Most missing deals have status: **{most_common_status}**")
                        st.write("• Check if combine_deals() filters this status")
                
                # Check for data quality issues in missing deals
                if 'purchase_price' in missing_deals_df.columns:
                    null_prices = missing_deals_df['purchase_price'].isna().sum()
                    if null_prices > len(missing_deals_df) * 0.5:  # If >50% have null prices
                        st.write("• Many missing deals have null purchase prices")
                        st.write("• Check if combine_deals() requires valid amounts")
                
                if 'funding_date' in missing_deals_df.columns:
                    null_dates = pd.to_datetime(missing_deals_df['funding_date'], errors='coerce').isna().sum()
                    if null_dates > len(missing_deals_df) * 0.5:  # If >50% have null dates
                        st.write("• Many missing deals have invalid funding dates")
                        st.write("• Check if combine_deals() requires valid dates")
    
    else:
        st.warning("⚠️ Cannot compare datasets - deal_number column missing in one or both datasets")

elif len(mca_deals_raw) == 0:
    st.warning("⚠️ No raw MCA data available for comparison")
elif len(mca_deals_combined) == 0:
    st.warning("⚠️ No combined MCA data available for comparison")

# QA Check 5: Date Validation
st.write("### 5. Date Validation")

if 'funding_date' in mca_deals_combined.columns:
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Funding Date Issues:**")
        
        # Future funding dates
        future_funding = mca_deals_combined[
            (mca_deals_combined['funding_date'] > pd.Timestamp.now()) &
            (mca_deals_combined['funding_date'].notna())
        ]
        st.metric("Future Funding Dates", len(future_funding))
        
        # Very old funding dates (>10 years)
        very_old_funding = mca_deals_combined[
            (mca_deals_combined['funding_date'] < pd.Timestamp.now() - pd.Timedelta(days=3650)) &
            (mca_deals_combined['funding_date'].notna())
        ]
        st.metric("Very Old Funding (>10 years)", len(very_old_funding))
        
        # Missing funding dates
        missing_funding_dates = mca_deals_combined['funding_date'].isna().sum()
        st.metric("Missing Funding Dates", missing_funding_dates)
    
    with col2:
        if len(mca_deals_combined[mca_deals_combined['funding_date'].notna()]) > 0:
            st.write("**Funding Date Range:**")
            min_date = mca_deals_combined['funding_date'].min()
            max_date = mca_deals_combined['funding_date'].max()
            st.write(f"Earliest: {min_date.strftime('%Y-%m-%d') if pd.notna(min_date) else 'N/A'}")
            st.write(f"Latest: {max_date.strftime('%Y-%m-%d') if pd.notna(max_date) else 'N/A'}")
            
            # Recent activity (last 30 days)
            recent_cutoff = pd.Timestamp.now() - pd.Timedelta(days=30)
            recent_deals = mca_deals_combined[mca_deals_combined['funding_date'] >= recent_cutoff]
            st.metric("Recent Deals (30 days)", len(recent_deals))

# Full MCA Deals List for Scraper Verification
st.subheader("Complete MCA Deals List (Scraper Verification)")
st.caption("Use this section to verify all deals are being captured by your scraper")

# Create comprehensive deals list
if len(mca_deals_combined) > 0:
    # Select key columns for verification
    verification_columns = ['deal_number', 'dba', 'funding_date', 'status_category', 'purchase_price', 
                           'total_funded_amount', 'current_balance', 'past_due_amount']
    existing_verification_columns = [col for col in verification_columns if col in mca_deals_combined.columns]
    
    verification_df = mca_deals_combined[existing_verification_columns].copy()
    
    # Format for display
    if 'funding_date' in verification_df.columns:
        verification_df['funding_date'] = verification_df['funding_date'].dt.strftime('%Y-%m-%d')
    
    for col in ['purchase_price', 'total_funded_amount', 'current_balance', 'past_due_amount']:
        if col in verification_df.columns:
            verification_df[col] = verification_df[col].apply(lambda x: f"${x:,.0f}" if pd.notna(x) else "N/A")
    
    # Rename columns for display
    column_rename = {
        'deal_number': 'Deal Number',
        'dba': 'Business Name',
        'funding_date': 'Funding Date',
        'status_category': 'Status',
        'purchase_price': 'Purchase Price',
        'total_funded_amount': 'Total Funded',
        'current_balance': 'Current Balance',
        'past_due_amount': 'Past Due Amount'
    }
    
    verification_df = verification_df.rename(columns=column_rename)
    
    # Add search functionality
    search_term = st.text_input("Search deals (by deal number or business name):", "")
    if search_term:
        mask = (
            verification_df['Deal Number'].astype(str).str.contains(search_term, case=False, na=False) |
            verification_df['Business Name'].astype(str).str.contains(search_term, case=False, na=False)
        )
        verification_df = verification_df[mask]
    
    # Display the table
    st.dataframe(verification_df, use_container_width=True, hide_index=True)
    
    # Summary statistics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Deals Displayed", len(verification_df))
    with col2:
        if 'Status' in verification_df.columns:
            active_deals = len(verification_df[verification_df['Status'].isin(['Current', 'Not Current'])])
            st.metric("Active Deals", active_deals)
    with col3:
        if 'Status' in verification_df.columns:
            matured_deals = len(verification_df[verification_df['Status'] == 'Matured'])
            st.metric("Matured Deals", matured_deals)
    
    # Download option for verification
    csv_data = mca_deals_combined[existing_verification_columns].to_csv(index=False).encode("utf-8")
    st.download_button(
        label="📥 Download Complete MCA Deals List",
        data=csv_data,
        file_name=f"mca_deals_complete_{pd.Timestamp.now().strftime('%Y%m%d')}.csv",
        mime="text/csv"
    )

# ----------------------------
# DEAL DATA AUDIT SECTION (Original)
# ----------------------------
st.header("📋 Deal Data Audit")

# Debug Info Expander
with st.expander("Deal Data Debug Info", expanded=False):
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
        st.success("All won deals have loan IDs assigned!")
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
            st.info(f"Using '{deal_name_field}' for deal names")
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
            label="Download Missing Loan IDs as CSV",
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
        # Get deals with non-null, non-empty loan IDs
        valid_loan_ids = deals_df[deals_df["loan_id"].notna() & (deals_df["loan_id"] != "")].copy()
        duplicate_loan_ids = valid_loan_ids["loan_id"].duplicated().sum()
        
        st.metric("Duplicate Loan IDs", duplicate_loan_ids)
        if duplicate_loan_ids == 0:
            st.success("No duplicate loan IDs found")
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
# QBO DATA ANALYSIS SECTION (Updated for new schema)
# ----------------------------
st.header("💰 QBO Financial Data Analysis")

# QBO Data Quality Overview
col1, col2 = st.columns(2)

with col1:
    st.subheader("Invoice/Payment Data Quality")
    if len(qbo_txn_df) > 0:
        st.metric("Total Records", len(qbo_txn_df))
        if 'txn_date' in qbo_txn_df.columns:
            date_range = f"{qbo_txn_df['txn_date'].min().strftime('%m/%d/%y') if qbo_txn_df['txn_date'].min() else 'N/A'} to {qbo_txn_df['txn_date'].max().strftime('%m/%d/%y') if qbo_txn_df['txn_date'].max() else 'N/A'}"
            st.metric("Date Range", date_range)
        st.metric("Null Values", qbo_txn_df.isnull().sum().sum())
        
        # Transaction type breakdown
        if 'transaction_type' in qbo_txn_df.columns:
            txn_type_counts = qbo_txn_df['transaction_type'].value_counts()
            st.write("**Transaction Types:**")
            for txn_type, count in txn_type_counts.items():
                st.write(f"• {txn_type}: {count}")
    else:
        st.warning("No QBO invoice/payment data available")

with col2:
    st.subheader("General Ledger Quality")
    if len(qbo_gl_df) > 0:
        st.metric("Total Records", len(qbo_gl_df))
        if 'txn_date' in qbo_gl_df.columns:
            date_range = f"{qbo_gl_df['txn_date'].min().strftime('%m/%d/%y') if qbo_gl_df['txn_date'].min() else 'N/A'} to {qbo_gl_df['txn_date'].max().strftime('%m/%d/%y') if qbo_gl_df['txn_date'].max() else 'N/A'}"
            st.metric("Date Range", date_range)
        st.metric("Null Values", qbo_gl_df.isnull().sum().sum())
    else:
        st.warning("No QBO general ledger data available")

# QBO Analysis Tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs(["General Ledger Analysis", "Invoice/Payment Analysis", "Outstanding Balances", "Loan Performance", "Data Quality Issues"])

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
    st.subheader("Invoice/Payment Analysis")
    
    if not qbo_txn_df.empty:
        # Group by transaction type
        if 'transaction_type' in qbo_txn_df.columns and 'total_amount' in qbo_txn_df.columns:
            txn_by_type = qbo_txn_df.groupby('transaction_type')['total_amount'].agg(['sum', 'count', 'mean']).round(2)
            txn_by_type.columns = ['Total Amount', 'Transaction Count', 'Average Amount']
            txn_by_type = txn_by_type.sort_values('Total Amount', ascending=False)
            
            st.write("**By Transaction Type:**")
            st.dataframe(txn_by_type.style.format({
                'Total Amount': '${:,.2f}',
                'Average Amount': '${:,.2f}'
            }), use_container_width=True)
        
        # Group by customer
        if 'customer_name' in qbo_txn_df.columns:
            txn_by_customer = qbo_txn_df.groupby('customer_name')['total_amount'].agg(['sum', 'count', 'mean']).round(2)
            txn_by_customer.columns = ['Total Amount', 'Transaction Count', 'Average Amount']
            txn_by_customer = txn_by_customer.sort_values('Total Amount', ascending=False).head(15)
            
            st.write("**Top 15 by Customer:**")
            st.dataframe(txn_by_customer.style.format({
                'Total Amount': '${:,.2f}',
                'Average Amount': '${:,.2f}'
            }), use_container_width=True)
        
        # Payment method analysis
        if 'payment_method' in qbo_txn_df.columns:
            payment_methods = qbo_txn_df[qbo_txn_df['transaction_type'] == 'Payment']['payment_method'].value_counts()
            if len(payment_methods) > 0:
                st.write("**Payment Methods:**")
                for method, count in payment_methods.items():
                    st.write(f"• {method}: {count}")
    else:
        st.info("No invoice/payment data available for analysis")

with tab3:
    st.subheader("Outstanding Balances Analysis")
    
    if not qbo_txn_df.empty and 'balance' in qbo_txn_df.columns:
        # Filter for invoices with outstanding balances
        outstanding_invoices = qbo_txn_df[
            (qbo_txn_df['transaction_type'] == 'Invoice') & 
            (qbo_txn_df['balance'] > 0)
        ].copy()
        
        if len(outstanding_invoices) > 0:
            # Summary statistics
            col1, col2, col3 = st.columns(3)
            
            with col1:
                total_outstanding = outstanding_invoices['balance'].sum()
                st.metric("Total Outstanding", f"${total_outstanding:,.2f}")
            
            with col2:
                avg_outstanding = outstanding_invoices['balance'].mean()
                st.metric("Average Outstanding", f"${avg_outstanding:,.2f}")
            
            with col3:
                st.metric("Outstanding Invoices", len(outstanding_invoices))
            
            # Outstanding by customer
            if 'customer_name' in outstanding_invoices.columns:
                outstanding_by_customer = outstanding_invoices.groupby('customer_name')['balance'].agg(['sum', 'count']).round(2)
                outstanding_by_customer.columns = ['Outstanding Balance', 'Invoice Count']
                outstanding_by_customer = outstanding_by_customer.sort_values('Outstanding Balance', ascending=False).head(15)
                
                st.write("**Top 15 Outstanding Balances by Customer:**")
                st.dataframe(outstanding_by_customer.style.format({
                    'Outstanding Balance': '${:,.2f}'
                }), use_container_width=True)
            
            # Overdue analysis
            if 'due_date' in outstanding_invoices.columns:
                today = pd.Timestamp.now().date()
                outstanding_invoices['due_date'] = pd.to_datetime(outstanding_invoices['due_date'], errors='coerce')
                
                overdue_invoices = outstanding_invoices[
                    (outstanding_invoices['due_date'].notna()) & 
                    (outstanding_invoices['due_date'].dt.date < today)
                ]
                
                if len(overdue_invoices) > 0:
                    st.write("**⚠️ Overdue Invoices Summary:**")
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("Overdue Count", len(overdue_invoices))
                    with col2:
                        total_overdue = overdue_invoices['balance'].sum()
                        st.metric("Total Overdue", f"${total_overdue:,.2f}")
                    with col3:
                        overdue_pct = (total_overdue / total_outstanding * 100) if total_outstanding > 0 else 0
                        st.metric("% of Outstanding", f"{overdue_pct:.1f}%")
        else:
            st.info("No outstanding invoices found")
    else:
        st.info("No balance data available for outstanding analysis")

with tab4:
    st.subheader("Enhanced Loan Performance Analysis")
    
    # Use Invoice/Payment data for loan performance
    if not qbo_txn_df.empty:
        # Filter for loan-related transactions
        loan_transactions = qbo_txn_df[
            qbo_txn_df['transaction_type'].isin(['Invoice', 'Payment'])
        ].copy()
        
        if not loan_transactions.empty and 'customer_name' in loan_transactions.columns:
            # Enhanced pivot analysis using new schema
            loan_transactions['total_amount'] = loan_transactions['total_amount'].abs()
            
            pivot_enhanced = loan_transactions.pivot_table(
                index="customer_name",
                columns="transaction_type",
                values="total_amount",
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
            display_cols = ["customer_name", "total_invoiced", "total_payments", "outstanding_balance", "payment_ratio", "risk_score"]
            pivot_display = pivot_enhanced[display_cols].copy()
            pivot_display = pivot_display.sort_values("outstanding_balance", ascending=False)
            
            # Format currency columns
            currency_cols = ["total_invoiced", "total_payments", "outstanding_balance"]
            for col in currency_cols:
                if col in pivot_display.columns:
                    pivot_display[col] = pivot_display[col].apply(lambda x: f"${x:,.2f}")
            
            if "payment_ratio" in pivot_display.columns:
                pivot_display["payment_ratio"] = pivot_display["payment_ratio"].apply(lambda x: f"{x:.1%}")
            
            # Rename columns
            pivot_display = pivot_display.rename(columns={
                'customer_name': 'Customer Name',
                'total_invoiced': 'Total Invoiced',
                'total_payments': 'Total Payments',
                'outstanding_balance': 'Outstanding Balance',
                'payment_ratio': 'Payment Ratio',
                'risk_score': 'Risk Score'
            })
            
            st.dataframe(pivot_display, use_container_width=True, hide_index=True)
            
            # Risk summary
            if len(pivot_enhanced) > 0:
                risk_summary = pivot_enhanced['risk_score'].value_counts()
                st.write("**Risk Distribution:**")
                for risk, count in risk_summary.items():
                    st.write(f"• {risk}: {count}")
        else:
            st.info("Unable to perform loan performance analysis - missing required data")
    else:
        st.info("No QBO data available for loan performance analysis")

with tab5:
    st.subheader("Data Quality Issues")
    
    # NA Count Analysis
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Invoice/Payment Data - NA Counts:**")
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
            st.write("No invoice/payment data available")
    
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
    if 'txn_date' in qbo_txn_df.columns:
        null_dates_txn = qbo_txn_df['txn_date'].isnull().sum()
        if null_dates_txn > 0:
            issues.append(f"Invoice/Payment Data: {null_dates_txn} records with null transaction dates")
    
    if 'due_date' in qbo_txn_df.columns:
        null_due_dates = qbo_txn_df['due_date'].isnull().sum()
        if null_due_dates > 0:
            issues.append(f"Invoice/Payment Data: {null_due_dates} records with null due dates")
    
    if 'txn_date' in qbo_gl_df.columns:
        null_dates_gl = qbo_gl_df['txn_date'].isnull().sum()
        if null_dates_gl > 0:
            issues.append(f"General Ledger: {null_dates_gl} records with null dates")
    
    # Amount issues
    if 'total_amount' in qbo_txn_df.columns:
        null_amounts_txn = qbo_txn_df['total_amount'].isnull().sum()
        zero_amounts_txn = (qbo_txn_df['total_amount'] == 0).sum()
        if null_amounts_txn > 0:
            issues.append(f"Invoice/Payment Data: {null_amounts_txn} records with null total amounts")
        if zero_amounts_txn > 0:
            issues.append(f"Invoice/Payment Data: {zero_amounts_txn} records with zero amounts")
    
    if 'amount' in qbo_gl_df.columns:
        null_amounts_gl = qbo_gl_df['amount'].isnull().sum()
        zero_amounts_gl = (qbo_gl_df['amount'] == 0).sum()
        if null_amounts_gl > 0:
            issues.append(f"General Ledger: {null_amounts_gl} records with null amounts")
        if zero_amounts_gl > 0:
            issues.append(f"General Ledger: {zero_amounts_gl} records with zero amounts")
    
    # Customer/name issues
    if 'customer_name' in qbo_txn_df.columns:
        null_customers = qbo_txn_df['customer_name'].isnull().sum()
        if null_customers > 0:
            issues.append(f"Invoice/Payment Data: {null_customers} records with missing customer names")
    
    # Balance validation issues
    if 'balance' in qbo_txn_df.columns and 'total_amount' in qbo_txn_df.columns:
        # Check for balances greater than total amounts (shouldn't happen)
        invalid_balances = qbo_txn_df[
            (qbo_txn_df['balance'] > qbo_txn_df['total_amount']) & 
            (qbo_txn_df['balance'].notna()) & 
            (qbo_txn_df['total_amount'].notna())
        ]
        if len(invalid_balances) > 0:
            issues.append(f"Invoice/Payment Data: {len(invalid_balances)} records with balance > total amount")
    
    if issues:
        for issue in issues:
            st.write(f"• {issue}")
    else:
        st.success("No major data quality issues detected")

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

col1, col2, col3, col4 = st.columns(4)

with col1:
    if st.button("🔄 Refresh Pipeline Data", help="Clears cache for main pipeline dashboard (load_deals function)"):
        clear_pipeline_cache()
        st.success("Pipeline dashboard cache cleared!")
        
with col2:
    if st.button("🔄 Refresh MCA Data", help="Clears cache for MCA dashboard (load_mca_deals & combine_deals functions)"):
        clear_mca_cache()
        st.success("MCA dashboard cache cleared!")
        
with col3:
    if st.button("🔄 Refresh QBO Data", help="Clears cache for QBO dashboard (load_qbo_data function)"):
        clear_qbo_cache()
        st.success("QBO dashboard cache cleared!")

with col4:
    if st.button("🔄 Refresh All Caches", type="primary", help="Clears all cached data across the entire application"):
        st.cache_data.clear()
        st.success("All data caches cleared!")

# ----------------------------
# Cross-Dataset Analysis
# ----------------------------
st.header("📊 Cross-Dataset Analysis")

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
        'MCA Deals': [
            len(mca_deals_combined),
            mca_deals_combined['dba'].nunique() if 'dba' in mca_deals_combined.columns else 0,
            (mca_deals_combined['funding_date'].max() - mca_deals_combined['funding_date'].min()).days if 'funding_date' in mca_deals_combined.columns and mca_deals_combined['funding_date'].notna().any() else 0,
            mca_deals_combined['total_funded_amount'].sum() if 'total_funded_amount' in mca_deals_combined.columns else 0
        ],
        'QBO Invoice/Payments': [
            len(qbo_txn_df),
            qbo_txn_df['customer_name'].nunique() if 'customer_name' in qbo_txn_df.columns else 0,
            (qbo_txn_df['txn_date'].max() - qbo_txn_df['txn_date'].min()).days if 'txn_date' in qbo_txn_df.columns and qbo_txn_df['txn_date'].notna().any() else 0,
            qbo_txn_df['total_amount'].sum() if 'total_amount' in qbo_txn_df.columns else 0
        ]
    }
    comparison_df = pd.DataFrame(comparison_data)
    st.dataframe(comparison_df, use_container_width=True)

with col2:
    # Common names analysis between MCA and QBO data
    if 'dba' in mca_deals_combined.columns and 'customer_name' in qbo_txn_df.columns:
        mca_names = set(mca_deals_combined['dba'].dropna().unique())
        qbo_txn_names = set(qbo_txn_df['customer_name'].dropna().unique())
        qbo_gl_names = set(qbo_gl_df['name'].dropna().unique()) if 'name' in qbo_gl_df.columns else set()
        
        common_mca_qbo = mca_names.intersection(qbo_txn_names)
        common_all = mca_names.intersection(qbo_txn_names).intersection(qbo_gl_names)
        
        st.write("**Name Overlap Analysis:**")
        st.metric("MCA & QBO Invoice/Payment Names", len(common_mca_qbo))
        st.metric("Common Across All Datasets", len(common_all))
        st.metric("MCA Names Only", len(mca_names - qbo_txn_names - qbo_gl_names))

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

# MCA data health
mca_health_issues = 0
if len(mca_deals_raw) == 0:
    mca_health_issues += 1
if len(mca_deals_combined) == 0:
    mca_health_issues += 1

if mca_health_issues == 0:
    health_status.append("✅ MCA Data: Both raw and combined datasets available")
elif mca_health_issues == 1:
    health_status.append("🟡 MCA Data: One dataset missing or empty")
else:
    health_status.append("🔴 MCA Data: Multiple datasets missing or empty")

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
freshness_issues = []
if len(deals_df) > 0 and 'date_created' in deals_df.columns:
    latest_deal = deals_df["date_created"].max()
    days_since_last = (pd.Timestamp.now() - latest_deal).days
    if days_since_last > 7:
        freshness_issues.append(f"Deal Data ({days_since_last} days old)")

if len(mca_deals_combined) > 0 and 'funding_date' in mca_deals_combined.columns:
    latest_mca = mca_deals_combined["funding_date"].max()
    mca_days_since = (pd.Timestamp.now() - latest_mca).days
    if mca_days_since > 7:
        freshness_issues.append(f"MCA Data ({mca_days_since} days old)")

if len(qbo_txn_df) > 0 and 'txn_date' in qbo_txn_df.columns:
    qbo_txn_latest = qbo_txn_df["txn_date"].max()
    qbo_txn_days_since = (pd.Timestamp.now() - qbo_txn_latest).days
    if qbo_txn_days_since > 7:
        freshness_issues.append(f"QBO Invoice/Payments ({qbo_txn_days_since} days old)")

if len(freshness_issues) == 0:
    health_status.append("✅ Data Freshness: All data appears current")
elif len(freshness_issues) == 1:
    health_status.append(f"🟡 Data Freshness: {freshness_issues[0]} may be stale")
else:
    stale_datasets = ", ".join(freshness_issues)
    health_status.append(f"🔴 Data Freshness: Multiple datasets stale - {stale_datasets}")

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
    st.success(f"System Health Score: {health_score:.0f}% - Excellent!")
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
    if st.button("Generate Health Report", type="primary"):
        # Create summary report
        report_data = {
            "Report Generated": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
            "Deal Data Records": len(deals_df),
            "MCA Deals Records": len(mca_deals_combined),
            "QBO Invoice/Payment Records": len(qbo_txn_df),
            "QBO GL Records": len(qbo_gl_df),
            "System Health Score": f"{health_score:.0f}%",
            "Health Status": health_status
        }
        
        # Convert to DataFrame for easy export
        report_df = pd.DataFrame(list(report_data.items()), columns=['Metric', 'Value'])
        csv_data = report_df.to_csv(index=False).encode("utf-8")
        
        st.download_button(
            label="Download Health Report CSV",
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
                label="Export Missing Loan IDs",
                data=csv_data,
                file_name=f"missing_loan_ids_{pd.Timestamp.now().strftime('%Y%m%d')}.csv",
                mime="text/csv"
            )
    
    # Export MCA data quality issues
    if len(mca_deals_combined) > 0:
        if st.button("Export MCA Quality Report"):
            quality_data = []
            
            # Add MCA data quality info
            quality_data.append({
                "Dataset": "MCA Deals Combined",
                "Total Records": len(mca_deals_combined),
                "Null Values": mca_deals_combined.isnull().sum().sum(),
                "Date Range": f"{mca_deals_combined['funding_date'].min().strftime('%m/%d/%y') if 'funding_date' in mca_deals_combined.columns and mca_deals_combined['funding_date'].notna().any() else 'N/A'} to {mca_deals_combined['funding_date'].max().strftime('%m/%d/%y') if 'funding_date' in mca_deals_combined.columns and mca_deals_combined['funding_date'].notna().any() else 'N/A'}"
            })
            
            quality_df = pd.DataFrame(quality_data)
            csv_data = quality_df.to_csv(index=False).encode("utf-8")
            
            st.download_button(
                label="Download MCA Quality Report",
                data=csv_data,
                file_name=f"mca_quality_report_{pd.Timestamp.now().strftime('%Y%m%d')}.csv",
                mime="text/csv"
            )

# ----------------------------
# Footer with Last Updated
# ----------------------------
st.divider()
st.caption(f"Dashboard last updated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')} | Data sources: Supabase (deals, mca_deals, qbo_invoice_payments, qbo_general_ledger)")
