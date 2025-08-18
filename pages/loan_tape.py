# pages/loan_tape.py
from utils.imports import *
from utils.config import (
    inject_global_styles,
    inject_logo,
    get_supabase_client,
)

# Page config & branding
st.set_page_config(
    page_title="CSL Capital | Loan Tape",
    layout="wide",
)
inject_global_styles()
inject_logo()

# Load data
supabase = get_supabase_client()

@st.cache_data(ttl=3600)
def load_loan_summaries():
    res = supabase.table("loan_summaries").select("*").execute()
    return pd.DataFrame(res.data)

@st.cache_data(ttl=3600)
def load_deals():
    res = supabase.table("deals").select("loan_id,deal_name,partner_source,industry,commission").execute()
    return pd.DataFrame(res.data)

# Main content
st.title("Loan Tape")

# Load loan data
with st.spinner("Loading data..."):
    try:
        loans_df = load_loan_summaries()
        deals_df = load_deals()
        
        # Merge the dataframes to get the deal information
        if not deals_df.empty:
            loans_df = loans_df.merge(
                deals_df[["loan_id", "deal_name", "partner_source", "industry", "commission"]], 
                on="loan_id", 
                how="left"
            )
    except Exception as e:
        st.warning(f"Note: Some data could not be loaded. {str(e)}")
        if 'loans_df' not in locals():
            loans_df = pd.DataFrame()

if loans_df.empty:
    st.warning("No loan data available.")
else:
    # Convert commission to numeric if it's not already
    if 'commission' in loans_df.columns:
        loans_df['commission'] = pd.to_numeric(loans_df['commission'], errors='coerce').fillna(0)
    else:
        loans_df['commission'] = 0
        
    # Calculate capital at risk metrics
    loans_df['total_gross_csl_amount'] = loans_df['csl_participation_amount'] * (1 + loans_df['commission']) + (loans_df['csl_participation_amount'] * 1.03)
    loans_df['gross_csl_amount_balance'] = loans_df['total_gross_csl_amount'] - loans_df['total_paid']
    
    # Flag for unpaid balances (non-paid off loans)
    loans_df['is_unpaid'] = loans_df['loan_status'] != "Paid Off"
    
    # Status filter - multiple selection
    all_statuses = sorted(loans_df["loan_status"].unique().tolist())
    selected_statuses = st.multiselect("Filter by Status:", all_statuses, default=all_statuses)
    
    # Apply filters
    if selected_statuses:
        filtered_df = loans_df[loans_df["loan_status"].isin(selected_statuses)]
    else:
        filtered_df = loans_df
    
    # Dashboard metrics
    st.subheader("Portfolio Overview")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Loans", len(filtered_df))
        st.metric("Total Participation", f"${filtered_df['csl_participation_amount'].sum():,.2f}")
    with col2:
        st.metric("Total Gross CSL Amount", f"${filtered_df['total_gross_csl_amount'].sum():,.2f}")
        st.metric("Total Paid Back", f"${filtered_df['total_paid'].sum():,.2f}")
    with col3:
        st.metric("Gross CSL Amount Balance", f"${filtered_df['gross_csl_amount_balance'].sum():,.2f}")
        unpaid_balance = filtered_df[filtered_df['is_unpaid']]['gross_csl_amount_balance'].sum()
        st.metric("Unpaid Balance", f"${unpaid_balance:,.2f}")
    
    # Top 5 largest outstanding positions
    st.subheader("Top 5 Largest Outstanding Positions")
    top_positions = (
        filtered_df[filtered_df['is_unpaid']]
        .sort_values('gross_csl_amount_balance', ascending=False)
        .head(5)
    )
    
    top_positions_display = top_positions[['loan_id', 'deal_name', 'loan_status', 'gross_csl_amount_balance']].copy()
    top_positions_display['gross_csl_amount_balance'] = top_positions_display['gross_csl_amount_balance'].map(
        lambda x: f"${x:,.2f}" if pd.notnull(x) else ""
    )
    
    st.dataframe(
        top_positions_display,
        use_container_width=True,
        hide_index=True
    )
    
    # Data table with all loans
    st.subheader("All Loan Details")
    
    # Select priority columns for display
    display_columns = [
        "loan_id"
    ]
    
    # Add deal columns if available
    for col in ["deal_name", "partner_source", "industry"]:
        if col in filtered_df.columns:
            display_columns.append(col)
        
    # Add remaining columns
    display_columns.extend([
        "loan_status", 
        "csl_participation_amount", 
        "total_gross_csl_amount",
        "total_paid",
        "gross_csl_amount_balance",
        "participation_percentage", 
        "on_time_rate",
        "payment_performance"
    ])
    
    # Filter to only include columns that exist in the dataframe
    display_columns = [col for col in display_columns if col in filtered_df.columns]
    
    # Make a copy for display
    display_df = filtered_df[display_columns].copy()
    
    # Format numeric columns
    for col in display_df.select_dtypes(include=['float64', 'float32']).columns:
        if "rate" in col or "percentage" in col:
            display_df[col] = display_df[col].map(lambda x: f"{x:.2%}" if pd.notnull(x) else "")
        elif "amount" in col or "paid" in col or "balance" in col:
            display_df[col] = display_df[col].map(lambda x: f"${x:,.2f}" if pd.notnull(x) else "")
    
    # Display table with pagination
    st.dataframe(
        display_df, 
        use_container_width=True,
        hide_index=True
    )
    
    # Export functionality
    if st.button("Export to CSV"):
        csv = filtered_df.to_csv(index=False)
        st.download_button(
            label="Download CSV",
            data=csv,
            file_name="loan_tape_export.csv",
            mime="text/csv"
        )
