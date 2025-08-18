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

# Main content
st.title("Loan Tape")

# Load loan data
with st.spinner("Loading loan data..."):
    loans_df = load_loan_summaries()

if loans_df.empty:
    st.warning("No loan data available.")
else:
    # Status filter
    status_options = ["All"] + sorted(loans_df["loan_status"].unique().tolist())
    selected_status = st.selectbox("Filter by Status:", status_options)
    
    # Apply filters
    filtered_df = loans_df
    if selected_status != "All":
        filtered_df = filtered_df[filtered_df["loan_status"] == selected_status]
    
    # Dashboard metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Loans", len(filtered_df))
    with col2:
        st.metric("Total Funding", f"${filtered_df['total_funding'].sum():,.2f}")
    with col3:
        st.metric("Our RTR", f"${filtered_df['our_rtr'].sum():,.2f}")
    with col4:
        st.metric("Total Paid", f"${filtered_df['total_paid'].sum():,.2f}")
    
    # Data table
    st.subheader("Loan Details")
    
    # Format dates for display
    display_df = filtered_df.copy()
    for date_col in ["created_at", "updated_at", "projected_maturity_date", 
                     "status_changed_at", "payoff_date", "status_last_manual_update"]:
        if date_col in display_df.columns:
            display_df[date_col] = pd.to_datetime(display_df[date_col]).dt.strftime('%Y-%m-%d')
    
    # Select columns for display
    display_columns = [
        "loan_id", "loan_status", "total_funding", "factor_rate", 
        "our_rtr", "total_paid", "payment_performance", "on_time_rate",
        "funding_date", "projected_maturity_date", "maturity_confidence"
    ]
    
    # Format numeric columns
    for col in display_df.select_dtypes(include=['float64']).columns:
        if "rate" in col or "percentage" in col:
            display_df[col] = display_df[col].map(lambda x: f"{x:.2%}" if pd.notnull(x) else "")
        else:
            display_df[col] = display_df[col].map(lambda x: f"${x:,.2f}" if pd.notnull(x) else "")
    
    # Display table with pagination
    st.dataframe(
        display_df[display_columns], 
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
