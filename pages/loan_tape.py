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
    try:
        loans_df = load_loan_summaries()
        
        # Try to get deal names if available
        try:
            deals_response = supabase.table("deals").select("loan_id,name").execute()
            deals_df = pd.DataFrame(deals_response.data)
            
            # Merge with loans if we have deals data
            if not deals_df.empty:
                loans_df = loans_df.merge(
                    deals_df[["loan_id", "name"]], 
                    on="loan_id", 
                    how="left"
                )
                has_deal_names = True
            else:
                has_deal_names = False
        except Exception as e:
            st.warning(f"Note: Deal names could not be loaded. Using loan IDs only.")
            has_deal_names = False
            
    except Exception as e:
        st.error(f"Error loading loan data: {e}")
        loans_df = pd.DataFrame()
        has_deal_names = False

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
        st.metric("Total Participation", f"${filtered_df['csl_participation_amount'].sum():,.2f}")
    with col3:
        st.metric("Total Paid", f"${filtered_df['total_paid'].sum():,.2f}")
    with col4:
        avg_on_time = filtered_df['on_time_rate'].mean() if 'on_time_rate' in filtered_df.columns else 0
        st.metric("Avg On-Time Rate", f"{avg_on_time:.2%}")
    
    # Data table
    st.subheader("Loan Details")
    
    # Select priority columns for display
    display_columns = [
        "loan_id"
    ]
    
    # Add name column if available
    if has_deal_names and "name" in filtered_df.columns:
        display_columns.append("name")
        
    # Add remaining columns
    display_columns.extend([
        "loan_status", 
        "csl_participation_amount", 
        "participation_percentage", 
        "total_paid", 
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
        elif "amount" in col or "paid" in col:
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
