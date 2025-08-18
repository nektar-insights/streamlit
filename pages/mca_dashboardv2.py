# pages/mca_dashboardv2.py
from utils.imports import *
from utils.config import (
    inject_global_styles,
    inject_logo,
    get_supabase_client,
    PRIMARY_COLOR,
    COLOR_PALETTE,
    PLATFORM_FEE_RATE,
)

# Page configuration and branding
st.set_page_config(
    page_title="CSL Capital | Loan Tape",
    layout="wide",
)
inject_global_styles()
inject_logo()

# Define risk gradient color scheme
RISK_GRADIENT = [
    "#fff600",  # Light yellow
    "#ffc302",  # Yellow
    "#ff8f00",  # Orange
    "#ff5b00",  # Dark orange
    "#ff0505",  # Red
]

# ----------------------------
# Supabase connection
# ----------------------------
supabase = get_supabase_client()

# ----------------------------
# Load and prepare data
# ----------------------------
@st.cache_data(ttl=3600)
def load_loan_summaries():
    res = supabase.table("loan_summaries").select("*").execute()
    return pd.DataFrame(res.data)

@st.cache_data(ttl=3600)
def load_deals():
    res = supabase.table("deals").select("*").execute()
    return pd.DataFrame(res.data)

# Load data
loans_df = load_loan_summaries()
deals_df = load_deals()

# Merge the dataframes
if not loans_df.empty and not deals_df.empty:
    df = loans_df.merge(
        deals_df[["loan_id", "deal_name", "partner_source", "industry", "commission"]], 
        on="loan_id", 
        how="left"
    )
else:
    df = loans_df.copy()

# ----------------------------
# Data type conversions and basic calculations
# ----------------------------
# Convert dates
for date_col in ["funding_date", "maturity_date"]:
    if date_col in df.columns:
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce").dt.date

# Convert commission to numeric
if 'commission' in df.columns:
    df['commission'] = pd.to_numeric(df['commission'], errors='coerce').fillna(0)
else:
    df['commission'] = 0

# Calculate total invested (participation + platform fee + commission)
df['total_invested'] = (
    df['csl_participation_amount'] + 
    (df['csl_participation_amount'] * 0.03) +  # Fixed platform fee as 3%
    (df['csl_participation_amount'] * df['commission_fee'])  # Commission fee as percentage
)

# Calculate net balance (invested - paid)
df['net_balance'] = df['total_invested'] - df['total_paid']

# Calculate ROI
df['current_roi'] = df.apply(
    lambda x: (x['total_paid'] / x['total_invested']) - 1 if x['total_invested'] > 0 else 0, 
    axis=1
)

# Flag for unpaid balances (non-paid off loans)
df['is_unpaid'] = df['loan_status'] != "Paid Off"

# Calculate days since funding
df["days_since_funding"] = (pd.Timestamp.today() - pd.to_datetime(df["funding_date"])).dt.days

# ----------------------------
# Filters
# ----------------------------
st.title("Loan Tape Dashboard")

# Date filter
if 'funding_date' in df.columns and not df['funding_date'].isna().all():
    min_date = df["funding_date"].min()
    max_date = df["funding_date"].max()
    start_date, end_date = st.date_input(
        "Filter by Funding Date",
        [min_date, max_date],
        min_value=min_date,
        max_value=max_date
    )
    df = df[(df["funding_date"] >= start_date) & (df["funding_date"] <= end_date)]

# Status filter
all_statuses = sorted(df["loan_status"].unique().tolist())
selected_statuses = st.multiselect("Filter by Status:", all_statuses, default=all_statuses)

# Apply status filter
if selected_statuses:
    filtered_df = df[df["loan_status"].isin(selected_statuses)]
else:
    filtered_df = df

# ----------------------------
# Calculate portfolio metrics
# ----------------------------
# Portfolio summary
total_loans = len(filtered_df)
total_paid_off = (filtered_df["loan_status"] == "Paid Off").sum()
total_active = (filtered_df["loan_status"] != "Paid Off").sum()

# Financial metrics
total_participation = filtered_df['csl_participation_amount'].sum()
total_invested = filtered_df['total_invested'].sum()
total_returned = filtered_df['total_paid'].sum()
net_balance = filtered_df['net_balance'].sum()

# Calculate portfolio ROI
portfolio_roi = ((total_returned / total_invested) - 1) if total_invested > 0 else 0

# ----------------------------
# Display metrics sections
# ----------------------------
# Portfolio Overview
st.subheader("Portfolio Overview")
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Total Loans", total_loans)
    st.metric("Active Loans", total_active)
with col2:
    st.metric("Total Participation", f"${total_participation:,.2f}")
    st.metric("Total Invested (with fees)", f"${total_invested:,.2f}")
with col3:
    st.metric("Total Returned", f"${total_returned:,.2f}")
    roi_color = "normal" if portfolio_roi < 0 else "inverse"
    st.metric("Portfolio ROI", f"{portfolio_roi:.2%}", delta_color=roi_color)

# ----------------------------
# Top 5 largest outstanding positions
# ----------------------------
st.subheader("Top 5 Largest Outstanding Positions")
top_positions = (
    filtered_df[filtered_df['is_unpaid']]
    .sort_values('net_balance', ascending=False)
    .head(5)
)

if not top_positions.empty:
    display_columns = ['loan_id', 'deal_name', 'loan_status', 'total_invested', 'total_paid', 'net_balance']
    display_columns = [col for col in display_columns if col in top_positions.columns]
    
    top_positions_display = top_positions[display_columns].copy()
    
    # Format numeric columns
    for col in top_positions_display.select_dtypes(include=['float64', 'float32']).columns:
        if "roi" in col or "rate" in col or "percentage" in col:
            top_positions_display[col] = top_positions_display[col].map(lambda x: f"{x:.2%}" if pd.notnull(x) else "")
        elif "amount" in col or "paid" in col or "balance" in col or "invested" in col:
            top_positions_display[col] = top_positions_display[col].map(lambda x: f"${x:,.2f}" if pd.notnull(x) else "")
    
    st.dataframe(
        top_positions_display,
        use_container_width=True,
        hide_index=True
    )
else:
    st.info("No outstanding positions found with the current filters.")

# ----------------------------
# Loan Tape
# ----------------------------
st.subheader("Loan Tape")

# Select columns for display
display_columns = ["loan_id"]

# Add deal columns if available
for col in ["deal_name", "partner_source", "industry"]:
    if col in filtered_df.columns:
        display_columns.append(col)

# Add financial columns
display_columns.extend([
    "loan_status",
    "funding_date",
    "maturity_date",
    "csl_participation_amount",
    "total_invested",
    "total_paid",
    "net_balance",
    "current_roi",
    "participation_percentage",
    "on_time_rate",
    "payment_performance"
])

# Filter to only include columns that exist in the dataframe
display_columns = [col for col in display_columns if col in filtered_df.columns]

# Make a copy for display
loan_tape = filtered_df[display_columns].copy()

# Format numeric columns
for col in loan_tape.select_dtypes(include=['float64', 'float32']).columns:
    if "roi" in col or "rate" in col or "percentage" in col:
        loan_tape[col] = loan_tape[col].map(lambda x: f"{x:.2%}" if pd.notnull(x) else "")
    elif "amount" in col or "paid" in col or "balance" in col or "invested" in col:
        loan_tape[col] = loan_tape[col].map(lambda x: f"${x:,.2f}" if pd.notnull(x) else "")

st.dataframe(
    loan_tape,
    use_container_width=True,
    hide_index=True
)

# ----------------------------
# Status Distribution Chart
# ----------------------------
if 'loan_status' in df.columns and not df['loan_status'].isna().all():
    st.subheader("Distribution of Loan Status")
    
    # Calculate status percentages
    status_counts = filtered_df["loan_status"].value_counts(normalize=True)
    status_summary = pd.DataFrame({
        "loan_status": status_counts.index.astype(str),
        "percentage": status_counts.values,
        "count": filtered_df["loan_status"].value_counts().values
    })
    
    # Status bar chart
    status_chart = alt.Chart(status_summary).mark_bar().encode(
        x=alt.X("loan_status:N", title="Loan Status", axis=alt.Axis(labelAngle=-45)),
        y=alt.Y("percentage:Q", title="% of Total Loans", axis=alt.Axis(format=".0%")),
        color=alt.Color("loan_status:N", scale=alt.Scale(range=RISK_GRADIENT), legend=None),
        tooltip=[
            alt.Tooltip("loan_status:N", title="Loan Status"),
            alt.Tooltip("count:Q", title="Number of Loans"),
            alt.Tooltip("percentage:Q", title="% of Total", format=".1%")
        ]
    ).properties(
        width=700,
        height=350,
        title="Distribution of Loan Status"
    )
    
    st.altair_chart(status_chart, use_container_width=True)

# ----------------------------
# ROI Distribution Chart
# ----------------------------
st.subheader("ROI Distribution by Loan")

# Create ROI visualization for non-zero investment loans
roi_df = filtered_df[filtered_df['total_invested'] > 0].copy()
roi_df = roi_df.sort_values('current_roi', ascending=False)

if not roi_df.empty:
    roi_chart = alt.Chart(roi_df).mark_bar().encode(
        x=alt.X(
            "loan_id:N",
            title="Loan ID",
            sort="-y",
            axis=alt.Axis(labelAngle=-90)
        ),
        y=alt.Y("current_roi:Q", title="Current ROI"),
        color=alt.Color(
            "current_roi:Q",
            scale=alt.Scale(domain=[-0.5, 0, 0.5], range=["#ff0505", "#ffc302", "#2ca02c"]),
            legend=None
        ),
        tooltip=[
            alt.Tooltip("loan_id:N", title="Loan ID"),
            alt.Tooltip("deal_name:N", title="Deal Name"),
            alt.Tooltip("loan_status:N", title="Status"),
            alt.Tooltip("current_roi:Q", title="Current ROI", format=".2%"),
            alt.Tooltip("total_invested:Q", title="Total Invested", format="$,.2f"),
            alt.Tooltip("total_paid:Q", title="Total Paid", format="$,.2f")
        ]
    ).properties(
        width=800,
        height=400,
        title="ROI by Loan (Highest to Lowest)"
    )

    st.altair_chart(roi_chart, use_container_width=True)
else:
    st.info("No loans with investment data to display ROI distribution.")

# ----------------------------
# Export functionality
# ----------------------------
csv = loan_tape.to_csv(index=False).encode("utf-8")
st.download_button(
    label="Download Loan Tape as CSV",
    data=csv,
    file_name="loan_tape.csv",
    mime="text/csv"
)
