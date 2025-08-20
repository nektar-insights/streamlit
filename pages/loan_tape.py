# pages/loan_tape.py
from utils.imports import *
from utils.config import (
    inject_global_styles,
    inject_logo,
    get_supabase_client,
    PRIMARY_COLOR,
    COLOR_PALETTE,
    PLATFORM_FEE_RATE,
)
from datetime import datetime

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
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce")

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

# Calculate commission fees
df['commission_fees'] = df['csl_participation_amount'] * df['commission_fee']

# Calculate platform fees
df['platform_fees'] = df['csl_participation_amount'] * 0.03

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
df["days_since_funding"] = (pd.Timestamp.today() - df["funding_date"]).dt.days

# Calculate remaining maturity in months (only for active loans)
df["remaining_maturity_months"] = 0.0
active_loans_mask = (df['loan_status'] != "Paid Off") & (df['maturity_date'] > pd.Timestamp.today())
if 'maturity_date' in df.columns:
    df.loc[active_loans_mask, "remaining_maturity_months"] = (
        (df.loc[active_loans_mask, 'maturity_date'] - pd.Timestamp.today()).dt.days / 30
    )

# ----------------------------
# Filters
# ----------------------------
st.title("Loan Tape Dashboard")

# Date filter
if 'funding_date' in df.columns and not df['funding_date'].isna().all():
    min_date = df["funding_date"].min().date()
    max_date = df["funding_date"].max().date()
    start_date, end_date = st.date_input(
        "Filter by Funding Date",
        [min_date, max_date],
        min_value=min_date,
        max_value=max_date
    )
    df = df[(df["funding_date"].dt.date >= start_date) & (df["funding_date"].dt.date <= end_date)]

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
total_positions = len(filtered_df)
total_paid_off = (filtered_df["loan_status"] == "Paid Off").sum()
total_active = (filtered_df["loan_status"] != "Paid Off").sum()

# Financial metrics
total_capital_deployed = filtered_df['csl_participation_amount'].sum()
total_invested = filtered_df['total_invested'].sum()
total_capital_returned = filtered_df['total_paid'].sum()
net_balance = filtered_df['net_balance'].sum()

# Fee metrics
total_commission_fees = filtered_df['commission_fees'].sum()
total_platform_fees = filtered_df['platform_fees'].sum()
total_bad_debt_allowance = filtered_df['bad_debt_allowance'].sum()

# Average metrics
avg_total_paid = filtered_df['total_paid'].mean()
avg_payment_performance = filtered_df['payment_performance'].mean() if 'payment_performance' in filtered_df.columns else 0
avg_remaining_maturity = filtered_df.loc[active_loans_mask, 'remaining_maturity_months'].mean() if not filtered_df.loc[active_loans_mask].empty else 0

# Calculate portfolio ROI
portfolio_roi = ((total_capital_returned / total_invested) - 1) if total_invested > 0 else 0

# ----------------------------
# Display metrics sections
# ----------------------------
# Portfolio Overview
st.subheader("Portfolio Overview")

# First row: Position counts and capital metrics
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Total Positions", total_positions)
with col2:
    st.metric("Total Capital Deployed", f"${total_capital_deployed:,.2f}")
with col3:
    st.metric("Total Capital Returned", f"${total_capital_returned:,.2f}")

# Second row: Fee metrics
col4, col5, col6 = st.columns(3)
with col4:
    st.metric("Total Commission Fees", f"${total_commission_fees:,.2f}")
with col5:
    st.metric("Total Platform Fees", f"${total_platform_fees:,.2f}")
with col6:
    st.metric("Total Bad Debt Allowance", f"${total_bad_debt_allowance:,.2f}")

# Third row: Average metrics
col7, col8, col9 = st.columns(3)
with col7:
    st.metric("Average Total Paid", f"${avg_total_paid:,.2f}")
with col8:
    # Changed info message to tooltip on the metric
    st.metric(
        "Average Payment Performance", 
        f"{avg_payment_performance:.2%}", 
        help="Payment Performance measures the ratio of actual payments to expected payments. 100% means payments are on schedule."
    )
with col9:
    st.metric("Average Remaining Maturity", f"{avg_remaining_maturity:.1f} months")

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
    # Calculate total net balance of top 5 positions
    top_5_total_balance = top_positions['net_balance'].sum()
    # Calculate percentage of total net balance
    top_5_pct_of_total = (top_5_total_balance / net_balance * 100) if net_balance > 0 else 0
    
    st.caption(f"Total Value: ${top_5_total_balance:,.2f} ({top_5_pct_of_total:.1f}% of total net balance)")
    
    display_columns = ['loan_id', 'deal_name', 'loan_status', 'total_invested', 'total_paid', 'net_balance', 'remaining_maturity_months']
    display_columns = [col for col in display_columns if col in top_positions.columns]
    
    top_positions_display = top_positions[display_columns].copy()
    
    # Format numeric columns
    for col in top_positions_display.select_dtypes(include=['float64', 'float32']).columns:
        if col == 'remaining_maturity_months':
            top_positions_display[col] = top_positions_display[col].map(lambda x: f"{x:.1f}" if pd.notnull(x) else "")
        elif "roi" in col or "rate" in col or "percentage" in col:
            top_positions_display[col] = top_positions_display[col].map(lambda x: f"{x:.2%}" if pd.notnull(x) else "")
        elif "amount" in col or "paid" in col or "balance" in col or "invested" in col:
            top_positions_display[col] = top_positions_display[col].map(lambda x: f"${x:,.2f}" if pd.notnull(x) else "")
    
    # Rename columns for display only
    column_rename = {
        "loan_id": "Loan ID",
        "deal_name": "Deal Name",
        "loan_status": "Loan Status",
        "total_invested": "Total Invested",
        "total_paid": "Total Paid",
        "net_balance": "Net Balance",
        "remaining_maturity_months": "Months to Maturity"
    }
    top_positions_display.rename(columns=column_rename, inplace=True)
    
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
    "payment_performance",
    "remaining_maturity_months"
])

# Filter to only include columns that exist in the dataframe
display_columns = [col for col in display_columns if col in filtered_df.columns]

# Make a copy for display
loan_tape = filtered_df[display_columns].copy()

# Rename columns to more user-friendly names with spaces
column_rename = {
    "loan_id": "Loan ID",
    "deal_name": "Deal Name",
    "partner_source": "Partner Source",
    "industry": "Industry",
    "loan_status": "Loan Status",
    "funding_date": "Funding Date",
    "maturity_date": "Maturity Date",
    "csl_participation_amount": "Capital Deployed",
    "total_invested": "Total Invested",
    "total_paid": "Total Paid",
    "net_balance": "Net Balance",
    "current_roi": "Current ROI",
    "participation_percentage": "Participation Percentage",
    "on_time_rate": "On Time Rate",
    "payment_performance": "Payment Performance",
    "remaining_maturity_months": "Remaining Maturity Months"
}

# Apply renaming for columns that exist
loan_tape.rename(columns={k: v for k, v in column_rename.items() if k in loan_tape.columns}, inplace=True)

# Format numeric columns
for col in loan_tape.select_dtypes(include=['float64', 'float32']).columns:
    if "ROI" in col or "Rate" in col or "Percentage" in col or "Performance" in col:
        loan_tape[col] = loan_tape[col].map(lambda x: f"{x:.2%}" if pd.notnull(x) else "")
    elif "Maturity" in col:
        loan_tape[col] = loan_tape[col].map(lambda x: f"{x:.1f}" if pd.notnull(x) else "")
    elif col in ["Capital Deployed", "Total Invested", "Total Paid", "Net Balance"]:
        loan_tape[col] = loan_tape[col].map(lambda x: f"${x:,.2f}" if pd.notnull(x) else "")

# Convert dates to readable format
for col in ["Funding Date", "Maturity Date"]:
    if col in loan_tape.columns:
        loan_tape[col] = loan_tape[col].dt.strftime('%Y-%m-%d')

st.dataframe(
    loan_tape,
    use_container_width=True,
    hide_index=True
)

# ----------------------------
# Status Distribution Chart (Pie Chart, excluding Paid Off)
# ----------------------------
if 'loan_status' in df.columns and not df['loan_status'].isna().all():
    st.subheader("Distribution of Loan Status")
    
    # Create a copy of filtered_df without Paid Off loans
    active_df = filtered_df[filtered_df["loan_status"] != "Paid Off"].copy()
    
    if not active_df.empty:
        # Calculate status percentages for active loans
        status_counts = active_df["loan_status"].value_counts(normalize=True)
        status_summary = pd.DataFrame({
            "status": status_counts.index.astype(str),
            "percentage": status_counts.values,
            "count": active_df["loan_status"].value_counts().values
        })
        
        # Add note about excluding Paid Off loans
        st.caption("Note: 'Paid Off' loans are excluded from this chart")
        
        # Create pie chart - removed text labels, keeping legend and tooltip
        pie_chart = alt.Chart(status_summary).mark_arc().encode(
            theta=alt.Theta(field="percentage", type="quantitative"),
            color=alt.Color(
                field="status", 
                type="nominal", 
                scale=alt.Scale(range=RISK_GRADIENT),
                legend=alt.Legend(title="Loan Status")
            ),
            tooltip=[
                alt.Tooltip("status:N", title="Loan Status"),
                alt.Tooltip("count:Q", title="Number of Loans"),
                alt.Tooltip("percentage:Q", title="% of Active Loans", format=".1%")
            ]
        ).properties(
            width=600,
            height=400,
            title="Distribution of Active Loan Status"
        )
        
        st.altair_chart(pie_chart, use_container_width=True)
    else:
        st.info("No active loans to display in status distribution chart.")

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

# ----------------------------
# Capital Deployment vs Returns Over Time (Combined)
# ----------------------------
st.subheader("Capital Flow: Deployment vs. Returns")

# Convert funding_date to datetime if not already
df['funding_date'] = pd.to_datetime(df['funding_date'], errors='coerce')

# Capital Deployed
deploy_df = df[['funding_date', 'csl_participation_amount']].dropna().sort_values('funding_date')
deploy_df = deploy_df.groupby('funding_date').sum().cumsum().reset_index()
deploy_df.rename(columns={'csl_participation_amount': 'capital_deployed'}, inplace=True)

# Capital Returned
return_df = df[['funding_date', 'total_paid']].dropna().sort_values('funding_date')
return_df = return_df.groupby('funding_date').sum().cumsum().reset_index()
return_df.rename(columns={'total_paid': 'capital_returned'}, inplace=True)

# Merge
flow_df = pd.merge(deploy_df, return_df, on='funding_date', how='outer').sort_values('funding_date').fillna(method='ffill')

# Milestone Annotations
milestones = [500_000, 1_000_000, 2_000_000, 3_000_000]
milestone_df = pd.DataFrame()
for value in milestones:
    row = flow_df[flow_df['capital_deployed'] >= value].head(1)
    if not row.empty:
        row = row.copy()
        row['milestone'] = f"${value:,.0f}"
        milestone_df = pd.concat([milestone_df, row])

# Chart
deploy_line = alt.Chart(flow_df).mark_line(color="red").encode(
    x='funding_date:T',
    y='capital_deployed:Q',
    tooltip=['funding_date:T', alt.Tooltip('capital_deployed:Q', title="Capital Deployed", format="$,.0f")]
)

return_line = alt.Chart(flow_df).mark_line(color="green").encode(
    x='funding_date:T',
    y='capital_returned:Q',
    tooltip=['funding_date:T', alt.Tooltip('capital_returned:Q', title="Capital Returned", format="$,.0f")]
)

milestone_points = alt.Chart(milestone_df).mark_point(filled=True, size=80, color="red").encode(
    x='funding_date:T',
    y='capital_deployed:Q',
    tooltip=[alt.Tooltip('milestone:N', title="Milestone"), 'funding_date:T']
)

capital_chart = (deploy_line + return_line + milestone_points).properties(
    width=800,
    height=400,
    title="Capital Deployed (Red) vs. Capital Returned (Green)"
)

st.altair_chart(capital_chart, use_container_width=True)

# ----------------------------
# IRR Calculation & Display
# ----------------------------
st.subheader("IRR Analysis for Paid-Off Loans")

# Convert payoff_date to datetime
df['payoff_date'] = pd.to_datetime(df['payoff_date'], errors='coerce')
df['maturity_date'] = pd.to_datetime(df['maturity_date'], errors='coerce')

paid_df = df[df['loan_status'] == "Paid Off"].copy()

# Calculate Realized IRR and Expected IRR
def calc_irr(row):
    if pd.isna(row['funding_date']) or pd.isna(row['payoff_date']) or row['total_invested'] <= 0:
        return None
    return npf.irr([-row['total_invested'], row['total_paid']])

def calc_expected_irr(row):
    if pd.isna(row['funding_date']) or pd.isna(row['maturity_date']) or row['total_invested'] <= 0:
        return None
    return npf.irr([-row['total_invested'], row['total_paid']])

paid_df['realized_irr'] = paid_df.apply(calc_irr, axis=1)
paid_df['expected_irr'] = paid_df.apply(calc_expected_irr, axis=1)

# Average IRRs
avg_realized_irr = paid_df['realized_irr'].mean()
avg_expected_irr = paid_df['expected_irr'].mean()

col1, col2 = st.columns(2)
with col1:
    st.metric("Average Realized IRR", f"{avg_realized_irr:.2%}" if pd.notnull(avg_realized_irr) else "N/A")
with col2:
    st.metric("Average Expected IRR", f"{avg_expected_irr:.2%}" if pd.notnull(avg_expected_irr) else "N/A")

# ----------------------------
# IRR per Loan Table
# ----------------------------
st.subheader("IRR by Loan")

irr_display = paid_df[[
    'loan_id', 'deal_name', 'partner_source', 'funding_date', 'payoff_date',
    'total_invested', 'total_paid', 'realized_irr', 'expected_irr'
]].copy()

irr_display['realized_irr'] = irr_display['realized_irr'].map(lambda x: f"{x:.2%}" if pd.notnull(x) else "")
irr_display['expected_irr'] = irr_display['expected_irr'].map(lambda x: f"{x:.2%}" if pd.notnull(x) else "")
irr_display['total_invested'] = irr_display['total_invested'].map(lambda x: f"${x:,.2f}")
irr_display['total_paid'] = irr_display['total_paid'].map(lambda x: f"${x:,.2f}")

st.dataframe(irr_display.sort_values(by='realized_irr', ascending=False), use_container_width=True)

# ----------------------------
# IRR by Partner
# ----------------------------
st.subheader("Average IRR by Partner")

irr_by_partner = paid_df.groupby('partner_source').agg(
    avg_irr=('realized_irr', 'mean'),
    deal_count=('loan_id', 'count')
).dropna().reset_index()

irr_chart = alt.Chart(irr_by_partner).mark_bar().encode(
    x=alt.X('avg_irr:Q', title="Average IRR", axis=alt.Axis(format=".0%")),
    y=alt.Y('partner_source:N', title="Partner", sort='-x'),
    color=alt.value(PRIMARY_COLOR),
    tooltip=['partner_source:N', alt.Tooltip('avg_irr:Q', format=".2%"), 'deal_count:Q']
).properties(
    width=700,
    height=400,
    title="Average IRR by Partner"
)

st.altair_chart(irr_chart, use_container_width=True)

# ----------------------------
# Vintage/Cohort/Waterfall Scaffold
# ----------------------------
st.subheader("Cohort & Vintage Analysis (Prototype)")

# Create cohort column (e.g., 2024-Q1)
df['cohort'] = df['funding_date'].dt.to_period('Q').astype(str)
cohort_summary = df.groupby('cohort').agg(
    loans=('loan_id', 'count'),
    capital_deployed=('csl_participation_amount', 'sum'),
    capital_returned=('total_paid', 'sum'),
    avg_roi=('roi', 'mean')
).reset_index()

st.dataframe(cohort_summary.sort_values('cohort'), use_container_width=True)

# ----------------------------
# Cohort Repayment Heatmap (% of RTR Paid Over Time)
# ----------------------------
st.subheader("Cohort Repayment Heatmap (Cumulative % of RTR Paid)")

# Filter relevant loans
cohort_df = df[
    (df['funding_date'].notna()) &
    (df['our_rtr'] > 0) &
    (df['total_paid'] >= 0)
].copy()

# Extract cohort
cohort_df['cohort'] = cohort_df['funding_date'].dt.to_period('Q').astype(str)
cohort_df['funding_month'] = cohort_df['funding_date'].dt.to_period('M')

# Calculate "months since funding"
cohort_df['current_month'] = pd.Timestamp.today().to_period('M')
cohort_df['months_since_funding'] = ((cohort_df['current_month'] - cohort_df['funding_month']).apply(lambda x: x.n)).clip(lower=0)

# Create base repayment % for each loan
cohort_df['repayment_pct'] = (cohort_df['total_paid'] / cohort_df['our_rtr']).clip(upper=1.0)

# Group by cohort and months since funding
repayment_grid = cohort_df.groupby(['cohort', 'months_since_funding']).agg(
    avg_repayment_pct=('repayment_pct', 'mean'),
    loans=('loan_id', 'count')
).reset_index()

# Build heatmap
heatmap = alt.Chart(repayment_grid).mark_rect().encode(
    x=alt.X('months_since_funding:O', title="Months Since Funding"),
    y=alt.Y('cohort:N', title="Cohort (Funding Quarter)", sort='-y'),
    color=alt.Color('avg_repayment_pct:Q', title="% Repaid", scale=alt.Scale(scheme="greens")),
    tooltip=[
        alt.Tooltip('cohort:N', title="Cohort"),
        alt.Tooltip('months_since_funding:O', title="Months Since Funding"),
        alt.Tooltip('avg_repayment_pct:Q', title="Avg % Repaid", format=".0%"),
        alt.Tooltip('loans:Q', title="# of Loans")
    ]
).properties(
    width=700,
    height=400,
    title="Repayment Progress by Cohort"
)

st.altair_chart(heatmap, use_container_width=True)

# ----------------------------
# Capital Flow Waterfall Chart
# ----------------------------
st.subheader("Capital Flow Waterfall")

# Summarize values
total_invested = df['total_invested'].sum()
platform_fees = (df['csl_participation_amount'] * 0.03).sum()
commission_fees = df['commission_fees'].sum()
capital_returned = df['total_paid'].sum()

net_investment = total_invested - platform_fees - commission_fees
net_gain = capital_returned - net_investment

# Build waterfall data
waterfall_data = pd.DataFrame([
    {"label": "Capital Deployed", "value": total_invested, "type": "start"},
    {"label": "Less Platform Fees", "value": -platform_fees, "type": "decrease"},
    {"label": "Less Commission Fees", "value": -commission_fees, "type": "decrease"},
    {"label": "Net Investment", "value": net_investment, "type": "subtotal"},
    {"label": "Capital Returned", "value": capital_returned, "type": "increase"},
    {"label": "Net Gain/Loss", "value": net_gain, "type": "net"}
])

# Compute cumulative position
cumulative = []
running_total = 0
for _, row in waterfall_data.iterrows():
    if row['type'] == "start":
        start = row['value']
        end = start
    else:
        start = running_total
        end = start + row['value']
    cumulative.append({
        "label": row['label'],
        "start": start,
        "end": end,
        "color": (
            "#2ca02c" if row['value'] > 0 else "#d62728"
            if row['type'] not in ["start", "subtotal"] else "#1f77b4"
        ),
        "value": row['value']
    })
    running_total = end

wf_df = pd.DataFrame(cumulative)

# Create bars
bars = alt.Chart(wf_df).mark_bar().encode(
    x=alt.X('label:N', title=""),
    y=alt.Y('start:Q', title="Capital ($)"),
    y2='end:Q',
    color=alt.Color('color:N', scale=None, legend=None),
    tooltip=[
        alt.Tooltip('label:N'),
        alt.Tooltip('value:Q', format="$,.2f")
    ]
).properties(
    width=700,
    height=400,
    title="Capital Waterfall"
)

# Add labels
labels = alt.Chart(wf_df).mark_text(
    dy=-10,
    size=12,
    fontWeight="bold"
).encode(
    x='label:N',
    y='end:Q',
    text=alt.Text('value:Q', format="$,.0f")
)

st.altair_chart(bars + labels, use_container_width=True)
