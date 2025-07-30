# pages/mca_dashboard.py
from utils.imports import *
from utils.config import (
    inject_global_styles,
    inject_logo,
    get_supabase_client,
    PRIMARY_COLOR,
    COLOR_PALETTE,
    PLATFORM_FEE_RATE,
)

from scripts.combine_hubspot_mca import combine_deals
from scripts.get_naics_sector_risk import get_naics_sector_risk
from utils.loan_tape_loader import load_unified_loan_customer_data

# Page configuration and branding
st.set_page_config(
    page_title="CSL Capital | MCA Dashboard",
    layout="wide",
)
inject_global_styles()
inject_logo()

# Define risk gradient color scheme
RISK_GRADIENT = [
    "#fff600",
    "#ffc302",
    "#ff8f00",
    "#ff5b00",
    "#ff0505",
]

# ----------------------------
# Supabase connection
# ----------------------------
supabase = get_supabase_client()

# ----------------------------
# Load and prepare single dataframe
# ----------------------------
@st.cache_data(ttl=3600)
def load_mca_deals():
    res = supabase.table("mca_deals").select("*").execute()
    return pd.DataFrame(res.data)

# Use combined dataframe as the single source of truth
df = combine_deals()

# ----------------------------
# Data type conversions and basic calculations
# ----------------------------
df["funding_date"] = pd.to_datetime(df["funding_date"], errors="coerce").dt.date

# Convert all financial columns to numeric, handling any non-numeric values
for col in ["purchase_price", "receivables_amount", "current_balance", "past_due_amount", 
            "principal_amount", "rtr_balance", "amount_hubspot", "total_funded_amount", 
            "total_paid", "outstanding_balance"]:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

# CALCULATION 1: Set past_due_amount to 0 for Matured deals
# Logic: Matured deals have been closed out, so no amount should be past due
df.loc[df["status_category"] == "Matured", "past_due_amount"] = 0

# CALCULATION 2: Calculate past due percentage
# Formula: past_due_amount / current_balance
# Logic: Shows what percentage of the current outstanding balance is past due
df["past_due_pct"] = df.apply(
    lambda row: row["past_due_amount"] / row["current_balance"]
    if pd.notna(row["past_due_amount"]) and pd.notna(row["current_balance"]) and row["current_balance"] > 0
    else 0,
    axis=1
)

# CALCULATION 3: Calculate CSL participation ratio using amount_hubspot
# Formula: amount_hubspot / total_funded_amount
# Logic: CSL's percentage ownership/participation in each deal
df["participation_ratio"] = df["amount_hubspot"] / df["total_funded_amount"].replace(0, pd.NA)

# CALCULATION 4: Calculate CSL's portion of past due amount
# Formula: participation_ratio * past_due_amount
# Logic: CSL's proportional share of any past due amounts based on participation percentage
df["csl_past_due"] = df["participation_ratio"] * df["past_due_amount"]

# CALCULATION 5: Calculate remaining principal balance using logical approach
# Formula: principal_amount - total_paid (what's actually been paid back)
# Logic: Principal outstanding = original principal minus payments received
df["principal_remaining_actual"] = (df["principal_amount"] - df["total_paid"].fillna(0)).clip(lower=0)

# CALCULATION 6: Calculate CSL's principal outstanding 
# Formula: amount_hubspot - (amount_hubspot/principal_amount * total_paid)
# Logic: CSL's share of principal minus CSL's proportional share of payments received
df["participation_ratio"] = df["amount_hubspot"] / df["principal_amount"].replace(0, pd.NA)

df["csl_principal_outstanding"] = df.apply(
    lambda row: max(
        row["amount_hubspot"] -
        (row["amount_hubspot"] / row["principal_amount"] * row["total_paid"]
         if row["principal_amount"] > 0 else 0),
        0
    ) if pd.notna(row["amount_hubspot"]) and pd.notna(row["principal_amount"]) and pd.notna(row["total_paid"])
    else row["amount_hubspot"] if pd.notna(row["amount_hubspot"]) else 0,
    axis=1
)

# CALCULATION 7: Calculate CSL's principal at risk using logical approach
# Formula: Use csl_principal_outstanding for deals that are "Not Current"
# Logic: CSL's actual outstanding principal that could be at risk if deal defaults
df["csl_principal_at_risk"] = df.apply(
    lambda row: row["csl_principal_outstanding"] if row["status_category"] == "Not Current" 
    else 0,
    axis=1
)

# CALCULATION 8: Set CSL principal at risk to 0 for Matured deals
# Logic: Matured deals are closed, so there's no remaining principal at risk
df.loc[df["status_category"] == "Matured", "csl_principal_at_risk"] = 0
df.loc[df["status_category"] == "Matured", "csl_principal_outstanding"] = 0

# CALCULATION 9: Set CSL past due to 0 for Current deals
# Logic: Current deals by definition have no past due amounts
df.loc[df["status_category"] == "Current", "csl_past_due"] = 0

# CALCULATION 9a: Calculate CSL % of balance at risk
df["csl_at_risk_pct"] = df.apply(
    lambda row: row["csl_past_due"] / row["csl_principal_outstanding"]
    if pd.notna(row["csl_past_due"]) and pd.notna(row["csl_principal_outstanding"]) and row["csl_principal_outstanding"] > 0
    else 0,
    axis=1
)

# CALCULATION 10: Commission rate conversion
# Convert commission field to numeric percentage
df["commission_rate"] = pd.to_numeric(df["commission"], errors="coerce")
if "commission" in df.columns:
    df["commission_rate"] = pd.to_numeric(df["commission"], errors="coerce")
else:
    df["commission_rate"] = 0
    
# CALCULATION 11: Calculate days since funding
# Used for risk scoring - older deals may have higher risk
df["days_since_funding"] = (pd.Timestamp.today() - pd.to_datetime(df["funding_date"])).dt.days



# ----------------------------
# NEW PROJECTED PAYMENT ANALYSIS CALCULATIONS
# ----------------------------

# Convert necessary columns to numeric for payment projections
for col in ["loan_term", "factor_rate", "total_paid"]:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

# CALCULATION 12: Calculate Total Repayment Amount
# Formula: principal_amount * factor_rate
# Logic: Total amount merchant should pay back over the life of the deal
df["total_repayment"] = df["principal_amount"] * df["factor_rate"]

# CALCULATION 13: Calculate Expected Daily Payment
# Formula: total_repayment / loan_term (in days)
# Logic: Daily payment amount based on linear amortization schedule
df["expected_daily_payment"] = df.apply(
    lambda row: row["total_repayment"] / row["loan_term"]
    if pd.notna(row["total_repayment"]) and pd.notna(row["loan_term"]) and row["loan_term"] > 0
    else 0,
    axis=1
)

# CALCULATION 14: Calculate Expected Payments to Date
# Formula: expected_daily_payment * days_since_funding
# Logic: Total payments we should have received by today based on schedule
# Constraint: Cannot exceed total_repayment amount
df["expected_payments_to_date"] = df.apply(
    lambda row: min(
        row["expected_daily_payment"] * row["days_since_funding"],
        row["total_repayment"]
    ) if pd.notna(row["expected_daily_payment"]) and pd.notna(row["days_since_funding"]) and pd.notna(row["total_repayment"])
    else 0,
    axis=1
)

# CALCULATION 15: Calculate Payment Delta using combined dataset
# Formula: total_paid - expected_payments_to_date
# Logic: Positive = merchant ahead of schedule, Negative = merchant behind schedule
df["payment_delta"] = df.apply(
    lambda row: (row["total_paid"] - row["expected_payments_to_date"])
    if pd.notna(row["total_paid"]) and pd.notna(row["expected_payments_to_date"])
    else 0,
    axis=1
)

# CALCULATION 15a: Calculate performance ratio (actual paid vs expected)
df["performance_ratio"] = df.apply(
    lambda row: row["total_paid"] / row["expected_payments_to_date"]
    if pd.notna(row["total_paid"]) and pd.notna(row["expected_payments_to_date"]) and row["expected_payments_to_date"] > 0
    else 0,
    axis=1
)

# CALCULATION 16: Calculate Projected Status
# Logic: Simplified status - Current, Not Current, or Matured
def calculate_projected_status(row):
    # already Matured?
    if row["status_category"] == "Matured":
        return "Matured"

    exp = row["expected_payments_to_date"]
    delta = row["payment_delta"]
    perf = row["performance_ratio"]

    # new deals or missing data → assume Current
    if pd.isna(delta) or pd.isna(exp) or exp == 0:
        return "Current"

    # % variance
    var_pct = delta / exp

    # if ≥10% behind or performance_ratio < 0.9 → Not Current
    if var_pct <= -0.10 or perf < 0.90:
        return "Not Current"
    return "Current"

df["projected_status"] = df.apply(calculate_projected_status, axis=1)

# CALCULATION 17: Calculate RTR percentage
# Formula: (principal_amount - rtr_balance) / principal_amount
# Logic: Shows percentage of principal that has been recovered
df["rtr_pct"] = df.apply(
    lambda row: (row["principal_amount"] - row["rtr_balance"]) / row["principal_amount"]
    if pd.notna(row["principal_amount"]) and pd.notna(row["rtr_balance"]) and row["principal_amount"] > 0
    else 0,
    axis=1
)

# ----------------------------
# Industry/NAICS Processing
# ----------------------------
# Load NAICS sector risk data
naics_risk_df = get_naics_sector_risk()

# Extract 2-digit sector code from industry (full NAICS code)
if 'industry' in df.columns:
    df['sector_code'] = df['industry'].astype(str).str[:2].str.zfill(2)
    
    # Consolidate manufacturing sectors (31, 32, 33 -> Manufacturing)
    df['sector_code_consolidated'] = df['sector_code'].copy()
    df.loc[df['sector_code'].isin(['31', '32', '33']), 'sector_code_consolidated'] = 'Manufacturing'
    
# Join with NAICS sector risk data
if not naics_risk_df.empty:
    # Create consolidated risk data for manufacturing (31, 32, 33)
    manufacturing_codes = ['31', '32', '33']
    manufacturing_subset = naics_risk_df[naics_risk_df['sector_code'].isin(manufacturing_codes)]

    if not manufacturing_subset.empty:
        avg_risk_score = manufacturing_subset['risk_score'].mean()
        manufacturing_risk_row = pd.DataFrame([{
            'sector_code': 'Manufacturing',
            'sector_name': 'Manufacturing (31-33)',
            'risk_score': avg_risk_score,
            'risk_profile': 'Medium'  # You can improve this by computing based on avg
        }])

        naics_risk_consolidated = pd.concat([naics_risk_df, manufacturing_risk_row], ignore_index=True)
    else:
        naics_risk_consolidated = naics_risk_df

    # Join on consolidated sector codes
    df = df.merge(
        naics_risk_consolidated,
        left_on='sector_code_consolidated',
        right_on='sector_code',
        how='left',
        suffixes=('', '_risk')
    )
else:
    st.warning("Industry column not found in data")

# ----------------------------
# Filters
# ----------------------------
st.title("MCA Deals Dashboard")

min_date = df["funding_date"].min()
max_date = df["funding_date"].max()
start_date, end_date = st.date_input(
    "Filter by Funding Date",
    [min_date, max_date],
    min_value=min_date,
    max_value=max_date
)
df = df[(df["funding_date"] >= start_date) & (df["funding_date"] <= end_date)]

status_options = ["All"] + list(df["status_category"].dropna().unique())
status_category_filter = st.radio("Current Status", status_options, index=0,horizontal=True)
if status_category_filter != "All":
    df = df[df["status_category"] == status_category_filter]

# ----------------------------
# Calculate all metrics with detailed comments
# ----------------------------

# PORTFOLIO METRICS
# Count total deals (excluding canceled)
total_deals = len(df)

# Count deals by status category
total_matured = (df["status_category"] == "Matured").sum()
total_current = (df["status_category"] == "Current").sum()
total_non_current = (df["status_category"] == "Not Current").sum()

# Calculate outstanding deals (Current + Not Current, excludes Matured)
outstanding_total = total_current + total_non_current

# Calculate percentages of outstanding deals
pct_current = total_current / outstanding_total if outstanding_total > 0 else 0
pct_non_current = total_non_current / outstanding_total if outstanding_total > 0 else 0

# CSL INVESTMENT METRICS using combined dataset
# CALCULATION 18: Total CSL capital deployed across all deals
# Sum of all CSL participation amounts from amount_hubspot
csl_capital_deployed = df["amount_hubspot"].sum()

# CALCULATION 19: Total CSL past due exposure
# Sum of CSL's proportional share of all past due amounts
total_csl_past_due = df["csl_past_due"].sum()

# CALCULATION 20: Outstanding CSL Principal (Capital at Risk)
# This represents CSL's actual principal outstanding on deals that are "Not Current"
# Formula: Sum of csl_principal_outstanding for Not Current deals only
at_risk = df[df["status_category"] == "Not Current"]
total_csl_at_risk = at_risk["csl_principal_outstanding"].sum()

# COMMISSION METRICS
# CALCULATION 21: Average commission rate across all deals
average_commission_pct = df["commission_rate"].mean()

# CALCULATION 22: Total commission paid by CSL using combined dataset
# Formula: Sum of (amount_hubspot * commission_rate) for all deals
total_commission_paid = (df["amount_hubspot"] * df["commission_rate"]).sum()

# CALCULATION 23: Average commission rate weighted by CSL participation using combined dataset
# Formula: total_commission_paid / total_csl_participation
average_commission_on_loan = total_commission_paid / df["amount_hubspot"].sum() if df["amount_hubspot"].sum() > 0 else 0

# RISK ANALYSIS DATAFRAMES
# CALCULATION 24: Create dataframe for non-current, non-matured deals with risk metrics
not_current_df = df[(df["status_category"] != "Current") & (df["status_category"] != "Matured")].copy()

# Calculate at-risk percentage for visualization
# Formula: past_due_amount / current_balance
not_current_df["at_risk_pct"] = not_current_df["past_due_amount"] / not_current_df["current_balance"].clip(lower=1)

# Filter to only deals with actual risk (past due amount > 0)
not_current_df = not_current_df[not_current_df["at_risk_pct"] > 0]

# CALCULATION 25: Risk scoring for top 10 highest risk deals
# Create dataframe for deals meeting risk criteria:
# - More than 30 days old (seasoned deals)
# - Past due amount > 1% of current balance (meaningful delinquency)
# - Not Current or Matured (active problematic deals)
risk_df = df[
    (df["days_since_funding"] > 30) &
    (df["past_due_amount"] > df["current_balance"] * 0.01) &
    (df["status_category"] != "Current") &
    (df["status_category"] != "Matured")
].copy()

if len(risk_df) > 0:
    # CALCULATION 26: Calculate risk score components
    # Component 1: Past due percentage (70% weight)
    # Formula: past_due_amount / current_balance (minimum 1 to avoid division by zero)
    risk_df["past_due_pct_calc"] = risk_df["past_due_amount"] / risk_df["current_balance"].clip(lower=1)
    
    # Component 2: Age weight (30% weight)
    # Formula: days_since_funding / max_days_since_funding (normalized 0-1)
    max_days = risk_df["days_since_funding"].max()
    if max_days > 0:
        risk_df["age_weight"] = risk_df["days_since_funding"] / max_days
    else:
        risk_df["age_weight"] = 0
    
    # CALCULATION 27: Final risk score
    # Formula: (past_due_percentage * 0.7) + (age_weight * 0.3)
    # Range: 0.0 to 1.0 (higher = more risk)
    risk_df["risk_score"] = risk_df["past_due_pct_calc"] * 0.7 + risk_df["age_weight"] * 0.3
    
    # Get top 10 highest risk deals
    top_risk = risk_df.sort_values("risk_score", ascending=False).head(10).copy()

# ----------------------------
# Display metrics sections
# ----------------------------

# Portfolio Summary
st.subheader("CSL Portfolio Summary")
col1, col2, col3 = st.columns(3)
col1.metric("Total Deals", total_deals, help="Total number of deals in the portfolio (excludes Canceled deals)")
col2.metric("Matured Deals", total_matured, help="Number of deals that have been fully paid off and closed")
col3.metric("Current Deals", total_current, help="Number of deals that are performing and up-to-date on payments")

col4, col5, col6 = st.columns(3)
col4.metric("Not Current Deals", total_non_current, help="Number of deals that are delinquent or past due on payments")
col5.metric("Pct. Outstanding Deals Current", f"{pct_current:.1%}", help="Percentage of active deals (Current + Not Current) that are performing well")
col6.metric("Pct. Outstanding Deals Not Current", f"{pct_non_current:.1%}", help="Percentage of active deals (Current + Not Current) that are delinquent")

# CSL Investment Overview
st.subheader("CSL Investment Overview")
col7, col8, col9 = st.columns(3)
col7.metric("Capital Deployed", f"${csl_capital_deployed:,.0f}", help="Total amount of capital that CSL has invested across all deals (sum of CSL participation amounts)")
col8.metric("Past Due Exposure", f"${total_csl_past_due:,.0f}", help="CSL's proportional share of all past due amounts based on participation ratio")
col9.metric("Outstanding CSL Principal", f"${total_csl_at_risk:,.0f}", help="CSL's share of remaining principal on 'Not Current' deals - represents capital that could be at risk if deals default")

# CSL Commission Summary
st.subheader("CSL Commission Summary")
col10, col11, col12 = st.columns(3)
col10.metric("Avg. Commission Rate", f"{average_commission_pct:.2%}", help="Average commission rate across all deals (unweighted average)")
col11.metric("Avg. Applied to Participation", f"{average_commission_on_loan:.2%}", help="Average commission rate weighted by CSL participation amounts - shows effective commission rate paid")
col12.metric("Total Commission Paid", f"${total_commission_paid:,.0f}", help="Total commission payments made by CSL across all deals (sum of participation × commission rate)")

# ----------------------------
# Deal Type Composition Visual
# ----------------------------

# Deal Type Composition
if 'deal_type' in df.columns:
    st.subheader("Deal Type Composition")
    
    # Calculate deal type percentages
    deal_type_counts = df["deal_type"].fillna("Unknown").value_counts(normalize=True)
    deal_type_summary = pd.DataFrame({
        "deal_type": deal_type_counts.index.astype(str),
        "percentage": deal_type_counts.values,
        "count": df["deal_type"].fillna("Unknown").value_counts().values
    })
    
    # Deal type bar chart
    deal_type_chart = alt.Chart(deal_type_summary).mark_bar().encode(
        x=alt.X("deal_type:N", title="Deal Type", axis=alt.Axis(labelAngle=-45)),
        y=alt.Y("percentage:Q", title="% of Total Deals", axis=alt.Axis(format=".0%")),
        color=alt.Color("deal_type:N", scale=alt.Scale(range=RISK_GRADIENT), legend=None),
        tooltip=[
            alt.Tooltip("deal_type:N", title="Deal Type"),
            alt.Tooltip("count:Q", title="Number of Deals"),
            alt.Tooltip("percentage:Q", title="% of Total", format=".1%")
        ]
    ).properties(
        width=700,
        height=350,
        title="Distribution of Deal Types"
    )
    
    st.altair_chart(deal_type_chart, use_container_width=True)

# ----------------------------
# Loan Tape Filter and Display (ENHANCED WITH NEW FIELDS)
# ----------------------------

# Status Category Filter for Loan Tape
st.subheader("Loan Tape")
loan_tape_status_options = ["All"] + list(df["status_category"].dropna().unique())
loan_tape_status_filter = st.radio("Filter Loan Tape by Status Category", loan_tape_status_options, index=0, key="loan_tape_filter",horizontal=True)

# Apply filter to dataframe for loan tape
loan_tape_df = df.copy()
if loan_tape_status_filter != "All":
    loan_tape_df = loan_tape_df[loan_tape_df["status_category"] == loan_tape_status_filter]

# Enhanced loan tape with new projected payment fields using combined dataset
loan_tape = loan_tape_df[[
    "loan_id", "dba","nature_of_business_mca", "funding_date", "status_category","projected_status",
    "csl_past_due", "past_due_pct", "performance_ratio",
    "current_balance", "csl_principal_outstanding", "performance_details",
    "expected_payments_to_date", "payment_delta"
]].copy()

loan_tape.rename(columns={
    "loan_id": "Loan ID",
    "dba": "Deal",
    "funding_date": "Funding Date",
    "status_category": "Status Category",
    "csl_past_due": "CSL Past Due ($)",
    "past_due_pct": "Past Due %",
    "performance_ratio": "Performance Ratio",
    "current_balance": "Total Loan Remaining ($)",
    "csl_principal_outstanding": "CSL Principal Outstanding ($)",
    "performance_details": "Performance Notes",
    "expected_payments_to_date": "Expected Payments to Date ($)",
    "payment_delta": "Payment Delta ($)",
    "projected_status": "Projected Status"
}, inplace=True)

loan_tape["Past Due %"] = pd.to_numeric(loan_tape["Past Due %"], errors="coerce").fillna(0)*100
loan_tape["CSL Past Due ($)"] = pd.to_numeric(loan_tape["CSL Past Due ($)"], errors="coerce").fillna(0)
loan_tape["Total Loan Remaining ($)"] = pd.to_numeric(loan_tape["Total Loan Remaining ($)"], errors="coerce").fillna(0)
loan_tape["CSL Principal Outstanding ($)"] = pd.to_numeric(loan_tape["CSL Principal Outstanding ($)"], errors="coerce").fillna(0)
loan_tape["Performance Ratio"] = pd.to_numeric(loan_tape["Performance Ratio"], errors="coerce").fillna(0)
loan_tape["Expected Payments to Date ($)"] = pd.to_numeric(loan_tape["Expected Payments to Date ($)"], errors="coerce").fillna(0)
loan_tape["Payment Delta ($)"] = pd.to_numeric(loan_tape["Payment Delta ($)"], errors="coerce").fillna(0)

st.dataframe(
    loan_tape,
    use_container_width=True,
    column_config={
        "Past Due %": st.column_config.NumberColumn("Past Due %", format="%.2f"),
        "CSL Past Due ($)": st.column_config.NumberColumn("CSL Past Due ($)", format="$%.0f"),
        "Total Loan Remaining ($)": st.column_config.NumberColumn("Total Loan Remaining ($)", format="$%.0f", help="Full loan balance remaining including interest/fees"),
        "CSL Principal Outstanding ($)": st.column_config.NumberColumn("CSL Principal Outstanding ($)", format="$%.0f", help="CSL's principal investment minus payments received"),
        "Performance Ratio": st.column_config.NumberColumn("Performance Ratio", format="%.2f"),
        "Expected Payments to Date ($)": st.column_config.NumberColumn("Expected Payments to Date ($)", format="$%.0f", help="Total payments expected by today based on repayment schedule"),
        "Payment Delta ($)": st.column_config.NumberColumn("Payment Delta ($)", format="$%.0f", help="Variance between actual and expected payments. Positive = ahead, negative = behind"),
        "Projected Status": st.column_config.TextColumn("Projected Status", help="Current, Not Current, or Matured based on payment performance vs. expected schedule")
    }
)

# ----------------------------
# Top 5 Biggest Loans Outstanding
# ----------------------------

st.subheader("Top 5 Biggest CSL Investments Outstanding")

# Filter to non-matured deals and sort by CSL participation amount using combined dataset
biggest_csl_loans = df[df["status_category"] != "Matured"].copy()
biggest_csl_loans = biggest_csl_loans.sort_values("amount_hubspot", ascending=False).head(5)

biggest_csl_loans_display = biggest_csl_loans[[
    "loan_id", "dba", "status_category", "amount_hubspot", "csl_principal_outstanding", 
    "csl_past_due", "principal_amount", "principal_remaining_actual", "current_balance", 
    "total_paid", "participation_ratio"
]].copy()

biggest_csl_loans_display.rename(columns={
    "loan_id": "Loan ID",
    "dba": "Deal Name",
    "status_category": "Status",
    "amount_hubspot": "CSL Investment ($)",
    "csl_principal_outstanding": "CSL Principal Outstanding ($)",
    "csl_past_due": "CSL Past Due ($)",
    "principal_amount": "Original Principal ($)",
    "principal_remaining_actual": "Total Principal Outstanding ($)",
    "current_balance": "Total Loan Remaining ($)",
    "total_paid": "Total Paid ($)",
    "participation_ratio": "CSL Participation %"
}, inplace=True)

# Clean up numeric data using combined dataset fields
for col in ["CSL Investment ($)", "CSL Principal Outstanding ($)", "CSL Past Due ($)", 
            "Original Principal ($)", "Total Principal Outstanding ($)", "Total Loan Remaining ($)", 
            "Total Paid ($)"]:
    biggest_csl_loans_display[col] = pd.to_numeric(biggest_csl_loans_display[col], errors="coerce").fillna(0)

biggest_csl_loans_display["CSL Participation %"] = pd.to_numeric(biggest_csl_loans_display["CSL Participation %"], errors="coerce").fillna(0)

st.dataframe(
    biggest_csl_loans_display,
    use_container_width=True,
    column_config={
        "CSL Investment ($)": st.column_config.NumberColumn("CSL Investment ($)", format="$%.0f"),
        "CSL Principal Outstanding ($)": st.column_config.NumberColumn("CSL Principal Outstanding ($)", format="$%.0f", help="CSL's remaining principal (investment minus payments received)"),
        "CSL Past Due ($)": st.column_config.NumberColumn("CSL Past Due ($)", format="$%.0f"),
        "Original Principal ($)": st.column_config.NumberColumn("Original Principal ($)", format="$%.0f"),
        "Total Principal Outstanding ($)": st.column_config.NumberColumn("Total Principal Outstanding ($)", format="$%.0f"),
        "Total Loan Remaining ($)": st.column_config.NumberColumn("Total Loan Remaining ($)", format="$%.0f", help="Full loan balance including interest/fees"),
        "Total Paid ($)": st.column_config.NumberColumn("Total Paid ($)", format="$%.0f"),
        "CSL Participation %": st.column_config.NumberColumn("CSL Participation %", format="%.1%"),
    }
)

# ----------------------------
# Charts and visualizations
# ----------------------------

# Distribution of Deal Status (Bar Chart)
status_category_counts = df["status_category"].fillna("Unknown").value_counts(normalize=True)

# Calculate unpaid CSL Principal by status category
status_csl_principal = df.groupby(df["status_category"].fillna("Unknown"))["csl_principal_at_risk"].sum()

status_category_chart = pd.DataFrame({
    "status_category": status_category_counts.index.astype(str),
    "Share": status_category_counts.values,
    "unpaid_csl_principal": status_csl_principal.reindex(status_category_counts.index).fillna(0).values
})

bar = alt.Chart(status_category_chart).mark_bar().encode(
    x=alt.X(
        "status_category:N",
        title="Status Category",
        sort=alt.EncodingSortField(field="Share", order="descending"),
        axis=alt.Axis(labelAngle=-90)
    ),
    y=alt.Y("Share:Q", title="Percent of Deals", axis=alt.Axis(format=".0%")),
    color=alt.Color(
        "Share:Q",
        scale=alt.Scale(range=["#d73027", "#fc8d59", "#fee08b", "#d9ef8b", "#91bfdb", "#4575b4"]),
        legend=None
    ),
    tooltip=[
        alt.Tooltip("status_category", title="Status"),
        alt.Tooltip("Share:Q", title="Share", format=".2%"),
        alt.Tooltip("unpaid_csl_principal:Q", title="Unpaid CSL Principal (Est.)", format="$,.0f")
    ]
).properties(
    width=700,
    height=350,
    title="Distribution of Deal Status"
)

st.altair_chart(bar, use_container_width=True)

# Risk Chart: Percentage of Balance at Risk
if len(not_current_df) > 0:
    risk_chart = alt.Chart(not_current_df).mark_bar().encode(
        x=alt.X(
            "loan_id:N",
            title="Loan ID",
            sort="-y",
            axis=alt.Axis(labelAngle=-90)
        ),
               y=alt.Y("csl_at_risk_pct:Q", title="Pct. of CSL Balance at Risk"),
        tooltip=[
            alt.Tooltip("csl_past_due:Q", title="CSL Past Due ($)", format="$,.0f"),
            alt.Tooltip("csl_principal_outstanding:Q", title="CSL Outstanding ($)", format="$,.0f"),
            alt.Tooltip("csl_at_risk_pct:Q", title="% CSL at Risk", format=".2%")
        ]
            ).properties(
                width=850,
                height=400,
                title="Percentage of Balance at Risk (Non-Current, Non-Matured Deals)"
            )

    st.altair_chart(risk_chart, use_container_width=True)
else:
    st.info("No non-current, non-matured deals with past due amounts to display.")

# Top 10 Highest Risk Deals
st.subheader("Top 10 Highest Risk Deals")

if len(risk_df) > 0:
    # Risk score bar chart with Loan ID on x-axis
    bar_chart = alt.Chart(top_risk).mark_bar().encode(
        x=alt.X(
            "loan_id:N",
            title="Loan ID",
            sort="-y",
            axis=alt.Axis(labelAngle=-90)
        ),
        y=alt.Y("risk_score:Q", title="Risk Score"),
        color=alt.Color(
            "risk_score:Q",
            scale=alt.Scale(range=RISK_GRADIENT),
            legend=alt.Legend(title="Risk Score")
        ),
        tooltip=[
            alt.Tooltip("loan_id:N", title="Loan ID"),
            alt.Tooltip("dba:N", title="Deal Name"),
            alt.Tooltip("status_category:N", title="Status Category"),
            alt.Tooltip("funding_date:T", title="Funding Date"),
            alt.Tooltip("past_due_amount:Q", title="Past Due Amount", format="$,.0f"),
            alt.Tooltip("current_balance:Q", title="Current Balance", format="$,.0f"),
            alt.Tooltip("risk_score:Q", title="Risk Score", format=".3f")
        ]
    ).properties(
        width=700,
        height=400,
        title="(Excludes New and Performing Loans)"
    )

    st.altair_chart(bar_chart, use_container_width=True)

    # Top 10 Risk table
    top_risk_display = top_risk[[
        "loan_id", "dba", "status_category", "funding_date", "risk_score",
        "csl_past_due", "current_balance"
    ]].copy()

    top_risk_display.rename(columns={
        "loan_id": "Loan ID",
        "dba": "Deal",
        "status_category": "Status",
        "funding_date": "Funded",
        "risk_score": "Risk Score",
        "csl_past_due": "CSL Past Due ($)",
        "current_balance": "Current Balance ($)"
    }, inplace=True)

    # Clean up numeric data
    top_risk_display["Risk Score"] = pd.to_numeric(top_risk_display["Risk Score"], errors="coerce").fillna(0)
    top_risk_display["CSL Past Due ($)"] = pd.to_numeric(top_risk_display["CSL Past Due ($)"], errors="coerce").fillna(0)
    top_risk_display["Current Balance ($)"] = pd.to_numeric(top_risk_display["Current Balance ($)"], errors="coerce").fillna(0)

    st.dataframe(
        top_risk_display,
        use_container_width=True,
        column_config={
            "CSL Past Due ($)": st.column_config.NumberColumn(
                "CSL Past Due ($)",
                format="$%.0f",
                help="CSL portion of past due amount"
            ),
            "Current Balance ($)": st.column_config.NumberColumn(
                "Current Balance ($)",
                format="$%.0f", 
                help="Current outstanding balance"
            ),
            "Risk Score": st.column_config.NumberColumn(
                "Risk Score",
                format="%.3f",
                help="Calculated risk score"
            ),
        }
    )

    # Scatter Plot
    scatter = alt.Chart(risk_df).mark_circle(size=80).encode(
        x=alt.X("past_due_pct_calc:Q", title="% Past Due", axis=alt.Axis(format=".0%")),
        y=alt.Y("days_since_funding:Q", title="Days Since Funding"),
        size=alt.Size("risk_score:Q", title="Risk Score", scale=alt.Scale(range=[50, 400])),
        color=alt.Color(
            "risk_score:Q",
            scale=alt.Scale(range=RISK_GRADIENT),
            title="Risk Score"
        ),
        tooltip=[
            alt.Tooltip("loan_id:N", title="Loan ID"),
            alt.Tooltip("dba:N", title="Deal Name"),
            alt.Tooltip("status_category:N", title="Status"),
            alt.Tooltip("funding_date:T", title="Funded"),
            alt.Tooltip("risk_score:Q", title="Risk Score", format=".2f"),
            alt.Tooltip("past_due_pct_calc:Q", title="% Past Due", format=".2%"),
            alt.Tooltip("days_since_funding:Q", title="Days Since Funding")
        ]
    ).properties(
        width=700,
        height=400,
        title="Risk Score by Past Due % and Deal Age"
    )

    threshold_x = alt.Chart(pd.DataFrame({"x": [0.10]})).mark_rule(
        strokeDash=[4, 4], color="gray"
    ).encode(x="x:Q")

    threshold_y = alt.Chart(pd.DataFrame({"y": [90]})).mark_rule(
        strokeDash=[4, 4], color="gray"
    ).encode(y="y:Q")

    st.altair_chart(
        scatter + threshold_x + threshold_y,
        use_container_width=True
    )
else:
    st.info("No deals meet the risk criteria for analysis.")

st.markdown("""
#### Understanding the Risk Score

The **Risk Score** ranges between **0.00 and 1.00** and blends two key factors:
- **70% weight**: The **percentage of the balance that is past due**
- **30% weight**: The **age of the deal** (older deals score higher)

| Risk Score | Interpretation            |
|------------|---------------------------|
| 0.00–0.19  | Very Low Risk             |
| 0.20–0.49  | Low to Moderate Risk      |
| 0.50–0.74  | Elevated Risk             |
| 0.75–1.00  | High Risk of Loss         |

*Example*: A score of **0.80** implies the deal is either **very delinquent**, **very old**, or both.
New deals (< 30 days old), those with low delinquency (<1%), or with status 'Current' are excluded.
""")

# ----------------------------
# INDUSTRY & PORTFOLIO INSIGHTS
# ----------------------------

st.markdown("---")
st.header("Portfolio Composition & Risk Insights")

# Industry Analysis
if 'sector_name' in df.columns and not df['sector_name'].isna().all():
    st.subheader("Deal Distribution by Industry Sector")
    st.caption("*Risk scores are based on industry sector risk profiles from NAICS data. Risk Score 5 represents the highest industry risk (darkest red).*")
    
    # Group by risk_score from naics_sector_risk_profile table using combined dataset
    industry_summary = df.groupby(['risk_score']).agg({
        'loan_id': 'count',
        'amount_hubspot': 'sum',
        'csl_principal_outstanding': 'sum',
        'risk_profile': 'first',
        'sector_name': lambda x: ', '.join(x.unique()[:3])  # Show up to 3 sector names per risk score
    }).reset_index()
    
    industry_summary.columns = ['Risk Score', 'Deal Count', 'CSL Capital Deployed', 'CSL Principal Outstanding', 'Risk Profile', 'Sectors']
    industry_summary = industry_summary.sort_values('Deal Count', ascending=False)
    
    # Industry deals chart grouped by risk score
    industry_chart = alt.Chart(industry_summary).mark_bar().encode(
        x=alt.X('Risk Score:O', title='Risk Score', axis=alt.Axis(labelAngle=0)),
        y=alt.Y('Deal Count:Q', title='Number of Deals'),
        color=alt.Color('Risk Score:O', 
                       scale=alt.Scale(range=RISK_GRADIENT),
                       title='Industry Risk Score'),
        tooltip=[
            alt.Tooltip('Risk Score:O', title='Risk Score'),
            alt.Tooltip('Deal Count:Q', title='Number of Deals'),
            alt.Tooltip('CSL Capital Deployed:Q', title='Capital Deployed', format='$,.0f'),
            alt.Tooltip('CSL Principal Outstanding:Q', title='Principal Outstanding', format='$,.0f'),
            alt.Tooltip('Risk Profile:N', title='Risk Profile'),
            alt.Tooltip('Sectors:N', title='Primary Sectors')
        ]
    ).properties(
        width=800,
        height=400,
        title='Number of Deals by Risk Score'
    )
    
    st.altair_chart(industry_chart, use_container_width=True)
    
    # Capital exposure by risk score
    st.subheader("CSL Capital Exposure by Industry")
    st.caption("*Risk scores are based on industry sector risk profiles from NAICS data. Risk Score 5 represents the highest industry risk (darkest red).*")
    
    capital_chart = alt.Chart(industry_summary).mark_bar().encode(
        x=alt.X('Risk Score:O', title='Risk Score', axis=alt.Axis(labelAngle=0)),
        y=alt.Y('CSL Principal Outstanding:Q', title='CSL Principal Outstanding ($)', axis=alt.Axis(format='$,.0f')),
        color=alt.Color('Risk Score:O',
                       scale=alt.Scale(range=RISK_GRADIENT),
                       title='Industry Risk Score'),
        tooltip=[
            alt.Tooltip('Risk Score:O', title='Risk Score'),
            alt.Tooltip('CSL Capital Deployed:Q', title='Total Capital Deployed', format='$,.0f'),
            alt.Tooltip('CSL Principal Outstanding:Q', title='Principal Outstanding', format='$,.0f'),
            alt.Tooltip('Deal Count:Q', title='Number of Deals'),
            alt.Tooltip('Risk Profile:N', title='Risk Profile'),
            alt.Tooltip('Sectors:N', title='Primary Sectors')
        ]
    ).properties(
        width=800,
        height=400,
        title='CSL Principal Outstanding by Risk Score'
    )
    
    st.altair_chart(capital_chart, use_container_width=True)
    
    # Industry summary table
    st.dataframe(
        industry_summary,
        use_container_width=True,
        column_config={
            "CSL Capital Deployed": st.column_config.NumberColumn("CSL Capital Deployed", format="$%.0f"),
            "CSL Principal Outstanding": st.column_config.NumberColumn("CSL Principal Outstanding", format="$%.0f"),
        }
    )

    # Portfolio Summary Table by Sector
    st.subheader("Portfolio Summary by Industry Sector")
    
    if 'sector_code' in df.columns and not df['sector_code'].isna().all():
        # Create comprehensive sector summary using combined dataset
        sector_portfolio_summary = df.groupby(['sector_code', 'sector_name']).agg({
            'loan_id': 'count',
            'amount_hubspot': 'sum',
            'csl_principal_outstanding': 'sum',
            'risk_score': 'first'
        }).reset_index()
        
        # Calculate percentage of total deals
        sector_portfolio_summary['pct_of_total'] = (sector_portfolio_summary['loan_id'] / sector_portfolio_summary['loan_id'].sum()) * 100
        
        sector_portfolio_summary.columns = ['Sector Number', 'Industry Name', 'Count of Deals', 'Total Deployed', 'Principal Outstanding', 'Risk Score', '% of Total']
        
        # Sort by capital deployed descending
        sector_portfolio_summary = sector_portfolio_summary.sort_values('Total Deployed', ascending=False)
        
        # Add total row
        total_row_portfolio = pd.DataFrame({
            'Sector Number': ['TOTAL'],
            'Industry Name': ['ALL SECTORS'],
            'Count of Deals': [sector_portfolio_summary['Count of Deals'].sum()],
            'Total Deployed': [sector_portfolio_summary['Total Deployed'].sum()],
            'Principal Outstanding': [sector_portfolio_summary['Principal Outstanding'].sum()],
            'Risk Score': [''],
            '% of Total': [100.0]
        })
        sector_portfolio_summary = pd.concat([sector_portfolio_summary, total_row_portfolio], ignore_index=True)
        
        st.dataframe(
            sector_portfolio_summary,
            use_container_width=True,
            column_config={
                "Total Deployed": st.column_config.NumberColumn("Total Deployed", format="$%.0f"),
                "Principal Outstanding": st.column_config.NumberColumn("Principal Outstanding", format="$%.0f"),
                "% of Total": st.column_config.NumberColumn("% of Total", format="%.1f%%"),
            }
        )

# FICO Score Analysis
if 'fico' in df.columns:
    st.subheader("Portfolio Distribution by FICO Score")
    
    # Create FICO bands
    df['fico_band'] = pd.cut(df['fico'], 
                            bins=[0, 580, 620, 660, 700, 740, 850], 
                            labels=['<580', '580-619', '620-659', '660-699', '700-739', '740+'],
                            include_lowest=True)
    
    # FICO analysis using combined dataset
    fico_summary = df.groupby('fico_band').agg({
        'loan_id': 'count',
        'amount_hubspot': 'sum',
        'csl_principal_outstanding': 'sum'
    }).reset_index()
    
    # Calculate percentage of total deals (not capital)
    fico_summary['pct_of_total'] = (fico_summary['loan_id'] / fico_summary['loan_id'].sum()) * 100
    
    fico_summary.columns = ['FICO Band', 'Deal Count', 'CSL Capital Deployed', 'CSL Principal Outstanding', 'Pct of Total']
    
    # FICO deals chart
    fico_deals_chart = alt.Chart(fico_summary).mark_bar().encode(
        x=alt.X('FICO Band:N', title='FICO Score Band'),
        y=alt.Y('Deal Count:Q', title='Number of Deals'),
        color=alt.Color('FICO Band:N', 
                       scale=alt.Scale(range=['#d62728', '#ff7f0e', '#ffbb78', '#2ca02c', '#98df8a', '#1f77b4']),
                       legend=None),
        tooltip=[
            alt.Tooltip('FICO Band:N', title='FICO Band'),
            alt.Tooltip('Deal Count:Q', title='Number of Deals'),
            alt.Tooltip('Pct of Total:Q', title='% of Total Deals', format='.1f'),
            alt.Tooltip('CSL Capital Deployed:Q', title='Capital Deployed', format='$,.0f'),
            alt.Tooltip('CSL Principal Outstanding:Q', title='Principal Outstanding', format='$,.0f')
        ]
    ).properties(
        width=600,
        height=400,
        title='Deal Count by FICO Score Band'
    )
    
    # FICO capital exposure chart
    fico_capital_chart = alt.Chart(fico_summary).mark_bar().encode(
        x=alt.X('FICO Band:N', title='FICO Score Band'),
        y=alt.Y('CSL Principal Outstanding:Q', title='CSL Principal Outstanding ($)', axis=alt.Axis(format='$,.0f')),
        color=alt.Color('FICO Band:N',
                       scale=alt.Scale(range=['#d62728', '#ff7f0e', '#ffbb78', '#2ca02c', '#98df8a', '#1f77b4']),
                       legend=None),
        tooltip=[
            alt.Tooltip('FICO Band:N', title='FICO Band'),
            alt.Tooltip('CSL Capital Deployed:Q', title='Total Capital Deployed', format='$,.0f'),
            alt.Tooltip('CSL Principal Outstanding:Q', title='Principal Outstanding', format='$,.0f'),
            alt.Tooltip('Deal Count:Q', title='Number of Deals'),
            alt.Tooltip('Pct of Total:Q', title='% of Total Deals', format='.1f')
        ]
    ).properties(
        width=600,
        height=400,
        title='CSL Principal Outstanding by FICO Score Band'
    )
    
    col1, col2 = st.columns(2)
    with col1:
        st.altair_chart(fico_deals_chart, use_container_width=True)
    with col2:
        st.altair_chart(fico_capital_chart, use_container_width=True)
    
    # FICO summary table with total row
    fico_summary_with_total = fico_summary.copy()
    total_row = pd.DataFrame({
        'FICO Band': ['TOTAL'],
        'Deal Count': [fico_summary['Deal Count'].sum()],
        'CSL Capital Deployed': [fico_summary['CSL Capital Deployed'].sum()],
        'CSL Principal Outstanding': [fico_summary['CSL Principal Outstanding'].sum()],
        'Pct of Total': [100.0]
    })
    fico_summary_with_total = pd.concat([fico_summary_with_total, total_row], ignore_index=True)
    
    st.dataframe(
        fico_summary_with_total,
        use_container_width=True,
        column_config={
            "CSL Capital Deployed": st.column_config.NumberColumn("CSL Capital Deployed", format="$%.0f"),
            "CSL Principal Outstanding": st.column_config.NumberColumn("CSL Principal Outstanding", format="$%.0f"),
            "Pct of Total": st.column_config.NumberColumn("% of Total", format="%.1f%%"),
        }
    )

# Term Analysis
if 'tib' in df.columns:
    st.subheader("Capital Exposure by Time in Business")
    
    # Create TIB bands using years
    df['tib_band'] = pd.cut(df['tib'], 
                           bins=[0, 5, 10, 15, 20, 25, 1000], 
                           labels=['≤5', '5-10', '10-15', '15-20', '20-25', '25+'],
                           include_lowest=True)
    
    # TIB analysis using combined dataset
    tib_summary = df.groupby('tib_band').agg({
        'loan_id': 'count',
        'amount_hubspot': 'sum',
        'csl_principal_outstanding': 'sum'
    }).reset_index()
    
    tib_summary.columns = ['TIB Band', 'Deal Count', 'CSL Capital Deployed', 'CSL Principal Outstanding']
    
    # TIB capital exposure chart with forced x-axis order
    tib_chart = alt.Chart(tib_summary).mark_bar().encode(
        x=alt.X('TIB Band:N', 
                title='Time in Business (Years)',
                sort=['≤5', '5-10', '10-15', '15-20', '20-25', '25+']),
        y=alt.Y('CSL Principal Outstanding:Q', title='CSL Principal Outstanding ($)', axis=alt.Axis(format='$,.0f')),
        color=alt.Color('TIB Band:N',
                       scale=alt.Scale(range=RISK_GRADIENT),
                       legend=None),
        tooltip=[
            alt.Tooltip('TIB Band:N', title='TIB Band'),
            alt.Tooltip('Deal Count:Q', title='Number of Deals'),
            alt.Tooltip('CSL Capital Deployed:Q', title='Total Capital Deployed', format='$,.0f'),
            alt.Tooltip('CSL Principal Outstanding:Q', title='Principal Outstanding', format='$,.0f')
        ]
    ).properties(
        width=700,
        height=400,
        title='CSL Principal Outstanding by Time in Business'
    )
    
    st.altair_chart(tib_chart, use_container_width=True)
    
    # TIB summary table with total row
    tib_summary_with_total = tib_summary.copy()
    total_row_tib = pd.DataFrame({
        'TIB Band': ['TOTAL'],
        'Deal Count': [tib_summary['Deal Count'].sum()],
        'CSL Capital Deployed': [tib_summary['CSL Capital Deployed'].sum()],
        'CSL Principal Outstanding': [tib_summary['CSL Principal Outstanding'].sum()]
    })
    tib_summary_with_total = pd.concat([tib_summary_with_total, total_row_tib], ignore_index=True)
    
    st.dataframe(
        tib_summary_with_total,
        use_container_width=True,
        column_config={
            "CSL Capital Deployed": st.column_config.NumberColumn("CSL Capital Deployed", format="$%.0f"),
            "CSL Principal Outstanding": st.column_config.NumberColumn("CSL Principal Outstanding", format="$%.0f"),
        }
    )

# Download buttons
csv = loan_tape.to_csv(index=False).encode("utf-8")
st.download_button(
    label="Download Loan Tape as CSV",
    data=csv,
    file_name="loan_tape.csv",
    mime="text/csv"
)

if len(risk_df) > 0:
    csv_risk = top_risk_display.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Download Top 10 Risk Deals as CSV",
        data=csv_risk,
        file_name="top_risk_deals.csv",
        mime="text/csv"
    )
