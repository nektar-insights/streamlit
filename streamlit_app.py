# streamlit_app.py
from utils.imports import *
from utils.config import (
    setup_page,
    PRIMARY_COLOR,
    COLOR_PALETTE,
    PLATFORM_FEE_RATE,
)
from utils.data_loader import load_deals, load_loan_summaries
from utils.loan_tape_data import prepare_loan_data, calculate_irr
from utils.status_constants import PROBLEM_STATUSES

# ----------------------------
# Apply Branding
# ----------------------------
setup_page("CSL Capital | Dashboard")

# ----------------------------
# Load and prepare data
# ----------------------------
df = load_deals()
today = pd.to_datetime("today").normalize()

# Load loan summaries for principal outstanding calculation
loan_summaries_df = load_loan_summaries()
if not loan_summaries_df.empty and not df.empty:
    loan_data = prepare_loan_data(loan_summaries_df, df)
    # Calculate IRR (adds realized_irr column for paid-off loans)
    loan_data = calculate_irr(loan_data)
else:
    loan_data = pd.DataFrame()

# ----------------------------
# Data preprocessing
# ----------------------------
cols_to_convert = ["amount", "total_funded_amount", "factor_rate", "loan_term", "commission"]

# --- Canonical date handling ---
# Make sure date_created is tz-naive and valid
df["date_created"] = pd.to_datetime(df["date_created"], errors="coerce", utc=True).dt.tz_localize(None)
df = df.dropna(subset=["date_created"])

# Keep your original string month (for any legacy use)
df["month"] = df["date_created"].dt.to_period("M").astype(str)

# NEW: canonical month-start timestamp (no day ambiguity)
df["month_start"] = df["date_created"].dt.to_period("M").dt.to_timestamp(how="start")

df[cols_to_convert] = df[cols_to_convert].apply(pd.to_numeric, errors="coerce")
df["loan_id"] = df["loan_id"].astype("string")
df["is_participated"] = df["is_closed_won"] == True

# ----------------------------
# Sidebar Filters
# ----------------------------
st.sidebar.header("Filters")

min_date, max_date = df["date_created"].min(), df["date_created"].max()
start_date, end_date = st.sidebar.date_input(
    "Filter by Date Range",
    [min_date, max_date],
    min_value=min_date,
    max_value=max_date
)
df = df[(df["date_created"] >= pd.to_datetime(start_date)) &
        (df["date_created"] <= pd.to_datetime(end_date))]

partner_options = sorted(df["partner_source"].dropna().unique())
selected_partners = st.sidebar.multiselect(
    "Filter by Partner Source",
    options=partner_options,
    default=partner_options,
    help="Select one or more partners to filter"
)
if selected_partners:
    df = df[df["partner_source"].isin(selected_partners)]

participation_filter = st.sidebar.radio(
    "Show Deals",
    ["All Deals", "Participated Only", "Not Participated"],
)

if participation_filter == "Participated Only":
    df = df[df["is_closed_won"] == True]
elif participation_filter == "Not Participated":
    df = df[df["is_closed_won"] != True]

# ----------------------------
# Main Content
# ----------------------------
st.title("Pipeline Dashboard")

# ----------------------------
# Calculate all metrics
# ----------------------------
closed_won = df[df["is_closed_won"] == True]
participated = df[df["is_participated"] == True]  # Closed won AND has loan_id
total_deals = len(df)  # Total deals reviewed
total_participated = len(participated)  # Total deals participated
participation_ratio = total_participated / total_deals if total_deals > 0 else 0
months = df["month"].nunique()

# Calculate date range and deal flow metrics
date_range_days = (df["date_created"].max() - df["date_created"].min()).days + 1
date_range_weeks = date_range_days / 7
date_range_months = date_range_days / 30.44  # Average days per month

# Deal flow averages (across entire date range)
avg_deals_per_day = total_deals / date_range_days if date_range_days > 0 else 0
avg_deals_per_week = total_deals / date_range_weeks if date_range_weeks > 0 else 0
avg_deals_per_month = total_deals / date_range_months if date_range_months > 0 else 0

# Last 30 days metrics
last_30_days = df[df["date_created"] >= today - pd.Timedelta(days=30)]
deals_last_30 = len(last_30_days)
avg_deals_per_week_30d = deals_last_30 / (30 / 7)  # Average deals per week from last 30 days

# Average deal characteristics (across ALL deals, not just participated)
avg_total_funded = df["total_funded_amount"].mean()
avg_factor_all = df["factor_rate"].mean()
avg_commission_all = df["commission"].mean()
avg_term_all = df["loan_term"].mean()

# Deal characteristics
avg_amount = closed_won["amount"].mean()
avg_factor = closed_won["factor_rate"].mean()
avg_term = closed_won["loan_term"].mean()
avg_participation_pct = (closed_won["amount"] / closed_won["total_funded_amount"]).mean()
avg_commission = closed_won["commission"].mean()
has_tib_data = "tib" in closed_won.columns and closed_won["tib"].count() > 0
has_fico_data = "fico" in closed_won.columns and closed_won["fico"].count() > 0

avg_tib = closed_won["tib"].mean() if has_tib_data else None
avg_fico = closed_won["fico"].mean() if has_fico_data else None

# Financial calculations
total_capital_deployed = closed_won["amount"].sum()
total_commissions_paid = (closed_won["amount"] * closed_won["commission"]).sum()
total_platform_fee = total_capital_deployed * PLATFORM_FEE_RATE
total_expected_return = ((closed_won["amount"] * closed_won["factor_rate"]) -
                        closed_won["commission"] -
                        (closed_won["amount"] * PLATFORM_FEE_RATE))
total_expected_return_sum = total_expected_return.sum()
moic = total_expected_return_sum / total_capital_deployed if total_capital_deployed > 0 else 0
projected_irr = (moic ** (12 / avg_term) - 1) if avg_term > 0 else 0

# Principal Outstanding calculation from loan summaries
if not loan_data.empty and "csl_participation_amount" in loan_data.columns and "loan_status" in loan_data.columns:
    active_loans = loan_data[loan_data["loan_status"] != "Paid Off"]
    paid_off_loans = loan_data[loan_data["loan_status"] == "Paid Off"]

    # Principal = original capital deployed (excludes fees and commissions)
    total_principal_outstanding = active_loans["csl_participation_amount"].sum() if not active_loans.empty else 0
    active_loan_count = len(active_loans)

    # Collection rate: how much has been collected relative to principal deployed
    total_principal_deployed = loan_data["csl_participation_amount"].sum() if "csl_participation_amount" in loan_data.columns else 0
    total_collected = loan_data["total_paid"].sum() if "total_paid" in loan_data.columns else 0
    collection_rate = total_collected / total_principal_deployed if total_principal_deployed > 0 else 0

    # Net Profit/Loss: total collected minus total cost basis (principal + fees + commissions)
    total_cost_basis = loan_data["total_invested"].sum() if "total_invested" in loan_data.columns else 0
    net_profit_loss = total_collected - total_cost_basis

    # Realized IRR: average IRR from paid-off loans only
    if not paid_off_loans.empty and "realized_irr" in paid_off_loans.columns:
        realized_irr_values = paid_off_loans["realized_irr"].dropna()
        avg_realized_irr = realized_irr_values.mean() if len(realized_irr_values) > 0 else 0
    else:
        avg_realized_irr = 0

    # Delinquency Rate: % of active loans in problem statuses
    if not active_loans.empty:
        problem_loans = active_loans[active_loans["loan_status"].isin(PROBLEM_STATUSES)]
        delinquency_rate = len(problem_loans) / len(active_loans) if len(active_loans) > 0 else 0
        # At-Risk Balance: principal in problem statuses
        at_risk_balance = problem_loans["csl_participation_amount"].sum() if not problem_loans.empty else 0
    else:
        delinquency_rate = 0
        at_risk_balance = 0

    # Weighted Average Factor Rate (by participation amount)
    if "factor_rate" in loan_data.columns:
        valid_factor = loan_data[loan_data["factor_rate"].notna() & (loan_data["csl_participation_amount"] > 0)]
        if not valid_factor.empty:
            weighted_factor = (valid_factor["factor_rate"] * valid_factor["csl_participation_amount"]).sum()
            total_weight = valid_factor["csl_participation_amount"].sum()
            weighted_avg_factor = weighted_factor / total_weight if total_weight > 0 else 0
        else:
            weighted_avg_factor = 0
    else:
        weighted_avg_factor = 0

    # Average Days to Payoff (for paid-off loans)
    if not paid_off_loans.empty and "funding_date" in paid_off_loans.columns and "payoff_date" in paid_off_loans.columns:
        paid_off_with_dates = paid_off_loans[
            paid_off_loans["funding_date"].notna() &
            paid_off_loans["payoff_date"].notna()
        ].copy()
        if not paid_off_with_dates.empty:
            # Normalize both dates to tz-naive to avoid timezone mismatch
            payoff_dates = pd.to_datetime(paid_off_with_dates["payoff_date"], utc=True).dt.tz_localize(None)
            funding_dates = pd.to_datetime(paid_off_with_dates["funding_date"], utc=True).dt.tz_localize(None)
            paid_off_with_dates["days_to_payoff"] = (payoff_dates - funding_dates).dt.days
            avg_days_to_payoff = paid_off_with_dates["days_to_payoff"].mean()
        else:
            avg_days_to_payoff = 0
    else:
        avg_days_to_payoff = 0
else:
    total_principal_outstanding = 0
    active_loan_count = 0
    collection_rate = 0
    net_profit_loss = 0
    avg_realized_irr = 0
    delinquency_rate = 0
    at_risk_balance = 0
    weighted_avg_factor = 0
    avg_days_to_payoff = 0



# Rolling deal flow calculations - Fixed to look back from today
periods = [
    ("91-120 Days", 91, 120),
    ("61-90 Days", 61, 90), 
    ("31-60 Days", 31, 60), 
    ("0-30 Days", 0, 30)
]
flow_data = []
for label, start, end in periods:
    window = df[(df["date_created"] >= today - pd.Timedelta(days=end)) & 
                (df["date_created"] <= today - pd.Timedelta(days=start))]
    flow_data.append({
        "Period": label, 
        "Deals": len(window), 
        "Total Funded": window["total_funded_amount"].sum()
    })
flow_df = pd.DataFrame(flow_data)

# Monthly aggregations
monthly_funded = (
    df.groupby("month_start", as_index=False)["total_funded_amount"].sum()
    .rename(columns={"month_start": "month_date"})
)

monthly_deals = (
    df.groupby("month_start", as_index=False)
      .size()
      .rename(columns={"month_start": "month_date", "size": "deal_count"})
)

participated_only = df[df["is_participated"] == True]

monthly_participation = (
    participated_only.groupby("month_start", as_index=False)
    .agg(deal_count=("id", "count"), total_amount=("amount", "sum"))
    .rename(columns={"month_start": "month_date"})
)

monthly_participation_ratio = (
    df.groupby("month_start", as_index=False)
      .agg(total_deals=("id", "count"),
           participated_deals=("is_participated", "sum"))
      .assign(participation_pct=lambda x: x["participated_deals"] / x["total_deals"])
      .rename(columns={"month_start": "month_date"})
)

# Dollar-based participation ratio
monthly_participation_ratio_dollar = (
    df.groupby("month_start", as_index=False)
      .agg(total_funded_amount=("total_funded_amount", "sum"),
           participated_amount=("amount", "sum"))
      .assign(participation_pct_dollar=lambda x: (x["participated_amount"] / x["total_funded_amount"]).fillna(0))
      .rename(columns={"month_start": "month_date"})
)

# Partner summary calculations
all_deals = df.groupby("partner_source").agg(
    total_deals=("id", "count"),
    total_amount=("total_funded_amount", "sum")
)
won_deals = closed_won.groupby("partner_source").agg(
    participated_deals=("id", "count"),
    participated_amount=("amount", "sum"),
    total_won_amount=("total_funded_amount", "sum")
)
partner_summary = all_deals.join(won_deals, how="left").fillna(0)
partner_summary["closed_won_pct"] = partner_summary["participated_deals"] / partner_summary["total_deals"]
partner_summary["avg_participation_pct"] = partner_summary["participated_amount"] / partner_summary["total_won_amount"]

# ----------------------------
# Display metrics sections
# ----------------------------
st.subheader("Deal Overview")
col1, col2, col3 = st.columns(3)
col1.metric("Total Deals", total_deals)
col2.metric("Closed Won", len(closed_won))
col3.metric("Close Ratio", f"{participation_ratio:.1%}")

# Deal Flow Metrics
st.write("**Deal Flow Averages (Entire Date Range)**")
col4, col5, col6 = st.columns(3)
col4.metric("Avg Deals/Week", f"{avg_deals_per_week:.1f}")
col5.metric("Avg Deals/Day", f"{avg_deals_per_day:.2f}")
col6.metric("Avg Deals/Month", f"{avg_deals_per_month:.1f}")

# Last 30 Days Flow
st.write("**Deal Flow - Last 30 Days**")
col4b, col5b = st.columns(2)
col4b.metric("AVG Deals/Week (Last 30d)", f"{avg_deals_per_week_30d:.2f}")
col5b.metric("Total Deals (Last 30d)", f"{deals_last_30}")

st.write("**Average Participation Amounts**")
col4a, col5a, col6a = st.columns(3)
# Calculate average participation amount per week
avg_participation_per_week = closed_won["amount"].sum() / date_range_weeks if date_range_weeks > 0 else 0
col4a.metric("Avg Participation/Week", f"${avg_participation_per_week:,.0f}")

# Calculate average participation amount per month
avg_participation_per_month = closed_won["amount"].sum() / date_range_months if date_range_months > 0 else 0
col5a.metric("Avg Participation/Month", f"${avg_participation_per_month:,.0f}")

# Calculate participation amount for last 30 days
participation_last_30 = closed_won[closed_won["date_created"] >= today - pd.Timedelta(days=30)]["amount"].sum()
col6a.metric("Total Participation/30 Days", f"${participation_last_30:,.0f}")

# ----------------------------
# Current Week/Month vs Historical Participation
# ----------------------------
st.write("**Current Period vs Historical Participation**")

# Calculate current week boundaries (Monday to Sunday)
current_week_start = today - pd.Timedelta(days=today.weekday())
current_week_end = current_week_start + pd.Timedelta(days=6)

# Calculate current month boundaries
current_month_start = today.replace(day=1)

# Current week participation (closed won deals in current week)
current_week_deals = closed_won[
    (closed_won["date_created"] >= current_week_start) &
    (closed_won["date_created"] <= current_week_end)
]
current_week_participation = current_week_deals["amount"].sum()

# Current month participation (closed won deals in current month)
current_month_deals = closed_won[closed_won["date_created"] >= current_month_start]
current_month_participation = current_month_deals["amount"].sum()

# Historical weekly average (excluding current week)
historical_deals = closed_won[closed_won["date_created"] < current_week_start]
if not historical_deals.empty:
    historical_date_range_days = (historical_deals["date_created"].max() - historical_deals["date_created"].min()).days + 1
    historical_weeks = max(historical_date_range_days / 7, 1)
    historical_week_avg = historical_deals["amount"].sum() / historical_weeks
else:
    historical_week_avg = 0

# Historical monthly average (excluding current month)
historical_month_deals = closed_won[closed_won["date_created"] < current_month_start]
if not historical_month_deals.empty:
    historical_month_range_days = (historical_month_deals["date_created"].max() - historical_month_deals["date_created"].min()).days + 1
    historical_months = max(historical_month_range_days / 30.44, 1)
    historical_month_avg = historical_month_deals["amount"].sum() / historical_months
else:
    historical_month_avg = 0

# Calculate deltas (variance from historical)
week_delta = current_week_participation - historical_week_avg
month_delta = current_month_participation - historical_month_avg

# Display current vs historical metrics
col_cw1, col_cw2, col_cw3 = st.columns(3)
col_cw1.metric(
    "Current Week Participation",
    f"${current_week_participation:,.0f}",
    delta=f"{week_delta:+,.0f} vs avg" if historical_week_avg > 0 else None,
    delta_color="normal"
)
col_cw2.metric(
    "Current Month Participation",
    f"${current_month_participation:,.0f}",
    delta=f"{month_delta:+,.0f} vs avg" if historical_month_avg > 0 else None,
    delta_color="normal"
)
col_cw3.metric(
    "Week Deal Count",
    f"{len(current_week_deals)}",
    delta=f"{len(current_week_deals) - (len(historical_deals) / max(historical_weeks, 1)):.1f} vs avg" if not historical_deals.empty else None,
    delta_color="normal"
)

# Historical reference row
col_hw1, col_hw2, col_hw3 = st.columns(3)
col_hw1.metric("Historical Avg/Week", f"${historical_week_avg:,.0f}")
col_hw2.metric("Historical Avg/Month", f"${historical_month_avg:,.0f}")
col_hw3.metric("Month Deal Count", f"{len(current_month_deals)}")

# New Average Deal Characteristics 
st.write("**Average Deal Characteristics (All Deals)**")
col7, col8 = st.columns(2)
col7.metric("Avg Total Funded", f"${avg_total_funded:,.0f}")
col8.metric("Avg Factor", f"{avg_factor_all:.2f}")

col9, col10 = st.columns(2)
col9.metric("Avg Commission", f"{avg_commission_all:.1%}")
col10.metric("Avg Term (mo)", f"{avg_term_all:.1f}")

st.subheader("Financial Performance")
col11, col12, col13 = st.columns(3)
col11.metric("Total Capital Deployed", f"${total_capital_deployed:,.0f}")
col12.metric("Total Expected Return", f"${total_expected_return_sum:,.0f}")
col13.metric("MOIC", f"{moic:.2f}")

col14, col15, col16 = st.columns(3)
col14.metric("Projected IRR", f"{projected_irr:.1%}")
col15.metric("Avg % of Deal", f"{avg_participation_pct:.1%}")
col16.metric("Commission Paid", f"${total_commissions_paid:,.0f}")

col17a, col18a, col19a = st.columns(3)
col17a.metric("Principal Outstanding", f"${total_principal_outstanding:,.0f}")
col18a.metric("Active Loans", f"{active_loan_count}")
col19a.metric("Collection Rate", f"{collection_rate:.1%}")

col20a, col21a, col22a = st.columns(3)
col20a.metric("Net Profit/Loss", f"${net_profit_loss:,.0f}")
col21a.metric("Realized IRR", f"{avg_realized_irr:.1%}" if avg_realized_irr else "N/A")
col22a.metric("Avg Days to Payoff", f"{avg_days_to_payoff:.0f}" if avg_days_to_payoff else "N/A")

# Portfolio Health
st.subheader("Portfolio Health")
col23a, col24a, col25a = st.columns(3)
col23a.metric("Delinquency Rate", f"{delinquency_rate:.1%}")
col24a.metric("At-Risk Principal", f"${at_risk_balance:,.0f}")
col25a.metric("Weighted Avg Factor", f"{weighted_avg_factor:.3f}" if weighted_avg_factor else "N/A")

# Deal Characteristics
st.subheader("Deal Characteristics (Participated Only)")
col17, col18, col19 = st.columns(3)
col17.metric("Avg Participation ($)", f"${avg_amount:,.0f}")
col18.metric("Avg Factor", f"{avg_factor:.2f}")
col19.metric("Avg Term (mo)", f"{avg_term:.1f}")

# Second row - only show if we have data
if has_tib_data or has_fico_data:
    if has_tib_data and has_fico_data:
        # Both available - show in 2 columns
        col20, col21 = st.columns(2)
        col20.metric("Avg TIB", f"{avg_tib:,.0f}")
        col21.metric("Avg FICO", f"{avg_fico:.0f}")
    elif has_tib_data:
        # Only TIB available
        col20, _ = st.columns(2)
        col20.metric("Avg TIB", f"{avg_tib:,.0f}")
    elif has_fico_data:
        # Only FICO available
        col21, _ = st.columns(2)
        col21.metric("Avg FICO", f"{avg_fico:.0f}")
else:
    # No data available - show a note
    st.write("*TIB and FICO data not yet available*")

# Find true mins/maxes for box plots
amt_min, amt_max   = closed_won["amount"].min(),       closed_won["amount"].max()
fr_min,  fr_max    = closed_won["factor_rate"].min(),  closed_won["factor_rate"].max()
lt_min,  lt_max    = closed_won["loan_term"].min(),    closed_won["loan_term"].max()
fico_min, fico_max = closed_won["fico"].min(),         closed_won["fico"].max()

# Prepare capped TIB
tib_df = closed_won.dropna(subset=["tib"]).copy()
tib_df["tib_capped"] = tib_df["tib"].clip(upper=50)

# ----------------------------
# Box Plot Visualizations in Grid Layout
# ----------------------------
st.subheader("Deal Distribution Analysis")

# First row - Participation Amount and Factor Rate
col1, col2 = st.columns(2)

with col1:
    st.write("**Participation Amount Distribution**")
    participation_box = (
        alt.Chart(closed_won)
        .mark_boxplot(size=60, color=PRIMARY_COLOR, outliers={"color": COLOR_PALETTE[1], "size": 40})
        .encode(
            y=alt.Y(
                "amount:Q",
                title="Participation Amount ($)",
                axis=alt.Axis(format="$.2s"),
                scale=alt.Scale(domain=[amt_min, amt_max])
            ),
            tooltip=[alt.Tooltip("amount:Q", title="Amount", format="$,.0f")]
        )
        .properties(height=300)
    )
    st.altair_chart(participation_box, width='stretch')

with col2:
    st.write("**Factor Rate Distribution**")
    factor_box = (
        alt.Chart(closed_won)
        .mark_boxplot(size=60, color=COLOR_PALETTE[2], outliers={"color": COLOR_PALETTE[3], "size": 40})
        .encode(
            y=alt.Y(
                "factor_rate:Q",
                title="Factor Rate",
                axis=alt.Axis(format=".2f"),
                scale=alt.Scale(domain=[fr_min, fr_max])
            ),
            tooltip=[alt.Tooltip("factor_rate:Q", title="Factor Rate", format=".3f")]
        )
        .properties(height=300)
    )
    st.altair_chart(factor_box, width='stretch')

# Second row - Loan Term and TIB/FICO (when available)
col3, col4 = st.columns(2)

with col3:
    st.write("**Loan Term Distribution**")
    term_box = (
        alt.Chart(closed_won)
        .mark_boxplot(size=60, color=COLOR_PALETTE[4], outliers={"color": COLOR_PALETTE[0], "size": 40})
        .encode(
            y=alt.Y(
                "loan_term:Q",
                title="Loan Term (months)",
                axis=alt.Axis(format=".0f"),
                scale=alt.Scale(domain=[lt_min, lt_max])
            ),
            tooltip=[alt.Tooltip("loan_term:Q", title="Term (months)", format=".1f")]
        )
        .properties(height=300)
    )
    st.altair_chart(term_box, width='stretch')

with col4:
    if has_tib_data:
        st.write("**TIB Distribution (capped at 50)**")
        tib_box = (
            alt.Chart(tib_df)
            .mark_boxplot(size=60, color=COLOR_PALETTE[1], outliers={"color": COLOR_PALETTE[2], "size": 40})
            .encode(
                y=alt.Y(
                    "tib_capped:Q",
                    title="TIB (capped at 50)",
                    axis=alt.Axis(format=",.0f"),
                    scale=alt.Scale(domain=[tib_df["tib_capped"].min(), 50])
                ),
                tooltip=[alt.Tooltip("tib:Q", title="Original TIB", format=",.0f")]
            )
            .properties(height=300)
        )
        st.altair_chart(tib_box, width='stretch')

    elif has_fico_data:
        st.write("**FICO Score Distribution**")
        fico_box = (
            alt.Chart(closed_won.dropna(subset=["fico"]))
            .mark_boxplot(size=60, color=COLOR_PALETTE[3], outliers={"color": COLOR_PALETTE[4], "size": 40})
            .encode(
                y=alt.Y(
                    "fico:Q",
                    title="FICO Score",
                    axis=alt.Axis(format=".0f"),
                    scale=alt.Scale(domain=[fico_min, fico_max])
                ),
                tooltip=[alt.Tooltip("fico:Q", title="FICO Score", format=".0f")]
            )
            .properties(height=300)
        )
        st.altair_chart(fico_box, width='stretch')

    else:
        st.write("**Additional Data**")
        st.info("TIB and FICO data not yet available for visualization")

# Third row - Only if both TIB and FICO are available
if has_tib_data and has_fico_data:
    col5, col6 = st.columns(2)
    with col5:
        st.write("**FICO Score Distribution (300–850)**")
        fico_box2 = (
            alt.Chart(closed_won.dropna(subset=["fico"]))
            .mark_boxplot(size=60, color="#2E8B57", outliers={"color": "#DC143C", "size": 40})
            .encode(
                y=alt.Y(
                    "fico:Q",
                    title="FICO Score",
                    axis=alt.Axis(format=".0f"),
                    scale=alt.Scale(domain=[300, 850])
                ),
                tooltip=[alt.Tooltip("fico:Q", title="FICO Score", format=".0f")]
            )
            .properties(height=300)
        )
        st.altair_chart(fico_box2, width='stretch')
    
# ----------------------------
# Rolling Deal Flow
# ----------------------------
st.subheader("Rolling Deal Flow Trends")

# Add change calculations
flow_df["Deal Change"] = flow_df["Deals"].diff().fillna(0).astype(int)
flow_df["Deal Change %"] = flow_df["Deals"].pct_change().fillna(0).apply(lambda x: f"{x:.1%}")
flow_df["Funded Change"] = flow_df["Total Funded"].diff().fillna(0).astype(int)
flow_df["Funded Change %"] = flow_df["Total Funded"].pct_change().fillna(0).apply(lambda x: f"{x:.1%}")

# Display formatted table
flow_df_display = flow_df.copy()
flow_df_display["Total Funded Display"] = flow_df_display["Total Funded"].apply(lambda x: f"${x:,.0f}")
flow_df_display["Funded Change Display"] = flow_df_display["Funded Change"].apply(lambda x: f"${x:,.0f}")

st.dataframe(
    flow_df_display[["Period", "Deals", "Deal Change", "Deal Change %", 
                     "Total Funded Display", "Funded Change Display", "Funded Change %"]], 
    width='stretch',
    column_config={
        "Total Funded Display": "Total Funded",
        "Funded Change Display": "Funded Change"
    }
)

# Rolling flow charts
flow_chart = alt.Chart(flow_df).mark_bar(
    size=60, 
    color=PRIMARY_COLOR,
    cornerRadiusTopLeft=3,
    cornerRadiusTopRight=3
).encode(
    x=alt.X("Period:N", 
            sort=["91-120 Days", "61-90 Days", "31-60 Days", "0-30 Days"],
            axis=alt.Axis(labelAngle=0)),
    y=alt.Y("Deals:Q", title="Deal Count"),
    tooltip=[
        alt.Tooltip("Period", title="Period"),
        alt.Tooltip("Deals", title="Deal Count"),
        alt.Tooltip("Deal Change", title="Change vs Previous"),
        alt.Tooltip("Deal Change %", title="Percent Change")
    ]
).properties(
    height=350,
    title="Deal Count by Period"
)

funded_flow_chart = alt.Chart(flow_df).mark_bar(
    size=60, 
    color=COLOR_PALETTE[2],
    cornerRadiusTopLeft=3,
    cornerRadiusTopRight=3
).encode(
    x=alt.X("Period:N", 
            sort=["91-120 Days", "61-90 Days", "31-60 Days", "0-30 Days"],
            axis=alt.Axis(labelAngle=0)),
    y=alt.Y("Total Funded:Q", 
            title="Total Funded ($)", 
            axis=alt.Axis(format="$.1s", titlePadding=15)),
    tooltip=[
        alt.Tooltip("Period", title="Period"),
        alt.Tooltip("Total Funded", title="Total Funded", format="$,.0f"),
        alt.Tooltip("Funded Change", title="Change vs Previous", format="$,.0f"),
        alt.Tooltip("Funded Change %", title="Percent Change")
    ]
).properties(
    height=350,
    title="Total Funded by Period"
)

st.altair_chart(flow_chart, width='stretch')
st.altair_chart(funded_flow_chart, width='stretch')

# ----------------------------
# Partner Rolling Flow Metrics & Charts (all periods)
# ----------------------------
st.subheader("Partner Rolling Flow Trends")

# Build partner metrics for each rolling window
partner_metrics = []
for label, start, end in periods:
    window = df[
        (df["date_created"] >= today - pd.Timedelta(days=end)) &
        (df["date_created"] <= today - pd.Timedelta(days=start))
    ]
    grp = (
        window
        .dropna(subset=["partner_source"])          # remove null partners
        .groupby("partner_source")
        .agg(
            deal_count   = ("id", "size"),
            total_funded = ("total_funded_amount", "sum")
        )
        .reset_index()
    )
    grp["avg_funded"] = grp["total_funded"] / grp["deal_count"]
    grp["Period"] = label
    partner_metrics.append(grp)

partner_flow_df = pd.concat(partner_metrics, ignore_index=True)

# 1) Deal Count by Partner & Period
count_chart = (
    alt.Chart(partner_flow_df)
    .mark_bar(cornerRadiusTopLeft=3, cornerRadiusTopRight=3)
    .encode(
        x=alt.X("Period:N", title="Period", sort=[p[0] for p in periods]),
        y=alt.Y("deal_count:Q", title="Deal Count"),
        color=alt.Color("partner_source:N", title="Partner"),
        xOffset="partner_source:N",
        tooltip=[
            alt.Tooltip("partner_source:N", title="Partner"),
            alt.Tooltip("deal_count:Q",     title="Deal Count")
        ]
    )
    .properties(height=300, title="Deals by Partner")
)

# Compute global maxes (if you still need them)
max_total = partner_flow_df["total_funded"].max()
max_avg   = partner_flow_df["avg_funded"].max()

# Total Funded chart
funded_chart = (
    alt.Chart(partner_flow_df)
    .mark_bar(cornerRadiusTopLeft=3, cornerRadiusTopRight=3)
    .encode(
        x=alt.X("Period:N", title="Period", sort=[p[0] for p in periods]),
        y=alt.Y(
            "total_funded:Q",
            title="Total Funded ($)",
            axis=alt.Axis(format="$,.0f", titlePadding=10)
            # remove scale domain if you don't care about zero baseline
        ),
        color=alt.Color("partner_source:N", title="Partner"),
        xOffset="partner_source:N",
        tooltip=[
            alt.Tooltip("partner_source:N", title="Partner"),
            alt.Tooltip("total_funded:Q",   title="Total Funded", format="$,.0f")
        ]
    )
    .properties(height=300, title="Total Funded by Partner")
)

# Avg Funded chart
avg_chart = (
    alt.Chart(partner_flow_df)
    .mark_bar(cornerRadiusTopLeft=3, cornerRadiusTopRight=3)
    .encode(
        x=alt.X("Period:N", title="Period", sort=[p[0] for p in periods]),
        y=alt.Y(
            "avg_funded:Q",
            title="Avg Funded ($)",
            axis=alt.Axis(format="$,.0f", titlePadding=10)
        ),
        color=alt.Color("partner_source:N", title="Partner"),
        xOffset="partner_source:N",
        tooltip=[
            alt.Tooltip("partner_source:N", title="Partner"),
            alt.Tooltip("avg_funded:Q",     title="Avg Funded", format="$,.0f")
        ]
    )
    .properties(height=300, title="Avg Funded per Deal by Partner")
)


st.altair_chart(count_chart,   width='stretch')
st.altair_chart(funded_chart, width='stretch')
st.altair_chart(avg_chart,    width='stretch')

# ----------------------------
# ADDITIONAL DATA PREPARATION FOR DOLLAR-BASED PARTICIPATION RATE
# ----------------------------
# Monthly participation rate by DOLLAR amount
monthly_participation_ratio_dollar = df.groupby("month").agg(
    total_funded_amount=("total_funded_amount", "sum"),
    participated_amount=("amount", "sum")
).reset_index()
monthly_participation_ratio_dollar["participation_pct_dollar"] = (
    monthly_participation_ratio_dollar["participated_amount"] / 
    monthly_participation_ratio_dollar["total_funded_amount"]
).fillna(0)
monthly_participation_ratio_dollar["month_date"] = pd.to_datetime(monthly_participation_ratio_dollar["month"])

# ===================== MONTHLY TREND CHARTS =====================

# Small helper for the dashed average rule
def _avg_rule(df_, field, title, fmt):
    return (
        alt.Chart(df_)
        .transform_aggregate(avg_val=f"mean({field})")
        .mark_rule(color="gray", strokeDash=[4, 2], strokeWidth=2)
        .encode(
            y="avg_val:Q",
            tooltip=alt.Tooltip("avg_val:Q", title=f"Avg {title}", format=fmt),
        )
    )

# Generic renderer used by monthly line charts
def _render_monthly_line(df_, y_field, y_title, fmt, color):
    df_ = (
        df_.copy()
        .dropna(subset=["month_date"])           # ensure temporal axis is valid
        .sort_values("month_date")
        .drop_duplicates(subset=["month_date"])  # just in case
    )
    if df_.empty:
        st.info(f"No data for {y_title.lower()} in the selected range.")
        return

    # y must be numeric
    df_[y_field] = pd.to_numeric(df_[y_field], errors="coerce").fillna(0)

    base = alt.Chart(df_).encode(
        x=alt.X("month_date:T", title="", axis=alt.Axis(format="%b %Y", labelAngle=-45)),
        y=alt.Y(f"{y_field}:Q", title=y_title, axis=alt.Axis(grid=True)),
        tooltip=[
            alt.Tooltip("month_date:T", title="Month", format="%B %Y"),
            alt.Tooltip(f"{y_field}:Q", title=y_title, format=fmt),
        ],
    )
    line = base.mark_line(color=color, strokeWidth=3)
    pts  = base.mark_point(size=60, filled=True, color=color)
    avg  = _avg_rule(df_, y_field, y_title, fmt)

    st.altair_chart(
        (line + pts + avg).properties(height=350, padding={"bottom": 80}),
        width='stretch',
    )

# ----------------------------
# MONTHLY VOLUME CHARTS
# ----------------------------

# 1) Total Deal Count by Month
st.subheader("Total Deal Count by Month")
_render_monthly_line(
    monthly_deals,
    y_field="deal_count",
    y_title="Deal Count",
    fmt=",.0f",
    color=COLOR_PALETTE[2],
)

# 2) Total Funded Amount by Month (Total Opportunity Value)
st.subheader("Total Funded Amount by Month")
_render_monthly_line(
    monthly_funded,
    y_field="total_funded_amount",
    y_title="Total Funded ($)",
    fmt="$,.0f",
    color=PRIMARY_COLOR,
)

# ----------------------------
# CSL DEPLOYMENT CHARTS
# ----------------------------

# 3) Total Amount Deployed by Month (CSL Capital Deployed)
st.subheader("Total Amount Deployed by Month")
_render_monthly_line(
    monthly_participation,
    y_field="total_amount",
    y_title="Amount Deployed ($)",
    fmt="$,.0f",
    color=COLOR_PALETTE[6],  # Teal
)

# 4) Participated Deal Count by Month
st.subheader("Participated Deal Count by Month")
_render_monthly_line(
    monthly_participation,
    y_field="deal_count",
    y_title="Participated Deals",
    fmt=",.0f",
    color=PRIMARY_COLOR,
)

# ----------------------------
# PARTICIPATION RATE CHARTS
# ----------------------------

# 5) Monthly Participation Rate by Deal Count
st.subheader("Monthly Participation Rate by Deal Count")
rate_line = alt.Chart(monthly_participation_ratio).mark_line(
    color="#e45756", strokeWidth=4,
    point=alt.OverlayMarkDef(color="#e45756", size=80, filled=True)
).encode(
    x=alt.X("yearmonth(month_date):T", title="Month",
            axis=alt.Axis(labelAngle=-45, format="%b %Y", labelPadding=10), sort="ascending"),
    y=alt.Y("participation_pct:Q", title="Participation Rate (by Count)",
            axis=alt.Axis(format=".0%", titlePadding=20, labelPadding=5)),
    tooltip=[
        alt.Tooltip("yearmonth(month_date):T", title="Month", format="%B %Y"),
        alt.Tooltip("participation_pct:Q", title="Participation Rate", format=".1%")
    ]
).properties(height=350, width=800, padding={"left": 80, "top": 20, "right": 20, "bottom": 60})

st.altair_chart(rate_line, width='stretch')

# 6) Monthly Participation Rate by Dollar Amount
st.subheader("Monthly Participation Rate by Dollar Amount")
rate_line_dollar = alt.Chart(monthly_participation_ratio_dollar).mark_line(
    color="#17a2b8", strokeWidth=4,
    point=alt.OverlayMarkDef(color="#17a2b8", size=80, filled=True)
).encode(
    x=alt.X("yearmonth(month_date):T", title="Month",
            axis=alt.Axis(labelAngle=-45, format="%b %Y", labelPadding=10), sort="ascending"),
    y=alt.Y("participation_pct_dollar:Q", title="Participation Rate (by $)",
            axis=alt.Axis(format=".0%", titlePadding=20, labelPadding=5)),
    tooltip=[
        alt.Tooltip("yearmonth(month_date):T", title="Month", format="%B %Y"),
        alt.Tooltip("participation_pct_dollar:Q", title="Participation Rate ($)", format=".1%"),
        alt.Tooltip("total_funded_amount:Q", title="Total Opportunities", format="$,.0f"),
        alt.Tooltip("participated_amount:Q", title="Amount Participated", format="$,.0f")
    ]
).properties(height=350, width=800, padding={"left": 80, "top": 20, "right": 20, "bottom": 60})

st.altair_chart(rate_line_dollar, width='stretch')
# ----------------------------
# PARTNER SUMMARY TABLES
# ----------------------------
st.subheader("Partner Performance Summary")

# Calculate additional metrics for partner summary
partner_summary_enhanced = partner_summary.copy()

# Deal-based metrics
partner_summary_enhanced["participated_deal_count"] = partner_summary_enhanced["participated_deals"].astype(int)
partner_summary_enhanced["deal_participation_rate"] = (
    partner_summary_enhanced["participated_deals"] / partner_summary_enhanced["total_deals"]
).fillna(0)

# Dollar-based metrics  
partner_summary_enhanced["dollar_participation_rate"] = (
    partner_summary_enhanced["participated_amount"] / partner_summary_enhanced["total_amount"]
).fillna(0)

# ----------------------------
# COMBINED PARTNER SUMMARY TABLE
# ----------------------------
# Build comprehensive summary combining deal count and dollar metrics
combined_summary = partner_summary_enhanced.reset_index()[[
    "partner_source", "total_deals", "participated_deal_count", "deal_participation_rate",
    "total_amount", "participated_amount"
]].copy()

# Calculate additional metrics
combined_summary["avg_deal_size"] = combined_summary["total_amount"] / combined_summary["total_deals"]
combined_summary["pct_cap_deployed"] = combined_summary["participated_amount"] / total_capital_deployed
combined_summary["avg_participation_size"] = (
    combined_summary["participated_amount"] / combined_summary["participated_deal_count"]
).fillna(0)

# Format for display
combined_summary_display = pd.DataFrame({
    "Partner": combined_summary["partner_source"],
    "Total Deals": combined_summary["total_deals"].astype(int),
    "CSL Deals": combined_summary["participated_deal_count"].astype(int),
    "Deal Rate": combined_summary["deal_participation_rate"].apply(lambda x: f"{x:.1%}"),
    "Avg Deal Size": combined_summary["avg_deal_size"].apply(lambda x: f"${x:,.0f}"),
    "$ Participated": combined_summary["participated_amount"].apply(lambda x: f"${x:,.0f}"),
    "% of Capital": combined_summary["pct_cap_deployed"].apply(lambda x: f"{x:.1%}"),
    "Avg Participation": combined_summary["avg_participation_size"].apply(lambda x: f"${x:,.0f}")
})

# Calculate totals row with proper weighted averages
total_deals_sum = combined_summary["total_deals"].sum()
csl_deals_sum = combined_summary["participated_deal_count"].sum()
total_amount_sum = combined_summary["total_amount"].sum()
participated_amount_sum = combined_summary["participated_amount"].sum()

# Calculate derived metrics for totals
total_deal_rate = csl_deals_sum / total_deals_sum if total_deals_sum > 0 else 0
total_avg_deal_size = total_amount_sum / total_deals_sum if total_deals_sum > 0 else 0
total_pct_capital = participated_amount_sum / total_capital_deployed if total_capital_deployed > 0 else 0
total_avg_participation = participated_amount_sum / csl_deals_sum if csl_deals_sum > 0 else 0

# Create totals row
totals_row = pd.DataFrame({
    "Partner": ["TOTAL"],
    "Total Deals": [int(total_deals_sum)],
    "CSL Deals": [int(csl_deals_sum)],
    "Deal Rate": [f"{total_deal_rate:.1%}"],
    "Avg Deal Size": [f"${total_avg_deal_size:,.0f}"],
    "$ Participated": [f"${participated_amount_sum:,.0f}"],
    "% of Capital": [f"{total_pct_capital:.1%}"],
    "Avg Participation": [f"${total_avg_participation:,.0f}"]
})

# Append totals row to display DataFrame
combined_summary_display = pd.concat([combined_summary_display, totals_row], ignore_index=True)

st.dataframe(combined_summary_display, width='stretch', height=450)

# ----------------------------
# DOWNLOAD FUNCTIONS
# ----------------------------
def create_pdf_from_html(html: str):
    result = io.BytesIO()
    pisa.CreatePDF(io.StringIO(html), dest=result)
    return result.getvalue()

# ----------------------------
# DOWNLOAD BUTTONS FOR COMBINED TABLE
# ----------------------------
col_download1, col_download2 = st.columns(2)

with col_download1:
    # Combined Summary CSV Download
    combined_csv = combined_summary_display.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Download Partner Summary (CSV)",
        data=combined_csv,
        file_name="partner_performance_summary.csv",
        mime="text/csv"
    )

with col_download2:
    # Combined Summary PDF Download
    combined_html = combined_summary_display.to_html(index=False)
    combined_pdf = create_pdf_from_html(combined_html)
    st.download_button(
        label="Download Partner Summary (PDF)",
        data=combined_pdf,
        file_name="partner_performance_summary.pdf",
        mime="application/pdf"
    )
