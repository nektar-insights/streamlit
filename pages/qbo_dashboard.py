# pages/qbo_dashboard.py
"""
QBO Dashboard - Cash Performance & Portfolio Insights

Provides actionable insights into portfolio cash performance:
- Executive summary of cash in vs expected
- Loan health monitoring with attention flags
- Cash flow forecasting
- Decision support recommendations
"""

from utils.imports import *
from utils.config import (
    setup_page,
    PRIMARY_COLOR,
    COLOR_PALETTE,
    PLATFORM_FEE_RATE,
)
from utils.data_loader import load_qbo_data, load_deals, load_mca_deals
from utils.preprocessing import preprocess_dataframe, add_derived_date_features
from utils.display_components import create_date_range_filter, create_status_filter
from utils.loan_tape_loader import (
    load_loan_tape_data,
    load_unified_loan_customer_data,
    get_customer_payment_summary,
    get_data_diagnostics,
)
import numpy as np
from datetime import datetime, timedelta
import warnings

warnings.filterwarnings('ignore')

# -------------------------
# Page config & branding
# -------------------------
setup_page("CSL Capital | QBO Dashboard")

# -------------------------
# Load Data
# -------------------------
df, gl_df = load_qbo_data()
deals_df = load_deals()
mca_deals_df = load_mca_deals()
loan_tape_df = load_loan_tape_data()
unified_data_df = load_unified_loan_customer_data()
diagnostics = get_data_diagnostics()

# -------------------------
# Preprocess data
# -------------------------
df = preprocess_dataframe(
    df,
    numeric_cols=["total_amount", "balance", "amount"],
    date_cols=["date", "txn_date", "due_date", "created_date"]
)

if "txn_date" in df.columns:
    df = add_derived_date_features(df, "txn_date", prefix="")
    df["year_month"] = df["txn_date"].dt.to_period("M")

# -------------------------
# Helper Functions
# -------------------------
def calculate_portfolio_metrics(deals_df, unified_df, qbo_df):
    """Calculate comprehensive portfolio metrics"""
    metrics = {}

    # Filter to closed/won deals only
    if "is_closed_won" in deals_df.columns:
        active_deals = deals_df[deals_df["is_closed_won"] == True].copy()
    else:
        active_deals = deals_df.copy()

    # Total deployed capital
    metrics["total_deployed"] = active_deals["amount"].sum() if "amount" in active_deals.columns else 0

    # Expected return (factor_rate * amount - amount = profit)
    if "factor_rate" in active_deals.columns and "amount" in active_deals.columns:
        active_deals["expected_total_return"] = active_deals["amount"] * active_deals["factor_rate"]
        metrics["expected_total_return"] = active_deals["expected_total_return"].sum()
        metrics["expected_profit"] = metrics["expected_total_return"] - metrics["total_deployed"]
    else:
        metrics["expected_total_return"] = metrics["total_deployed"] * 1.3  # Default 30% return assumption
        metrics["expected_profit"] = metrics["expected_total_return"] - metrics["total_deployed"]

    # Actual cash received (from QBO payments)
    # Note: Exclude "Deposit" to avoid double-counting - Deposits often contain the same payments
    payment_types = ["Payment", "Receipt"]
    if "transaction_type" in qbo_df.columns:
        payments = qbo_df[qbo_df["transaction_type"].isin(payment_types)].copy()
        # Exclude internal transfers
        if "customer_name" in payments.columns:
            payments = payments[~payments["customer_name"].isin(["CSL", "VEEM", "CSL Capital", "Internal"])]
        metrics["total_cash_received"] = payments["total_amount"].abs().sum()
    else:
        metrics["total_cash_received"] = 0

    # Calculate key ratios
    if metrics["expected_total_return"] > 0:
        metrics["collection_rate"] = (metrics["total_cash_received"] / metrics["expected_total_return"]) * 100
    else:
        metrics["collection_rate"] = 0

    if metrics["total_deployed"] > 0:
        metrics["capital_returned_pct"] = (metrics["total_cash_received"] / metrics["total_deployed"]) * 100
    else:
        metrics["capital_returned_pct"] = 0

    # Outstanding amount
    metrics["outstanding"] = max(0, metrics["expected_total_return"] - metrics["total_cash_received"])

    # Loan counts
    metrics["total_loans"] = len(active_deals)

    # From unified data - loans with payments
    if not unified_df.empty and "RTR Amount" in unified_df.columns:
        metrics["loans_with_payments"] = (unified_df["RTR Amount"] > 0).sum()
        metrics["loans_without_payments"] = metrics["total_loans"] - metrics["loans_with_payments"]
    else:
        metrics["loans_with_payments"] = 0
        metrics["loans_without_payments"] = metrics["total_loans"]

    return metrics


def calculate_loan_health(unified_df, qbo_df):
    """Calculate loan health metrics with attention flags"""
    if unified_df.empty:
        return pd.DataFrame()

    health_df = unified_df.copy()

    # Ensure numeric columns
    for col in ["RTR Amount", "Participation Amount", "Expected Return", "Days Since Last Payment"]:
        if col in health_df.columns:
            health_df[col] = pd.to_numeric(health_df[col], errors="coerce").fillna(0)

    # Calculate collection percentage
    if "RTR Amount" in health_df.columns and "Expected Return" in health_df.columns:
        health_df["Collection %"] = np.where(
            health_df["Expected Return"] > 0,
            (health_df["RTR Amount"] / health_df["Expected Return"]) * 100,
            0
        )
    else:
        health_df["Collection %"] = 0

    # Calculate remaining to collect
    if "Expected Return" in health_df.columns and "RTR Amount" in health_df.columns:
        health_df["Remaining"] = health_df["Expected Return"] - health_df["RTR Amount"]
    else:
        health_df["Remaining"] = 0

    # Health status based on multiple factors
    def get_health_status(row):
        days_since = row.get("Days Since Last Payment", 999)
        collection_pct = row.get("Collection %", 0)
        rtr = row.get("RTR Amount", 0)

        # Fully collected
        if collection_pct >= 95:
            return "✅ Complete"

        # No payments ever
        if rtr == 0 or pd.isna(rtr):
            return "🔴 No Payments"

        # Stale - no payment in 45+ days
        if days_since > 45:
            return "🟠 Stale (45+ days)"

        # Slow - no payment in 30+ days
        if days_since > 30:
            return "🟡 Slow (30+ days)"

        # On track
        return "🟢 Active"

    health_df["Health Status"] = health_df.apply(get_health_status, axis=1)

    # Priority score for sorting (higher = needs more attention)
    def get_priority(row):
        status = row.get("Health Status", "")
        remaining = row.get("Remaining", 0)

        if "Complete" in status:
            return 0
        elif "No Payments" in status:
            return 100 + remaining / 1000
        elif "Stale" in status:
            return 80 + remaining / 1000
        elif "Slow" in status:
            return 60 + remaining / 1000
        else:
            return 20 + remaining / 1000

    health_df["Priority Score"] = health_df.apply(get_priority, axis=1)

    return health_df


def calculate_cash_forecast(deals_df, qbo_df, months_forward=6):
    """
    Calculate cash flow forecast based on historical patterns and outstanding loans
    """
    forecast = []

    # Get historical monthly inflows
    # Note: Exclude "Deposit" to avoid double-counting - Deposits often contain the same payments
    payment_types = ["Payment", "Receipt"]
    if "transaction_type" in qbo_df.columns:
        payments = qbo_df[qbo_df["transaction_type"].isin(payment_types)].copy()
        if "customer_name" in payments.columns:
            payments = payments[~payments["customer_name"].isin(["CSL", "VEEM", "CSL Capital", "Internal"])]
    else:
        payments = pd.DataFrame()

    if payments.empty or "txn_date" not in payments.columns:
        return pd.DataFrame()

    # Monthly aggregation
    payments["month"] = payments["txn_date"].dt.to_period("M")
    monthly = payments.groupby("month")["total_amount"].sum().reset_index()
    monthly["month"] = monthly["month"].dt.to_timestamp()
    monthly = monthly.sort_values("month")

    if len(monthly) < 2:
        return pd.DataFrame()

    # Calculate trends
    recent_months = monthly.tail(6)
    avg_monthly = recent_months["total_amount"].mean()

    # Calculate trend (growth/decline rate)
    if len(recent_months) >= 3:
        trend = (recent_months["total_amount"].iloc[-1] - recent_months["total_amount"].iloc[0]) / len(recent_months)
    else:
        trend = 0

    # Get total outstanding
    if "is_closed_won" in deals_df.columns:
        active_deals = deals_df[deals_df["is_closed_won"] == True].copy()
    else:
        active_deals = deals_df.copy()

    if "factor_rate" in active_deals.columns and "amount" in active_deals.columns:
        expected_total = (active_deals["amount"] * active_deals["factor_rate"]).sum()
    else:
        expected_total = active_deals["amount"].sum() * 1.3 if "amount" in active_deals.columns else 0

    total_received = payments["total_amount"].abs().sum()
    remaining_to_collect = max(0, expected_total - total_received)

    # Build forecast
    last_month = monthly["month"].max()
    cumulative_forecast = 0

    for i in range(1, months_forward + 1):
        forecast_month = last_month + pd.DateOffset(months=i)

        # Apply trend with dampening
        dampening = 0.9 ** i  # Trend effect decreases over time
        forecast_amount = avg_monthly + (trend * i * dampening)

        # Can't forecast more than what's remaining
        forecast_amount = min(forecast_amount, remaining_to_collect - cumulative_forecast)
        forecast_amount = max(0, forecast_amount)

        cumulative_forecast += forecast_amount

        forecast.append({
            "Month": forecast_month,
            "Forecast": forecast_amount,
            "Type": "Forecast",
            "Cumulative Forecast": cumulative_forecast
        })

    # Combine historical and forecast
    historical = monthly.copy()
    historical["Type"] = "Actual"
    historical = historical.rename(columns={"month": "Month", "total_amount": "Amount"})
    historical["Cumulative"] = historical["Amount"].cumsum()

    return historical, pd.DataFrame(forecast), remaining_to_collect


def get_partner_performance(unified_df):
    """Analyze performance by partner source"""
    if unified_df.empty or "Partner Source" not in unified_df.columns:
        return pd.DataFrame()

    partner_df = unified_df.copy()

    # Ensure numeric
    for col in ["Participation Amount", "Expected Return", "RTR Amount"]:
        if col in partner_df.columns:
            partner_df[col] = pd.to_numeric(partner_df[col], errors="coerce").fillna(0)

    # Aggregate by partner
    partner_perf = partner_df.groupby("Partner Source").agg({
        "Loan ID": "count",
        "Participation Amount": "sum",
        "Expected Return": "sum",
        "RTR Amount": "sum"
    }).reset_index()

    partner_perf.columns = ["Partner", "Loans", "Deployed", "Expected Return", "Cash Collected"]

    # Calculate metrics
    partner_perf["Collection Rate"] = np.where(
        partner_perf["Expected Return"] > 0,
        (partner_perf["Cash Collected"] / partner_perf["Expected Return"]) * 100,
        0
    )

    partner_perf["Outstanding"] = partner_perf["Expected Return"] - partner_perf["Cash Collected"]
    partner_perf["Avg Loan Size"] = partner_perf["Deployed"] / partner_perf["Loans"]

    return partner_perf.sort_values("Cash Collected", ascending=False)


def get_monthly_cohort_performance(unified_df, deals_df):
    """Analyze performance by origination cohort"""
    if unified_df.empty:
        return pd.DataFrame()

    cohort_df = unified_df.copy()

    # Get deal dates
    if "Deal Date" in cohort_df.columns:
        date_col = "Deal Date"
    elif "date_created" in deals_df.columns:
        # Merge deal dates
        deals_dates = deals_df[["loan_id", "date_created"]].copy()
        deals_dates["loan_id"] = deals_dates["loan_id"].astype(str).str.strip()
        cohort_df["Loan ID"] = cohort_df["Loan ID"].astype(str).str.strip()
        cohort_df = cohort_df.merge(deals_dates, left_on="Loan ID", right_on="loan_id", how="left")
        date_col = "date_created"
    else:
        return pd.DataFrame()

    # Create monthly cohort
    cohort_df[date_col] = pd.to_datetime(cohort_df[date_col], errors="coerce")
    cohort_df["Cohort"] = cohort_df[date_col].dt.to_period("M")

    # Ensure numeric
    for col in ["Participation Amount", "Expected Return", "RTR Amount"]:
        if col in cohort_df.columns:
            cohort_df[col] = pd.to_numeric(cohort_df[col], errors="coerce").fillna(0)

    # Aggregate by cohort
    cohort_perf = cohort_df.groupby("Cohort").agg({
        "Loan ID": "count",
        "Participation Amount": "sum",
        "Expected Return": "sum",
        "RTR Amount": "sum"
    }).reset_index()

    cohort_perf.columns = ["Cohort", "Loans", "Deployed", "Expected", "Collected"]

    # Calculate metrics
    cohort_perf["Collection %"] = np.where(
        cohort_perf["Expected"] > 0,
        (cohort_perf["Collected"] / cohort_perf["Expected"]) * 100,
        0
    )

    cohort_perf["Remaining"] = cohort_perf["Expected"] - cohort_perf["Collected"]

    # Sort by cohort date
    cohort_perf = cohort_perf.sort_values("Cohort")
    cohort_perf["Cohort"] = cohort_perf["Cohort"].astype(str)

    return cohort_perf


# ===================================================================
# DASHBOARD START
# ===================================================================

st.title("💰 Cash Performance Dashboard")
st.markdown("**Real-time portfolio cash flow insights and decision support**")
st.markdown("---")

# Calculate portfolio metrics
metrics = calculate_portfolio_metrics(deals_df, unified_data_df, df)

# ===================================================================
# SECTION 1: EXECUTIVE CASH PERFORMANCE SUMMARY
# ===================================================================
st.header("📊 Executive Summary: Cash In vs Expected")

# Key metrics in prominent display
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "Total Deployed",
        f"${metrics['total_deployed']:,.0f}",
        help="Total capital invested across all closed deals"
    )

with col2:
    st.metric(
        "Expected Return",
        f"${metrics['expected_total_return']:,.0f}",
        help="Total expected return based on factor rates (principal + profit)"
    )

with col3:
    st.metric(
        "Cash Received",
        f"${metrics['total_cash_received']:,.0f}",
        delta=f"{metrics['collection_rate']:.1f}% of expected",
        help="Total cash collected from loan repayments"
    )

with col4:
    st.metric(
        "Outstanding",
        f"${metrics['outstanding']:,.0f}",
        delta=f"-{100 - metrics['collection_rate']:.1f}% remaining",
        delta_color="inverse",
        help="Remaining amount to collect"
    )

# Visual progress bar
st.markdown("### Collection Progress")
progress_pct = min(metrics['collection_rate'] / 100, 1.0)

# Create a visual progress indicator
progress_col1, progress_col2 = st.columns([3, 1])
with progress_col1:
    st.progress(progress_pct)
with progress_col2:
    st.markdown(f"**{metrics['collection_rate']:.1f}%** collected")

# Loan health summary
st.markdown("### Portfolio Health Snapshot")
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Total Loans", f"{metrics['total_loans']:,}")

with col2:
    st.metric(
        "Loans with Payments",
        f"{metrics['loans_with_payments']:,}",
        delta=f"{(metrics['loans_with_payments']/metrics['total_loans']*100) if metrics['total_loans'] > 0 else 0:.0f}%"
    )

with col3:
    st.metric(
        "Loans w/o Payments",
        f"{metrics['loans_without_payments']:,}",
        delta="needs attention" if metrics['loans_without_payments'] > 0 else None,
        delta_color="inverse"
    )

with col4:
    # Capital returned percentage
    st.metric(
        "Capital Returned",
        f"{metrics['capital_returned_pct']:.1f}%",
        help="Percentage of original capital that has been returned"
    )

st.markdown("---")

# ===================================================================
# SECTION 2: CASH IN VS EXPECTED ANALYSIS
# ===================================================================
st.header("📈 Cash Collection Analysis")

tab1, tab2, tab3 = st.tabs(["By Time Period", "By Partner", "By Cohort"])

with tab1:
    st.subheader("Monthly Cash Collection Trend")

    # Get monthly payment data
    # Note: Exclude "Deposit" to avoid double-counting - Deposits often contain the same payments
    payment_types = ["Payment", "Receipt"]
    if "transaction_type" in df.columns:
        payments = df[df["transaction_type"].isin(payment_types)].copy()
        if "customer_name" in payments.columns:
            payments = payments[~payments["customer_name"].isin(["CSL", "VEEM", "CSL Capital", "Internal"])]

        if not payments.empty and "txn_date" in payments.columns:
            payments["month"] = payments["txn_date"].dt.to_period("M")
            monthly_payments = payments.groupby("month")["total_amount"].sum().reset_index()
            monthly_payments["month"] = monthly_payments["month"].dt.to_timestamp()
            monthly_payments = monthly_payments.sort_values("month")
            monthly_payments["cumulative"] = monthly_payments["total_amount"].cumsum()

            # Calculate expected cumulative line (linear assumption)
            total_expected = metrics["expected_total_return"]
            first_month = monthly_payments["month"].min()
            months_elapsed = len(monthly_payments)
            monthly_payments["expected_cumulative"] = [
                (i+1) / (months_elapsed + 6) * total_expected
                for i in range(len(monthly_payments))
            ]

            # Monthly trend chart
            base = alt.Chart(monthly_payments).encode(
                x=alt.X("month:T", title="Month", axis=alt.Axis(format="%b %Y"))
            )

            bars = base.mark_bar(color=PRIMARY_COLOR, opacity=0.7).encode(
                y=alt.Y("total_amount:Q", title="Monthly Cash ($)", axis=alt.Axis(format="$,.0f")),
                tooltip=[
                    alt.Tooltip("month:T", title="Month", format="%B %Y"),
                    alt.Tooltip("total_amount:Q", title="Cash Collected", format="$,.0f")
                ]
            )

            # Add trend line
            trend_line = base.mark_line(color="orange", strokeWidth=2, strokeDash=[5, 5]).encode(
                y=alt.Y("total_amount:Q"),
            ).transform_regression("month", "total_amount")

            chart = (bars + trend_line).properties(
                width=700,
                height=400,
                title="Monthly Cash Collections with Trend"
            )

            st.altair_chart(chart, use_container_width=True)

            # Cumulative chart
            st.subheader("Cumulative Collections vs Target")

            cumulative_chart = alt.Chart(monthly_payments).mark_area(
                color=PRIMARY_COLOR,
                opacity=0.5,
                line={"color": PRIMARY_COLOR, "strokeWidth": 2}
            ).encode(
                x=alt.X("month:T", title="Month"),
                y=alt.Y("cumulative:Q", title="Cumulative Cash ($)", axis=alt.Axis(format="$,.0f")),
                tooltip=[
                    alt.Tooltip("month:T", title="Month", format="%B %Y"),
                    alt.Tooltip("cumulative:Q", title="Cumulative", format="$,.0f"),
                ]
            ).properties(
                width=700,
                height=300,
                title="Cumulative Cash Collections"
            )

            # Add expected line
            expected_line = alt.Chart(pd.DataFrame({
                "month": monthly_payments["month"],
                "expected": monthly_payments["expected_cumulative"]
            })).mark_line(color="red", strokeDash=[5, 5], strokeWidth=2).encode(
                x="month:T",
                y=alt.Y("expected:Q")
            )

            st.altair_chart(cumulative_chart + expected_line, use_container_width=True)

            # Monthly stats
            col1, col2, col3 = st.columns(3)
            with col1:
                avg_monthly = monthly_payments["total_amount"].mean()
                st.metric("Avg Monthly Collection", f"${avg_monthly:,.0f}")
            with col2:
                last_month_val = monthly_payments["total_amount"].iloc[-1] if len(monthly_payments) > 0 else 0
                prev_month_val = monthly_payments["total_amount"].iloc[-2] if len(monthly_payments) > 1 else last_month_val
                change = ((last_month_val - prev_month_val) / prev_month_val * 100) if prev_month_val > 0 else 0
                st.metric("Last Month", f"${last_month_val:,.0f}", delta=f"{change:+.1f}%")
            with col3:
                months_to_complete = metrics["outstanding"] / avg_monthly if avg_monthly > 0 else float("inf")
                st.metric("Est. Months to Complete", f"{months_to_complete:.1f}" if months_to_complete < 100 else "N/A")
        else:
            st.warning("No payment data available for time analysis")
    else:
        st.warning("No transaction type data available")

with tab2:
    st.subheader("Performance by Partner Source")

    partner_perf = get_partner_performance(unified_data_df)

    if not partner_perf.empty:
        # Partner performance chart
        partner_chart = alt.Chart(partner_perf).mark_bar().encode(
            x=alt.X("Partner:N", sort="-y", title="Partner Source"),
            y=alt.Y("Cash Collected:Q", title="Cash Collected ($)", axis=alt.Axis(format="$,.0f")),
            color=alt.Color("Collection Rate:Q",
                          scale=alt.Scale(scheme="redyellowgreen", domain=[0, 100]),
                          title="Collection Rate %"),
            tooltip=[
                alt.Tooltip("Partner:N", title="Partner"),
                alt.Tooltip("Loans:Q", title="# Loans"),
                alt.Tooltip("Deployed:Q", title="Deployed", format="$,.0f"),
                alt.Tooltip("Cash Collected:Q", title="Collected", format="$,.0f"),
                alt.Tooltip("Collection Rate:Q", title="Collection %", format=".1f"),
                alt.Tooltip("Outstanding:Q", title="Outstanding", format="$,.0f")
            ]
        ).properties(
            width=700,
            height=400,
            title="Cash Collected by Partner (colored by collection rate)"
        )

        st.altair_chart(partner_chart, use_container_width=True)

        # Partner table
        st.dataframe(
            partner_perf,
            column_config={
                "Partner": st.column_config.TextColumn("Partner"),
                "Loans": st.column_config.NumberColumn("Loans"),
                "Deployed": st.column_config.NumberColumn("Deployed", format="$%.0f"),
                "Expected Return": st.column_config.NumberColumn("Expected", format="$%.0f"),
                "Cash Collected": st.column_config.NumberColumn("Collected", format="$%.0f"),
                "Collection Rate": st.column_config.NumberColumn("Collection %", format="%.1f%%"),
                "Outstanding": st.column_config.NumberColumn("Outstanding", format="$%.0f"),
                "Avg Loan Size": st.column_config.NumberColumn("Avg Loan", format="$%.0f"),
            },
            hide_index=True,
            use_container_width=True
        )
    else:
        st.info("No partner performance data available")

with tab3:
    st.subheader("Performance by Origination Cohort")

    cohort_perf = get_monthly_cohort_performance(unified_data_df, deals_df)

    if not cohort_perf.empty:
        # Cohort chart
        cohort_chart = alt.Chart(cohort_perf).mark_bar().encode(
            x=alt.X("Cohort:N", title="Origination Month"),
            y=alt.Y("Collected:Q", title="Cash Collected ($)", axis=alt.Axis(format="$,.0f")),
            color=alt.Color("Collection %:Q",
                          scale=alt.Scale(scheme="redyellowgreen", domain=[0, 100]),
                          title="Collection %"),
            tooltip=[
                alt.Tooltip("Cohort:N", title="Cohort"),
                alt.Tooltip("Loans:Q", title="# Loans"),
                alt.Tooltip("Deployed:Q", title="Deployed", format="$,.0f"),
                alt.Tooltip("Collected:Q", title="Collected", format="$,.0f"),
                alt.Tooltip("Collection %:Q", title="Collection %", format=".1f"),
                alt.Tooltip("Remaining:Q", title="Remaining", format="$,.0f")
            ]
        ).properties(
            width=700,
            height=400,
            title="Cash Collected by Origination Cohort"
        )

        st.altair_chart(cohort_chart, use_container_width=True)

        # Cohort table
        st.dataframe(
            cohort_perf,
            column_config={
                "Cohort": st.column_config.TextColumn("Cohort"),
                "Loans": st.column_config.NumberColumn("Loans"),
                "Deployed": st.column_config.NumberColumn("Deployed", format="$%.0f"),
                "Expected": st.column_config.NumberColumn("Expected", format="$%.0f"),
                "Collected": st.column_config.NumberColumn("Collected", format="$%.0f"),
                "Collection %": st.column_config.NumberColumn("Collection %", format="%.1f%%"),
                "Remaining": st.column_config.NumberColumn("Remaining", format="$%.0f"),
            },
            hide_index=True,
            use_container_width=True
        )
    else:
        st.info("No cohort performance data available")

st.markdown("---")

# ===================================================================
# SECTION 3: LOAN HEALTH DASHBOARD
# ===================================================================
st.header("🏥 Loan Health Monitor")

health_df = calculate_loan_health(unified_data_df, df)

if not health_df.empty:
    # Health summary cards
    health_counts = health_df["Health Status"].value_counts()

    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        complete = health_counts.get("✅ Complete", 0)
        st.metric("Complete", complete, help="Loans that are 95%+ collected")

    with col2:
        active = health_counts.get("🟢 Active", 0)
        st.metric("Active", active, help="Loans with recent payments (within 30 days)")

    with col3:
        slow = health_counts.get("🟡 Slow (30+ days)", 0)
        st.metric("Slow", slow, help="No payment in 30-45 days", delta="attention" if slow > 0 else None, delta_color="off")

    with col4:
        stale = health_counts.get("🟠 Stale (45+ days)", 0)
        st.metric("Stale", stale, help="No payment in 45+ days", delta="review" if stale > 0 else None, delta_color="inverse")

    with col5:
        no_payments = health_counts.get("🔴 No Payments", 0)
        st.metric("No Payments", no_payments, help="Loans with zero payments", delta="urgent" if no_payments > 0 else None, delta_color="inverse")

    # Loans needing attention
    st.subheader("⚠️ Loans Requiring Attention")

    # Filter to loans needing attention (not complete, not active)
    attention_df = health_df[~health_df["Health Status"].isin(["✅ Complete", "🟢 Active"])].copy()
    attention_df = attention_df.sort_values("Priority Score", ascending=False)

    if not attention_df.empty:
        # Select key columns for display
        display_cols = ["Deal Name", "Health Status", "Participation Amount", "Expected Return",
                       "RTR Amount", "Collection %", "Remaining", "Days Since Last Payment"]
        available_cols = [c for c in display_cols if c in attention_df.columns]

        st.dataframe(
            attention_df[available_cols].head(20),
            column_config={
                "Deal Name": st.column_config.TextColumn("Deal", width="medium"),
                "Health Status": st.column_config.TextColumn("Status", width="small"),
                "Participation Amount": st.column_config.NumberColumn("Deployed", format="$%.0f"),
                "Expected Return": st.column_config.NumberColumn("Expected", format="$%.0f"),
                "RTR Amount": st.column_config.NumberColumn("Collected", format="$%.0f"),
                "Collection %": st.column_config.NumberColumn("Collected %", format="%.1f%%"),
                "Remaining": st.column_config.NumberColumn("Remaining", format="$%.0f"),
                "Days Since Last Payment": st.column_config.NumberColumn("Days Silent", format="%.0f"),
            },
            hide_index=True,
            use_container_width=True
        )

        # Calculate total at risk
        total_at_risk = attention_df["Remaining"].sum()
        st.warning(f"**${total_at_risk:,.0f}** outstanding across {len(attention_df)} loans requiring attention")
    else:
        st.success("✅ All loans are either complete or actively paying!")

    # Full loan health table (expandable)
    with st.expander("View All Loans Health Status"):
        all_display_cols = ["Deal Name", "Health Status", "Participation Amount", "Expected Return",
                          "RTR Amount", "Collection %", "Remaining", "Days Since Last Payment", "Partner Source"]
        available_cols = [c for c in all_display_cols if c in health_df.columns]

        # Sort by Priority Score first (if exists), then select display columns
        sorted_health_df = health_df.sort_values("Priority Score", ascending=False) if "Priority Score" in health_df.columns else health_df
        st.dataframe(
            sorted_health_df[available_cols],
            column_config={
                "Deal Name": st.column_config.TextColumn("Deal", width="medium"),
                "Health Status": st.column_config.TextColumn("Status"),
                "Participation Amount": st.column_config.NumberColumn("Deployed", format="$%.0f"),
                "Expected Return": st.column_config.NumberColumn("Expected", format="$%.0f"),
                "RTR Amount": st.column_config.NumberColumn("Collected", format="$%.0f"),
                "Collection %": st.column_config.NumberColumn("Collected %", format="%.1f%%"),
                "Remaining": st.column_config.NumberColumn("Remaining", format="$%.0f"),
                "Days Since Last Payment": st.column_config.NumberColumn("Days Silent", format="%.0f"),
                "Partner Source": st.column_config.TextColumn("Partner"),
            },
            hide_index=True,
            use_container_width=True
        )

else:
    st.warning("No loan health data available")

st.markdown("---")

# ===================================================================
# SECTION 4: CASH FLOW FORECAST
# ===================================================================
st.header("🔮 Cash Flow Forecast")

forecast_result = calculate_cash_forecast(deals_df, df, months_forward=6)

if forecast_result and len(forecast_result) == 3:
    historical, forecast_df, remaining = forecast_result

    if not historical.empty and not forecast_df.empty:
        # Create combined chart data
        hist_for_chart = historical.tail(12).copy()  # Last 12 months
        hist_for_chart["Type"] = "Actual"
        hist_for_chart = hist_for_chart.rename(columns={"Amount": "Value"})

        forecast_for_chart = forecast_df.copy()
        forecast_for_chart = forecast_for_chart.rename(columns={"Forecast": "Value"})

        combined = pd.concat([
            hist_for_chart[["Month", "Value", "Type"]],
            forecast_for_chart[["Month", "Value", "Type"]]
        ], ignore_index=True)

        # Forecast chart
        forecast_chart = alt.Chart(combined).mark_bar().encode(
            x=alt.X("Month:T", title="Month", axis=alt.Axis(format="%b %Y")),
            y=alt.Y("Value:Q", title="Cash Flow ($)", axis=alt.Axis(format="$,.0f")),
            color=alt.Color("Type:N",
                          scale=alt.Scale(domain=["Actual", "Forecast"], range=[PRIMARY_COLOR, "orange"]),
                          title="Type"),
            opacity=alt.condition(
                alt.datum.Type == "Forecast",
                alt.value(0.6),
                alt.value(1)
            ),
            tooltip=[
                alt.Tooltip("Month:T", title="Month", format="%B %Y"),
                alt.Tooltip("Value:Q", title="Amount", format="$,.0f"),
                alt.Tooltip("Type:N", title="Type")
            ]
        ).properties(
            width=700,
            height=400,
            title="6-Month Cash Flow Forecast"
        )

        st.altair_chart(forecast_chart, use_container_width=True)

        # Forecast summary
        col1, col2, col3 = st.columns(3)

        with col1:
            total_forecast = forecast_df["Forecast"].sum()
            st.metric("6-Month Forecast", f"${total_forecast:,.0f}")

        with col2:
            avg_forecast = forecast_df["Forecast"].mean()
            st.metric("Avg Monthly Forecast", f"${avg_forecast:,.0f}")

        with col3:
            # Months to collect remaining
            if avg_forecast > 0:
                months_remaining = remaining / avg_forecast
                st.metric("Months to Complete", f"{months_remaining:.1f}" if months_remaining < 50 else "50+")
            else:
                st.metric("Months to Complete", "N/A")

        # Forecast table
        st.subheader("Monthly Forecast Details")
        forecast_display = forecast_df.copy()
        forecast_display["Month"] = forecast_display["Month"].dt.strftime("%B %Y")

        st.dataframe(
            forecast_display[["Month", "Forecast", "Cumulative Forecast"]],
            column_config={
                "Month": st.column_config.TextColumn("Month"),
                "Forecast": st.column_config.NumberColumn("Forecast", format="$%.0f"),
                "Cumulative Forecast": st.column_config.NumberColumn("Cumulative", format="$%.0f"),
            },
            hide_index=True,
            use_container_width=True
        )

        st.info(f"**Note:** Forecast is based on historical collection patterns. ${remaining:,.0f} remaining to collect.")
    else:
        st.warning("Insufficient historical data for forecasting")
else:
    st.warning("Unable to generate forecast - insufficient data")

st.markdown("---")

# ===================================================================
# SECTION 5: DECISION SUPPORT & RECOMMENDATIONS
# ===================================================================
st.header("🎯 Decision Support & Recommendations")

# Generate actionable recommendations
recommendations = []

# Check collection rate
if metrics["collection_rate"] < 50:
    recommendations.append({
        "priority": "HIGH",
        "area": "Collection Rate",
        "insight": f"Collection rate is only {metrics['collection_rate']:.1f}%",
        "action": "Review loans without payments and prioritize collection outreach"
    })
elif metrics["collection_rate"] < 75:
    recommendations.append({
        "priority": "MEDIUM",
        "area": "Collection Rate",
        "insight": f"Collection rate is {metrics['collection_rate']:.1f}%",
        "action": "Identify slow-paying loans and establish follow-up cadence"
    })

# Check loans without payments
if metrics["loans_without_payments"] > 0:
    pct_no_payments = metrics["loans_without_payments"] / metrics["total_loans"] * 100
    if pct_no_payments > 20:
        recommendations.append({
            "priority": "HIGH",
            "area": "Non-Performing Loans",
            "insight": f"{metrics['loans_without_payments']} loans ({pct_no_payments:.0f}%) have no payments",
            "action": "Immediate review of loans with zero collections - verify loan status and customer contact"
        })
    else:
        recommendations.append({
            "priority": "MEDIUM",
            "area": "Non-Performing Loans",
            "insight": f"{metrics['loans_without_payments']} loans have no payments",
            "action": "Schedule review of non-paying loans with respective partners"
        })

# Check stale loans
if not health_df.empty:
    stale_count = len(health_df[health_df["Health Status"].str.contains("Stale|No Payments", na=False)])
    if stale_count > 0:
        stale_amount = health_df[health_df["Health Status"].str.contains("Stale|No Payments", na=False)]["Remaining"].sum()
        recommendations.append({
            "priority": "HIGH" if stale_amount > 100000 else "MEDIUM",
            "area": "Stale Loans",
            "insight": f"{stale_count} loans are stale/non-paying with ${stale_amount:,.0f} outstanding",
            "action": "Escalate collection efforts on stale accounts"
        })

# Partner performance issues
partner_perf = get_partner_performance(unified_data_df)
if not partner_perf.empty:
    poor_partners = partner_perf[partner_perf["Collection Rate"] < 50]
    if not poor_partners.empty:
        worst_partner = poor_partners.iloc[0]["Partner"]
        worst_rate = poor_partners.iloc[0]["Collection Rate"]
        recommendations.append({
            "priority": "MEDIUM",
            "area": "Partner Performance",
            "insight": f"Partner '{worst_partner}' has only {worst_rate:.0f}% collection rate",
            "action": "Review partner relationship and loan quality from this source"
        })

# If no issues, add positive note
if not recommendations:
    recommendations.append({
        "priority": "INFO",
        "area": "Portfolio Health",
        "insight": "Portfolio is performing within expected parameters",
        "action": "Continue monitoring and maintain current collection cadence"
    })

# Display recommendations
for rec in recommendations:
    if rec["priority"] == "HIGH":
        st.error(f"🔴 **{rec['area']}:** {rec['insight']}")
        st.markdown(f"   → **Action:** {rec['action']}")
    elif rec["priority"] == "MEDIUM":
        st.warning(f"🟡 **{rec['area']}:** {rec['insight']}")
        st.markdown(f"   → **Action:** {rec['action']}")
    else:
        st.info(f"🟢 **{rec['area']}:** {rec['insight']}")
        st.markdown(f"   → **Action:** {rec['action']}")

st.markdown("---")

# ===================================================================
# SECTION 6: DATA DETAILS (Collapsed)
# ===================================================================
with st.expander("📋 Data Details & Diagnostics"):

    tab1, tab2, tab3 = st.tabs(["Unified Loan Data", "Data Diagnostics", "Customer Summary"])

    with tab1:
        st.subheader("Complete Loan Performance Data")

        if not unified_data_df.empty:
            st.dataframe(
                unified_data_df,
                column_config={
                    "Loan ID": st.column_config.TextColumn("Loan ID", width="small"),
                    "Deal Name": st.column_config.TextColumn("Deal Name", width="medium"),
                    "QBO Customer": st.column_config.TextColumn("QBO Customer", width="medium"),
                    "Factor Rate": st.column_config.NumberColumn("Factor Rate", format="%.3f"),
                    "Participation Amount": st.column_config.NumberColumn("Participation", format="$%.0f"),
                    "Expected Return": st.column_config.NumberColumn("Expected Return", format="$%.0f"),
                    "RTR Amount": st.column_config.NumberColumn("RTR Amount", format="$%.0f"),
                    "RTR %": st.column_config.NumberColumn("RTR %", format="%.1f%%"),
                },
                hide_index=True,
                use_container_width=True
            )

            # Download button
            csv = unified_data_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="Download Full Data (CSV)",
                data=csv,
                file_name=f"portfolio_data_{pd.Timestamp.now().strftime('%Y%m%d')}.csv",
                mime="text/csv"
            )
        else:
            st.warning("No unified data available")

    with tab2:
        st.subheader("Data Join Diagnostics")

        if diagnostics:
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Total QBO Transactions", f"{diagnostics.get('raw_qbo_count', 0):,}")
                st.metric("Total Deals", f"{diagnostics.get('raw_deals_count', 0):,}")
            with col2:
                st.metric("Closed Won Deals", f"{diagnostics.get('closed_won_deals', 0):,}")
                overlap = diagnostics.get('overlapping_loan_ids', 0)
                total_deals = diagnostics.get('unique_deal_loan_ids', 1)
                st.metric("Loan ID Match Rate", f"{(overlap/total_deals*100):.0f}%")

            # Data warnings
            if diagnostics.get("raw_qbo_count", 0) == 1000:
                st.warning("⚠️ QBO data may be truncated at 1000 records")
        else:
            st.warning("Diagnostics data not available")

    with tab3:
        st.subheader("Customer Payment Summary")

        customer_summary = get_customer_payment_summary()

        if not customer_summary.empty:
            st.dataframe(
                customer_summary,
                column_config={
                    "Customer": st.column_config.TextColumn("Customer", width="medium"),
                    "Total Payments": st.column_config.NumberColumn("Total Payments", format="$%.0f"),
                    "Payment Count": st.column_config.NumberColumn("Payments"),
                    "Unique Loans": st.column_config.NumberColumn("Loans"),
                    "Unattributed Amount": st.column_config.NumberColumn("Unattributed", format="$%.0f"),
                },
                hide_index=True,
                use_container_width=True
            )
        else:
            st.info("No customer data available")

st.markdown("---")
st.markdown(f"*Dashboard updated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}*")
