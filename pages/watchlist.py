# pages/watchlist.py
"""
Loan Watchlist - Surface loans needing attention before they become problems.

This page identifies:
1. Past Maturity: Active loans past their maturity date
2. Approaching Maturity: Loans due soon with low payment performance
3. Payment Performance Declining: Loans with deteriorating payment patterns
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from utils.config import setup_page, PRIMARY_COLOR
from utils.data_loader import load_loan_summaries, load_deals
from utils.loan_tape_data import prepare_loan_data
from utils.status_constants import PROBLEM_STATUSES, STATUS_COLORS, TERMINAL_STATUSES

# Page setup
setup_page("CSL Capital | Loan Watchlist")

# =============================================================================
# CONSTANTS
# =============================================================================

# Severity thresholds
HIGH_SEVERITY_PERFORMANCE = 0.70  # Below 70% performance = high severity
MEDIUM_SEVERITY_PERFORMANCE = 0.80  # Below 80% performance = medium severity
APPROACHING_MATURITY_DAYS = 60  # Loans due within 60 days

# Severity color codes
SEVERITY_COLORS = {
    "High": "#dc3545",     # Red
    "Medium": "#fd7e14",   # Orange
    "Low": "#ffc107",      # Yellow
}

SEVERITY_BADGES = {
    "High": "🔴",
    "Medium": "🟠",
    "Low": "🟡",
}


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def calculate_severity(row: pd.Series, category: str) -> str:
    """
    Calculate severity level based on loan metrics.

    Severity Levels:
    - High: Past maturity + performance < 70%
    - Medium: Past maturity OR approaching with low performance
    - Low: Minor concerns
    """
    is_past_maturity = row.get("is_past_maturity", False)
    payment_performance = row.get("payment_performance", 1.0) or 1.0

    if category == "past_maturity":
        if payment_performance < HIGH_SEVERITY_PERFORMANCE:
            return "High"
        elif payment_performance < MEDIUM_SEVERITY_PERFORMANCE:
            return "Medium"
        else:
            return "Low"
    elif category == "approaching_maturity":
        if payment_performance < HIGH_SEVERITY_PERFORMANCE:
            return "High"
        elif payment_performance < MEDIUM_SEVERITY_PERFORMANCE:
            return "Medium"
        else:
            return "Low"
    elif category == "declining_performance":
        # For declining performance, base on the severity of decline
        decline_rate = row.get("performance_decline", 0)
        if decline_rate > 0.20:  # More than 20% decline
            return "High"
        elif decline_rate > 0.10:  # More than 10% decline
            return "Medium"
        else:
            return "Low"

    return "Low"


def format_currency(value: float) -> str:
    """Format value as currency."""
    if pd.isna(value):
        return "-"
    return f"${value:,.0f}"


def format_percentage(value: float) -> str:
    """Format value as percentage."""
    if pd.isna(value):
        return "-"
    return f"{value * 100:.1f}%"


def get_days_past_maturity(maturity_date: pd.Timestamp, today: pd.Timestamp) -> int:
    """Calculate days past maturity."""
    if pd.isna(maturity_date):
        return 0
    return max(0, (today - maturity_date).days)


def get_days_to_maturity(maturity_date: pd.Timestamp, today: pd.Timestamp) -> int:
    """Calculate days to maturity."""
    if pd.isna(maturity_date):
        return 999
    return (maturity_date - today).days


def display_severity_badge(severity: str) -> str:
    """Return severity badge HTML."""
    emoji = SEVERITY_BADGES.get(severity, "⚪")
    color = SEVERITY_COLORS.get(severity, "#6c757d")
    return f'<span style="color:{color}">{emoji} {severity}</span>'


def create_watchlist_table(df: pd.DataFrame, columns: list, title: str, severity_col: str = "severity"):
    """Create a styled watchlist table."""
    if df.empty:
        st.info(f"No loans in {title.lower()}")
        return

    # Sort by severity (High first) then by net_balance descending
    severity_order = {"High": 0, "Medium": 1, "Low": 2}
    df = df.copy()
    df["_severity_order"] = df[severity_col].map(severity_order)
    df = df.sort_values(["_severity_order", "net_balance"], ascending=[True, False])

    # Display table
    display_df = df[columns].copy()

    # Format columns
    if "net_balance" in display_df.columns:
        display_df["net_balance"] = display_df["net_balance"].apply(format_currency)
    if "total_invested" in display_df.columns:
        display_df["total_invested"] = display_df["total_invested"].apply(format_currency)
    if "total_paid" in display_df.columns:
        display_df["total_paid"] = display_df["total_paid"].apply(format_currency)
    if "payment_performance" in display_df.columns:
        display_df["payment_performance"] = display_df["payment_performance"].apply(format_percentage)

    # Rename columns for display
    column_names = {
        "deal_name": "Deal Name",
        "loan_id": "Loan ID",
        "loan_status": "Status",
        "severity": "Severity",
        "days_past_maturity": "Days Past",
        "days_to_maturity": "Days Left",
        "payment_performance": "Performance",
        "net_balance": "Net Balance",
        "total_invested": "Invested",
        "total_paid": "Paid",
        "maturity_date": "Maturity Date",
        "partner_source": "Partner",
    }
    display_df = display_df.rename(columns=column_names)

    st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True,
    )


# =============================================================================
# MAIN PAGE
# =============================================================================

st.title("Loan Watchlist")
st.markdown("*Loans requiring attention - sorted by exposure*")

# Load data
with st.spinner("Loading loan data..."):
    loans_df = load_loan_summaries()
    deals_df = load_deals()

    if loans_df.empty:
        st.error("No loan data available")
        st.stop()

    # Prepare loan data
    df = prepare_loan_data(loans_df, deals_df)

# Get today's date
today = pd.Timestamp.today().tz_localize(None)

# Ensure maturity_date is datetime
if "maturity_date" in df.columns:
    df["maturity_date"] = pd.to_datetime(df["maturity_date"], errors="coerce")

# =============================================================================
# SECTION 1: PAST MATURITY LOANS
# =============================================================================

st.header("Past Maturity")
st.markdown("*Active loans that have passed their maturity date*")

# Filter: Active status but past maturity
active_statuses = ["Active", "Active - Frequently Late"]
past_maturity_mask = (
    (df["loan_status"].isin(active_statuses)) &
    (df["maturity_date"].notna()) &
    (df["maturity_date"] < today)
)

past_maturity_df = df[past_maturity_mask].copy()

if not past_maturity_df.empty:
    # Calculate days past maturity
    past_maturity_df["days_past_maturity"] = past_maturity_df["maturity_date"].apply(
        lambda x: get_days_past_maturity(x, today)
    )

    # Calculate severity
    past_maturity_df["severity"] = past_maturity_df.apply(
        lambda row: calculate_severity(row, "past_maturity"), axis=1
    )

    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Loans", len(past_maturity_df))
    with col2:
        high_severity = (past_maturity_df["severity"] == "High").sum()
        st.metric("High Severity", high_severity)
    with col3:
        total_exposure = past_maturity_df["net_balance"].sum()
        st.metric("Total Exposure", format_currency(total_exposure))
    with col4:
        avg_days_past = past_maturity_df["days_past_maturity"].mean()
        st.metric("Avg Days Past", f"{avg_days_past:.0f}")

    # Display table
    create_watchlist_table(
        past_maturity_df,
        ["severity", "deal_name", "loan_status", "days_past_maturity", "payment_performance", "net_balance", "partner_source"],
        "Past Maturity"
    )
else:
    st.success("No active loans are past maturity")

st.divider()

# =============================================================================
# SECTION 2: APPROACHING MATURITY
# =============================================================================

st.header("Approaching Maturity")
st.markdown(f"*Loans due within {APPROACHING_MATURITY_DAYS} days with payment performance below 80%*")

# Filter: Due within 60 days with low performance
approaching_mask = (
    (~df["loan_status"].isin(TERMINAL_STATUSES)) &
    (df["maturity_date"].notna()) &
    (df["maturity_date"] >= today) &
    (df["maturity_date"] <= today + timedelta(days=APPROACHING_MATURITY_DAYS)) &
    (df["payment_performance"] < MEDIUM_SEVERITY_PERFORMANCE)
)

approaching_df = df[approaching_mask].copy()

if not approaching_df.empty:
    # Calculate days to maturity
    approaching_df["days_to_maturity"] = approaching_df["maturity_date"].apply(
        lambda x: get_days_to_maturity(x, today)
    )

    # Calculate severity
    approaching_df["severity"] = approaching_df.apply(
        lambda row: calculate_severity(row, "approaching_maturity"), axis=1
    )

    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Loans", len(approaching_df))
    with col2:
        high_severity = (approaching_df["severity"] == "High").sum()
        st.metric("High Severity", high_severity)
    with col3:
        total_exposure = approaching_df["net_balance"].sum()
        st.metric("Total Exposure", format_currency(total_exposure))
    with col4:
        avg_performance = approaching_df["payment_performance"].mean()
        st.metric("Avg Performance", format_percentage(avg_performance))

    # Display table
    create_watchlist_table(
        approaching_df,
        ["severity", "deal_name", "loan_status", "days_to_maturity", "payment_performance", "net_balance", "partner_source"],
        "Approaching Maturity"
    )
else:
    st.success("No at-risk loans approaching maturity")

st.divider()

# =============================================================================
# SECTION 3: PROBLEM STATUS LOANS
# =============================================================================

st.header("Problem Status Loans")
st.markdown("*Loans currently in problem statuses requiring monitoring*")

# Filter: Loans in problem statuses (not terminal)
problem_mask = (
    (df["loan_status"].isin(PROBLEM_STATUSES)) &
    (~df["loan_status"].isin(TERMINAL_STATUSES))
)

problem_df = df[problem_mask].copy()

if not problem_df.empty:
    # Add days past maturity for context
    problem_df["days_past_maturity"] = problem_df["maturity_date"].apply(
        lambda x: get_days_past_maturity(x, today) if pd.notna(x) and x < today else 0
    )

    # Calculate severity based on status and performance
    def problem_severity(row):
        status = row.get("loan_status", "")
        perf = row.get("payment_performance", 1.0) or 1.0

        # High severity statuses
        if status in ["Default", "Non-Performing", "Bankruptcy", "Legal Action"]:
            return "High"
        elif status in ["Severe Delinquency", "In Collections", "NSF / Suspended"]:
            if perf < HIGH_SEVERITY_PERFORMANCE:
                return "High"
            return "Medium"
        else:
            return "Medium" if perf < MEDIUM_SEVERITY_PERFORMANCE else "Low"

    problem_df["severity"] = problem_df.apply(problem_severity, axis=1)

    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Loans", len(problem_df))
    with col2:
        high_severity = (problem_df["severity"] == "High").sum()
        st.metric("High Severity", high_severity)
    with col3:
        total_exposure = problem_df["net_balance"].sum()
        st.metric("Total Exposure", format_currency(total_exposure))
    with col4:
        avg_performance = problem_df["payment_performance"].mean()
        st.metric("Avg Performance", format_percentage(avg_performance))

    # Group by status for quick view
    with st.expander("View by Status", expanded=False):
        status_summary = problem_df.groupby("loan_status").agg({
            "loan_id": "count",
            "net_balance": "sum",
            "payment_performance": "mean"
        }).rename(columns={
            "loan_id": "Count",
            "net_balance": "Total Exposure",
            "payment_performance": "Avg Performance"
        })
        status_summary["Total Exposure"] = status_summary["Total Exposure"].apply(format_currency)
        status_summary["Avg Performance"] = status_summary["Avg Performance"].apply(format_percentage)
        st.dataframe(status_summary, use_container_width=True)

    # Display table
    create_watchlist_table(
        problem_df,
        ["severity", "deal_name", "loan_status", "days_past_maturity", "payment_performance", "net_balance", "partner_source"],
        "Problem Status"
    )
else:
    st.success("No loans in problem status")

st.divider()

# =============================================================================
# SECTION 4: STALLED LOANS
# =============================================================================

st.header("Stalled Loans")
st.markdown("*Loans with no recent payment activity (appears stalled)*")

# Filter: Loans flagged as stalled
stalled_mask = (
    (~df["loan_status"].isin(TERMINAL_STATUSES)) &
    (df.get("is_stalled", False) == True)
)

stalled_df = df[stalled_mask].copy()

if not stalled_df.empty:
    # Add days past maturity
    stalled_df["days_past_maturity"] = stalled_df["maturity_date"].apply(
        lambda x: get_days_past_maturity(x, today) if pd.notna(x) and x < today else 0
    )

    # All stalled loans are at least medium severity
    stalled_df["severity"] = stalled_df["payment_performance"].apply(
        lambda x: "High" if (x or 1.0) < HIGH_SEVERITY_PERFORMANCE else "Medium"
    )

    # Summary metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Stalled", len(stalled_df))
    with col2:
        total_exposure = stalled_df["net_balance"].sum()
        st.metric("Total Exposure", format_currency(total_exposure))
    with col3:
        avg_performance = stalled_df["payment_performance"].mean()
        st.metric("Avg Performance", format_percentage(avg_performance))

    # Display table
    create_watchlist_table(
        stalled_df,
        ["severity", "deal_name", "loan_status", "days_past_maturity", "payment_performance", "net_balance", "partner_source"],
        "Stalled Loans"
    )
else:
    st.success("No stalled loans detected")

st.divider()

# =============================================================================
# SUMMARY SECTION
# =============================================================================

st.header("Watchlist Summary")

# Combine all watchlist items for summary
all_watchlist = pd.concat([
    past_maturity_df.assign(category="Past Maturity") if not past_maturity_df.empty else pd.DataFrame(),
    approaching_df.assign(category="Approaching Maturity") if not approaching_df.empty else pd.DataFrame(),
    problem_df.assign(category="Problem Status") if not problem_df.empty else pd.DataFrame(),
    stalled_df.assign(category="Stalled") if not stalled_df.empty else pd.DataFrame(),
], ignore_index=True)

if not all_watchlist.empty:
    # Remove duplicates (a loan might appear in multiple categories)
    all_watchlist_unique = all_watchlist.drop_duplicates(subset=["loan_id"], keep="first")

    # Summary by severity
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        total_unique = len(all_watchlist_unique)
        st.metric("Total Unique Loans", total_unique)

    with col2:
        high_count = (all_watchlist_unique["severity"] == "High").sum()
        st.metric("High Severity", high_count, delta=None, delta_color="inverse")

    with col3:
        medium_count = (all_watchlist_unique["severity"] == "Medium").sum()
        st.metric("Medium Severity", medium_count)

    with col4:
        total_exposure = all_watchlist_unique["net_balance"].sum()
        st.metric("Total Exposure", format_currency(total_exposure))

    # Category breakdown
    with st.expander("Breakdown by Category", expanded=True):
        category_summary = all_watchlist.groupby("category").agg({
            "loan_id": "count",
            "net_balance": "sum"
        }).rename(columns={
            "loan_id": "Loans",
            "net_balance": "Exposure"
        })
        category_summary["Exposure"] = category_summary["Exposure"].apply(format_currency)

        # Reorder categories
        category_order = ["Past Maturity", "Approaching Maturity", "Problem Status", "Stalled"]
        category_summary = category_summary.reindex([c for c in category_order if c in category_summary.index])

        st.dataframe(category_summary, use_container_width=True)
else:
    st.success("No loans currently require attention on the watchlist")

# Footer
st.markdown("---")
st.caption(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
