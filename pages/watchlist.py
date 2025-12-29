# pages/watchlist.py
"""
Loan Watchlist Dashboard - Proactive loan monitoring and alerts

This page surfaces loans needing attention before they become problems:
- Past Maturity: Active loans past their maturity date
- Approaching Maturity: Loans due soon with low payment performance
- Payment Performance Declining: Loans with deteriorating payment patterns
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Core utilities
from utils.config import setup_page, PRIMARY_COLOR
from utils.data_loader import (
    load_loan_summaries,
    load_deals,
    load_loan_schedules,
    get_last_updated,
)
from utils.loan_tape_data import prepare_loan_data
from utils.loan_tape_analytics import get_payment_behavior_features
from utils.status_constants import STATUS_COLORS

# ---------------------------
# Page Configuration
# ---------------------------
setup_page("CSL Capital | Loan Watchlist")

# ---------------------------
# Constants
# ---------------------------
# Severity thresholds
HIGH_SEVERITY_PERFORMANCE = 0.70  # Below 70% is high severity
MEDIUM_SEVERITY_PERFORMANCE = 0.80  # Below 80% is medium severity
APPROACHING_MATURITY_DAYS = 60  # Days to maturity threshold

# Severity colors
SEVERITY_COLORS = {
    "High": "#dc3545",  # Red
    "Medium": "#fd7e14",  # Orange
    "Low": "#ffc107",  # Yellow
}


# ---------------------------
# Helper Functions
# ---------------------------
def calculate_severity(row: pd.Series) -> str:
    """
    Calculate severity level for a watchlist item.

    High: Past maturity + performance < 70%
    Medium: Past maturity OR approaching with low performance
    Low: Minor concerns
    """
    is_past_maturity = row.get("is_past_maturity", False)
    payment_performance = row.get("payment_performance", 1.0)
    days_to_maturity = row.get("days_to_maturity", 999)

    # Ensure payment_performance is a float
    if pd.isna(payment_performance):
        payment_performance = 0.0

    # High severity: Past maturity AND poor performance
    if is_past_maturity and payment_performance < HIGH_SEVERITY_PERFORMANCE:
        return "High"

    # Medium severity: Past maturity OR (approaching maturity with low performance)
    if is_past_maturity:
        return "Medium"

    if 0 < days_to_maturity <= APPROACHING_MATURITY_DAYS and payment_performance < MEDIUM_SEVERITY_PERFORMANCE:
        return "Medium"

    # Low severity: Minor concerns
    return "Low"


def get_severity_badge(severity: str) -> str:
    """Return HTML badge for severity level."""
    color = SEVERITY_COLORS.get(severity, "#6c757d")
    return f'<span style="background-color: {color}; color: white; padding: 2px 8px; border-radius: 4px; font-size: 12px;">{severity}</span>'


def format_currency(value) -> str:
    """Format value as currency."""
    if pd.isna(value):
        return "$0"
    return f"${value:,.0f}"


def format_percentage(value) -> str:
    """Format value as percentage."""
    if pd.isna(value):
        return "N/A"
    return f"{value:.1%}"


def prepare_watchlist_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare loan data for watchlist analysis.
    Adds calculated fields for maturity analysis.
    """
    if df.empty:
        return df

    result = df.copy()
    today = pd.Timestamp.today().normalize()

    # Ensure maturity_date is datetime
    if "maturity_date" in result.columns:
        result["maturity_date"] = pd.to_datetime(result["maturity_date"], errors="coerce")

    # Calculate days past maturity and days to maturity
    result["days_past_maturity"] = 0
    result["days_to_maturity"] = 999

    if "maturity_date" in result.columns:
        mask_valid_maturity = result["maturity_date"].notna()

        # Days past maturity (positive if past)
        result.loc[mask_valid_maturity, "days_past_maturity"] = result.loc[mask_valid_maturity, "maturity_date"].apply(
            lambda x: max(0, (today - x).days)
        )

        # Days to maturity (positive if future, negative if past)
        result.loc[mask_valid_maturity, "days_to_maturity"] = result.loc[mask_valid_maturity, "maturity_date"].apply(
            lambda x: (x - today).days
        )

    # Ensure is_past_maturity is set correctly
    result["is_past_maturity"] = (
        (result.get("loan_status", "") != "Paid Off") &
        (result["maturity_date"].notna()) &
        (result["maturity_date"] < today)
    )

    return result


def display_watchlist_table(
    df: pd.DataFrame,
    title: str,
    severity_filter: str = None,
    show_severity: bool = True
):
    """
    Display a watchlist table with consistent formatting.
    """
    if df.empty:
        st.info(f"No loans found for: {title}")
        return

    # Calculate severity if not already present
    if "severity" not in df.columns:
        df = df.copy()
        df["severity"] = df.apply(calculate_severity, axis=1)

    # Apply severity filter if specified
    if severity_filter:
        df = df[df["severity"] == severity_filter]
        if df.empty:
            st.info(f"No {severity_filter.lower()} severity loans found.")
            return

    # Sort by net_balance descending (biggest exposure first)
    df = df.sort_values("net_balance", ascending=False)

    # Prepare display columns
    display_df = pd.DataFrame()

    if show_severity:
        display_df["Severity"] = df["severity"]

    display_df["Loan ID"] = df["loan_id"]
    display_df["Deal Name"] = df["deal_name"]
    display_df["Status"] = df["loan_status"]

    if "days_past_maturity" in df.columns:
        display_df["Days Past Maturity"] = df["days_past_maturity"].astype(int)

    if "days_to_maturity" in df.columns and "days_past_maturity" not in df.columns:
        display_df["Days to Maturity"] = df["days_to_maturity"].astype(int)

    display_df["Payment Perf"] = df["payment_performance"].apply(format_percentage)
    display_df["Net Balance"] = df["net_balance"].apply(format_currency)

    if "partner_source" in df.columns:
        display_df["Partner"] = df["partner_source"]

    # Apply conditional formatting based on severity
    def highlight_severity(row):
        severity = row.get("Severity", "Low")
        if severity == "High":
            return ["background-color: rgba(220, 53, 69, 0.2)"] * len(row)
        elif severity == "Medium":
            return ["background-color: rgba(253, 126, 20, 0.2)"] * len(row)
        else:
            return [""] * len(row)

    if show_severity:
        styled_df = display_df.style.apply(highlight_severity, axis=1)
        st.dataframe(styled_df, hide_index=True, use_container_width=True)
    else:
        st.dataframe(display_df, hide_index=True, use_container_width=True)

    # Summary metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Loans", len(df))
    with col2:
        st.metric("Total Exposure", format_currency(df["net_balance"].sum()))
    with col3:
        avg_perf = df["payment_performance"].mean()
        st.metric("Avg Payment Perf", format_percentage(avg_perf))


# ---------------------------
# Main Page
# ---------------------------
def main():
    st.title("Loan Watchlist")
    st.markdown("*Proactive monitoring for loans needing attention*")

    last_updated = get_last_updated()
    st.caption(f"Data last updated: {last_updated}")

    # Load and prepare data
    with st.spinner("Loading loan data..."):
        loans_df = load_loan_summaries()
        deals_df = load_deals()

        if loans_df.empty:
            st.error("Unable to load loan data. Please check your database connection.")
            return

        # Prepare loan data with calculations
        df = prepare_loan_data(loans_df, deals_df)
        df = prepare_watchlist_data(df)

        # Load payment behavior features if available
        schedules_df = load_loan_schedules()
        if not schedules_df.empty:
            payment_behavior = get_payment_behavior_features(schedules_df)
            if not payment_behavior.empty:
                payment_behavior["loan_id"] = (
                    payment_behavior["loan_id"]
                    .astype(str).str.strip().str.replace(r'\.0$', '', regex=True)
                )
                df["loan_id"] = df["loan_id"].astype(str).str.strip().str.replace(r'\.0$', '', regex=True)
                df = df.merge(
                    payment_behavior[["loan_id", "pct_on_time", "pct_late", "pct_missed", "consecutive_missed"]],
                    on="loan_id",
                    how="left"
                )

    # Filter to only active loans (non-paid off)
    active_loans = df[df["loan_status"] != "Paid Off"].copy()

    if active_loans.empty:
        st.info("No active loans found in the portfolio.")
        return

    # Calculate severity for all loans
    active_loans["severity"] = active_loans.apply(calculate_severity, axis=1)

    # ---------------------------
    # Summary Metrics
    # ---------------------------
    st.header("Watchlist Summary")

    # Count by severity
    severity_counts = active_loans["severity"].value_counts()
    high_count = severity_counts.get("High", 0)
    medium_count = severity_counts.get("Medium", 0)
    low_count = severity_counts.get("Low", 0)

    # Calculate exposure by severity
    high_exposure = active_loans[active_loans["severity"] == "High"]["net_balance"].sum()
    medium_exposure = active_loans[active_loans["severity"] == "Medium"]["net_balance"].sum()

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "High Severity",
            f"{high_count} loans",
            delta=format_currency(high_exposure),
            delta_color="inverse"
        )

    with col2:
        st.metric(
            "Medium Severity",
            f"{medium_count} loans",
            delta=format_currency(medium_exposure),
            delta_color="inverse"
        )

    with col3:
        st.metric(
            "Low Severity",
            f"{low_count} loans"
        )

    with col4:
        total_watchlist = high_count + medium_count
        st.metric(
            "Total Watchlist",
            f"{total_watchlist} loans",
            delta=f"{total_watchlist / len(active_loans) * 100:.1f}% of portfolio"
        )

    st.markdown("---")

    # ---------------------------
    # Section 1: Past Maturity
    # ---------------------------
    st.header("Past Maturity Loans")
    st.markdown("""
    *Loans where status = "Active" but maturity_date < today.
    Sorted by net balance (biggest exposure first).*
    """)

    past_maturity = active_loans[
        (active_loans["is_past_maturity"] == True) &
        (active_loans["loan_status"].isin(["Active", "Active - Frequently Late", "Minor Delinquency", "Moderate Delinquency", "Severe Delinquency"]))
    ].copy()

    if not past_maturity.empty:
        # Add severity filter
        severity_options = ["All", "High", "Medium", "Low"]
        selected_severity = st.selectbox(
            "Filter by Severity",
            severity_options,
            key="past_maturity_severity"
        )

        filter_severity = None if selected_severity == "All" else selected_severity
        display_watchlist_table(past_maturity, "Past Maturity Loans", severity_filter=filter_severity)

        # Download option
        csv = past_maturity.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="Download Past Maturity Report",
            data=csv,
            file_name=f"past_maturity_loans_{pd.Timestamp.today().strftime('%Y%m%d')}.csv",
            mime="text/csv",
        )
    else:
        st.success("No past maturity loans found!")

    st.markdown("---")

    # ---------------------------
    # Section 2: Approaching Maturity
    # ---------------------------
    st.header("Approaching Maturity - At Risk")
    st.markdown(f"""
    *Loans due within {APPROACHING_MATURITY_DAYS} days with payment performance < {MEDIUM_SEVERITY_PERFORMANCE:.0%}.
    These are at risk of not paying off on time.*
    """)

    today = pd.Timestamp.today().normalize()
    approaching_maturity = active_loans[
        (active_loans["is_past_maturity"] == False) &
        (active_loans["days_to_maturity"] <= APPROACHING_MATURITY_DAYS) &
        (active_loans["days_to_maturity"] > 0) &
        (active_loans["payment_performance"] < MEDIUM_SEVERITY_PERFORMANCE)
    ].copy()

    if not approaching_maturity.empty:
        # Sort by days to maturity (most urgent first)
        approaching_maturity = approaching_maturity.sort_values("days_to_maturity")

        # Prepare display
        display_df = pd.DataFrame()
        display_df["Loan ID"] = approaching_maturity["loan_id"]
        display_df["Deal Name"] = approaching_maturity["deal_name"]
        display_df["Status"] = approaching_maturity["loan_status"]
        display_df["Days to Maturity"] = approaching_maturity["days_to_maturity"].astype(int)
        display_df["Maturity Date"] = approaching_maturity["maturity_date"].dt.strftime("%Y-%m-%d")
        display_df["Payment Perf"] = approaching_maturity["payment_performance"].apply(format_percentage)
        display_df["Net Balance"] = approaching_maturity["net_balance"].apply(format_currency)
        display_df["Partner"] = approaching_maturity.get("partner_source", "")

        st.dataframe(display_df, hide_index=True, use_container_width=True)

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("At-Risk Loans", len(approaching_maturity))
        with col2:
            st.metric("At-Risk Exposure", format_currency(approaching_maturity["net_balance"].sum()))
        with col3:
            st.metric("Avg Days to Maturity", f"{approaching_maturity['days_to_maturity'].mean():.0f}")

        # Download option
        csv = approaching_maturity.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="Download Approaching Maturity Report",
            data=csv,
            file_name=f"approaching_maturity_loans_{pd.Timestamp.today().strftime('%Y%m%d')}.csv",
            mime="text/csv",
        )
    else:
        st.success("No at-risk approaching maturity loans found!")

    st.markdown("---")

    # ---------------------------
    # Section 3: Payment Performance Declining
    # ---------------------------
    st.header("Payment Performance Concerns")
    st.markdown("""
    *Loans with declining payment patterns or missed payments.*
    """)

    # Check if we have payment behavior data
    has_payment_behavior = "consecutive_missed" in active_loans.columns

    if has_payment_behavior:
        # Loans with consecutive missed payments
        consecutive_missed = active_loans[
            (active_loans["consecutive_missed"] >= 2) &
            (~active_loans["is_past_maturity"])  # Not already in past maturity section
        ].copy()

        if not consecutive_missed.empty:
            st.subheader("Consecutive Missed Payments")
            st.markdown("*Loans with 2+ consecutive missed payments*")

            consecutive_missed = consecutive_missed.sort_values("consecutive_missed", ascending=False)

            display_df = pd.DataFrame()
            display_df["Loan ID"] = consecutive_missed["loan_id"]
            display_df["Deal Name"] = consecutive_missed["deal_name"]
            display_df["Status"] = consecutive_missed["loan_status"]
            display_df["Consecutive Missed"] = consecutive_missed["consecutive_missed"].astype(int)
            display_df["% On-Time"] = consecutive_missed["pct_on_time"].apply(format_percentage)
            display_df["% Missed"] = consecutive_missed["pct_missed"].apply(format_percentage)
            display_df["Payment Perf"] = consecutive_missed["payment_performance"].apply(format_percentage)
            display_df["Net Balance"] = consecutive_missed["net_balance"].apply(format_currency)

            st.dataframe(display_df, hide_index=True, use_container_width=True)

            st.metric("Loans with Missed Payments", len(consecutive_missed))
        else:
            st.success("No loans with consecutive missed payments!")

    # Low performing loans (not already in other sections)
    low_performing = active_loans[
        (active_loans["payment_performance"] < HIGH_SEVERITY_PERFORMANCE) &
        (~active_loans["is_past_maturity"]) &
        (active_loans["days_to_maturity"] > APPROACHING_MATURITY_DAYS)
    ].copy()

    if not low_performing.empty:
        st.subheader("Low Payment Performance")
        st.markdown(f"*Active loans with payment performance below {HIGH_SEVERITY_PERFORMANCE:.0%}*")

        low_performing = low_performing.sort_values("payment_performance")

        display_df = pd.DataFrame()
        display_df["Loan ID"] = low_performing["loan_id"]
        display_df["Deal Name"] = low_performing["deal_name"]
        display_df["Status"] = low_performing["loan_status"]
        display_df["Payment Perf"] = low_performing["payment_performance"].apply(format_percentage)
        display_df["Days to Maturity"] = low_performing["days_to_maturity"].astype(int)
        display_df["Net Balance"] = low_performing["net_balance"].apply(format_currency)
        display_df["Partner"] = low_performing.get("partner_source", "")

        st.dataframe(display_df, hide_index=True, use_container_width=True)

        col1, col2 = st.columns(2)
        with col1:
            st.metric("Low Performing Loans", len(low_performing))
        with col2:
            st.metric("Exposure", format_currency(low_performing["net_balance"].sum()))
    elif not has_payment_behavior:
        st.info("Payment behavior data not available. Load loan schedules for detailed payment analysis.")

    st.markdown("---")

    # ---------------------------
    # Section 4: Action Items
    # ---------------------------
    st.header("Quick Actions")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Export Watchlist")

        # Combine all watchlist items
        all_watchlist = active_loans[
            (active_loans["severity"].isin(["High", "Medium"])) |
            (active_loans["is_past_maturity"] == True)
        ].copy()

        if not all_watchlist.empty:
            csv = all_watchlist.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="Download Full Watchlist",
                data=csv,
                file_name=f"loan_watchlist_{pd.Timestamp.today().strftime('%Y%m%d')}.csv",
                mime="text/csv",
                key="full_watchlist_download"
            )

    with col2:
        st.subheader("Severity Guide")
        st.markdown(f"""
        - **High**: Past maturity + performance < {HIGH_SEVERITY_PERFORMANCE:.0%}
        - **Medium**: Past maturity OR approaching with low performance
        - **Low**: Minor concerns, monitor regularly
        """)

    # ---------------------------
    # Loan Detail Lookup
    # ---------------------------
    st.markdown("---")
    st.header("Loan Detail Lookup")

    watchlist_loan_ids = active_loans[
        (active_loans["severity"].isin(["High", "Medium"])) |
        (active_loans["is_past_maturity"] == True)
    ]["loan_id"].tolist()

    if watchlist_loan_ids:
        selected_loan = st.selectbox(
            "Select a loan from the watchlist to view details",
            options=[""] + watchlist_loan_ids,
            key="watchlist_loan_lookup"
        )

        if selected_loan:
            loan_data = active_loans[active_loans["loan_id"] == selected_loan].iloc[0]

            col1, col2 = st.columns(2)

            with col1:
                st.subheader("Loan Information")
                st.write(f"**Loan ID:** {loan_data['loan_id']}")
                st.write(f"**Deal Name:** {loan_data.get('deal_name', 'N/A')}")
                st.write(f"**Status:** {loan_data['loan_status']}")
                st.write(f"**Partner:** {loan_data.get('partner_source', 'N/A')}")

                if pd.notna(loan_data.get("maturity_date")):
                    st.write(f"**Maturity Date:** {loan_data['maturity_date'].strftime('%Y-%m-%d')}")

                if loan_data.get("is_past_maturity"):
                    st.warning(f"Days Past Maturity: {int(loan_data['days_past_maturity'])}")
                elif loan_data.get("days_to_maturity", 999) < 999:
                    st.info(f"Days to Maturity: {int(loan_data['days_to_maturity'])}")

            with col2:
                st.subheader("Financial Summary")
                st.write(f"**Net Balance:** {format_currency(loan_data['net_balance'])}")
                st.write(f"**Payment Performance:** {format_percentage(loan_data['payment_performance'])}")
                st.write(f"**Capital Deployed:** {format_currency(loan_data.get('csl_participation_amount', 0))}")
                st.write(f"**Total Paid:** {format_currency(loan_data.get('total_paid', 0))}")

                severity = loan_data.get("severity", "Unknown")
                severity_color = SEVERITY_COLORS.get(severity, "#6c757d")
                st.markdown(f"**Severity:** <span style='color: {severity_color}; font-weight: bold;'>{severity}</span>", unsafe_allow_html=True)

                # Show payment behavior if available
                if has_payment_behavior and pd.notna(loan_data.get("pct_on_time")):
                    st.write(f"**% On-Time:** {format_percentage(loan_data['pct_on_time'])}")
                    st.write(f"**Consecutive Missed:** {int(loan_data.get('consecutive_missed', 0))}")


if __name__ == "__main__":
    main()
