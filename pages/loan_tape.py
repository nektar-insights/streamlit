# pages/loan_tape.py
"""
Loan Tape Dashboard - Portfolio analysis and risk management

This page provides comprehensive loan portfolio analytics including:
- Capital flow tracking
- Performance analysis
- Risk scoring
- IRR calculations
- ML-based predictions
"""

import streamlit as st
import pandas as pd
import numpy as np
import altair as alt

# Core utilities
from utils.config import (
    setup_page,
    PRIMARY_COLOR,
    COLOR_PALETTE,
    PLATFORM_FEE_RATE,
)
from utils.data_loader import (
    load_loan_summaries,
    load_deals,
    load_naics_sector_risk,
    load_loan_schedules,
    get_last_updated,
)

# Loan tape specific utilities
from utils.loan_tape_data import (
    prepare_loan_data,
    calculate_irr,
    calculate_risk_scores,
    calculate_expected_payment_to_date,
    format_dataframe_for_display,
)
from utils.loan_tape_analytics import (
    PROBLEM_STATUSES,
    get_display_name,
)
from utils.status_constants import (
    STATUS_COLORS,
    STATUS_GROUPS,
    STATUS_GROUP_COLORS,
    ALL_VALID_STATUSES,
)
from utils.display_components import (
    create_date_range_filter,
    create_partner_source_filter,
    create_status_filter,
)
from utils.manual_status_editor import render_manual_status_editor, render_status_badge

# ---------------------------
# Page Configuration & Styles
# ---------------------------
setup_page("CSL Capital | Loan Tape")

# -------------
# Constants
# -------------
PLATFORM_FEE = PLATFORM_FEE_RATE

# Status risk multipliers for all valid statuses
# Higher multipliers indicate higher risk
STATUS_RISK_MULTIPLIERS = {
    # Active statuses
    "Active": 1.0,
    "Active - Frequently Late": 1.3,
    # Delinquency statuses (escalating severity)
    "Minor Delinquency": 1.5,
    "Moderate Delinquency": 2.0,
    "Severe Delinquency": 3.0,
    "Past Delinquency": 1.2,
    # Problem statuses
    "Default": 4.0,
    "NSF / Suspended": 3.5,
    "Non-Performing": 4.5,
    "In Collections": 4.0,
    "Legal Action": 4.5,
    # Terminal statuses
    "Paid Off": 0.0,
    "Charged Off": 5.0,
    "Bankruptcy": 5.0,
}


# -------------------
# Page-Specific Visualizations
# -------------------

def plot_status_distribution(df: pd.DataFrame):
    """Plot loan status distribution with all valid statuses colored"""
    status_counts = df["loan_status"].value_counts().reset_index()
    status_counts.columns = ["status", "count"]

    # Get colors for statuses in the data
    status_list = status_counts["status"].tolist()
    status_colors = [STATUS_COLORS.get(s, "#7f7f7f") for s in status_list]

    chart = alt.Chart(status_counts).mark_bar().encode(
        x=alt.X("status:N", title="Loan Status", sort="-y"),
        y=alt.Y("count:Q", title="Number of Loans"),
        color=alt.Color(
            "status:N",
            scale=alt.Scale(
                domain=status_list,
                range=status_colors
            ),
            legend=None
        ),
        tooltip=[
            alt.Tooltip("status:N", title="Status"),
            alt.Tooltip("count:Q", title="Count"),
        ]
    ).properties(width=600, height=350)

    st.altair_chart(chart, width='stretch')


def plot_roi_distribution(df: pd.DataFrame):
    """Plot ROI distribution histogram"""
    if "current_roi" not in df.columns:
        st.info("ROI data not available")
        return

    roi_data = df[df["current_roi"].notna()].copy()

    if roi_data.empty:
        st.info("No ROI data available for visualization")
        return

    chart = alt.Chart(roi_data).mark_bar().encode(
        alt.X("current_roi:Q", bin=alt.Bin(maxbins=30), title="Current ROI"),
        y=alt.Y("count():Q", title="Number of Loans"),
        color=alt.value(PRIMARY_COLOR),
        tooltip=[
            alt.Tooltip("current_roi:Q", title="ROI", format=".1%", bin=True),
            alt.Tooltip("count():Q", title="Count"),
        ]
    ).properties(width=600, height=350, title="Distribution of Current ROI")

    st.altair_chart(chart, width='stretch')


def plot_fico_histogram(df: pd.DataFrame):
    """Plot FICO score distribution histogram"""
    if "fico" not in df.columns:
        st.info("FICO data not available")
        return

    fico_data = df[df["fico"].notna() & (df["fico"] > 0)].copy()

    if fico_data.empty:
        st.info("No FICO data available for visualization")
        return

    chart = alt.Chart(fico_data).mark_bar().encode(
        alt.X("fico:Q", bin=alt.Bin(maxbins=25), title="FICO Score"),
        y=alt.Y("count():Q", title="Number of Loans"),
        color=alt.value(PRIMARY_COLOR),
        tooltip=[
            alt.Tooltip("fico:Q", title="FICO", bin=True),
            alt.Tooltip("count():Q", title="Count"),
        ]
    ).properties(height=300, title="FICO Score Distribution")

    st.altair_chart(chart, width='stretch')


def plot_factor_histogram(df: pd.DataFrame):
    """Plot Factor Rate distribution histogram"""
    if "factor_rate" not in df.columns:
        st.info("Factor Rate data not available")
        return

    factor_data = df[df["factor_rate"].notna() & (df["factor_rate"] > 0)].copy()

    if factor_data.empty:
        st.info("No Factor Rate data available for visualization")
        return

    chart = alt.Chart(factor_data).mark_bar().encode(
        alt.X("factor_rate:Q", bin=alt.Bin(maxbins=25), title="Factor Rate"),
        y=alt.Y("count():Q", title="Number of Loans"),
        color=alt.value(PRIMARY_COLOR),
        tooltip=[
            alt.Tooltip("factor_rate:Q", title="Factor", format=".3f", bin=True),
            alt.Tooltip("count():Q", title="Count"),
        ]
    ).properties(height=300, title="Factor Rate Distribution")

    st.altair_chart(chart, width='stretch')


def plot_term_histogram(df: pd.DataFrame):
    """Plot Loan Term distribution histogram"""
    if "loan_term" not in df.columns:
        st.info("Loan Term data not available")
        return

    term_data = df[df["loan_term"].notna() & (df["loan_term"] > 0)].copy()

    if term_data.empty:
        st.info("No Loan Term data available for visualization")
        return

    chart = alt.Chart(term_data).mark_bar().encode(
        alt.X("loan_term:Q", bin=alt.Bin(maxbins=20), title="Term (months)"),
        y=alt.Y("count():Q", title="Number of Loans"),
        color=alt.value(PRIMARY_COLOR),
        tooltip=[
            alt.Tooltip("loan_term:Q", title="Term (months)", format=".0f", bin=True),
            alt.Tooltip("count():Q", title="Count"),
        ]
    ).properties(height=300, title="Loan Term Distribution")

    st.altair_chart(chart, width='stretch')


def plot_months_left_histogram(df: pd.DataFrame):
    """Plot Remaining Months distribution histogram"""
    if "remaining_maturity_months" not in df.columns:
        st.info("Months Left data not available")
        return

    # Filter to active loans with positive remaining months
    active_df = df[(df["loan_status"] != "Paid Off") & (df["remaining_maturity_months"] > 0)].copy()

    if active_df.empty:
        st.info("No active loans with remaining months for visualization")
        return

    chart = alt.Chart(active_df).mark_bar().encode(
        alt.X("remaining_maturity_months:Q", bin=alt.Bin(maxbins=20), title="Months Left"),
        y=alt.Y("count():Q", title="Number of Loans"),
        color=alt.value(PRIMARY_COLOR),
        tooltip=[
            alt.Tooltip("remaining_maturity_months:Q", title="Months Left", format=".0f", bin=True),
            alt.Tooltip("count():Q", title="Count"),
        ]
    ).properties(height=300, title="Remaining Months Distribution (Active Loans)")

    st.altair_chart(chart, width='stretch')


def plot_csl_participation_histogram(df: pd.DataFrame):
    """Plot CSL Participation Amount distribution histogram"""
    if "csl_participation_amount" not in df.columns:
        st.info("CSL Participation data not available")
        return

    csl_data = df[df["csl_participation_amount"].notna() & (df["csl_participation_amount"] > 0)].copy()

    if csl_data.empty:
        st.info("No CSL Participation data available for visualization")
        return

    chart = alt.Chart(csl_data).mark_bar().encode(
        alt.X("csl_participation_amount:Q", bin=alt.Bin(maxbins=25), title="CSL Participation ($)"),
        y=alt.Y("count():Q", title="Number of Loans"),
        color=alt.value(PRIMARY_COLOR),
        tooltip=[
            alt.Tooltip("csl_participation_amount:Q", title="Participation", format="$,.0f", bin=True),
            alt.Tooltip("count():Q", title="Count"),
        ]
    ).properties(height=300, title="CSL Participation Distribution")

    st.altair_chart(chart, width='stretch')


def plot_tib_histogram(df: pd.DataFrame):
    """Plot Time in Business distribution histogram (capped at 50 years)"""
    if "tib" not in df.columns:
        st.info("TIB data not available")
        return

    tib_data = df[df["tib"].notna() & (df["tib"] > 0)].copy()

    if tib_data.empty:
        st.info("No TIB data available for visualization")
        return

    # Cap TIB at 50 years
    tib_data["tib_capped"] = tib_data["tib"].clip(upper=50)

    chart = alt.Chart(tib_data).mark_bar().encode(
        alt.X("tib_capped:Q", bin=alt.Bin(maxbins=25), title="Time in Business (Years, capped at 50)"),
        y=alt.Y("count():Q", title="Number of Loans"),
        color=alt.value(PRIMARY_COLOR),
        tooltip=[
            alt.Tooltip("tib_capped:Q", title="TIB (Years)", format=".1f", bin=True),
            alt.Tooltip("count():Q", title="Count"),
        ]
    ).properties(height=300, title="Time in Business Distribution")

    st.altair_chart(chart, width='stretch')


def plot_capital_flow(df: pd.DataFrame):
    """Plot capital deployment vs returns over time"""
    st.subheader("Capital Flow: Deployment vs. Returns")

    schedules = load_loan_schedules()

    d = df.copy()
    d["funding_date"] = pd.to_datetime(d["funding_date"], errors="coerce").dt.tz_localize(None)

    # Calculate total deployed from the dataframe
    total_deployed = d["csl_participation_amount"].sum()

    # Calculate total returned from loan schedules actual payments to ensure consistency
    if not schedules.empty and "actual_payment" in schedules.columns:
        schedules_copy = schedules.copy()
        schedules_copy["actual_payment"] = pd.to_numeric(schedules_copy["actual_payment"], errors="coerce")
        total_returned = schedules_copy[schedules_copy["actual_payment"] > 0]["actual_payment"].sum()
    else:
        # Fallback to total_paid from dataframe if schedules not available
        total_returned = d["total_paid"].sum()

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Total Capital Deployed", f"${total_deployed:,.0f}")
    with col2:
        st.metric("Total Capital Returned", f"${total_returned:,.0f}")

    deploy_data = d[["funding_date", "csl_participation_amount"]].dropna()
    deploy_timeline = deploy_data.groupby("funding_date")["csl_participation_amount"].sum().sort_index().cumsum()

    if not schedules.empty and "payment_date" in schedules.columns:
        schedules["payment_date"] = pd.to_datetime(schedules["payment_date"], errors="coerce").dt.tz_localize(None)
        schedules["actual_payment"] = pd.to_numeric(schedules["actual_payment"], errors="coerce")
        payment_data = schedules[
            schedules["actual_payment"].notna() &
            (schedules["actual_payment"] > 0) &
            schedules["payment_date"].notna()
        ]
        return_timeline = payment_data.groupby("payment_date")["actual_payment"].sum().sort_index().cumsum()
    else:
        return_timeline = pd.Series(dtype=float)

    if not deploy_timeline.empty:
        min_date = deploy_timeline.index.min()
        max_date = pd.Timestamp.today().normalize()
        date_range = pd.date_range(start=min_date, end=max_date, freq="D")

        unified = pd.DataFrame(index=date_range)
        unified["capital_deployed"] = deploy_timeline.reindex(date_range).ffill().fillna(0)
        unified["capital_returned"] = return_timeline.reindex(date_range).ffill().fillna(0)

        # Aggregate by month to avoid duplicate dates and improve readability
        unified['month_date'] = unified.index.to_period('M').to_timestamp()
        monthly = unified.groupby('month_date').last().reset_index()

        st.caption(
            f"Chart shows: Deployed ${monthly['capital_deployed'].iloc[-1]:,.0f} | "
            f"Returned ${monthly['capital_returned'].iloc[-1]:,.0f}"
        )

        plot_df = pd.concat([
            pd.DataFrame({"month_date": monthly['month_date'], "amount": monthly["capital_deployed"].values, "series": "Capital Deployed"}),
            pd.DataFrame({"month_date": monthly['month_date'], "amount": monthly["capital_returned"].values, "series": "Capital Returned"})
        ], ignore_index=True)

        chart = alt.Chart(plot_df).mark_line(point=True).encode(
            x=alt.X("yearmonth(month_date):T", title="Month",
                   axis=alt.Axis(format="%b %Y", labelAngle=-45), sort="ascending"),
            y=alt.Y("amount:Q", title="Cumulative Amount ($)", axis=alt.Axis(format="$,.0f")),
            color=alt.Color(
                "series:N",
                scale=alt.Scale(domain=["Capital Deployed", "Capital Returned"], range=["#ff7f0e", "#2ca02c"]),
                legend=alt.Legend(title="Capital Flow")
            ),
            tooltip=[
                alt.Tooltip("yearmonth(month_date):T", title="Month", format="%B %Y"),
                alt.Tooltip("amount:Q", title="Amount", format="$,.0f"),
                alt.Tooltip("series:N", title="Type"),
            ],
        ).properties(width=800, height=400, title="Capital Deployed vs. Capital Returned Over Time")

        st.altair_chart(chart, width='stretch')
    else:
        st.info("Insufficient data to display capital flow chart.")


def plot_investment_net_position(df: pd.DataFrame):
    """Plot net investment position over time"""
    st.subheader("Net Investment Position Over Time", help="Shows capital at work: cumulative deployed minus cumulative returned")

    schedules = load_loan_schedules()

    d = df.copy()
    d["funding_date"] = pd.to_datetime(d["funding_date"], errors="coerce").dt.tz_localize(None)

    deploy_data = d[["funding_date", "csl_participation_amount"]].dropna()
    deploy_timeline = deploy_data.groupby("funding_date")["csl_participation_amount"].sum().sort_index().cumsum()

    if not schedules.empty and "payment_date" in schedules.columns:
        schedules["payment_date"] = pd.to_datetime(schedules["payment_date"], errors="coerce").dt.tz_localize(None)
        schedules["actual_payment"] = pd.to_numeric(schedules["actual_payment"], errors="coerce")
        payment_data = schedules[
            schedules["actual_payment"].notna() &
            (schedules["actual_payment"] > 0) &
            schedules["payment_date"].notna()
        ]
        return_timeline = payment_data.groupby("payment_date")["actual_payment"].sum().sort_index().cumsum()
    else:
        return_timeline = pd.Series(dtype=float)

    if not deploy_timeline.empty:
        min_date = deploy_timeline.index.min()
        max_date = pd.Timestamp.today().normalize()
        date_range = pd.date_range(start=min_date, end=max_date, freq="D")

        unified = pd.DataFrame(index=date_range)
        unified["cum_deployed"] = deploy_timeline.reindex(date_range).ffill().fillna(0)
        unified["cum_returned"] = return_timeline.reindex(date_range).ffill().fillna(0)
        unified["net_position"] = unified["cum_deployed"] - unified["cum_returned"]

        # Aggregate by month to avoid duplicate dates and improve readability
        unified['month_date'] = unified.index.to_period('M').to_timestamp()
        monthly = unified.groupby('month_date').last().reset_index()

        plot_df = pd.concat([
            pd.DataFrame({"month_date": monthly['month_date'], "amount": monthly["cum_deployed"].values, "Type": "Cumulative Deployed"}),
            pd.DataFrame({"month_date": monthly['month_date'], "amount": monthly["cum_returned"].values, "Type": "Cumulative Returned"}),
            pd.DataFrame({"month_date": monthly['month_date'], "amount": monthly["net_position"].values, "Type": "Net Position"}),
        ], ignore_index=True)

        chart = alt.Chart(plot_df).mark_line(point=True).encode(
            x=alt.X("yearmonth(month_date):T", title="Month",
                   axis=alt.Axis(format="%b %Y", labelAngle=-45), sort="ascending"),
            y=alt.Y("amount:Q", title="Amount ($)", axis=alt.Axis(format="$,.0f")),
            color=alt.Color(
                "Type:N",
                scale=alt.Scale(
                    domain=["Cumulative Deployed", "Cumulative Returned", "Net Position"],
                    range=["#ff7f0e", "#2ca02c", "#1f77b4"]
                ),
            ),
            tooltip=[
                alt.Tooltip("yearmonth(month_date):T", title="Month", format="%B %Y"),
                alt.Tooltip("amount:Q", title="Amount", format="$,.0f"),
                alt.Tooltip("Type:N", title="Metric"),
            ],
        ).properties(width=800, height=500, title="Portfolio Net Position Over Time")

        zero_line = alt.Chart(pd.DataFrame({"y": [0]})).mark_rule(strokeDash=[2, 2], color="gray", strokeWidth=1).encode(y="y:Q")

        st.altair_chart(chart + zero_line, width='stretch')
        st.caption("Net Position: Capital still deployed (positive) or profit after recovery (negative).")
    else:
        st.info("Insufficient data for net position analysis.")


def plot_payment_performance_by_cohort(df: pd.DataFrame):
    """Plot payment performance by funding cohort"""
    active_df = df[df.get("loan_status", "") != "Paid Off"].copy()
    if active_df.empty:
        st.info("No active loans to analyze.")
        return

    active_df["expected_paid_to_date"] = active_df.apply(calculate_expected_payment_to_date, axis=1)
    active_df["actual_paid"] = active_df["total_paid"]

    active_df["performance_pct_diff"] = active_df.apply(
        lambda x: ((x["actual_paid"] / x["expected_paid_to_date"]) - 1) if x["expected_paid_to_date"] > 0 else 0,
        axis=1
    )
    active_df["cohort"] = pd.to_datetime(active_df["funding_date"]).dt.to_period("Q").astype(str)

    cohort_perf = active_df.groupby("cohort").agg(
        expected_payment=("expected_paid_to_date", "sum"),
        actual_payment=("actual_paid", "sum"),
        loan_count=("loan_id", "count"),
    ).reset_index()

    cohort_perf["performance_pct_diff"] = (cohort_perf["actual_payment"] / cohort_perf["expected_payment"]) - 1
    cohort_perf["perf_label"] = cohort_perf["performance_pct_diff"].apply(lambda x: f"{x:+.1%}")
    cohort_perf = cohort_perf.sort_values("cohort")

    def classify(p):
        if p >= -0.05:
            return "On/Above Target"
        elif p >= -0.15:
            return "Slightly Below"
        else:
            return "Significantly Below"

    cohort_perf["performance_category"] = cohort_perf["performance_pct_diff"].apply(classify)

    bars = alt.Chart(cohort_perf).mark_bar().encode(
        x=alt.X("cohort:N", title="Funding Quarter", sort=None),
        y=alt.Y("performance_pct_diff:Q", title="Performance Difference from Expected", axis=alt.Axis(format=".0%")),
        color=alt.Color(
            "performance_category:N",
            scale=alt.Scale(
                domain=["On/Above Target", "Slightly Below", "Significantly Below"],
                range=["#2ca02c", "#ffbb78", "#d62728"]
            ),
            legend=alt.Legend(title="Performance"),
        ),
        tooltip=[
            alt.Tooltip("cohort:N", title="Cohort"),
            alt.Tooltip("expected_payment:Q", title="Expected Payment", format="$,.0f"),
            alt.Tooltip("actual_payment:Q", title="Actual Payment", format="$,.0f"),
            alt.Tooltip("performance_pct_diff:Q", title="Performance Difference", format="+.1%"),
            alt.Tooltip("loan_count:Q", title="Number of Loans"),
        ],
    ).properties(width=700, height=400, title="Payment Performance by Cohort")

    text = alt.Chart(cohort_perf).mark_text(align="center", baseline="bottom", dy=-5, fontSize=11, fontWeight="bold").encode(
        x=alt.X("cohort:N", sort=None),
        y=alt.Y("performance_pct_diff:Q"),
        text="perf_label:N",
    )

    ref_line = alt.Chart(pd.DataFrame({"y": [0]})).mark_rule(strokeDash=[4, 4], color="gray", strokeWidth=2).encode(y="y:Q")

    target_zone = alt.Chart(pd.DataFrame({"y": [-0.05], "y2": [0.05]})).mark_rect(opacity=0.2, color="green").encode(
        y="y:Q", y2="y2:Q"
    )

    st.altair_chart(target_zone + bars + text + ref_line, width='stretch')
    st.caption("On-Target Zone: -5% to +5%. Positive = ahead of schedule, negative = behind schedule.")


def plot_industry_performance_analysis(df: pd.DataFrame):
    """Plot performance metrics by industry (grouped by 2-digit NAICS sector code)"""
    st.header("Industry Performance Analysis")

    # Check if we have sector_code and industry_name columns
    if "sector_code" not in df.columns or df["sector_code"].isna().all():
        st.warning("Industry sector data not available.")
        return

    # Group by sector_code and get the industry_name
    industry_metrics = df.groupby("sector_code").agg(
        industry_name=("industry_name", "first"),  # Get the industry name
        deal_count=("loan_id", "count"),
        capital_deployed=("csl_participation_amount", "sum"),
        outstanding_balance=("net_balance", "sum"),
        avg_payment_performance=("payment_performance", "mean"),
        total_paid=("total_paid", "sum"),
        total_invested=("total_invested", "sum"),
    ).reset_index()

    industry_metrics = industry_metrics[industry_metrics["deal_count"] >= 3]  # Filter for significance
    industry_metrics["actual_return_rate"] = industry_metrics["total_paid"] / industry_metrics["total_invested"]

    # Calculate % of total outstanding
    total_outstanding = industry_metrics["outstanding_balance"].sum()
    industry_metrics["pct_of_total_outstanding"] = industry_metrics["outstanding_balance"] / total_outstanding if total_outstanding > 0 else 0

    industry_metrics = industry_metrics.sort_values("capital_deployed", ascending=False).head(15)

    # Create display label with sector code and name
    industry_metrics["display_label"] = industry_metrics["sector_code"] + " - " + industry_metrics["industry_name"]

    col1, col2 = st.columns(2)

    with col1:
        perf_chart = alt.Chart(industry_metrics).mark_bar().encode(
            x=alt.X("display_label:N", title="Industry (NAICS 2-Digit)", sort="-y"),
            y=alt.Y("avg_payment_performance:Q", title="Avg Payment Performance", axis=alt.Axis(format=".0%")),
            color=alt.Color(
                "avg_payment_performance:Q",
                scale=alt.Scale(domain=[0.5, 0.8, 1.0], range=["#d62728", "#ffbb78", "#2ca02c"]),
                legend=None,
            ),
            tooltip=[
                alt.Tooltip("display_label:N", title="Industry"),
                alt.Tooltip("avg_payment_performance:Q", title="Avg Payment Performance", format=".1%"),
                alt.Tooltip("deal_count:Q", title="Loan Count"),
            ],
        ).properties(width=350, height=400, title="Payment Performance by Industry (Top 15)")
        st.altair_chart(perf_chart, width='stretch')

    with col2:
        return_chart = alt.Chart(industry_metrics).mark_bar().encode(
            x=alt.X("display_label:N", title="Industry (NAICS 2-Digit)", sort="-y"),
            y=alt.Y("actual_return_rate:Q", title="Actual Return Rate", axis=alt.Axis(format=".0%")),
            color=alt.Color(
                "actual_return_rate:Q",
                scale=alt.Scale(domain=[0.5, 1.0, 1.3], range=["#d62728", "#ffbb78", "#2ca02c"]),
                legend=None,
            ),
            tooltip=[
                alt.Tooltip("display_label:N", title="Industry"),
                alt.Tooltip("actual_return_rate:Q", title="Return Rate", format=".1%"),
                alt.Tooltip("deal_count:Q", title="Loan Count"),
            ],
        ).properties(width=350, height=400, title="Return Rate by Industry (Top 15)")
        st.altair_chart(return_chart, width='stretch')

    st.subheader("Industry Performance Summary")
    display_df = industry_metrics.copy()
    display_df["outstanding_balance"] = display_df["outstanding_balance"].map(lambda x: f"${x:,.0f}")
    display_df["pct_of_total_outstanding"] = display_df["pct_of_total_outstanding"].map(lambda x: f"{x:.1%}")
    display_df["avg_payment_performance"] = display_df["avg_payment_performance"].map(lambda x: f"{x:.1%}")
    display_df["actual_return_rate"] = display_df["actual_return_rate"].map(lambda x: f"{x:.1%}")
    display_df = display_df[["display_label", "deal_count", "outstanding_balance", "pct_of_total_outstanding", "avg_payment_performance", "actual_return_rate"]]
    display_df.columns = ["Industry (NAICS 2-Digit)", "Loan Count", "Outstanding Balance", "% of Total Outstanding", "Avg Payment Performance", "Actual Return Rate"]
    st.dataframe(display_df, width='stretch', hide_index=True)

    # --- Capital Exposure Analysis Section ---
    st.subheader("Capital Exposure Analysis")

    # Filter to active loans only (loans with outstanding balance)
    active_df = df[df["loan_status"] != "Paid Off"].copy()

    if active_df.empty:
        st.info("No active loans with outstanding exposure.")
    else:
        # Get the top industries from industry_metrics for consistent filtering
        top_industries = set(industry_metrics["sector_code"].tolist())

        # Filter active_df to only include top industries
        active_filtered = active_df[active_df["sector_code"].isin(top_industries)].copy()

        if active_filtered.empty:
            st.info("No active exposure data available for top industries.")
        else:
            # Create display label for active loans
            active_filtered["display_label"] = active_filtered["sector_code"] + " - " + active_filtered["industry_name"]

            # --- Industry Exposure Donut Chart ---
            industry_exposure = active_filtered.groupby(["sector_code", "display_label"]).agg(
                outstanding_balance=("net_balance", "sum"),
                deal_count=("loan_id", "count")
            ).reset_index()
            industry_exposure = industry_exposure.sort_values("outstanding_balance", ascending=False)

            # --- Status Exposure Donut Chart ---
            status_exposure = active_filtered.groupby("loan_status").agg(
                outstanding_balance=("net_balance", "sum"),
                deal_count=("loan_id", "count")
            ).reset_index()
            status_exposure = status_exposure.sort_values("outstanding_balance", ascending=False)

            # Create two donut charts side by side
            col1, col2 = st.columns(2)

            with col1:
                industry_donut = alt.Chart(industry_exposure).mark_arc(innerRadius=80, outerRadius=150).encode(
                    theta=alt.Theta("outstanding_balance:Q", stack=True),
                    color=alt.Color(
                        "display_label:N",
                        legend=alt.Legend(title="Industry", orient="bottom", columns=2),
                        scale=alt.Scale(scheme="tableau20")
                    ),
                    tooltip=[
                        alt.Tooltip("display_label:N", title="Industry"),
                        alt.Tooltip("outstanding_balance:Q", title="Exposure", format="$,.0f"),
                        alt.Tooltip("deal_count:Q", title="Loans"),
                    ],
                ).properties(width=400, height=450, title="Exposure by Industry")

                # Add center text showing total
                total_exposure = industry_exposure["outstanding_balance"].sum()
                center_text = alt.Chart(pd.DataFrame({"text": [f"${total_exposure/1e6:.1f}M"]})).mark_text(
                    size=24, fontWeight="bold", color="#333"
                ).encode(text="text:N")

                st.altair_chart(industry_donut + center_text, width='stretch')

            with col2:
                # Get colors for statuses present in data (uses shared STATUS_COLORS)
                status_list = status_exposure["loan_status"].tolist()
                status_colors = [STATUS_COLORS.get(s, "#7f7f7f") for s in status_list]

                status_donut = alt.Chart(status_exposure).mark_arc(innerRadius=80, outerRadius=150).encode(
                    theta=alt.Theta("outstanding_balance:Q", stack=True),
                    color=alt.Color(
                        "loan_status:N",
                        legend=alt.Legend(title="Status", orient="bottom", columns=2),
                        scale=alt.Scale(domain=status_list, range=status_colors)
                    ),
                    tooltip=[
                        alt.Tooltip("loan_status:N", title="Status"),
                        alt.Tooltip("outstanding_balance:Q", title="Exposure", format="$,.0f"),
                        alt.Tooltip("deal_count:Q", title="Loans"),
                    ],
                ).properties(width=400, height=450, title="Exposure by Status")

                # Add center text
                center_text2 = alt.Chart(pd.DataFrame({"text": [f"${total_exposure/1e6:.1f}M"]})).mark_text(
                    size=24, fontWeight="bold", color="#333"
                ).encode(text="text:N")

                st.altair_chart(status_donut + center_text2, width='stretch')

            # --- Stacked Bar Chart: Industry x Status Breakdown ---
            st.markdown("##### Exposure by Industry & Status")

            # Group by industry and status for stacked bar
            industry_status_exposure = active_filtered.groupby(["display_label", "loan_status"]).agg(
                outstanding_balance=("net_balance", "sum"),
                deal_count=("loan_id", "count")
            ).reset_index()

            # Sort industries by total exposure
            industry_order = industry_exposure["display_label"].tolist()

            # Get all statuses for color scale (uses shared STATUS_COLORS)
            all_statuses = industry_status_exposure["loan_status"].unique().tolist()
            all_status_colors = [STATUS_COLORS.get(s, "#7f7f7f") for s in all_statuses]

            stacked_bar = alt.Chart(industry_status_exposure).mark_bar().encode(
                y=alt.Y("display_label:N", title="Industry", sort=industry_order),
                x=alt.X("outstanding_balance:Q", title="Outstanding Exposure ($)", axis=alt.Axis(format="$,.0f")),
                color=alt.Color(
                    "loan_status:N",
                    legend=alt.Legend(title="Status", orient="right"),
                    scale=alt.Scale(domain=all_statuses, range=all_status_colors)
                ),
                tooltip=[
                    alt.Tooltip("display_label:N", title="Industry"),
                    alt.Tooltip("loan_status:N", title="Status"),
                    alt.Tooltip("outstanding_balance:Q", title="Exposure", format="$,.0f"),
                    alt.Tooltip("deal_count:Q", title="Loans"),
                ],
                order=alt.Order("loan_status:N"),
            ).properties(width=800, height=500, title="Industry Exposure Breakdown by Loan Status")

            st.altair_chart(stacked_bar, width='stretch')


def plot_fico_performance_analysis(df: pd.DataFrame):
    """Plot performance metrics by FICO score bands"""
    st.header("FICO Score Performance Analysis")

    if "fico" not in df.columns or df["fico"].isna().all():
        st.warning("FICO score data not available.")
        return

    fico_bins = [0, 580, 620, 660, 700, 740, 850]
    fico_labels = ["<580", "580-619", "620-659", "660-699", "700-739", "740+"]

    fico_df = df.copy()
    fico_df["fico"] = pd.to_numeric(fico_df["fico"], errors="coerce")
    fico_df["fico_band"] = pd.cut(fico_df["fico"], bins=fico_bins, labels=fico_labels, right=False)

    fico_metrics = fico_df.groupby("fico_band", observed=True).agg(
        deal_count=("loan_id", "count"),
        capital_deployed=("csl_participation_amount", "sum"),
        outstanding_balance=("net_balance", "sum"),
        avg_payment_performance=("payment_performance", "mean"),
        total_paid=("total_paid", "sum"),
        total_invested=("total_invested", "sum"),
    ).reset_index()

    fico_metrics["actual_return_rate"] = fico_metrics["total_paid"] / fico_metrics["total_invested"]

    status_by_fico = fico_df.groupby(["fico_band", "loan_status"], observed=True).size().reset_index(name="count")
    total_by_fico = fico_df.groupby("fico_band", observed=True).size().reset_index(name="total")
    status_by_fico = status_by_fico.merge(total_by_fico, on="fico_band")
    status_by_fico["pct"] = status_by_fico["count"] / status_by_fico["total"]

    problem_loans = status_by_fico[status_by_fico["loan_status"].isin(PROBLEM_STATUSES)]
    problem_rate = problem_loans.groupby("fico_band", observed=True)["pct"].sum().reset_index(name="problem_rate")

    fico_metrics = fico_metrics.merge(problem_rate, on="fico_band", how="left")
    fico_metrics["problem_rate"] = fico_metrics["problem_rate"].fillna(0)

    col1, col2 = st.columns(2)

    with col1:
        perf_chart = alt.Chart(fico_metrics).mark_bar().encode(
            x=alt.X("fico_band:N", title="FICO Score Band", sort=fico_labels),
            y=alt.Y("avg_payment_performance:Q", title="Avg Payment Performance", axis=alt.Axis(format=".0%")),
            color=alt.Color(
                "avg_payment_performance:Q",
                scale=alt.Scale(domain=[0.5, 0.8, 1.0], range=["#d62728", "#ffbb78", "#2ca02c"]),
                legend=None,
            ),
            tooltip=[
                alt.Tooltip("fico_band:N", title="FICO Band"),
                alt.Tooltip("avg_payment_performance:Q", title="Avg Payment Performance", format=".1%"),
                alt.Tooltip("deal_count:Q", title="Loan Count"),
            ],
        ).properties(width=350, height=300, title="Payment Performance by FICO Score")
        st.altair_chart(perf_chart, width='stretch')

    with col2:
        problem_chart = alt.Chart(fico_metrics).mark_bar().encode(
            x=alt.X("fico_band:N", title="FICO Score Band", sort=fico_labels),
            y=alt.Y("problem_rate:Q", title="Problem Loan Rate", axis=alt.Axis(format=".0%")),
            color=alt.Color(
                "problem_rate:Q",
                scale=alt.Scale(domain=[0, 0.2, 0.4], range=["#2ca02c", "#ffbb78", "#d62728"]),
                legend=None,
            ),
            tooltip=[
                alt.Tooltip("fico_band:N", title="FICO Band"),
                alt.Tooltip("problem_rate:Q", title="Problem Loan Rate", format=".1%"),
                alt.Tooltip("deal_count:Q", title="Total Loans"),
            ],
        ).properties(width=350, height=300, title="Problem Loan Rate by FICO Score")
        st.altair_chart(problem_chart, width='stretch')

    return_chart = alt.Chart(fico_metrics).mark_bar().encode(
        x=alt.X("fico_band:N", title="FICO Score Band", sort=fico_labels),
        y=alt.Y("actual_return_rate:Q", title="Actual Return Rate", axis=alt.Axis(format=".0%")),
        color=alt.Color(
            "actual_return_rate:Q",
            scale=alt.Scale(domain=[0.5, 1.0, 1.3], range=["#d62728", "#ffbb78", "#2ca02c"]),
            legend=None,
        ),
        tooltip=[
            alt.Tooltip("fico_band:N", title="FICO Band"),
            alt.Tooltip("actual_return_rate:Q", title="Return Rate", format=".1%"),
            alt.Tooltip("deal_count:Q", title="Loan Count"),
        ],
    ).properties(width=700, height=300, title="Actual Return Rate by FICO Score")
    st.altair_chart(return_chart, width='stretch')

    st.subheader("FICO Performance Summary")
    display_df = fico_metrics.copy()
    display_df["outstanding_balance"] = display_df["outstanding_balance"].map(lambda x: f"${x:,.0f}")
    display_df["avg_payment_performance"] = display_df["avg_payment_performance"].map(lambda x: f"{x:.1%}")
    display_df["actual_return_rate"] = display_df["actual_return_rate"].map(lambda x: f"{x:.1%}")
    display_df["problem_rate"] = display_df["problem_rate"].map(lambda x: f"{x:.1%}")
    display_df = display_df[[
        "fico_band", "deal_count", "outstanding_balance",
        "avg_payment_performance", "actual_return_rate", "problem_rate"
    ]]
    display_df.columns = ["FICO Band", "Loan Count", "Outstanding Balance", "Avg Payment Performance", "Actual Return Rate", "Problem Loan Rate"]
    st.dataframe(display_df, width='stretch', hide_index=True)


def plot_tib_performance_analysis(df: pd.DataFrame):
    """Plot performance metrics by Time in Business bands"""
    st.header("Time in Business Performance Analysis")

    if "tib" not in df.columns or df["tib"].isna().all():
        st.warning("Time in Business data not available.")
        return

    tib_bins = [0, 5, 10, 15, 20, 25, 100]
    tib_labels = ["≤5", "5-10", "10-15", "15-20", "20-25", "25+"]

    tib_df = df.copy()
    tib_df["tib"] = pd.to_numeric(tib_df["tib"], errors="coerce")
    tib_df["tib_band"] = pd.cut(tib_df["tib"], bins=tib_bins, labels=tib_labels, right=False)

    tib_metrics = tib_df.groupby("tib_band", observed=True).agg(
        deal_count=("loan_id", "count"),
        capital_deployed=("csl_participation_amount", "sum"),
        outstanding_balance=("net_balance", "sum"),
        avg_payment_performance=("payment_performance", "mean"),
        total_paid=("total_paid", "sum"),
        total_invested=("total_invested", "sum"),
    ).reset_index()

    tib_metrics["actual_return_rate"] = tib_metrics["total_paid"] / tib_metrics["total_invested"]

    status_by_tib = tib_df.groupby(["tib_band", "loan_status"], observed=True).size().reset_index(name="count")
    total_by_tib = tib_df.groupby("tib_band", observed=True).size().reset_index(name="total")
    status_by_tib = status_by_tib.merge(total_by_tib, on="tib_band")
    status_by_tib["pct"] = status_by_tib["count"] / status_by_tib["total"]

    problem_loans = status_by_tib[status_by_tib["loan_status"].isin(PROBLEM_STATUSES)]
    problem_rate = problem_loans.groupby("tib_band", observed=True)["pct"].sum().reset_index(name="problem_rate")

    tib_metrics = tib_metrics.merge(problem_rate, on="tib_band", how="left")
    tib_metrics["problem_rate"] = tib_metrics["problem_rate"].fillna(0)

    col1, col2 = st.columns(2)

    with col1:
        perf_chart = alt.Chart(tib_metrics).mark_bar().encode(
            x=alt.X("tib_band:N", title="TIB Band (Years)", sort=tib_labels),
            y=alt.Y("avg_payment_performance:Q", title="Avg Payment Performance", axis=alt.Axis(format=".0%")),
            color=alt.Color(
                "avg_payment_performance:Q",
                scale=alt.Scale(domain=[0.5, 0.8, 1.0], range=["#d62728", "#ffbb78", "#2ca02c"]),
                legend=None,
            ),
            tooltip=[
                alt.Tooltip("tib_band:N", title="TIB Band"),
                alt.Tooltip("avg_payment_performance:Q", title="Avg Payment Performance", format=".1%"),
                alt.Tooltip("deal_count:Q", title="Loan Count"),
            ],
        ).properties(width=350, height=300, title="Payment Performance by Time in Business")
        st.altair_chart(perf_chart, width='stretch')

    with col2:
        problem_chart = alt.Chart(tib_metrics).mark_bar().encode(
            x=alt.X("tib_band:N", title="TIB Band (Years)", sort=tib_labels),
            y=alt.Y("problem_rate:Q", title="Problem Loan Rate", axis=alt.Axis(format=".0%")),
            color=alt.Color(
                "problem_rate:Q",
                scale=alt.Scale(domain=[0, 0.2, 0.4], range=["#2ca02c", "#ffbb78", "#d62728"]),
                legend=None,
            ),
            tooltip=[
                alt.Tooltip("tib_band:N", title="TIB Band"),
                alt.Tooltip("problem_rate:Q", title="Problem Loan Rate", format=".1%"),
                alt.Tooltip("deal_count:Q", title="Total Loans"),
            ],
        ).properties(width=350, height=300, title="Problem Loan Rate by Time in Business")
        st.altair_chart(problem_chart, width='stretch')

    return_chart = alt.Chart(tib_metrics).mark_bar().encode(
        x=alt.X("tib_band:N", title="TIB Band (Years)", sort=tib_labels),
        y=alt.Y("actual_return_rate:Q", title="Actual Return Rate", axis=alt.Axis(format=".0%")),
        color=alt.Color(
            "actual_return_rate:Q",
            scale=alt.Scale(domain=[0.5, 1.0, 1.3], range=["#d62728", "#ffbb78", "#2ca02c"]),
            legend=None,
        ),
        tooltip=[
            alt.Tooltip("tib_band:N", title="TIB Band"),
            alt.Tooltip("actual_return_rate:Q", title="Return Rate", format=".1%"),
            alt.Tooltip("deal_count:Q", title="Loan Count"),
        ],
    ).properties(width=700, height=300, title="Actual Return Rate by Time in Business")
    st.altair_chart(return_chart, width='stretch')

    st.subheader("TIB Performance Summary")
    display_df = tib_metrics.copy()
    display_df["outstanding_balance"] = display_df["outstanding_balance"].map(lambda x: f"${x:,.0f}")
    display_df["avg_payment_performance"] = display_df["avg_payment_performance"].map(lambda x: f"{x:.1%}")
    display_df["actual_return_rate"] = display_df["actual_return_rate"].map(lambda x: f"{x:.1%}")
    display_df["problem_rate"] = display_df["problem_rate"].map(lambda x: f"{x:.1%}")
    display_df = display_df[["tib_band", "deal_count", "outstanding_balance", "avg_payment_performance", "actual_return_rate", "problem_rate"]]
    display_df.columns = ["TIB Band", "Loan Count", "Outstanding Balance", "Avg Payment Performance", "Actual Return Rate", "Problem Loan Rate"]
    st.dataframe(display_df, width='stretch', hide_index=True)


def display_irr_analysis(df: pd.DataFrame):
    """Display IRR analysis for paid off loans"""
    st.subheader("IRR Analysis (Paid Off Loans)")

    paid_off = df[df["loan_status"] == "Paid Off"].copy()

    if paid_off.empty or "realized_irr" not in paid_off.columns:
        st.info("No paid-off loans with realized IRR data.")
        return

    paid_off_with_irr = paid_off[paid_off["realized_irr"].notna()]

    if paid_off_with_irr.empty:
        st.info("No paid-off loans with valid IRR data.")
        return

    avg_realized_irr = paid_off_with_irr["realized_irr"].mean()

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Paid Off Loans with IRR", len(paid_off_with_irr))
    with col2:
        st.metric("Simple Avg Realized IRR", f"{avg_realized_irr:.1%}" if pd.notnull(avg_realized_irr) else "N/A")


def plot_irr_by_partner(df: pd.DataFrame):
    """Plot average IRR by partner source"""
    paid_off = df[df["loan_status"] == "Paid Off"].copy()

    if paid_off.empty or "realized_irr" not in paid_off.columns or "partner_source" not in paid_off.columns:
        st.info("Insufficient data for IRR by partner analysis.")
        return

    paid_off_with_irr = paid_off[paid_off["realized_irr"].notna()]

    if paid_off_with_irr.empty:
        st.info("No IRR data available for partner analysis.")
        return

    partner_irr = paid_off_with_irr.groupby("partner_source").agg(
        avg_irr=("realized_irr", "mean"),
        loan_count=("loan_id", "count")
    ).reset_index()

    partner_irr = partner_irr[partner_irr["loan_count"] >= 2]  # Filter for significance

    if partner_irr.empty:
        st.info("Not enough paid-off loans per partner for IRR analysis.")
        return

    chart = alt.Chart(partner_irr).mark_bar().encode(
        x=alt.X("partner_source:N", title="Partner Source", sort="-y"),
        y=alt.Y("avg_irr:Q", title="Average Realized IRR", axis=alt.Axis(format=".1%")),
        color=alt.Color(
            "avg_irr:Q",
            scale=alt.Scale(domain=[0, 0.2, 0.4], range=["#d62728", "#ffbb78", "#2ca02c"]),
            legend=None
        ),
        tooltip=[
            alt.Tooltip("partner_source:N", title="Partner"),
            alt.Tooltip("avg_irr:Q", title="Avg IRR", format=".1%"),
            alt.Tooltip("loan_count:Q", title="Paid Off Loans"),
        ]
    ).properties(width=600, height=350, title="Average Realized IRR by Partner (Paid Off Loans, ≥2 loans)")

    st.altair_chart(chart, width='stretch')


# -----------
# Main Page
# -----------
def main():
    st.title("Loan Tape Dashboard")

    # Initialize session state for tab persistence
    if "loan_tape_active_tab" not in st.session_state:
        st.session_state.loan_tape_active_tab = 0

    # Check if we need to stay on Loan Tape tab (after status update)
    if st.session_state.get("stay_on_loan_tape_tab", False):
        st.session_state.loan_tape_active_tab = 4  # Loan Tape tab index
        st.session_state.stay_on_loan_tape_tab = False

    last_updated = get_last_updated()
    st.caption(f"Data last updated: {last_updated}")

    # Load data
    loans_df = load_loan_summaries()
    deals_df = load_deals()

    # Prepare data with calculations
    df = prepare_loan_data(loans_df, deals_df)
    df = calculate_irr(df)

    # Sidebar Filters
    st.sidebar.header("Filters")

    # Standardized date range filter
    with st.sidebar:
        filtered_df, _ = create_date_range_filter(
            df,
            date_col="funding_date",
            label="Select Date Range",
            checkbox_label="Filter by Funding Date",
            default_enabled=False,
            key_prefix="loan_tape_date"
        )

    # Standardized partner source filter
    with st.sidebar:
        filtered_df, selected_partners = create_partner_source_filter(
            filtered_df,
            partner_col="partner_source",
            label="Select Partner Sources",
            default_all=True,
            key_prefix="loan_tape_partner"
        )

    # Standardized status filter
    with st.sidebar:
        filtered_df, selected_status = create_status_filter(
            filtered_df,
            status_col="loan_status",
            label="Filter by Status",
            include_all_option=True,
            key_prefix="loan_tape_status"
        )

    st.sidebar.markdown("---")
    st.sidebar.write(f"**Showing:** {len(filtered_df)} of {len(df)} loans")

    # Tabs with persistence support
    tab_names = ["Summary", "Capital Flow", "Performance Analysis", "Risk Analytics", "Loan Tape"]
    tabs = st.tabs(tab_names)

    # Inject JavaScript to click the correct tab if we need to stay on Loan Tape
    if st.session_state.loan_tape_active_tab == 4:
        import streamlit.components.v1 as components
        components.html(
            """
            <script>
                // Wait for the page to load, then click the Loan Tape tab
                setTimeout(function() {
                    const tabs = window.parent.document.querySelectorAll('[data-baseweb="tab"]');
                    if (tabs.length >= 5) {
                        tabs[4].click();
                    }
                }, 100);
            </script>
            """,
            height=0
        )
        # Reset after injection
        st.session_state.loan_tape_active_tab = 0

    with tabs[0]:
        st.header("Portfolio Overview")

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            total_positions = len(filtered_df)
            paid_off = (filtered_df["loan_status"] == "Paid Off").sum()
            st.metric("Total Positions", f"{total_positions}")
            st.caption(f"({paid_off} paid off)")
        with col2:
            total_deployed = filtered_df["csl_participation_amount"].sum()
            st.metric("Capital Deployed", f"${total_deployed:,.0f}")
        with col3:
            total_returned = filtered_df["total_paid"].sum()
            st.metric("Capital Returned", f"${total_returned:,.0f}")
        with col4:
            net_balance = filtered_df["net_balance"].sum()
            st.metric("Net Outstanding", f"${net_balance:,.0f}")

        st.markdown("---")
        st.subheader("Loan Status Distribution")
        plot_status_distribution(filtered_df)

        st.markdown("---")
        st.subheader("ROI Distribution by Loan")
        plot_roi_distribution(filtered_df)

        # Portfolio Distribution Histograms
        st.markdown("---")
        st.subheader("Portfolio Distributions")

        # Row 1: FICO and Factor Rate
        hist_col1, hist_col2 = st.columns(2)
        with hist_col1:
            plot_fico_histogram(filtered_df)
        with hist_col2:
            plot_factor_histogram(filtered_df)

        # Row 2: Term and Months Left
        hist_col3, hist_col4 = st.columns(2)
        with hist_col3:
            plot_term_histogram(filtered_df)
        with hist_col4:
            plot_months_left_histogram(filtered_df)

        # Row 3: CSL Participation and TIB
        hist_col5, hist_col6 = st.columns(2)
        with hist_col5:
            plot_csl_participation_histogram(filtered_df)
        with hist_col6:
            plot_tib_histogram(filtered_df)

    with tabs[1]:
        st.header("Capital Flow Analysis")
        plot_capital_flow(filtered_df)

        st.markdown("---")
        plot_investment_net_position(filtered_df)

        st.markdown("---")
        st.subheader("Payment Performance by Cohort")
        plot_payment_performance_by_cohort(filtered_df)

        st.markdown("---")
        display_irr_analysis(filtered_df)

        st.markdown("---")
        st.subheader("Average IRR by Partner")
        plot_irr_by_partner(filtered_df)

    with tabs[2]:
        st.header("Performance Analysis")

        plot_industry_performance_analysis(filtered_df)

        st.markdown("---")
        plot_fico_performance_analysis(filtered_df)

        st.markdown("---")
        plot_tib_performance_analysis(filtered_df)

    with tabs[3]:
        st.header("Risk Analytics")

        risk_df = calculate_risk_scores(filtered_df)

        if not risk_df.empty:
            with st.expander("How Risk Scores are Calculated"):
                st.markdown(
                    """
**Risk Score Formula:**
**Components:**
- **Performance Gap**: 1 - Payment Performance
- **Status Multipliers**: Active=1.0, Late=2.5, Default=4.0, Bankrupt=5.0
- **Overdue Factor**: Months past maturity / 12

**Risk Bands:**
- Low: 0-0.5
- Moderate: 0.5-1.0
- Elevated: 1.0-1.5
- High: 1.5-2.0
- Severe: 2.0+
"""
                )

            col1, col2, col3 = st.columns(3)
            with col1:
                avg_risk = risk_df["risk_score"].mean()
                st.metric("Average Risk Score", f"{avg_risk:.2f}")
            with col2:
                high_risk_count = (risk_df["risk_score"] >= 1.5).sum()
                st.metric("High/Severe Risk Loans", f"{high_risk_count}")
            with col3:
                high_risk_balance = risk_df[risk_df["risk_score"] >= 1.5]["net_balance"].sum()
                st.metric("High Risk Balance", f"${high_risk_balance:,.0f}")

            st.markdown("---")
            st.subheader("Top 10 Highest Risk Loans")

            top_risk = risk_df.nlargest(10, "risk_score")[
                [
                    "loan_id", "deal_name", "loan_status", "payment_performance",
                    "days_since_funding", "days_past_maturity", "status_multiplier",
                    "risk_score", "net_balance",
                ]
            ].copy()

            top_risk_display = top_risk.copy()
            top_risk_display["payment_performance"] = top_risk_display["payment_performance"].map(lambda x: f"{x:.1%}" if pd.notnull(x) else "N/A")
            top_risk_display["risk_score"] = top_risk_display["risk_score"].map(lambda x: f"{x:.2f}" if pd.notnull(x) else "N/A")
            top_risk_display["status_multiplier"] = top_risk_display["status_multiplier"].map(lambda x: f"{x:.1f}x" if pd.notnull(x) else "N/A")
            top_risk_display["net_balance"] = top_risk_display["net_balance"].map(lambda x: f"${x:,.0f}" if pd.notnull(x) else "N/A")

            top_risk_display.columns = [
                "Loan ID", "Deal Name", "Status", "Payment Perf",
                "Days Funded", "Days Overdue", "Status Mult",
                "Risk Score", "Net Balance",
            ]
            st.dataframe(top_risk_display, width='stretch', hide_index=True)

            st.markdown("---")
            st.subheader("Risk Score Distribution")

            band_summary = risk_df.groupby("risk_band", observed=True).agg(
                loan_count=("loan_id", "count"),
                net_balance=("net_balance", "sum"),
            ).reset_index()

            if not band_summary.empty:
                risk_band_order = ["Low (0-0.5)", "Moderate (0.5-1.0)", "Elevated (1.0-1.5)", "High (1.5-2.0)", "Severe (2.0+)"]

                risk_bar = alt.Chart(band_summary).mark_bar().encode(
                    x=alt.X("risk_band:N", title="Risk Band", sort=risk_band_order),
                    y=alt.Y("loan_count:Q", title="Number of Loans"),
                    color=alt.Color(
                        "risk_band:N",
                        scale=alt.Scale(
                            domain=risk_band_order,
                            range=["#2ca02c", "#98df8a", "#ffbb78", "#ff7f0e", "#d62728"],
                        ),
                        legend=alt.Legend(title="Risk Level", orient="right"),
                        sort=risk_band_order,
                    ),
                    tooltip=[
                        alt.Tooltip("risk_band:N", title="Risk Band"),
                        alt.Tooltip("loan_count:Q", title="Loan Count"),
                        alt.Tooltip("net_balance:Q", title="Net Balance", format="$,.0f"),
                    ],
                ).properties(width=700, height=350, title="Loan Count by Risk Band (Active Loans Only)")

                st.altair_chart(risk_bar, width='stretch')
        else:
            st.info("No active loans to calculate risk scores.")

    with tabs[4]:
        st.header("Complete Loan Tape")

        display_columns = [
            "loan_id", "deal_name", "partner_source", "loan_status",
            "industry_name", "sector_code", "fico", "tib",
            "funding_date", "maturity_date", "projected_payoff_date",
            "factor_rate", "commission_fee",
            "csl_participation_amount", "total_invested", "total_paid", "net_balance",
            "current_roi", "payment_performance", "remaining_maturity_months",
            "is_past_maturity",
        ]

        # Add IRR columns for Paid Off loans
        if "realized_irr" in filtered_df.columns:
            display_columns.append("realized_irr")

        column_rename = {
            "loan_id": "Loan ID",
            "deal_name": "Deal Name",
            "partner_source": "Partner",
            "loan_status": "Status",
            "industry_name": "Industry",
            "sector_code": "NAICS 2-Digit",
            "fico": "FICO",
            "tib": "TIB (Years)",
            "funding_date": "Funded",
            "maturity_date": "Maturity",
            "projected_payoff_date": "Proj. Payoff",
            "factor_rate": "Factor Rate",
            "commission_fee": "Commission %",
            "csl_participation_amount": "Capital Deployed",
            "total_invested": "Total Cost Basis",
            "total_paid": "Total Paid",
            "net_balance": "Net Balance",
            "current_roi": "ROI",
            "payment_performance": "Payment Perf",
            "remaining_maturity_months": "Months Left",
            "is_past_maturity": "Past Maturity",
            "realized_irr": "IRR (Paid Off)",
        }

        loan_tape = format_dataframe_for_display(filtered_df, display_columns, column_rename)

        # Apply conditional formatting for past maturity dates
        def highlight_past_maturity(row):
            """Highlight rows with maturity date in the past"""
            try:
                maturity_date = pd.to_datetime(row.get("Maturity"), errors="coerce")
                today = pd.Timestamp.today().normalize()

                if pd.notna(maturity_date) and maturity_date < today and row.get("Status") != "Paid Off":
                    # Return red color for all cells in the row
                    return ["color: red"] * len(row)
                return [""] * len(row)
            except:
                return [""] * len(row)

        # Display with conditional formatting
        styled_df = loan_tape.style.apply(highlight_past_maturity, axis=1)
        st.dataframe(styled_df, width='stretch', hide_index=True)

        csv = loan_tape.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="Download Loan Tape as CSV",
            data=csv,
            file_name=f"loan_tape_{pd.Timestamp.today().strftime('%Y%m%d')}.csv",
            mime="text/csv",
        )

        # Manual Status Override Section
        st.markdown("---")
        st.subheader("Manual Status Override")

        # Callback to keep user on Loan Tape tab when selecting a loan
        def on_loan_select_change():
            st.session_state.stay_on_loan_tape_tab = True

        # Let user select a loan to edit
        loan_options = filtered_df["loan_id"].tolist()
        selected_loan = st.selectbox(
            "Select Loan to Edit",
            options=[""] + loan_options,
            key="status_editor_loan_select",
            on_change=on_loan_select_change
        )

        if selected_loan:
            loan_row = filtered_df[filtered_df["loan_id"] == selected_loan].iloc[0]

            col1, col2 = st.columns(2)
            with col1:
                st.write(f"**Loan ID:** {selected_loan}")
                st.write(f"**Deal Name:** {loan_row.get('deal_name', 'N/A')}")
                st.write(f"**Current Status:** {loan_row['loan_status']}")
                payment_perf = loan_row.get('payment_performance', 0)
                st.write(f"**Payment Performance:** {payment_perf:.1%}" if pd.notna(payment_perf) else "**Payment Performance:** N/A")

                # Show manual status indicator and last update time
                is_manual = loan_row.get("manual_status", False)
                if is_manual:
                    last_update = loan_row.get("status_last_manual_update") or loan_row.get("status_changed_at")
                    if pd.notna(last_update):
                        update_time = pd.to_datetime(last_update).strftime("%Y-%m-%d %H:%M")
                        st.info(f"Manual override active (set {update_time})")
                    else:
                        st.info("Manual override active")

            with col2:
                render_manual_status_editor(
                    loan_id=selected_loan,
                    current_status=loan_row["loan_status"],
                    is_manual=loan_row.get("manual_status", False)
                )



if __name__ == "__main__":
    main()
