# pages/loan_tape.py
"""
Loan Tape Dashboard - Portfolio analysis and risk management

This page provides comprehensive loan portfolio analytics including:
- Capital flow tracking
- Performance analysis
- Risk scoring
- IRR calculations
- ML-based predictions
- Watchlist monitoring
- Portfolio insights
"""

import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
from datetime import datetime, timedelta

# Core utilities
from utils.config import (
    setup_page,
    PRIMARY_COLOR,
    COLOR_PALETTE,
    PLATFORM_FEE_RATE,
    GRADE_INDICATORS,
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
    add_performance_grades,
    get_grade_summary,
)
from utils.loan_tape_analytics import (
    PROBLEM_STATUSES,
    get_display_name,
    get_payment_behavior_features,
)
from utils.status_constants import (
    STATUS_COLORS,
    STATUS_GROUPS,
    STATUS_GROUP_COLORS,
    ALL_VALID_STATUSES,
    PROBLEM_STATUSES as STATUS_PROBLEM_STATUSES,
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

# ---------------------------
# Watchlist Constants
# ---------------------------
HIGH_SEVERITY_PERFORMANCE = 0.70  # Below 70% is high severity
MEDIUM_SEVERITY_PERFORMANCE = 0.80  # Below 80% is medium severity
APPROACHING_MATURITY_DAYS = 60  # Days to maturity threshold

SEVERITY_COLORS = {
    "High": "#dc3545",  # Red
    "Medium": "#fd7e14",  # Orange
    "Low": "#ffc107",  # Yellow
}

# ---------------------------
# Portfolio Insights Constants
# ---------------------------
TARGET_MOIC = 1.20
TARGET_PROBLEM_RATE = 0.10  # 10%
PARTNER_HIGH_CONCENTRATION = 0.50  # 50% = red alert
PARTNER_MEDIUM_CONCENTRATION = 0.30  # 30% = yellow alert
INDUSTRY_HIGH_CONCENTRATION = 0.20  # 20% = red alert


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


def plot_grade_distribution(df: pd.DataFrame):
    """
    Plot grade distribution across ROI, On-Time Payment, and IRR (Speed) grades.
    Shows bar charts with grade colors for each metric.
    """
    grade_summary = get_grade_summary(df)

    if not grade_summary:
        st.info("No grade data available")
        return

    # Prepare data for visualization
    grades_order = ["A", "B", "C", "D", "F"]
    grade_colors = {g: GRADE_INDICATORS[g]["color"] for g in grades_order}

    # Create three columns for the three grade types
    col1, col2, col3 = st.columns(3)

    def create_grade_chart(grade_col: str, title: str, container):
        if grade_col not in grade_summary:
            container.info(f"No {title} data")
            return

        data = grade_summary[grade_col]
        # Filter out N/A and only include grades with counts > 0
        chart_data = pd.DataFrame([
            {"Grade": g, "Count": data.get(g, 0), "Color": grade_colors[g]}
            for g in grades_order
            if data.get(g, 0) > 0
        ])

        if chart_data.empty:
            container.info(f"No graded loans for {title}")
            return

        # Calculate total for percentages
        total = chart_data["Count"].sum()

        chart = alt.Chart(chart_data).mark_bar().encode(
            x=alt.X("Grade:N", sort=grades_order, title=None, axis=alt.Axis(labelAngle=0)),
            y=alt.Y("Count:Q", title="Loans"),
            color=alt.Color("Color:N", scale=None),
            tooltip=[
                alt.Tooltip("Grade:N", title="Grade"),
                alt.Tooltip("Count:Q", title="Count"),
            ]
        ).properties(
            height=200,
            title=title
        )

        container.altair_chart(chart, use_container_width=True)

        # Show percentages below
        pct_str = " | ".join([f"{g}: {data.get(g, 0)/total*100:.0f}%" for g in grades_order if data.get(g, 0) > 0])
        container.caption(pct_str)

    with col1:
        create_grade_chart("payment_grade", "On-Time Payment Grade", col1)

    with col2:
        create_grade_chart("roi_grade", "ROI Grade", col2)

    with col3:
        create_grade_chart("irr_grade", "Speed (IRR) Grade", col3)


def _build_return_timeline_from_summaries(df: pd.DataFrame) -> pd.Series:
    """
    Build a return timeline from loan_summaries when loan_schedules data is unavailable.

    Uses payoff_date for paid off loans and today for active loans with payments.
    This provides a reasonable approximation of when capital was returned.

    Args:
        df: DataFrame with funding_date, payoff_date, total_paid, loan_status columns

    Returns:
        pd.Series: Timeline of payments indexed by date
    """
    today = pd.Timestamp.today().normalize()

    # Prepare data
    d = df.copy()
    d["payoff_date"] = pd.to_datetime(d.get("payoff_date"), errors="coerce").dt.tz_localize(None)
    d["funding_date"] = pd.to_datetime(d.get("funding_date"), errors="coerce").dt.tz_localize(None)
    d["total_paid"] = pd.to_numeric(d.get("total_paid", 0), errors="coerce").fillna(0)

    # Only include loans with payments
    d = d[d["total_paid"] > 0].copy()

    if d.empty:
        return pd.Series(dtype=float)

    # Determine the payment attribution date for each loan:
    # - Paid Off loans: use payoff_date
    # - Active loans: use today (representing payments received so far)
    d["payment_attribution_date"] = d.apply(
        lambda row: row["payoff_date"] if pd.notnull(row.get("payoff_date")) and row.get("loan_status") == "Paid Off"
        else today,
        axis=1
    )

    # Group payments by attribution date
    payment_data = d[["payment_attribution_date", "total_paid"]].dropna()
    if payment_data.empty:
        return pd.Series(dtype=float)

    return_timeline = payment_data.groupby("payment_attribution_date")["total_paid"].sum().sort_index().cumsum()
    return return_timeline


def _get_return_timeline(df: pd.DataFrame, schedules: pd.DataFrame) -> pd.Series:
    """
    Get the return timeline, preferring loan_schedules data if available,
    otherwise falling back to loan_summaries.

    Args:
        df: Loan summaries dataframe
        schedules: Loan schedules dataframe (may be empty or missing actual_payment)

    Returns:
        pd.Series: Cumulative return timeline indexed by date
    """
    # Normalize loan_ids in df for filtering
    loan_ids_in_df = set(df["loan_id"].astype(str).str.strip().str.replace(r'\.0$', '', regex=True).dropna())

    # Try to use loan_schedules if available
    if not schedules.empty and "payment_date" in schedules.columns and "actual_payment" in schedules.columns:
        schedules = schedules.copy()

        # Filter schedules to only loans in df
        if "loan_id" in schedules.columns:
            schedules["loan_id_norm"] = schedules["loan_id"].astype(str).str.strip().str.replace(r'\.0$', '', regex=True)
            schedules = schedules[schedules["loan_id_norm"].isin(loan_ids_in_df)]

        schedules["payment_date"] = pd.to_datetime(schedules["payment_date"], errors="coerce").dt.tz_localize(None)
        schedules["actual_payment"] = pd.to_numeric(schedules["actual_payment"], errors="coerce")

        payment_data = schedules[
            schedules["actual_payment"].notna() &
            (schedules["actual_payment"] > 0) &
            schedules["payment_date"].notna()
        ]

        if not payment_data.empty:
            return payment_data.groupby("payment_date")["actual_payment"].sum().sort_index().cumsum()

    # Fallback to loan_summaries-based timeline
    return _build_return_timeline_from_summaries(df)


def plot_capital_flow(df: pd.DataFrame):
    """Plot capital deployment vs returns over time"""
    st.subheader("Capital Flow: Deployment vs. Returns")

    schedules = load_loan_schedules()

    d = df.copy()
    d["funding_date"] = pd.to_datetime(d["funding_date"], errors="coerce").dt.tz_localize(None)

    # Calculate total deployed from the dataframe
    total_deployed = d["csl_participation_amount"].sum()

    # Calculate total returned from loan_summaries total_paid
    # This represents actual payments received by CSL
    total_returned = d["total_paid"].sum()

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Total Capital Deployed", f"${total_deployed:,.0f}")
    with col2:
        st.metric("Total Capital Returned", f"${total_returned:,.0f}")

    deploy_data = d[["funding_date", "csl_participation_amount"]].dropna()
    deploy_timeline = deploy_data.groupby("funding_date")["csl_participation_amount"].sum().sort_index().cumsum()

    # Get return timeline from schedules or fallback to summaries
    return_timeline = _get_return_timeline(d, schedules)

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

    # Get return timeline from schedules or fallback to summaries
    return_timeline = _get_return_timeline(d, schedules)

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


def plot_metric_correlations(df: pd.DataFrame):
    """
    Display correlation analysis between key performance metrics.
    Shows correlation coefficients and scatter plots for key metric pairs.
    """
    st.header("Metric Correlation Analysis")

    # Filter to paid-off loans with both metrics available
    paid_off = df[df["loan_status"] == "Paid Off"].copy()

    if paid_off.empty:
        st.info("No paid-off loans available for correlation analysis.")
        return

    # Check for required columns
    has_pct_on_time = "pct_on_time" in paid_off.columns
    has_realized_irr = "realized_irr" in paid_off.columns
    has_roi = "current_roi" in paid_off.columns

    if not (has_pct_on_time and has_realized_irr):
        st.info("Insufficient data for correlation analysis. Need on-time rate and IRR data.")
        return

    # Filter to loans with valid data for both metrics
    correlation_df = paid_off[
        paid_off["pct_on_time"].notna() &
        paid_off["realized_irr"].notna()
    ].copy()

    if len(correlation_df) < 5:
        st.info("Need at least 5 paid-off loans with both on-time rate and IRR data for correlation analysis.")
        return

    # Calculate correlation coefficients
    corr_ontime_irr = correlation_df["pct_on_time"].corr(correlation_df["realized_irr"])

    # Display correlation coefficients as metrics
    st.subheader("Key Correlations")

    col1, col2, col3 = st.columns(3)

    def interpret_correlation(r):
        """Interpret correlation strength."""
        if abs(r) < 0.1:
            return "No correlation"
        elif abs(r) < 0.3:
            return "Weak"
        elif abs(r) < 0.5:
            return "Moderate"
        elif abs(r) < 0.7:
            return "Strong"
        else:
            return "Very strong"

    with col1:
        st.metric(
            "On-Time Rate ↔ IRR",
            f"r = {corr_ontime_irr:.3f}",
            interpret_correlation(corr_ontime_irr)
        )

    # Additional correlations if data available
    if has_roi:
        roi_df = correlation_df[correlation_df["current_roi"].notna()]
        if len(roi_df) >= 5:
            corr_ontime_roi = roi_df["pct_on_time"].corr(roi_df["current_roi"])
            corr_roi_irr = roi_df["current_roi"].corr(roi_df["realized_irr"])

            with col2:
                st.metric(
                    "On-Time Rate ↔ ROI",
                    f"r = {corr_ontime_roi:.3f}",
                    interpret_correlation(corr_ontime_roi)
                )

            with col3:
                st.metric(
                    "ROI ↔ IRR",
                    f"r = {corr_roi_irr:.3f}",
                    interpret_correlation(corr_roi_irr)
                )

    # Scatter plot: On-Time Rate vs IRR
    st.subheader("On-Time Rate vs Speed (IRR)")

    scatter = alt.Chart(correlation_df).mark_circle(size=60, opacity=0.6).encode(
        x=alt.X("pct_on_time:Q", title="On-Time Payment Rate", scale=alt.Scale(domain=[0, 1]), axis=alt.Axis(format=".0%")),
        y=alt.Y("realized_irr:Q", title="Realized IRR (Annualized)", axis=alt.Axis(format=".0%")),
        color=alt.Color("payment_grade:N", scale=alt.Scale(
            domain=["A", "B", "C", "D", "F"],
            range=[GRADE_INDICATORS["A"]["color"], GRADE_INDICATORS["B"]["color"],
                   GRADE_INDICATORS["C"]["color"], GRADE_INDICATORS["D"]["color"],
                   GRADE_INDICATORS["F"]["color"]]
        ), legend=alt.Legend(title="Payment Grade")),
        tooltip=[
            alt.Tooltip("loan_id:N", title="Loan ID"),
            alt.Tooltip("pct_on_time:Q", title="On-Time Rate", format=".1%"),
            alt.Tooltip("realized_irr:Q", title="Realized IRR", format=".1%"),
            alt.Tooltip("payment_grade:N", title="Payment Grade"),
        ]
    ).properties(
        width=700,
        height=400,
        title=f"On-Time Rate vs IRR (r = {corr_ontime_irr:.3f})"
    )

    # Add regression line
    regression_line = scatter.transform_regression(
        "pct_on_time", "realized_irr"
    ).mark_line(color="red", strokeDash=[4, 4])

    st.altair_chart(scatter + regression_line, use_container_width=True)

    # Interpretation text
    with st.expander("How to Interpret Correlations"):
        st.markdown("""
        **Correlation Coefficient (r) Interpretation:**
        - **r ≈ 0**: No linear relationship between metrics
        - **|r| < 0.3**: Weak correlation
        - **0.3 ≤ |r| < 0.5**: Moderate correlation
        - **0.5 ≤ |r| < 0.7**: Strong correlation
        - **|r| ≥ 0.7**: Very strong correlation

        **Key Insight:** A low correlation between on-time rate and IRR suggests these metrics
        capture independent aspects of loan performance:
        - **On-Time Rate**: Measures payment behavior consistency
        - **IRR (Speed)**: Measures how quickly capital was returned

        This means a borrower can have excellent payment timing but average speed to payoff,
        or vice versa. Both metrics provide unique information for assessing loan performance.
        """)


# -------------------
# Watchlist Helper Functions
# -------------------
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

    if pd.isna(payment_performance):
        payment_performance = 0.0

    if is_past_maturity and payment_performance < HIGH_SEVERITY_PERFORMANCE:
        return "High"

    if is_past_maturity:
        return "Medium"

    if 0 < days_to_maturity <= APPROACHING_MATURITY_DAYS and payment_performance < MEDIUM_SEVERITY_PERFORMANCE:
        return "Medium"

    return "Low"


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

    if "maturity_date" in result.columns:
        result["maturity_date"] = pd.to_datetime(result["maturity_date"], errors="coerce")

    result["days_past_maturity"] = 0
    result["days_to_maturity"] = 999

    if "maturity_date" in result.columns:
        mask_valid_maturity = result["maturity_date"].notna()

        result.loc[mask_valid_maturity, "days_past_maturity"] = result.loc[mask_valid_maturity, "maturity_date"].apply(
            lambda x: max(0, (today - x).days)
        )

        result.loc[mask_valid_maturity, "days_to_maturity"] = result.loc[mask_valid_maturity, "maturity_date"].apply(
            lambda x: (x - today).days
        )

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
    """Display a watchlist table with consistent formatting."""
    if df.empty:
        st.info(f"No loans found for: {title}")
        return

    if "severity" not in df.columns:
        df = df.copy()
        df["severity"] = df.apply(calculate_severity, axis=1)

    if severity_filter:
        df = df[df["severity"] == severity_filter]
        if df.empty:
            st.info(f"No {severity_filter.lower()} severity loans found.")
            return

    df = df.sort_values("net_balance", ascending=False)

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

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Loans", len(df))
    with col2:
        st.metric("Total Exposure", format_currency(df["net_balance"].sum()))
    with col3:
        avg_perf = df["payment_performance"].mean()
        st.metric("Avg Payment Perf", format_percentage(avg_perf))


def render_watchlist_tab(df: pd.DataFrame):
    """Render the Watchlist tab content."""
    st.header("Loan Watchlist")
    st.markdown("*Proactive monitoring for loans needing attention*")

    # Prepare watchlist data
    watchlist_df = prepare_watchlist_data(df)

    # Filter to only active loans
    active_loans = watchlist_df[watchlist_df["loan_status"] != "Paid Off"].copy()

    if active_loans.empty:
        st.info("No active loans found in the portfolio.")
        return

    # Calculate severity for all loans
    active_loans["severity"] = active_loans.apply(calculate_severity, axis=1)

    # Summary Metrics
    severity_counts = active_loans["severity"].value_counts()
    high_count = severity_counts.get("High", 0)
    medium_count = severity_counts.get("Medium", 0)
    low_count = severity_counts.get("Low", 0)

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
        st.metric("Low Severity", f"{low_count} loans")

    with col4:
        total_watchlist = high_count + medium_count
        st.metric(
            "Total Watchlist",
            f"{total_watchlist} loans",
            delta=f"{total_watchlist / len(active_loans) * 100:.1f}% of portfolio"
        )

    st.markdown("---")

    # Past Maturity Section
    st.subheader("Past Maturity Loans")

    past_maturity = active_loans[
        (active_loans["is_past_maturity"] == True) &
        (active_loans["loan_status"].isin(["Active", "Active - Frequently Late", "Minor Delinquency", "Moderate Delinquency", "Severe Delinquency"]))
    ].copy()

    if not past_maturity.empty:
        severity_options = ["All", "High", "Medium", "Low"]
        selected_severity = st.selectbox(
            "Filter by Severity",
            severity_options,
            key="past_maturity_severity_tab"
        )

        filter_severity = None if selected_severity == "All" else selected_severity
        display_watchlist_table(past_maturity, "Past Maturity Loans", severity_filter=filter_severity)

        csv = past_maturity.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="Download Past Maturity Report",
            data=csv,
            file_name=f"past_maturity_loans_{pd.Timestamp.today().strftime('%Y%m%d')}.csv",
            mime="text/csv",
            key="past_maturity_download_tab"
        )
    else:
        st.success("No past maturity loans found!")

    st.markdown("---")

    # Approaching Maturity Section
    st.subheader("Approaching Maturity - At Risk")
    st.caption(f"Loans due within {APPROACHING_MATURITY_DAYS} days with payment performance < {MEDIUM_SEVERITY_PERFORMANCE:.0%}")

    approaching_maturity = active_loans[
        (active_loans["is_past_maturity"] == False) &
        (active_loans["days_to_maturity"] <= APPROACHING_MATURITY_DAYS) &
        (active_loans["days_to_maturity"] > 0) &
        (active_loans["payment_performance"] < MEDIUM_SEVERITY_PERFORMANCE)
    ].copy()

    if not approaching_maturity.empty:
        approaching_maturity = approaching_maturity.sort_values("days_to_maturity")

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
    else:
        st.success("No at-risk approaching maturity loans found!")

    st.markdown("---")

    # Payment Performance Concerns
    st.subheader("Payment Performance Concerns")

    has_payment_behavior = "consecutive_missed" in active_loans.columns

    if has_payment_behavior:
        consecutive_missed = active_loans[
            (active_loans["consecutive_missed"] >= 2) &
            (~active_loans["is_past_maturity"])
        ].copy()

        if not consecutive_missed.empty:
            st.markdown("**Consecutive Missed Payments** (2+ consecutive missed)")

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

    # Low performing loans
    low_performing = active_loans[
        (active_loans["payment_performance"] < HIGH_SEVERITY_PERFORMANCE) &
        (~active_loans["is_past_maturity"]) &
        (active_loans["days_to_maturity"] > APPROACHING_MATURITY_DAYS)
    ].copy()

    if not low_performing.empty:
        st.markdown(f"**Low Payment Performance** (below {HIGH_SEVERITY_PERFORMANCE:.0%})")

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


# -------------------
# Portfolio Insights Helper Functions
# -------------------
def get_problem_loans(df: pd.DataFrame) -> pd.DataFrame:
    """Filter to loans with problem status."""
    return df[df["loan_status"].isin(PROBLEM_STATUSES)].copy()


def get_paid_off_loans(df: pd.DataFrame) -> pd.DataFrame:
    """Filter to paid off loans."""
    return df[df["loan_status"] == "Paid Off"].copy()


def calculate_moic(df: pd.DataFrame) -> float:
    """Calculate Multiple on Invested Capital (total_paid / total_invested)."""
    total_invested = df["total_invested"].sum()
    total_paid = df["total_paid"].sum()
    if total_invested > 0:
        return total_paid / total_invested
    return 0.0


def render_portfolio_insights_tab(df: pd.DataFrame):
    """Render the Portfolio Insights tab content."""
    st.header("Portfolio Insights")
    st.caption("Executive-level view of portfolio health and performance")

    # Executive Summary
    st.subheader("Executive Summary")

    realized_moic = calculate_moic(df)
    problem_df = get_problem_loans(df)
    problem_count = len(problem_df)
    total_count = len(df)
    problem_rate = problem_count / total_count if total_count > 0 else 0
    capital_at_risk = problem_df["net_balance"].sum() if not problem_df.empty else 0
    total_invested = df["total_invested"].sum()
    total_returned = df["total_paid"].sum()

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        moic_delta = realized_moic - TARGET_MOIC
        st.metric(
            label="Realized MOIC",
            value=f"{realized_moic:.2f}x",
            delta=f"{moic_delta:+.2f}x vs {TARGET_MOIC:.2f}x target",
            delta_color="normal" if realized_moic >= TARGET_MOIC else "inverse"
        )

    with col2:
        rate_delta = problem_rate - TARGET_PROBLEM_RATE
        st.metric(
            label="Problem Rate",
            value=f"{problem_rate:.1%}",
            delta=f"{rate_delta:+.1%} vs {TARGET_PROBLEM_RATE:.0%} target",
            delta_color="normal" if problem_rate <= TARGET_PROBLEM_RATE else "inverse"
        )

    with col3:
        st.metric(
            label="Capital at Risk",
            value=f"${capital_at_risk:,.0f}",
            delta=f"{problem_count} problem loans"
        )

    with col4:
        net_position = total_returned - total_invested
        st.metric(
            label="Net Position",
            value=f"${net_position:,.0f}",
            delta=f"${total_returned:,.0f} returned on ${total_invested:,.0f} invested",
            delta_color="normal" if net_position >= 0 else "inverse"
        )

    # Health assessment
    if realized_moic >= TARGET_MOIC and problem_rate <= TARGET_PROBLEM_RATE:
        st.success("Portfolio is performing above targets on both MOIC and problem rate.")
    elif realized_moic >= TARGET_MOIC:
        st.warning(f"MOIC is on target, but problem rate ({problem_rate:.1%}) exceeds {TARGET_PROBLEM_RATE:.0%} threshold.")
    elif problem_rate <= TARGET_PROBLEM_RATE:
        st.warning(f"Problem rate is acceptable, but MOIC ({realized_moic:.2f}x) is below {TARGET_MOIC:.2f}x target.")
    else:
        st.error(f"Both MOIC ({realized_moic:.2f}x) and problem rate ({problem_rate:.1%}) are below targets.")

    st.markdown("---")

    # Winners vs Losers Analysis
    st.subheader("Winners vs Losers Analysis")
    st.caption("Comparing characteristics of Paid Off loans (Winners) vs Problem loans (Losers)")

    paid_off = get_paid_off_loans(df)
    problem = get_problem_loans(df)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown(f"**Winners (Paid Off): {len(paid_off)}**")
        if not paid_off.empty:
            metric_col1, metric_col2 = st.columns(2)
            with metric_col1:
                avg_fico = paid_off["fico"].mean() if "fico" in paid_off.columns and paid_off["fico"].notna().any() else None
                st.metric("Avg FICO", f"{avg_fico:.0f}" if avg_fico else "N/A")
                avg_size = paid_off["csl_participation_amount"].mean() if "csl_participation_amount" in paid_off.columns else None
                st.metric("Avg Deal Size", f"${avg_size:,.0f}" if avg_size else "N/A")
            with metric_col2:
                avg_tib = paid_off["tib"].mean() if "tib" in paid_off.columns and paid_off["tib"].notna().any() else None
                st.metric("Avg TIB (Years)", f"{avg_tib:.1f}" if avg_tib else "N/A")
                avg_pos = paid_off["ahead_positions"].mean() if "ahead_positions" in paid_off.columns and paid_off["ahead_positions"].notna().any() else None
                st.metric("Avg Position", f"{avg_pos:.1f}" if avg_pos is not None else "N/A")

    with col2:
        st.markdown(f"**Losers (Problem): {len(problem)}**")
        if not problem.empty:
            metric_col1, metric_col2 = st.columns(2)
            with metric_col1:
                avg_fico = problem["fico"].mean() if "fico" in problem.columns and problem["fico"].notna().any() else None
                st.metric("Avg FICO", f"{avg_fico:.0f}" if avg_fico else "N/A")
                avg_size = problem["csl_participation_amount"].mean() if "csl_participation_amount" in problem.columns else None
                st.metric("Avg Deal Size", f"${avg_size:,.0f}" if avg_size else "N/A")
            with metric_col2:
                avg_tib = problem["tib"].mean() if "tib" in problem.columns and problem["tib"].notna().any() else None
                st.metric("Avg TIB (Years)", f"{avg_tib:.1f}" if avg_tib else "N/A")
                avg_pos = problem["ahead_positions"].mean() if "ahead_positions" in problem.columns and problem["ahead_positions"].notna().any() else None
                st.metric("Avg Position", f"{avg_pos:.1f}" if avg_pos is not None else "N/A")

    st.markdown("---")

    # Concentration Alerts
    st.subheader("Concentration Alerts")

    active_df = df[df["loan_status"] != "Paid Off"].copy()
    total_exposure = active_df["net_balance"].sum()

    if total_exposure > 0 and "partner_source" in active_df.columns:
        partner_exposure = active_df.groupby("partner_source").agg(
            exposure=("net_balance", "sum"),
            deal_count=("loan_id", "count")
        ).reset_index()
        partner_exposure["pct_of_total"] = partner_exposure["exposure"] / total_exposure
        partner_exposure = partner_exposure.sort_values("exposure", ascending=False)

        partner_chart = alt.Chart(partner_exposure.head(10)).mark_bar().encode(
            x=alt.X("partner_source:N", title="Partner", sort="-y"),
            y=alt.Y("pct_of_total:Q", title="% of Total Exposure", axis=alt.Axis(format=".0%")),
            color=alt.when(
                alt.datum.pct_of_total >= PARTNER_HIGH_CONCENTRATION
            ).then(alt.value("#d62728")).when(
                alt.datum.pct_of_total >= PARTNER_MEDIUM_CONCENTRATION
            ).then(alt.value("#ff7f0e")).otherwise(alt.value(PRIMARY_COLOR)),
            tooltip=[
                alt.Tooltip("partner_source:N", title="Partner"),
                alt.Tooltip("pct_of_total:Q", title="% of Exposure", format=".1%"),
                alt.Tooltip("exposure:Q", title="Exposure", format="$,.0f"),
                alt.Tooltip("deal_count:Q", title="Deals"),
            ]
        ).properties(width=600, height=300, title="Partner Exposure (Top 10)")

        st.altair_chart(partner_chart, use_container_width=True)

        # Show alerts
        alerts = []
        for _, row in partner_exposure.iterrows():
            if row["pct_of_total"] >= PARTNER_HIGH_CONCENTRATION:
                st.error(f"**Partner**: {row['partner_source']} has {row['pct_of_total']:.1%} of exposure (${row['exposure']:,.0f})")
            elif row["pct_of_total"] >= PARTNER_MEDIUM_CONCENTRATION:
                st.warning(f"**Partner**: {row['partner_source']} has {row['pct_of_total']:.1%} of exposure (${row['exposure']:,.0f})")

    st.markdown("---")

    # Vintage Analysis
    st.subheader("Vintage Analysis")
    st.caption("Problem rate by funding month")

    if "funding_date" in df.columns:
        df_vintage = df.copy()
        df_vintage["funding_date"] = pd.to_datetime(df_vintage["funding_date"], errors="coerce")
        df_vintage = df_vintage[df_vintage["funding_date"].notna()]

        if not df_vintage.empty:
            df_vintage["funding_month"] = df_vintage["funding_date"].dt.to_period("M").astype(str)
            df_vintage["is_problem"] = df_vintage["loan_status"].isin(PROBLEM_STATUSES).astype(int)

            vintage_metrics = df_vintage.groupby("funding_month").agg(
                total_loans=("loan_id", "count"),
                problem_loans=("is_problem", "sum"),
                total_invested=("csl_participation_amount", "sum"),
                total_returned=("total_paid", "sum"),
            ).reset_index()

            vintage_metrics["problem_rate"] = vintage_metrics["problem_loans"] / vintage_metrics["total_loans"]
            vintage_metrics = vintage_metrics.sort_values("funding_month").tail(24)

            problem_rate_chart = alt.Chart(vintage_metrics).mark_bar().encode(
                x=alt.X("funding_month:N", title="Funding Month", sort=None,
                       axis=alt.Axis(labelAngle=-45)),
                y=alt.Y("problem_rate:Q", title="Problem Rate", axis=alt.Axis(format=".0%")),
                color=alt.condition(
                    alt.datum.problem_rate >= 0.20,
                    alt.value("#d62728"),
                    alt.value(PRIMARY_COLOR)
                ),
                tooltip=[
                    alt.Tooltip("funding_month:N", title="Month"),
                    alt.Tooltip("problem_rate:Q", title="Problem Rate", format=".1%"),
                    alt.Tooltip("total_loans:Q", title="Total Loans"),
                    alt.Tooltip("problem_loans:Q", title="Problem Loans"),
                ]
            ).properties(width=800, height=350, title="Problem Rate by Vintage (Last 24 Months)")

            threshold_line = alt.Chart(pd.DataFrame([{"threshold": 0.20}])).mark_rule(
                strokeDash=[4, 4], color="#d62728", strokeWidth=2
            ).encode(y="threshold:Q")

            target_line = alt.Chart(pd.DataFrame([{"threshold": 0.10}])).mark_rule(
                strokeDash=[2, 2], color="#ff7f0e", strokeWidth=1
            ).encode(y="threshold:Q")

            st.altair_chart(problem_rate_chart + threshold_line + target_line, use_container_width=True)

            # Highlight problematic vintages
            high_problem_months = vintage_metrics[vintage_metrics["problem_rate"] >= 0.20]
            if not high_problem_months.empty:
                st.warning(f"**{len(high_problem_months)} vintage(s) with >20% problem rate:**")
                for _, row in high_problem_months.iterrows():
                    st.markdown(f"- **{row['funding_month']}**: {row['problem_rate']:.1%} problem rate ({row['problem_loans']:.0f} of {row['total_loans']:.0f} loans)")

    st.markdown("---")

    # Top 5 Problem Loans
    st.subheader("Top 5 Problem Loans")

    problem_active = get_problem_loans(active_df)
    if not problem_active.empty:
        total_problem_exposure = problem_active["net_balance"].sum()
        top_problems = problem_active.nlargest(5, "net_balance")[
            ["loan_id", "deal_name", "partner_source", "loan_status", "net_balance"]
        ].copy()
        top_problems["pct_of_problem"] = top_problems["net_balance"] / total_problem_exposure if total_problem_exposure > 0 else 0

        display_df = top_problems.copy()
        display_df["net_balance"] = display_df["net_balance"].map(lambda x: f"${x:,.0f}")
        display_df["pct_of_problem"] = display_df["pct_of_problem"].map(lambda x: f"{x:.1%}")
        display_df.columns = ["Loan ID", "Deal Name", "Partner", "Status", "Net Balance", "% of Problem Exposure"]

        st.dataframe(display_df, hide_index=True, use_container_width=True)
        st.caption(f"Total problem exposure: ${total_problem_exposure:,.0f}")
    else:
        st.info("No problem loans in the active portfolio.")


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
        st.session_state.loan_tape_active_tab = 1  # Loan Tape tab index
        st.session_state.stay_on_loan_tape_tab = False

    last_updated = get_last_updated()
    st.caption(f"Data last updated: {last_updated}")

    # Load data
    loans_df = load_loan_summaries()
    deals_df = load_deals()

    # Prepare data with calculations
    df = prepare_loan_data(loans_df, deals_df)
    df = calculate_irr(df)

    # Load payment behavior features for on-time performance grading
    schedules_df = load_loan_schedules()
    if not schedules_df.empty:
        payment_behavior = get_payment_behavior_features(schedules_df)
        if not payment_behavior.empty:
            # Normalize loan_id for merge
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

    # Add performance grades to filtered data
    filtered_df = add_performance_grades(filtered_df)

    # Tabs with persistence support
    tab_names = ["Summary", "Loan Tape", "Capital Flow", "Performance Analysis", "Risk Analytics", "Watchlist", "Portfolio Insights"]
    tabs = st.tabs(tab_names)

    # Inject JavaScript to click the correct tab if we need to stay on Loan Tape
    if st.session_state.loan_tape_active_tab == 1:
        import streamlit.components.v1 as components
        components.html(
            """
            <script>
                // Wait for the page to load, then click the Loan Tape tab
                setTimeout(function() {
                    const tabs = window.parent.document.querySelectorAll('[data-baseweb="tab"]');
                    if (tabs.length >= 2) {
                        tabs[1].click();
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

        # Performance Grade Distribution
        st.markdown("---")
        st.subheader("Performance Grade Distribution")
        plot_grade_distribution(filtered_df)

    with tabs[2]:
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

    with tabs[3]:
        st.header("Performance Analysis")

        plot_industry_performance_analysis(filtered_df)

        st.markdown("---")
        plot_fico_performance_analysis(filtered_df)

        st.markdown("---")
        plot_tib_performance_analysis(filtered_df)

        st.markdown("---")
        plot_metric_correlations(filtered_df)

    with tabs[4]:
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

    with tabs[1]:
        st.header("Complete Loan Tape")

        display_columns = [
            "loan_id", "deal_name", "partner_source", "loan_status",
            "industry_name", "sector_code", "fico", "tib",
            "funding_date", "maturity_date", "projected_payoff_date",
            "factor_rate", "commission_fee", "net_moic",
            "csl_participation_amount", "total_invested", "total_paid", "net_balance",
            "roi_with_grade", "payment_perf_with_grade",
            "remaining_maturity_months", "is_past_maturity",
        ]

        # Add IRR column for Paid Off loans (with grade icon)
        if "irr_with_grade" in filtered_df.columns:
            display_columns.append("irr_with_grade")

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
            "net_moic": "Net MOIC",
            "csl_participation_amount": "Capital Deployed",
            "total_invested": "Total Cost Basis",
            "total_paid": "Total Paid",
            "net_balance": "Net Balance",
            "roi_with_grade": "ROI",
            "payment_perf_with_grade": "Payment Perf",
            "remaining_maturity_months": "Months Left",
            "is_past_maturity": "Past Maturity",
            "irr_with_grade": "IRR (Paid Off)",
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

    with tabs[5]:
        render_watchlist_tab(filtered_df)

    with tabs[6]:
        render_portfolio_insights_tab(filtered_df)


if __name__ == "__main__":
    main()
