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
from utils.loan_tape_ml import (
    train_classification_small,
    train_regression_small,
    render_corr_outputs,
    render_fico_tib_heatmap,
)

# ---------------------------
# Page Configuration & Styles
# ---------------------------
setup_page("CSL Capital | Loan Tape")

# -------------
# Constants
# -------------
PLATFORM_FEE = PLATFORM_FEE_RATE

LOAN_STATUS_COLORS = {
    "Active": "#2ca02c",
    "Late": "#ffbb78",
    "Default": "#ff7f0e",
    "Paid Off": "#1f77b4",
    "Bankrupt": "#d62728",
}

STATUS_RISK_MULTIPLIERS = {
    "Active": 1.0,
    "Active - Frequently Late": 1.3,
    "Minor Delinquency": 1.5,
    "Past Delinquency": 1.2,
    "Moderate Delinquency": 2.0,
    "Late": 2.5,
    "Severe Delinquency": 3.0,
    "Default": 4.0,
    "Bankrupt": 5.0,
    "Severe": 5.0,
    "Paid Off": 0.0,
}

PROBLEM_STATUSES = {
    "Late", "Default", "Bankrupt", "Severe", "Severe Delinquency",
    "Moderate Delinquency", "Active - Frequently Late"
}

# -------------------
# Page-Specific Visualizations
# -------------------

def plot_status_distribution(df: pd.DataFrame):
    """Plot loan status distribution"""
    status_counts = df["loan_status"].value_counts().reset_index()
    status_counts.columns = ["status", "count"]

    chart = alt.Chart(status_counts).mark_bar().encode(
        x=alt.X("status:N", title="Loan Status", sort="-y"),
        y=alt.Y("count:Q", title="Number of Loans"),
        color=alt.Color(
            "status:N",
            scale=alt.Scale(
                domain=list(LOAN_STATUS_COLORS.keys()),
                range=list(LOAN_STATUS_COLORS.values())
            ),
            legend=None
        ),
        tooltip=[
            alt.Tooltip("status:N", title="Status"),
            alt.Tooltip("count:Q", title="Count"),
        ]
    ).properties(width=600, height=350)

    st.altair_chart(chart, use_container_width=True)


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

    st.altair_chart(chart, use_container_width=True)


def plot_capital_flow(df: pd.DataFrame):
    """Plot capital deployment vs returns over time"""
    st.subheader("Capital Flow: Deployment vs. Returns")

    schedules = load_loan_schedules()

    d = df.copy()
    d["funding_date"] = pd.to_datetime(d["funding_date"], errors="coerce").dt.tz_localize(None)

    total_deployed = d["csl_participation_amount"].sum()
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
        unified["date"] = unified.index

        st.caption(
            f"Chart shows: Deployed ${unified['capital_deployed'].iloc[-1]:,.0f} | "
            f"Returned ${unified['capital_returned'].iloc[-1]:,.0f}"
        )

        plot_df = pd.concat([
            pd.DataFrame({"date": unified.index, "amount": unified["capital_deployed"].values, "series": "Capital Deployed"}),
            pd.DataFrame({"date": unified.index, "amount": unified["capital_returned"].values, "series": "Capital Returned"})
        ], ignore_index=True)

        chart = alt.Chart(plot_df).mark_line().encode(
            x=alt.X("date:T", title="Date", axis=alt.Axis(format="%b %Y")),
            y=alt.Y("amount:Q", title="Cumulative Amount ($)", axis=alt.Axis(format="$,.0f")),
            color=alt.Color(
                "series:N",
                scale=alt.Scale(domain=["Capital Deployed", "Capital Returned"], range=["#ff7f0e", "#2ca02c"]),
                legend=alt.Legend(title="Capital Flow")
            ),
            tooltip=[
                alt.Tooltip("date:T", title="Date", format="%Y-%m-%d"),
                alt.Tooltip("amount:Q", title="Amount", format="$,.0f"),
                alt.Tooltip("series:N", title="Type"),
            ],
        ).properties(width=800, height=400, title="Capital Deployed vs. Capital Returned Over Time")

        st.altair_chart(chart, use_container_width=True)
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
        unified["date"] = unified.index

        plot_df = pd.concat([
            pd.DataFrame({"date": unified.index, "amount": unified["cum_deployed"].values, "Type": "Cumulative Deployed"}),
            pd.DataFrame({"date": unified.index, "amount": unified["cum_returned"].values, "Type": "Cumulative Returned"}),
            pd.DataFrame({"date": unified.index, "amount": unified["net_position"].values, "Type": "Net Position"}),
        ], ignore_index=True)

        chart = alt.Chart(plot_df).mark_line().encode(
            x=alt.X("date:T", title="Date"),
            y=alt.Y("amount:Q", title="Amount ($)", axis=alt.Axis(format="$,.0f")),
            color=alt.Color(
                "Type:N",
                scale=alt.Scale(
                    domain=["Cumulative Deployed", "Cumulative Returned", "Net Position"],
                    range=["#ff7f0e", "#2ca02c", "#1f77b4"]
                ),
            ),
            tooltip=[
                alt.Tooltip("date:T", title="Date", format="%Y-%m-%d"),
                alt.Tooltip("amount:Q", title="Amount", format="$,.2f"),
                alt.Tooltip("Type:N", title="Metric"),
            ],
        ).properties(width=800, height=500, title="Portfolio Net Position Over Time")

        zero_line = alt.Chart(pd.DataFrame({"y": [0]})).mark_rule(strokeDash=[2, 2], color="gray", strokeWidth=1).encode(y="y:Q")

        st.altair_chart(chart + zero_line, use_container_width=True)
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

    st.altair_chart(target_zone + bars + text + ref_line, use_container_width=True)
    st.caption("On-Target Zone: -5% to +5%. Positive = ahead of schedule, negative = behind schedule.")


def plot_industry_performance_analysis(df: pd.DataFrame):
    """Plot performance metrics by industry"""
    st.header("Industry Performance Analysis")

    if "industry" not in df.columns or df["industry"].isna().all():
        st.warning("Industry data not available.")
        return

    industry_metrics = df.groupby("industry").agg(
        deal_count=("loan_id", "count"),
        capital_deployed=("csl_participation_amount", "sum"),
        outstanding_balance=("net_balance", "sum"),
        avg_payment_performance=("payment_performance", "mean"),
        total_paid=("total_paid", "sum"),
        total_invested=("total_invested", "sum"),
    ).reset_index()

    industry_metrics = industry_metrics[industry_metrics["deal_count"] >= 3]  # Filter for significance
    industry_metrics["actual_return_rate"] = industry_metrics["total_paid"] / industry_metrics["total_invested"]
    industry_metrics = industry_metrics.sort_values("capital_deployed", ascending=False).head(15)

    col1, col2 = st.columns(2)

    with col1:
        perf_chart = alt.Chart(industry_metrics).mark_bar().encode(
            x=alt.X("industry:N", title="Industry", sort="-y"),
            y=alt.Y("avg_payment_performance:Q", title="Avg Payment Performance", axis=alt.Axis(format=".0%")),
            color=alt.Color(
                "avg_payment_performance:Q",
                scale=alt.Scale(domain=[0.5, 0.8, 1.0], range=["#d62728", "#ffbb78", "#2ca02c"]),
                legend=None,
            ),
            tooltip=[
                alt.Tooltip("industry:N", title="Industry"),
                alt.Tooltip("avg_payment_performance:Q", title="Avg Payment Performance", format=".1%"),
                alt.Tooltip("deal_count:Q", title="Loan Count"),
            ],
        ).properties(width=350, height=400, title="Payment Performance by Industry (Top 15)")
        st.altair_chart(perf_chart, use_container_width=True)

    with col2:
        return_chart = alt.Chart(industry_metrics).mark_bar().encode(
            x=alt.X("industry:N", title="Industry", sort="-y"),
            y=alt.Y("actual_return_rate:Q", title="Actual Return Rate", axis=alt.Axis(format=".0%")),
            color=alt.Color(
                "actual_return_rate:Q",
                scale=alt.Scale(domain=[0.5, 1.0, 1.3], range=["#d62728", "#ffbb78", "#2ca02c"]),
                legend=None,
            ),
            tooltip=[
                alt.Tooltip("industry:N", title="Industry"),
                alt.Tooltip("actual_return_rate:Q", title="Return Rate", format=".2%"),
                alt.Tooltip("deal_count:Q", title="Loan Count"),
            ],
        ).properties(width=350, height=400, title="Return Rate by Industry (Top 15)")
        st.altair_chart(return_chart, use_container_width=True)

    st.subheader("Industry Performance Summary")
    display_df = industry_metrics.copy()
    display_df["outstanding_balance"] = display_df["outstanding_balance"].map(lambda x: f"${x:,.0f}")
    display_df["avg_payment_performance"] = display_df["avg_payment_performance"].map(lambda x: f"{x:.1%}")
    display_df["actual_return_rate"] = display_df["actual_return_rate"].map(lambda x: f"{x:.2%}")
    display_df = display_df[["industry", "deal_count", "outstanding_balance", "avg_payment_performance", "actual_return_rate"]]
    display_df.columns = ["Industry", "Loan Count", "Outstanding Balance", "Avg Payment Performance", "Actual Return Rate"]
    st.dataframe(display_df, use_container_width=True, hide_index=True)


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
        st.altair_chart(perf_chart, use_container_width=True)

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
        st.altair_chart(problem_chart, use_container_width=True)

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
            alt.Tooltip("actual_return_rate:Q", title="Return Rate", format=".2%"),
            alt.Tooltip("deal_count:Q", title="Loan Count"),
        ],
    ).properties(width=700, height=300, title="Actual Return Rate by FICO Score")
    st.altair_chart(return_chart, use_container_width=True)

    st.subheader("FICO Performance Summary")
    display_df = fico_metrics.copy()
    display_df["outstanding_balance"] = display_df["outstanding_balance"].map(lambda x: f"${x:,.0f}")
    display_df["avg_payment_performance"] = display_df["avg_payment_performance"].map(lambda x: f"{x:.1%}")
    display_df["actual_return_rate"] = display_df["actual_return_rate"].map(lambda x: f"{x:.2%}")
    display_df["problem_rate"] = display_df["problem_rate"].map(lambda x: f"{x:.1%}")
    display_df = display_df[[
        "fico_band", "deal_count", "outstanding_balance",
        "avg_payment_performance", "actual_return_rate", "problem_rate"
    ]]
    display_df.columns = ["FICO Band", "Loan Count", "Outstanding Balance", "Avg Payment Performance", "Actual Return Rate", "Problem Loan Rate"]
    st.dataframe(display_df, use_container_width=True, hide_index=True)


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
        st.altair_chart(perf_chart, use_container_width=True)

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
        st.altair_chart(problem_chart, use_container_width=True)

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
            alt.Tooltip("actual_return_rate:Q", title="Return Rate", format=".2%"),
            alt.Tooltip("deal_count:Q", title="Loan Count"),
        ],
    ).properties(width=700, height=300, title="Actual Return Rate by Time in Business")
    st.altair_chart(return_chart, use_container_width=True)

    st.subheader("TIB Performance Summary")
    display_df = tib_metrics.copy()
    display_df["outstanding_balance"] = display_df["outstanding_balance"].map(lambda x: f"${x:,.0f}")
    display_df["avg_payment_performance"] = display_df["avg_payment_performance"].map(lambda x: f"{x:.1%}")
    display_df["actual_return_rate"] = display_df["actual_return_rate"].map(lambda x: f"{x:.2%}")
    display_df["problem_rate"] = display_df["problem_rate"].map(lambda x: f"{x:.1%}")
    display_df = display_df[["tib_band", "deal_count", "outstanding_balance", "avg_payment_performance", "actual_return_rate", "problem_rate"]]
    display_df.columns = ["TIB Band", "Loan Count", "Outstanding Balance", "Avg Payment Performance", "Actual Return Rate", "Problem Loan Rate"]
    st.dataframe(display_df, use_container_width=True, hide_index=True)


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
        st.metric("Simple Avg Realized IRR", f"{avg_realized_irr:.2%}" if pd.notnull(avg_realized_irr) else "N/A")


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
            alt.Tooltip("avg_irr:Q", title="Avg IRR", format=".2%"),
            alt.Tooltip("loan_count:Q", title="Paid Off Loans"),
        ]
    ).properties(width=600, height=350, title="Average Realized IRR by Partner (Paid Off Loans, ≥2 loans)")

    st.altair_chart(chart, use_container_width=True)


# -----------
# Main Page
# -----------
def main():
    st.title("Loan Tape Dashboard")

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

    if "funding_date" in df.columns and not df["funding_date"].isna().all():
        min_date = df["funding_date"].min().date()
        max_date = df["funding_date"].max().date()

        use_date_filter = st.sidebar.checkbox("Filter by Funding Date", value=False)

        if use_date_filter:
            date_range = st.sidebar.date_input(
                "Select Date Range", [min_date, max_date], min_value=min_date, max_value=max_date
            )
            if isinstance(date_range, (list, tuple)) and len(date_range) == 2:
                filtered_df = df[
                    (df["funding_date"].dt.date >= date_range[0]) &
                    (df["funding_date"].dt.date <= date_range[1])
                ].copy()
            else:
                filtered_df = df.copy()
        else:
            filtered_df = df.copy()
    else:
        filtered_df = df.copy()

    all_statuses = ["All"] + sorted(df["loan_status"].dropna().unique().tolist())
    selected_status = st.sidebar.selectbox("Filter by Status", all_statuses, index=0)

    if selected_status != "All":
        filtered_df = filtered_df[filtered_df["loan_status"] == selected_status]

    st.sidebar.markdown("---")
    st.sidebar.write(f"**Showing:** {len(filtered_df)} of {len(df)} loans")

    # Tabs
    tabs = st.tabs(["Summary", "Capital Flow", "Performance Analysis", "Risk Analytics", "Loan Tape", "Diagnostics & ML"])

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
            st.dataframe(top_risk_display, use_container_width=True, hide_index=True)

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

                st.altair_chart(risk_bar, use_container_width=True)
        else:
            st.info("No active loans to calculate risk scores.")

    with tabs[4]:
        st.header("Complete Loan Tape")

        display_columns = [
            "loan_id", "deal_name", "partner_source", "loan_status",
            "funding_date", "maturity_date",
            "csl_participation_amount", "total_invested", "total_paid", "net_balance",
            "current_roi", "payment_performance", "remaining_maturity_months",
        ]

        column_rename = {
            "loan_id": "Loan ID",
            "deal_name": "Deal Name",
            "partner_source": "Partner",
            "loan_status": "Status",
            "funding_date": "Funded",
            "maturity_date": "Maturity",
            "csl_participation_amount": "Capital Deployed",
            "total_invested": "Total Invested",
            "total_paid": "Total Paid",
            "net_balance": "Net Balance",
            "current_roi": "ROI",
            "payment_performance": "Payment Perf",
            "remaining_maturity_months": "Months Left",
        }

        loan_tape = format_dataframe_for_display(filtered_df, display_columns, column_rename)
        st.dataframe(loan_tape, use_container_width=True, hide_index=True)

        csv = loan_tape.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="Download Loan Tape as CSV",
            data=csv,
            file_name=f"loan_tape_{pd.Timestamp.today().strftime('%Y%m%d')}.csv",
            mime="text/csv",
        )

    with tabs[5]:
        st.header("Diagnostics & ML")

        st.markdown("##### Correlations")
        render_corr_outputs(filtered_df)

        st.markdown("---")
        render_fico_tib_heatmap(filtered_df)

        st.markdown("---")
        st.subheader("Small Classification Model: Predict Problem Loans")
        try:
            model, metrics, top_pos, top_neg = train_classification_small(filtered_df)
            col1, col2, col3, col4 = st.columns(4)
            with col1: st.metric("ROC AUC (CV)", f"{metrics['ROC AUC'][0]:.3f}" if pd.notnull(metrics['ROC AUC'][0]) else "N/A")
            with col2: st.metric("Precision (CV)", f"{metrics['Precision'][0]:.3f}")
            with col3: st.metric("Recall (CV)", f"{metrics['Recall'][0]:.3f}")
            with col4: st.caption(f"n={metrics['n_samples']}, positive rate={metrics['pos_rate']:.2f}")

            st.write("**Top Risk-Increasing Signals (coefficients)**")
            st.dataframe(top_pos.assign(coef=lambda s: s["coef"].map(lambda x: f"{x:.3f}")), use_container_width=True, hide_index=True)
            st.write("**Top Risk-Decreasing Signals (coefficients)**")
            st.dataframe(top_neg.assign(coef=lambda s: s["coef"].map(lambda x: f"{x:.3f}")), use_container_width=True, hide_index=True)
        except ImportError:
            st.warning("scikit-learn or scipy not installed. `pip install scikit-learn scipy` to enable modeling.")
        except Exception as e:
            st.warning(f"Classification model could not run: {e}")

        st.markdown("---")
        st.subheader("Small Regression Model: Predict Payment Performance")
        try:
            r_model, r_metrics = train_regression_small(filtered_df)
            c1, c2 = st.columns(2)
            with c1: st.metric("R² (CV)", f"{r_metrics['R2'][0]:.3f}" if pd.notnull(r_metrics['R2'][0]) else "N/A")
            with c2: st.metric("RMSE (CV)", f"{r_metrics['RMSE'][0]:.3f}")
            st.caption(f"n={r_metrics['n_samples']} (rows with non-null payment_performance)")
        except ImportError:
            st.warning("scikit-learn not installed. `pip install scikit-learn`.")
        except Exception as e:
            st.warning(f"Regression model could not run: {e}")


if __name__ == "__main__":
    main()
