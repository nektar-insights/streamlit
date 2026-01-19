# pages/performance_analysis.py
"""
Performance Analysis Dashboard - Industry, FICO, TIB, and Correlation Analysis

This page provides comprehensive borrower quality analytics including:
- Industry performance by NAICS sector
- FICO score band analysis
- Time in Business (TIB) analysis
- Metric correlation analysis
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
    GRADE_INDICATORS,
)
from utils.data_loader import (
    load_loan_summaries,
    load_deals,
)

# Loan tape specific utilities
from utils.loan_tape_data import (
    prepare_loan_data,
    calculate_irr,
    add_performance_grades,
)
from utils.loan_tape_analytics import (
    PROBLEM_STATUSES,
)
from utils.status_constants import (
    STATUS_COLORS,
)
from utils.display_components import (
    create_date_range_filter,
    create_partner_source_filter,
    create_status_filter,
)

# ---------------------------
# Page Configuration & Styles
# ---------------------------
setup_page("CSL Capital | Performance Analysis")


# ---------------------------
# Analysis Functions
# ---------------------------

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
        st.altair_chart(perf_chart, width="stretch")

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
        st.altair_chart(return_chart, width="stretch")

    st.subheader("Industry Performance Summary")
    display_df = industry_metrics.copy()
    display_df["outstanding_balance"] = display_df["outstanding_balance"].map(lambda x: f"${x:,.0f}")
    display_df["pct_of_total_outstanding"] = display_df["pct_of_total_outstanding"].map(lambda x: f"{x:.1%}")
    display_df["avg_payment_performance"] = display_df["avg_payment_performance"].map(lambda x: f"{x:.1%}")
    display_df["actual_return_rate"] = display_df["actual_return_rate"].map(lambda x: f"{x:.1%}")
    display_df = display_df[["display_label", "deal_count", "outstanding_balance", "pct_of_total_outstanding", "avg_payment_performance", "actual_return_rate"]]
    display_df.columns = ["Industry (NAICS 2-Digit)", "Loan Count", "Outstanding Balance", "% of Total Outstanding", "Avg Payment Performance", "Actual Return Rate"]
    st.dataframe(display_df, width="stretch", hide_index=True)

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

                st.altair_chart(industry_donut + center_text, width="stretch")

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

                st.altair_chart(status_donut + center_text2, width="stretch")

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

            st.altair_chart(stacked_bar, width="stretch")


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
        st.altair_chart(perf_chart, width="stretch")

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
        st.altair_chart(problem_chart, width="stretch")

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
    st.altair_chart(return_chart, width="stretch")

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
    st.dataframe(display_df, width="stretch", hide_index=True)


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
        st.altair_chart(perf_chart, width="stretch")

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
        st.altair_chart(problem_chart, width="stretch")

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
    st.altair_chart(return_chart, width="stretch")

    st.subheader("TIB Performance Summary")
    display_df = tib_metrics.copy()
    display_df["outstanding_balance"] = display_df["outstanding_balance"].map(lambda x: f"${x:,.0f}")
    display_df["avg_payment_performance"] = display_df["avg_payment_performance"].map(lambda x: f"{x:.1%}")
    display_df["actual_return_rate"] = display_df["actual_return_rate"].map(lambda x: f"{x:.1%}")
    display_df["problem_rate"] = display_df["problem_rate"].map(lambda x: f"{x:.1%}")
    display_df = display_df[["tib_band", "deal_count", "outstanding_balance", "avg_payment_performance", "actual_return_rate", "problem_rate"]]
    display_df.columns = ["TIB Band", "Loan Count", "Outstanding Balance", "Avg Payment Performance", "Actual Return Rate", "Problem Loan Rate"]
    st.dataframe(display_df, width="stretch", hide_index=True)


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

    st.altair_chart(scatter + regression_line, width="stretch")

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


# ---------------------------
# Main Application
# ---------------------------
def main():
    st.title("Performance Analysis")
    st.caption("Borrower quality analysis by industry, FICO score, and time in business")

    # Load data
    loan_summaries_df = load_loan_summaries()
    deals_df = load_deals()

    if loan_summaries_df.empty or deals_df.empty:
        st.error("Unable to load loan data. Please check the data source.")
        return

    # Prepare loan data
    df = prepare_loan_data(loan_summaries_df, deals_df)

    if df.empty:
        st.warning("No loan data available after preparation.")
        return

    # Calculate IRR and add grades
    df = calculate_irr(df)
    df = add_performance_grades(df)

    # Sidebar filters
    st.sidebar.header("Filters")

    # Date filter
    if "funding_date" in df.columns:
        df["funding_date"] = pd.to_datetime(df["funding_date"], errors="coerce")
        valid_dates = df[df["funding_date"].notna()]
        if not valid_dates.empty:
            min_date = valid_dates["funding_date"].min()
            max_date = valid_dates["funding_date"].max()
            date_range = st.sidebar.date_input(
                "Funding Date Range",
                value=(min_date, max_date),
                min_value=min_date,
                max_value=max_date
            )
            if len(date_range) == 2:
                df = df[
                    (df["funding_date"] >= pd.Timestamp(date_range[0])) &
                    (df["funding_date"] <= pd.Timestamp(date_range[1]))
                ]

    # Partner filter
    if "partner_source" in df.columns:
        partners = sorted(df["partner_source"].dropna().unique())
        selected_partners = st.sidebar.multiselect(
            "Partner Source",
            options=partners,
            default=partners
        )
        if selected_partners:
            df = df[df["partner_source"].isin(selected_partners)]

    # Status filter
    if "loan_status" in df.columns:
        statuses = sorted(df["loan_status"].dropna().unique())
        selected_statuses = st.sidebar.multiselect(
            "Loan Status",
            options=statuses,
            default=statuses
        )
        if selected_statuses:
            df = df[df["loan_status"].isin(selected_statuses)]

    st.sidebar.markdown("---")
    st.sidebar.write(f"**Showing:** {len(df)} loans")

    # Display analysis sections
    plot_industry_performance_analysis(df)

    st.markdown("---")
    plot_fico_performance_analysis(df)

    st.markdown("---")
    plot_tib_performance_analysis(df)

    st.markdown("---")
    plot_metric_correlations(df)


main()
