# pages/portfolio_insights.py
"""
Portfolio Insights Dashboard - Executive-level portfolio health analysis

This page provides high-level portfolio insights answering:
- Are we making money? (Realized MOIC, Problem Rate)
- Where are the problems? (Winners vs Losers, Top Problem Loans)
- What's our concentration risk? (Partner/Industry Exposure)
- How are vintages performing? (Vintage Analysis)
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
)
from utils.data_loader import (
    load_loan_summaries,
    load_deals,
    get_last_updated,
)
from utils.loan_tape_data import prepare_loan_data
from utils.status_constants import PROBLEM_STATUSES, STATUS_COLORS

# ---------------------------
# Page Configuration
# ---------------------------
setup_page("CSL Capital | Portfolio Insights")

# ---------------------------
# Target Thresholds
# ---------------------------
TARGET_MOIC = 1.20
TARGET_PROBLEM_RATE = 0.10  # 10%
PARTNER_HIGH_CONCENTRATION = 0.50  # 50% = red alert
PARTNER_MEDIUM_CONCENTRATION = 0.30  # 30% = yellow alert
INDUSTRY_HIGH_CONCENTRATION = 0.20  # 20% = red alert


# ---------------------------
# Helper Functions
# ---------------------------
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


def format_delta_color(value: float, target: float, higher_is_better: bool = True) -> str:
    """Return 'normal' or 'inverse' for st.metric delta_color based on comparison."""
    if higher_is_better:
        return "normal" if value >= target else "inverse"
    else:
        return "normal" if value <= target else "inverse"


# ---------------------------
# Section 1: Executive Summary
# ---------------------------
def render_executive_summary(df: pd.DataFrame):
    """Render executive summary metrics at top of page."""
    st.header("Executive Summary")

    # Calculate key metrics
    realized_moic = calculate_moic(df)
    problem_df = get_problem_loans(df)
    problem_count = len(problem_df)
    total_count = len(df)
    problem_rate = problem_count / total_count if total_count > 0 else 0
    capital_at_risk = problem_df["net_balance"].sum() if not problem_df.empty else 0
    total_invested = df["total_invested"].sum()
    total_returned = df["total_paid"].sum()

    # Display metrics in columns
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

    # Status summary
    st.markdown("---")

    # Overall health assessment
    if realized_moic >= TARGET_MOIC and problem_rate <= TARGET_PROBLEM_RATE:
        st.success("Portfolio is performing above targets on both MOIC and problem rate.")
    elif realized_moic >= TARGET_MOIC:
        st.warning(f"MOIC is on target, but problem rate ({problem_rate:.1%}) exceeds {TARGET_PROBLEM_RATE:.0%} threshold.")
    elif problem_rate <= TARGET_PROBLEM_RATE:
        st.warning(f"Problem rate is acceptable, but MOIC ({realized_moic:.2f}x) is below {TARGET_MOIC:.2f}x target.")
    else:
        st.error(f"Both MOIC ({realized_moic:.2f}x) and problem rate ({problem_rate:.1%}) are below targets.")


# ---------------------------
# Section 2: Winners vs Losers Analysis
# ---------------------------
def render_winners_vs_losers(df: pd.DataFrame):
    """Compare characteristics of paid off loans vs problem loans."""
    st.header("Winners vs Losers Analysis")
    st.caption("Comparing characteristics of Paid Off loans (Winners) vs Problem loans (Losers)")

    paid_off = get_paid_off_loans(df)
    problem = get_problem_loans(df)

    if paid_off.empty and problem.empty:
        st.info("Not enough data for winners vs losers comparison.")
        return

    # Calculate average metrics for each group
    def calc_group_metrics(group_df: pd.DataFrame, group_name: str) -> dict:
        if group_df.empty:
            return {
                "Group": group_name,
                "Count": 0,
                "Avg FICO": None,
                "Avg TIB (Years)": None,
                "Avg Deal Size": None,
                "Avg Position": None,
                "Avg Factor Rate": None,
            }
        return {
            "Group": group_name,
            "Count": len(group_df),
            "Avg FICO": group_df["fico"].mean() if "fico" in group_df.columns else None,
            "Avg TIB (Years)": group_df["tib"].mean() if "tib" in group_df.columns else None,
            "Avg Deal Size": group_df["csl_participation_amount"].mean() if "csl_participation_amount" in group_df.columns else None,
            "Avg Position": group_df["ahead_positions"].mean() if "ahead_positions" in group_df.columns else None,
            "Avg Factor Rate": group_df["factor_rate"].mean() if "factor_rate" in group_df.columns else None,
        }

    winners_metrics = calc_group_metrics(paid_off, "Winners (Paid Off)")
    losers_metrics = calc_group_metrics(problem, "Losers (Problem)")

    comparison_df = pd.DataFrame([winners_metrics, losers_metrics])

    # Display comparison in columns
    col1, col2 = st.columns(2)

    with col1:
        st.subheader(f"Winners (Paid Off): {len(paid_off)}")
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
        st.subheader(f"Losers (Problem): {len(problem)}")
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

    # Highlight significant differences
    st.markdown("---")
    st.subheader("Key Differences")

    differences = []

    # FICO comparison
    if not paid_off.empty and not problem.empty:
        winner_fico = paid_off["fico"].mean() if "fico" in paid_off.columns and paid_off["fico"].notna().any() else None
        loser_fico = problem["fico"].mean() if "fico" in problem.columns and problem["fico"].notna().any() else None
        if winner_fico and loser_fico:
            fico_diff = winner_fico - loser_fico
            if abs(fico_diff) >= 20:
                if fico_diff > 0:
                    differences.append(f"Winners have **{fico_diff:.0f} higher FICO** on average ({winner_fico:.0f} vs {loser_fico:.0f})")
                else:
                    differences.append(f"Losers have **{abs(fico_diff):.0f} higher FICO** on average - unexpected pattern")

        # TIB comparison
        winner_tib = paid_off["tib"].mean() if "tib" in paid_off.columns and paid_off["tib"].notna().any() else None
        loser_tib = problem["tib"].mean() if "tib" in problem.columns and problem["tib"].notna().any() else None
        if winner_tib and loser_tib:
            tib_diff = winner_tib - loser_tib
            if abs(tib_diff) >= 2:
                if tib_diff > 0:
                    differences.append(f"Winners have **{tib_diff:.1f} more years in business** on average ({winner_tib:.1f} vs {loser_tib:.1f})")
                else:
                    differences.append(f"Losers have **{abs(tib_diff):.1f} more years in business** on average - unexpected pattern")

        # Deal size comparison
        winner_size = paid_off["csl_participation_amount"].mean() if "csl_participation_amount" in paid_off.columns else None
        loser_size = problem["csl_participation_amount"].mean() if "csl_participation_amount" in problem.columns else None
        if winner_size and loser_size:
            size_ratio = loser_size / winner_size if winner_size > 0 else 1
            if size_ratio > 1.25:
                differences.append(f"Problem loans are **{size_ratio:.1f}x larger** on average (${loser_size:,.0f} vs ${winner_size:,.0f})")
            elif size_ratio < 0.75:
                differences.append(f"Problem loans are **{1/size_ratio:.1f}x smaller** on average")

        # Position comparison
        winner_pos = paid_off["ahead_positions"].mean() if "ahead_positions" in paid_off.columns and paid_off["ahead_positions"].notna().any() else None
        loser_pos = problem["ahead_positions"].mean() if "ahead_positions" in problem.columns and problem["ahead_positions"].notna().any() else None
        if winner_pos is not None and loser_pos is not None:
            pos_diff = loser_pos - winner_pos
            if pos_diff > 0.3:
                differences.append(f"Problem loans have **{pos_diff:.1f} worse average position** ({loser_pos:.1f} vs {winner_pos:.1f})")

    if differences:
        for diff in differences:
            st.markdown(f"- {diff}")
    else:
        st.info("No significant differences detected between winners and losers on key metrics.")


# ---------------------------
# Section 3: Concentration Alerts
# ---------------------------
def render_concentration_alerts(df: pd.DataFrame):
    """Show concentration risk alerts for partners and industries."""
    st.header("Concentration Alerts")

    # Calculate total exposure (net_balance for active loans)
    active_df = df[df["loan_status"] != "Paid Off"].copy()
    total_exposure = active_df["net_balance"].sum()

    if total_exposure <= 0:
        st.info("No active exposure to analyze for concentration risk.")
        return

    alerts = []

    # Partner Concentration Analysis
    st.subheader("Partner Concentration")

    if "partner_source" in active_df.columns:
        partner_exposure = active_df.groupby("partner_source").agg(
            exposure=("net_balance", "sum"),
            deal_count=("loan_id", "count")
        ).reset_index()
        partner_exposure["pct_of_total"] = partner_exposure["exposure"] / total_exposure
        partner_exposure = partner_exposure.sort_values("exposure", ascending=False)

        # Check for concentration alerts
        for _, row in partner_exposure.iterrows():
            if row["pct_of_total"] >= PARTNER_HIGH_CONCENTRATION:
                alerts.append({
                    "type": "Partner",
                    "name": row["partner_source"],
                    "pct": row["pct_of_total"],
                    "exposure": row["exposure"],
                    "severity": "high"
                })
            elif row["pct_of_total"] >= PARTNER_MEDIUM_CONCENTRATION:
                alerts.append({
                    "type": "Partner",
                    "name": row["partner_source"],
                    "pct": row["pct_of_total"],
                    "exposure": row["exposure"],
                    "severity": "medium"
                })

        # Display partner breakdown chart
        partner_chart = alt.Chart(partner_exposure.head(10)).mark_bar().encode(
            x=alt.X("partner_source:N", title="Partner", sort="-y"),
            y=alt.Y("pct_of_total:Q", title="% of Total Exposure", axis=alt.Axis(format=".0%")),
            color=alt.condition(
                alt.datum.pct_of_total >= PARTNER_HIGH_CONCENTRATION,
                alt.value("#d62728"),  # Red for > 50%
                alt.condition(
                    alt.datum.pct_of_total >= PARTNER_MEDIUM_CONCENTRATION,
                    alt.value("#ff7f0e"),  # Orange for > 30%
                    alt.value(PRIMARY_COLOR)  # Green otherwise
                )
            ),
            tooltip=[
                alt.Tooltip("partner_source:N", title="Partner"),
                alt.Tooltip("pct_of_total:Q", title="% of Exposure", format=".1%"),
                alt.Tooltip("exposure:Q", title="Exposure", format="$,.0f"),
                alt.Tooltip("deal_count:Q", title="Deals"),
            ]
        ).properties(width=600, height=300, title="Partner Exposure (Top 10)")

        # Add threshold lines
        threshold_df = pd.DataFrame([
            {"threshold": PARTNER_HIGH_CONCENTRATION, "label": "50% High Risk"},
            {"threshold": PARTNER_MEDIUM_CONCENTRATION, "label": "30% Medium Risk"},
        ])

        threshold_lines = alt.Chart(threshold_df).mark_rule(strokeDash=[4, 4]).encode(
            y="threshold:Q",
            color=alt.condition(
                alt.datum.threshold == PARTNER_HIGH_CONCENTRATION,
                alt.value("#d62728"),
                alt.value("#ff7f0e")
            )
        )

        st.altair_chart(partner_chart + threshold_lines, use_container_width=True)

    # Industry Concentration Analysis
    st.subheader("Industry Concentration")

    if "sector_code" in active_df.columns or "industry_name" in active_df.columns:
        industry_col = "industry_name" if "industry_name" in active_df.columns else "sector_code"

        industry_exposure = active_df.groupby(industry_col).agg(
            exposure=("net_balance", "sum"),
            deal_count=("loan_id", "count")
        ).reset_index()
        industry_exposure["pct_of_total"] = industry_exposure["exposure"] / total_exposure
        industry_exposure = industry_exposure.sort_values("exposure", ascending=False)

        # Check for industry concentration alerts
        for _, row in industry_exposure.iterrows():
            if row["pct_of_total"] >= INDUSTRY_HIGH_CONCENTRATION:
                alerts.append({
                    "type": "Industry",
                    "name": row[industry_col],
                    "pct": row["pct_of_total"],
                    "exposure": row["exposure"],
                    "severity": "high"
                })

        # Display industry breakdown chart
        industry_chart = alt.Chart(industry_exposure.head(10)).mark_bar().encode(
            x=alt.X(f"{industry_col}:N", title="Industry", sort="-y"),
            y=alt.Y("pct_of_total:Q", title="% of Total Exposure", axis=alt.Axis(format=".0%")),
            color=alt.condition(
                alt.datum.pct_of_total >= INDUSTRY_HIGH_CONCENTRATION,
                alt.value("#d62728"),  # Red for > 20%
                alt.value(PRIMARY_COLOR)
            ),
            tooltip=[
                alt.Tooltip(f"{industry_col}:N", title="Industry"),
                alt.Tooltip("pct_of_total:Q", title="% of Exposure", format=".1%"),
                alt.Tooltip("exposure:Q", title="Exposure", format="$,.0f"),
                alt.Tooltip("deal_count:Q", title="Deals"),
            ]
        ).properties(width=600, height=300, title="Industry Exposure (Top 10)")

        # Add 20% threshold line
        industry_threshold = alt.Chart(pd.DataFrame([{"threshold": INDUSTRY_HIGH_CONCENTRATION}])).mark_rule(
            strokeDash=[4, 4], color="#d62728"
        ).encode(y="threshold:Q")

        st.altair_chart(industry_chart + industry_threshold, use_container_width=True)

    # Display concentration alerts
    st.markdown("---")
    st.subheader("Alert Summary")

    if alerts:
        for alert in alerts:
            if alert["severity"] == "high":
                st.error(f"**{alert['type']}**: {alert['name']} has {alert['pct']:.1%} of exposure (${alert['exposure']:,.0f})")
            else:
                st.warning(f"**{alert['type']}**: {alert['name']} has {alert['pct']:.1%} of exposure (${alert['exposure']:,.0f})")
    else:
        st.success("No concentration alerts. Portfolio is well diversified.")

    # Top 5 Problem Loans as % of Total Problem Exposure
    st.markdown("---")
    st.subheader("Top 5 Problem Loans")

    problem_df = get_problem_loans(active_df)
    if not problem_df.empty:
        total_problem_exposure = problem_df["net_balance"].sum()
        top_problems = problem_df.nlargest(5, "net_balance")[
            ["loan_id", "deal_name", "partner_source", "loan_status", "net_balance"]
        ].copy()
        top_problems["pct_of_problem"] = top_problems["net_balance"] / total_problem_exposure if total_problem_exposure > 0 else 0

        # Format for display
        display_df = top_problems.copy()
        display_df["net_balance"] = display_df["net_balance"].map(lambda x: f"${x:,.0f}")
        display_df["pct_of_problem"] = display_df["pct_of_problem"].map(lambda x: f"{x:.1%}")
        display_df.columns = ["Loan ID", "Deal Name", "Partner", "Status", "Net Balance", "% of Problem Exposure"]

        st.dataframe(display_df, hide_index=True, use_container_width=True)
        st.caption(f"Total problem exposure: ${total_problem_exposure:,.0f}")
    else:
        st.info("No problem loans in the active portfolio.")


# ---------------------------
# Section 4: Vintage Analysis
# ---------------------------
def render_vintage_analysis(df: pd.DataFrame):
    """Show problem rate by funding month."""
    st.header("Vintage Analysis")
    st.caption("Problem rate by funding month - highlights months with >20% problem rate")

    # Prepare vintage data
    if "funding_date" not in df.columns:
        st.warning("Funding date not available for vintage analysis.")
        return

    df_vintage = df.copy()
    df_vintage["funding_date"] = pd.to_datetime(df_vintage["funding_date"], errors="coerce")
    df_vintage = df_vintage[df_vintage["funding_date"].notna()]

    if df_vintage.empty:
        st.warning("No loans with valid funding dates for vintage analysis.")
        return

    df_vintage["funding_month"] = df_vintage["funding_date"].dt.to_period("M").astype(str)
    df_vintage["is_problem"] = df_vintage["loan_status"].isin(PROBLEM_STATUSES).astype(int)

    # Calculate problem rate by month
    vintage_metrics = df_vintage.groupby("funding_month").agg(
        total_loans=("loan_id", "count"),
        problem_loans=("is_problem", "sum"),
        total_invested=("csl_participation_amount", "sum"),
        total_returned=("total_paid", "sum"),
    ).reset_index()

    vintage_metrics["problem_rate"] = vintage_metrics["problem_loans"] / vintage_metrics["total_loans"]
    vintage_metrics["moic"] = vintage_metrics["total_returned"] / vintage_metrics["total_invested"]
    vintage_metrics = vintage_metrics.sort_values("funding_month")

    # Filter to last 24 months for readability
    vintage_metrics = vintage_metrics.tail(24)

    # Problem rate chart with highlighting for >20%
    problem_rate_chart = alt.Chart(vintage_metrics).mark_bar().encode(
        x=alt.X("funding_month:N", title="Funding Month", sort=None,
               axis=alt.Axis(labelAngle=-45)),
        y=alt.Y("problem_rate:Q", title="Problem Rate", axis=alt.Axis(format=".0%")),
        color=alt.condition(
            alt.datum.problem_rate >= 0.20,
            alt.value("#d62728"),  # Red for >20%
            alt.value(PRIMARY_COLOR)
        ),
        tooltip=[
            alt.Tooltip("funding_month:N", title="Month"),
            alt.Tooltip("problem_rate:Q", title="Problem Rate", format=".1%"),
            alt.Tooltip("total_loans:Q", title="Total Loans"),
            alt.Tooltip("problem_loans:Q", title="Problem Loans"),
        ]
    ).properties(width=800, height=350, title="Problem Rate by Vintage (Last 24 Months)")

    # Add 20% threshold line
    threshold_line = alt.Chart(pd.DataFrame([{"threshold": 0.20}])).mark_rule(
        strokeDash=[4, 4], color="#d62728", strokeWidth=2
    ).encode(y="threshold:Q")

    # Add 10% target line
    target_line = alt.Chart(pd.DataFrame([{"threshold": 0.10}])).mark_rule(
        strokeDash=[2, 2], color="#ff7f0e", strokeWidth=1
    ).encode(y="threshold:Q")

    st.altair_chart(problem_rate_chart + threshold_line + target_line, use_container_width=True)
    st.caption("Red bars indicate months with >20% problem rate. Orange dashed line is 10% target. Red dashed line is 20% alert threshold.")

    # Highlight problematic vintages
    high_problem_months = vintage_metrics[vintage_metrics["problem_rate"] >= 0.20]
    if not high_problem_months.empty:
        st.warning(f"**{len(high_problem_months)} vintage(s) with >20% problem rate:**")
        for _, row in high_problem_months.iterrows():
            st.markdown(f"- **{row['funding_month']}**: {row['problem_rate']:.1%} problem rate ({row['problem_loans']:.0f} of {row['total_loans']:.0f} loans)")

    # MOIC by vintage
    st.markdown("---")
    st.subheader("MOIC by Vintage")

    moic_chart = alt.Chart(vintage_metrics).mark_line(point=True).encode(
        x=alt.X("funding_month:N", title="Funding Month", sort=None,
               axis=alt.Axis(labelAngle=-45)),
        y=alt.Y("moic:Q", title="MOIC", axis=alt.Axis(format=".2f")),
        color=alt.value(PRIMARY_COLOR),
        tooltip=[
            alt.Tooltip("funding_month:N", title="Month"),
            alt.Tooltip("moic:Q", title="MOIC", format=".2f"),
            alt.Tooltip("total_invested:Q", title="Invested", format="$,.0f"),
            alt.Tooltip("total_returned:Q", title="Returned", format="$,.0f"),
        ]
    ).properties(width=800, height=300, title="MOIC by Vintage (Last 24 Months)")

    # Add 1.20x target line
    moic_target = alt.Chart(pd.DataFrame([{"target": TARGET_MOIC}])).mark_rule(
        strokeDash=[4, 4], color="#ff7f0e", strokeWidth=2
    ).encode(y="target:Q")

    # Add 1.0x break-even line
    breakeven = alt.Chart(pd.DataFrame([{"target": 1.0}])).mark_rule(
        strokeDash=[2, 2], color="#d62728", strokeWidth=1
    ).encode(y="target:Q")

    st.altair_chart(moic_chart + moic_target + breakeven, use_container_width=True)
    st.caption(f"Orange dashed line is {TARGET_MOIC:.2f}x target. Red dashed line is 1.0x break-even.")


# ---------------------------
# Section 5: Partner Performance Table
# ---------------------------
def render_partner_performance(df: pd.DataFrame):
    """Show partner performance summary table."""
    st.header("Partner Performance")

    if "partner_source" not in df.columns:
        st.warning("Partner source data not available.")
        return

    # Calculate metrics by partner
    def calc_partner_metrics(group):
        total_invested = group["csl_participation_amount"].sum()
        total_paid = group["total_paid"].sum()
        problem_count = group["loan_status"].isin(PROBLEM_STATUSES).sum()
        total_count = len(group)

        # Calculate average recovery for paid off loans
        paid_off = group[group["loan_status"] == "Paid Off"]
        avg_recovery = paid_off["total_paid"].sum() / paid_off["total_invested"].sum() if not paid_off.empty and paid_off["total_invested"].sum() > 0 else None

        return pd.Series({
            "deal_count": total_count,
            "capital_deployed": total_invested,
            "capital_returned": total_paid,
            "problem_count": problem_count,
            "problem_rate": problem_count / total_count if total_count > 0 else 0,
            "moic": total_paid / total_invested if total_invested > 0 else 0,
            "avg_recovery": avg_recovery,
        })

    partner_metrics = df.groupby("partner_source").apply(calc_partner_metrics).reset_index()
    partner_metrics = partner_metrics.sort_values("capital_deployed", ascending=False)

    # Format for display
    display_df = partner_metrics.copy()
    display_df["capital_deployed"] = display_df["capital_deployed"].map(lambda x: f"${x:,.0f}")
    display_df["capital_returned"] = display_df["capital_returned"].map(lambda x: f"${x:,.0f}")
    display_df["problem_rate"] = display_df["problem_rate"].map(lambda x: f"{x:.1%}")
    display_df["moic"] = display_df["moic"].map(lambda x: f"{x:.2f}x")
    display_df["avg_recovery"] = display_df["avg_recovery"].map(lambda x: f"{x:.1%}" if pd.notna(x) else "N/A")
    display_df["deal_count"] = display_df["deal_count"].astype(int)
    display_df["problem_count"] = display_df["problem_count"].astype(int)

    display_df.columns = [
        "Partner", "Deal Count", "Capital Deployed", "Capital Returned",
        "Problem Loans", "Problem Rate", "MOIC", "Avg Recovery (Paid Off)"
    ]

    st.dataframe(display_df, hide_index=True, use_container_width=True)

    # Highlight best and worst performers
    st.markdown("---")

    if len(partner_metrics) >= 2:
        # Filter to partners with meaningful volume (at least 5 deals)
        significant_partners = partner_metrics[partner_metrics["deal_count"] >= 5]

        if not significant_partners.empty:
            best_moic = significant_partners.loc[significant_partners["moic"].idxmax()]
            worst_problem = significant_partners.loc[significant_partners["problem_rate"].idxmax()]

            col1, col2 = st.columns(2)

            with col1:
                st.success(f"**Best MOIC**: {best_moic['partner_source']} ({best_moic['moic']:.2f}x)")
                st.caption(f"{int(best_moic['deal_count'])} deals, ${best_moic['capital_deployed']:,.0f} deployed")

            with col2:
                if worst_problem["problem_rate"] > TARGET_PROBLEM_RATE:
                    st.error(f"**Highest Problem Rate**: {worst_problem['partner_source']} ({worst_problem['problem_rate']:.1%})")
                else:
                    st.info(f"**Highest Problem Rate**: {worst_problem['partner_source']} ({worst_problem['problem_rate']:.1%})")
                st.caption(f"{int(worst_problem['deal_count'])} deals, {int(worst_problem['problem_count'])} problems")


# ---------------------------
# Main Page
# ---------------------------
def main():
    st.title("Portfolio Insights Dashboard")
    st.caption("Executive-level view of portfolio health and performance")

    # Load data
    last_updated = get_last_updated()
    st.caption(f"Data last updated: {last_updated}")

    loans_df = load_loan_summaries()
    deals_df = load_deals()

    if loans_df.empty:
        st.error("No loan data available. Please check your data connection.")
        return

    # Prepare data
    df = prepare_loan_data(loans_df, deals_df)

    st.sidebar.header("Quick Stats")
    st.sidebar.metric("Total Loans", len(df))
    st.sidebar.metric("Active Loans", len(df[df["loan_status"] != "Paid Off"]))
    st.sidebar.metric("Problem Loans", len(get_problem_loans(df)))

    # Render all sections
    render_executive_summary(df)

    st.markdown("---")
    render_winners_vs_losers(df)

    st.markdown("---")
    render_concentration_alerts(df)

    st.markdown("---")
    render_vintage_analysis(df)

    st.markdown("---")
    render_partner_performance(df)


if __name__ == "__main__":
    main()
