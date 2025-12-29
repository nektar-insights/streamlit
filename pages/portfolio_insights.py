# pages/portfolio_insights.py
"""
Portfolio Insights Dashboard - Executive-level portfolio analytics

This page provides high-level insights for management including:
- Executive summary with key metrics (MOIC, Problem Rate, Capital at Risk)
- Winners vs Losers analysis comparing successful vs problem loans
- Concentration risk alerts (partner, industry, top problem loans)
- Vintage analysis showing problem rates by funding month
- Partner performance table with key metrics
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
)

# Loan tape specific utilities
from utils.loan_tape_data import prepare_loan_data
from utils.status_constants import PROBLEM_STATUSES, STATUS_COLORS

# ---------------------------
# Page Configuration
# ---------------------------
setup_page("CSL Capital | Portfolio Insights")

# ---------------------------
# Constants
# ---------------------------
MOIC_TARGET = 1.20
PROBLEM_RATE_TARGET = 0.10  # 10%

# Alert thresholds for concentration
PARTNER_RED_THRESHOLD = 0.50  # > 50% is red
PARTNER_YELLOW_THRESHOLD = 0.30  # > 30% is yellow
INDUSTRY_THRESHOLD = 0.20  # > 20% is flagged

# Colors for charts
SUCCESS_COLOR = "#28a745"  # Green
WARNING_COLOR = "#ffc107"  # Yellow
DANGER_COLOR = "#dc3545"  # Red
NEUTRAL_COLOR = "#6c757d"  # Gray


# ---------------------------
# Data Loading
# ---------------------------
@st.cache_data(ttl=3600)
def load_portfolio_data():
    """Load and prepare loan portfolio data."""
    loans_df = load_loan_summaries()
    deals_df = load_deals()

    if loans_df.empty:
        return pd.DataFrame()

    df = prepare_loan_data(loans_df, deals_df)
    return df


# ---------------------------
# Metric Calculations
# ---------------------------
def calculate_executive_metrics(df: pd.DataFrame) -> dict:
    """Calculate executive summary metrics."""
    if df.empty:
        return {
            "realized_moic": 0,
            "problem_rate": 0,
            "capital_at_risk": 0,
            "total_invested": 0,
            "total_paid": 0,
            "active_count": 0,
            "problem_count": 0,
            "paid_off_count": 0,
        }

    # Get status column
    status_col = "loan_status" if "loan_status" in df.columns else "status"

    # Total metrics
    total_invested = df["total_invested"].sum()
    total_paid = df["total_paid"].sum()

    # Realized MOIC (only for terminal/paid off loans)
    paid_off_mask = df[status_col] == "Paid Off"
    paid_off_df = df[paid_off_mask]
    if not paid_off_df.empty and paid_off_df["total_invested"].sum() > 0:
        realized_moic = paid_off_df["total_paid"].sum() / paid_off_df["total_invested"].sum()
    else:
        realized_moic = 0

    # Problem rate
    problem_mask = df[status_col].isin(PROBLEM_STATUSES)
    problem_count = problem_mask.sum()
    total_count = len(df)
    problem_rate = problem_count / total_count if total_count > 0 else 0

    # Capital at risk (net balance of problem loans)
    problem_df = df[problem_mask]
    capital_at_risk = problem_df["net_balance"].sum() if not problem_df.empty else 0

    # Active count (non-terminal, non-problem)
    active_mask = ~paid_off_mask & ~problem_mask & ~df[status_col].isin(["Charged Off", "Bankruptcy"])
    active_count = active_mask.sum()

    return {
        "realized_moic": realized_moic,
        "problem_rate": problem_rate,
        "capital_at_risk": capital_at_risk,
        "total_invested": total_invested,
        "total_paid": total_paid,
        "active_count": active_count,
        "problem_count": problem_count,
        "paid_off_count": paid_off_mask.sum(),
        "total_count": total_count,
    }


def get_winners_vs_losers(df: pd.DataFrame) -> dict:
    """Compare Paid Off loans vs Problem loans on key metrics."""
    if df.empty:
        return {"winners": None, "losers": None, "comparison": None}

    status_col = "loan_status" if "loan_status" in df.columns else "status"

    # Winners = Paid Off loans
    winners = df[df[status_col] == "Paid Off"].copy()

    # Losers = Problem status loans
    losers = df[df[status_col].isin(PROBLEM_STATUSES)].copy()

    if winners.empty or losers.empty:
        return {"winners": winners, "losers": losers, "comparison": None}

    # Calculate comparison metrics
    metrics = ["fico", "tib", "csl_participation_amount", "ahead_positions"]
    comparison = []

    for metric in metrics:
        if metric not in df.columns:
            continue

        winners_vals = pd.to_numeric(winners[metric], errors="coerce").dropna()
        losers_vals = pd.to_numeric(losers[metric], errors="coerce").dropna()

        if len(winners_vals) == 0 or len(losers_vals) == 0:
            continue

        winner_avg = winners_vals.mean()
        loser_avg = losers_vals.mean()

        # Calculate percentage difference
        if loser_avg != 0:
            pct_diff = ((winner_avg - loser_avg) / abs(loser_avg)) * 100
        elif winner_avg != 0:
            pct_diff = 100
        else:
            pct_diff = 0

        comparison.append({
            "metric": metric,
            "winner_avg": winner_avg,
            "loser_avg": loser_avg,
            "pct_diff": pct_diff,
            "significant": abs(pct_diff) > 10,  # >10% difference is significant
        })

    return {
        "winners": winners,
        "losers": losers,
        "comparison": pd.DataFrame(comparison) if comparison else None,
    }


def get_concentration_alerts(df: pd.DataFrame) -> dict:
    """Calculate concentration risk alerts."""
    if df.empty:
        return {"partner_alerts": [], "industry_alerts": [], "top_problem_loans": pd.DataFrame()}

    status_col = "loan_status" if "loan_status" in df.columns else "status"
    total_exposure = df["net_balance"].sum()

    # Partner concentration
    partner_alerts = []
    if "partner_source" in df.columns and total_exposure > 0:
        partner_exposure = df.groupby("partner_source")["net_balance"].sum().sort_values(ascending=False)

        for partner, exposure in partner_exposure.items():
            if pd.isna(partner) or partner == "":
                continue
            pct = exposure / total_exposure
            if pct > PARTNER_RED_THRESHOLD:
                partner_alerts.append({
                    "partner": partner,
                    "exposure": exposure,
                    "pct": pct,
                    "level": "red",
                })
            elif pct > PARTNER_YELLOW_THRESHOLD:
                partner_alerts.append({
                    "partner": partner,
                    "exposure": exposure,
                    "pct": pct,
                    "level": "yellow",
                })

    # Industry concentration
    industry_alerts = []
    if "sector_code" in df.columns and total_exposure > 0:
        # Use sector_code if available, else industry
        industry_col = "sector_code"
        if "industry_name" in df.columns:
            # Group by sector_code but show industry_name
            industry_exposure = df.groupby(industry_col).agg({
                "net_balance": "sum",
                "industry_name": "first"
            }).sort_values("net_balance", ascending=False)

            for sector, row in industry_exposure.iterrows():
                if pd.isna(sector) or sector == "":
                    continue
                pct = row["net_balance"] / total_exposure
                if pct > INDUSTRY_THRESHOLD:
                    industry_alerts.append({
                        "industry": row["industry_name"] if pd.notna(row["industry_name"]) else f"Sector {sector}",
                        "sector_code": sector,
                        "exposure": row["net_balance"],
                        "pct": pct,
                    })
        else:
            industry_exposure = df.groupby(industry_col)["net_balance"].sum().sort_values(ascending=False)
            for sector, exposure in industry_exposure.items():
                if pd.isna(sector) or sector == "":
                    continue
                pct = exposure / total_exposure
                if pct > INDUSTRY_THRESHOLD:
                    industry_alerts.append({
                        "industry": f"Sector {sector}",
                        "sector_code": sector,
                        "exposure": exposure,
                        "pct": pct,
                    })

    # Top 5 problem loans by exposure
    problem_mask = df[status_col].isin(PROBLEM_STATUSES)
    problem_df = df[problem_mask].copy()

    if not problem_df.empty:
        total_problem_exposure = problem_df["net_balance"].sum()
        top_problem = problem_df.nlargest(5, "net_balance")[
            ["loan_id", "deal_name", "loan_status", "net_balance", "partner_source"]
        ].copy()
        top_problem["pct_of_problem"] = top_problem["net_balance"] / total_problem_exposure if total_problem_exposure > 0 else 0
    else:
        top_problem = pd.DataFrame()

    return {
        "partner_alerts": partner_alerts,
        "industry_alerts": industry_alerts,
        "top_problem_loans": top_problem,
    }


def get_vintage_analysis(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate problem rate by funding month."""
    if df.empty or "funding_month" not in df.columns:
        return pd.DataFrame()

    status_col = "loan_status" if "loan_status" in df.columns else "status"

    # Group by funding month
    df_copy = df.copy()
    df_copy["is_problem"] = df_copy[status_col].isin(PROBLEM_STATUSES)

    vintage = df_copy.groupby("funding_month").agg(
        total_loans=("loan_id", "count"),
        problem_loans=("is_problem", "sum"),
        total_invested=("total_invested", "sum"),
        total_paid=("total_paid", "sum"),
    ).reset_index()

    vintage["problem_rate"] = vintage["problem_loans"] / vintage["total_loans"]
    vintage["problem_rate_pct"] = vintage["problem_rate"] * 100
    vintage["is_high_problem"] = vintage["problem_rate"] > 0.20

    # Convert period to datetime for charting
    vintage["funding_month_dt"] = vintage["funding_month"].apply(
        lambda x: x.to_timestamp() if pd.notna(x) else pd.NaT
    )

    # Sort by month
    vintage = vintage.sort_values("funding_month_dt")

    return vintage


def get_partner_performance(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate partner performance metrics."""
    if df.empty or "partner_source" not in df.columns:
        return pd.DataFrame()

    status_col = "loan_status" if "loan_status" in df.columns else "status"

    df_copy = df.copy()
    df_copy["is_problem"] = df_copy[status_col].isin(PROBLEM_STATUSES)
    df_copy["is_paid_off"] = df_copy[status_col] == "Paid Off"

    # Calculate recovery rate (total_paid / total_invested) for each loan
    df_copy["recovery_rate"] = np.where(
        df_copy["total_invested"] > 0,
        df_copy["total_paid"] / df_copy["total_invested"],
        0
    )

    partner_perf = df_copy.groupby("partner_source").agg(
        deal_count=("loan_id", "count"),
        capital_deployed=("total_invested", "sum"),
        total_paid=("total_paid", "sum"),
        problem_loans=("is_problem", "sum"),
        paid_off_loans=("is_paid_off", "sum"),
        avg_recovery=("recovery_rate", "mean"),
    ).reset_index()

    partner_perf["problem_rate"] = partner_perf["problem_loans"] / partner_perf["deal_count"]
    partner_perf["problem_rate_pct"] = partner_perf["problem_rate"] * 100
    partner_perf["moic"] = np.where(
        partner_perf["capital_deployed"] > 0,
        partner_perf["total_paid"] / partner_perf["capital_deployed"],
        0
    )

    # Sort by capital deployed descending
    partner_perf = partner_perf.sort_values("capital_deployed", ascending=False)

    return partner_perf


# ---------------------------
# Display Functions
# ---------------------------
def render_executive_summary(metrics: dict):
    """Render the executive summary section."""
    st.header("Executive Summary")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        moic_color = SUCCESS_COLOR if metrics["realized_moic"] >= MOIC_TARGET else DANGER_COLOR
        moic_delta = f"{((metrics['realized_moic'] / MOIC_TARGET) - 1) * 100:+.1f}% vs target" if MOIC_TARGET > 0 else ""
        st.metric(
            "Realized MOIC (Paid Off)",
            f"{metrics['realized_moic']:.2f}x",
            delta=moic_delta,
            delta_color="normal" if metrics["realized_moic"] >= MOIC_TARGET else "inverse"
        )
        st.caption(f"Target: {MOIC_TARGET:.2f}x")

    with col2:
        problem_color = SUCCESS_COLOR if metrics["problem_rate"] <= PROBLEM_RATE_TARGET else DANGER_COLOR
        problem_delta = f"{(PROBLEM_RATE_TARGET - metrics['problem_rate']) * 100:+.1f}pp vs target"
        st.metric(
            "Problem Rate",
            f"{metrics['problem_rate']:.1%}",
            delta=problem_delta,
            delta_color="normal" if metrics["problem_rate"] <= PROBLEM_RATE_TARGET else "inverse"
        )
        st.caption(f"Target: <{PROBLEM_RATE_TARGET:.0%}")

    with col3:
        st.metric(
            "Capital at Risk",
            f"${metrics['capital_at_risk']:,.0f}",
            delta=f"{metrics['problem_count']} problem loans",
            delta_color="off"
        )
        st.caption("Net balance of problem loans")

    with col4:
        st.metric(
            "Total Portfolio",
            f"${metrics['total_invested']:,.0f}",
            delta=f"{metrics['total_count']} loans",
            delta_color="off"
        )
        st.caption(f"Paid Off: {metrics['paid_off_count']} | Active: {metrics['active_count']}")


def render_winners_vs_losers(analysis: dict):
    """Render the winners vs losers comparison."""
    st.header("Winners vs Losers Analysis")

    winners = analysis.get("winners")
    losers = analysis.get("losers")
    comparison = analysis.get("comparison")

    if winners is None or winners.empty:
        st.info("No Paid Off loans to analyze as 'Winners'")
        return

    if losers is None or losers.empty:
        st.info("No Problem loans to analyze as 'Losers'")
        return

    # Summary stats
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Winners (Paid Off)")
        st.metric("Count", f"{len(winners)}")
        st.metric("Total Invested", f"${winners['total_invested'].sum():,.0f}")
        st.metric("Total Recovered", f"${winners['total_paid'].sum():,.0f}")

    with col2:
        st.subheader("Losers (Problem Status)")
        st.metric("Count", f"{len(losers)}")
        st.metric("Total Invested", f"${losers['total_invested'].sum():,.0f}")
        st.metric("Capital at Risk", f"${losers['net_balance'].sum():,.0f}")

    # Comparison table
    if comparison is not None and not comparison.empty:
        st.subheader("Factor Comparison")

        metric_labels = {
            "fico": "FICO Score",
            "tib": "Time in Business (yrs)",
            "csl_participation_amount": "Deal Size ($)",
            "ahead_positions": "Lien Position",
        }

        for _, row in comparison.iterrows():
            metric_name = metric_labels.get(row["metric"], row["metric"])
            significant = row["significant"]

            cols = st.columns([2, 2, 2, 1])
            with cols[0]:
                st.write(f"**{metric_name}**")
            with cols[1]:
                if row["metric"] == "csl_participation_amount":
                    st.write(f"Winners: ${row['winner_avg']:,.0f}")
                else:
                    st.write(f"Winners: {row['winner_avg']:.1f}")
            with cols[2]:
                if row["metric"] == "csl_participation_amount":
                    st.write(f"Losers: ${row['loser_avg']:,.0f}")
                else:
                    st.write(f"Losers: {row['loser_avg']:.1f}")
            with cols[3]:
                if significant:
                    color = SUCCESS_COLOR if row["pct_diff"] > 0 else DANGER_COLOR
                    st.markdown(f"<span style='color:{color}; font-weight:bold;'>{row['pct_diff']:+.0f}%</span>", unsafe_allow_html=True)
                else:
                    st.write(f"{row['pct_diff']:+.0f}%")


def render_concentration_alerts(alerts: dict):
    """Render concentration risk alerts."""
    st.header("Concentration Alerts")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Partner Concentration")
        partner_alerts = alerts.get("partner_alerts", [])

        if not partner_alerts:
            st.success("No partner concentration issues detected")
        else:
            for alert in partner_alerts:
                color = DANGER_COLOR if alert["level"] == "red" else WARNING_COLOR
                icon = "🔴" if alert["level"] == "red" else "🟡"
                threshold = ">50%" if alert["level"] == "red" else ">30%"
                st.markdown(
                    f"{icon} **{alert['partner']}**: {alert['pct']:.1%} of exposure "
                    f"(${alert['exposure']:,.0f}) - {threshold}",
                    unsafe_allow_html=True
                )

    with col2:
        st.subheader("Industry Concentration (>20%)")
        industry_alerts = alerts.get("industry_alerts", [])

        if not industry_alerts:
            st.success("No industry concentration issues detected")
        else:
            for alert in industry_alerts:
                st.markdown(
                    f"🟡 **{alert['industry']}**: {alert['pct']:.1%} of exposure "
                    f"(${alert['exposure']:,.0f})"
                )

    # Top problem loans
    st.subheader("Top 5 Problem Loans (by Exposure)")
    top_problem = alerts.get("top_problem_loans", pd.DataFrame())

    if top_problem.empty:
        st.success("No problem loans in portfolio")
    else:
        # Format for display
        display_df = top_problem.copy()
        display_df["Net Balance"] = display_df["net_balance"].apply(lambda x: f"${x:,.0f}")
        display_df["% of Problem Exposure"] = display_df["pct_of_problem"].apply(lambda x: f"{x:.1%}")
        display_df = display_df.rename(columns={
            "loan_id": "Loan ID",
            "deal_name": "Deal Name",
            "loan_status": "Status",
            "partner_source": "Partner",
        })

        st.dataframe(
            display_df[["Loan ID", "Deal Name", "Status", "Partner", "Net Balance", "% of Problem Exposure"]],
            use_container_width=True,
            hide_index=True,
        )


def render_vintage_analysis(vintage: pd.DataFrame):
    """Render vintage analysis chart."""
    st.header("Vintage Analysis")

    if vintage.empty:
        st.info("No vintage data available")
        return

    st.caption("Problem rate by funding month - Red bars indicate >20% problem rate")

    # Create color scale based on problem rate
    vintage_chart = vintage.copy()
    vintage_chart["color"] = vintage_chart["is_high_problem"].apply(
        lambda x: DANGER_COLOR if x else PRIMARY_COLOR
    )
    vintage_chart["month_label"] = vintage_chart["funding_month_dt"].dt.strftime("%b %Y")

    # Bar chart
    bars = alt.Chart(vintage_chart).mark_bar().encode(
        x=alt.X("month_label:N", title="Funding Month", sort=None),
        y=alt.Y("problem_rate_pct:Q", title="Problem Rate (%)"),
        color=alt.Color(
            "is_high_problem:N",
            scale=alt.Scale(domain=[False, True], range=[PRIMARY_COLOR, DANGER_COLOR]),
            legend=alt.Legend(title="High Problem Rate", labelExpr="datum.value ? '>20%' : '<=20%'")
        ),
        tooltip=[
            alt.Tooltip("month_label:N", title="Month"),
            alt.Tooltip("problem_rate_pct:Q", title="Problem Rate (%)", format=".1f"),
            alt.Tooltip("problem_loans:Q", title="Problem Loans"),
            alt.Tooltip("total_loans:Q", title="Total Loans"),
            alt.Tooltip("total_invested:Q", title="Capital Deployed", format="$,.0f"),
        ]
    ).properties(
        height=350
    )

    # Add threshold line
    threshold_line = alt.Chart(pd.DataFrame({"y": [20]})).mark_rule(
        color=DANGER_COLOR,
        strokeDash=[5, 5],
        strokeWidth=2
    ).encode(y="y:Q")

    chart = (bars + threshold_line).properties(
        title="Problem Rate by Vintage"
    )

    st.altair_chart(chart, use_container_width=True)

    # Summary table
    with st.expander("View Vintage Details"):
        display_vintage = vintage.copy()
        display_vintage["Funding Month"] = display_vintage["funding_month_dt"].dt.strftime("%b %Y")
        display_vintage["Total Loans"] = display_vintage["total_loans"]
        display_vintage["Problem Loans"] = display_vintage["problem_loans"]
        display_vintage["Problem Rate"] = display_vintage["problem_rate"].apply(lambda x: f"{x:.1%}")
        display_vintage["Capital Deployed"] = display_vintage["total_invested"].apply(lambda x: f"${x:,.0f}")

        st.dataframe(
            display_vintage[["Funding Month", "Total Loans", "Problem Loans", "Problem Rate", "Capital Deployed"]],
            use_container_width=True,
            hide_index=True,
        )


def render_partner_performance(partner_perf: pd.DataFrame):
    """Render partner performance table."""
    st.header("Partner Performance")

    if partner_perf.empty:
        st.info("No partner data available")
        return

    # Create display dataframe
    display_df = partner_perf.copy()
    display_df["Partner"] = display_df["partner_source"].fillna("Unknown")
    display_df["Deals"] = display_df["deal_count"]
    display_df["Capital Deployed"] = display_df["capital_deployed"].apply(lambda x: f"${x:,.0f}")
    display_df["Total Paid"] = display_df["total_paid"].apply(lambda x: f"${x:,.0f}")
    display_df["MOIC"] = display_df["moic"].apply(lambda x: f"{x:.2f}x")
    display_df["Problem Rate"] = display_df["problem_rate"].apply(lambda x: f"{x:.1%}")
    display_df["Avg Recovery"] = display_df["avg_recovery"].apply(lambda x: f"{x:.1%}")

    # Highlight problem rates above target
    def highlight_problem_rate(val):
        try:
            pct = float(val.strip("%")) / 100
            if pct > PROBLEM_RATE_TARGET:
                return f"color: {DANGER_COLOR}; font-weight: bold"
        except:
            pass
        return ""

    st.dataframe(
        display_df[["Partner", "Deals", "Capital Deployed", "Total Paid", "MOIC", "Problem Rate", "Avg Recovery"]],
        use_container_width=True,
        hide_index=True,
    )

    # Summary chart - Problem Rate by Partner
    st.subheader("Problem Rate by Partner")

    chart_df = partner_perf[partner_perf["partner_source"].notna()].copy()
    chart_df["above_target"] = chart_df["problem_rate"] > PROBLEM_RATE_TARGET

    chart = alt.Chart(chart_df).mark_bar().encode(
        x=alt.X("partner_source:N", title="Partner", sort="-y"),
        y=alt.Y("problem_rate_pct:Q", title="Problem Rate (%)"),
        color=alt.Color(
            "above_target:N",
            scale=alt.Scale(domain=[False, True], range=[PRIMARY_COLOR, DANGER_COLOR]),
            legend=alt.Legend(title="Above Target", labelExpr="datum.value ? '>10%' : '<=10%'")
        ),
        tooltip=[
            alt.Tooltip("partner_source:N", title="Partner"),
            alt.Tooltip("problem_rate_pct:Q", title="Problem Rate (%)", format=".1f"),
            alt.Tooltip("deal_count:Q", title="Deal Count"),
            alt.Tooltip("capital_deployed:Q", title="Capital Deployed", format="$,.0f"),
        ]
    ).properties(
        height=300
    )

    # Add threshold line
    threshold_line = alt.Chart(pd.DataFrame({"y": [PROBLEM_RATE_TARGET * 100]})).mark_rule(
        color=DANGER_COLOR,
        strokeDash=[5, 5],
        strokeWidth=2
    ).encode(y="y:Q")

    st.altair_chart((chart + threshold_line), use_container_width=True)


# ---------------------------
# Main App
# ---------------------------
def main():
    st.title("Portfolio Insights")
    st.caption("Executive-level portfolio analytics and risk monitoring")

    # Load data
    with st.spinner("Loading portfolio data..."):
        df = load_portfolio_data()

    if df.empty:
        st.error("No loan data available. Please check data sources.")
        return

    # Calculate all metrics
    metrics = calculate_executive_metrics(df)
    winners_losers = get_winners_vs_losers(df)
    concentration = get_concentration_alerts(df)
    vintage = get_vintage_analysis(df)
    partner_perf = get_partner_performance(df)

    # Render sections
    render_executive_summary(metrics)

    st.divider()
    render_winners_vs_losers(winners_losers)

    st.divider()
    render_concentration_alerts(concentration)

    st.divider()
    render_vintage_analysis(vintage)

    st.divider()
    render_partner_performance(partner_perf)

    # Footer with data info
    st.divider()
    st.caption(f"Data as of: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}")


if __name__ == "__main__":
    main()
else:
    main()
