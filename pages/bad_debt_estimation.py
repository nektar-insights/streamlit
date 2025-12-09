# pages/bad_debt_estimation.py
"""
Bad Debt Expense Estimation - Pre-screen deals and estimate portfolio losses.

Uses recovery scoring to estimate bad debt expense per deal based on:
- Status: Current loan performance status
- Industry: NAICS sector risk profile
- Collateral: Type and quality of security
- Lien Position: Priority in capital structure
- Communication: Borrower engagement level
"""

import streamlit as st
import pandas as pd
import numpy as np
import altair as alt

from utils.config import setup_page, PRIMARY_COLOR, COLOR_PALETTE
from utils.data_loader import DataLoader
from utils.loan_tape_data import prepare_loan_data
from utils.recovery_scoring import (
    calculate_recovery_score,
    score_portfolio,
    get_portfolio_summary,
    get_recovery_color,
    format_currency,
    format_percentage,
    COLLATERAL_SCORES,
    COMMUNICATION_SCORES,
    STATUS_SCORES,
    INDUSTRY_SCORES,
    WEIGHTS,
)
from utils.loan_tape_ml import NAICS_SECTOR_NAMES

# Page setup
setup_page("CSL Capital | Bad Debt Estimation")

st.title("Bad Debt Expense Estimation")
st.markdown("*Pre-screen deals and estimate portfolio bad debt exposure*")
st.markdown("---")

# =============================================================================
# DATA LOADING
# =============================================================================

@st.cache_data(ttl=3600)
def load_portfolio_data():
    """Load and prepare portfolio data for scoring."""
    loader = DataLoader()
    loans_df = loader.load_loan_summaries()
    deals_df = loader.load_deals()

    if loans_df.empty or deals_df.empty:
        return pd.DataFrame()

    # Prepare loan data
    df = prepare_loan_data(loans_df, deals_df)

    # Derive sector_code if not present
    if "sector_code" not in df.columns and "industry" in df.columns:
        df["sector_code"] = df["industry"].astype(str).str[:2].str.zfill(2)

    return df


# =============================================================================
# TABS
# =============================================================================

tab1, tab2 = st.tabs(["Single Deal Scorer", "Portfolio Analysis"])

# =============================================================================
# TAB 1: SINGLE DEAL SCORER
# =============================================================================

with tab1:
    st.subheader("Score a Prospective Deal")
    st.markdown("Enter deal characteristics to estimate recovery probability and bad debt expense.")

    col1, col2 = st.columns(2)

    with col1:
        # Loan Status
        status_options = ["Select Status..."] + sorted(STATUS_SCORES.keys())
        selected_status = st.selectbox(
            "Loan Status",
            options=status_options,
            index=status_options.index("Active") if "Active" in status_options else 0,
            help="Current status of the loan"
        )
        loan_status = selected_status if selected_status != "Select Status..." else "Active"

        # Industry
        sector_options = {"Select Industry...": None}
        sector_options.update({f"{code} - {name}": code for code, name in sorted(NAICS_SECTOR_NAMES.items())})
        selected_sector = st.selectbox(
            "Industry (NAICS Sector)",
            options=list(sector_options.keys()),
            index=0,
            help="2-digit NAICS sector classification"
        )
        sector_code = sector_options[selected_sector]

        # Collateral Type
        collateral_options = {
            "None / Unsecured": "none",
            "Intangible (IP, Licenses)": "intangible",
            "Inventory / Consumer Goods": "inventory",
            "Accounts Receivable (Unverified)": "receivables_unverified",
            "Accounts Receivable (Verified)": "receivables_verified",
            "Equipment (Specialized)": "equipment_specialized",
            "Equipment (Generic/Liquid)": "equipment_generic",
            "Property (Encumbered/Leased)": "property_encumbered",
            "Lockbox (No Control Agreement)": "lockbox_no_daca",
            "Property (Owned Outright)": "property_owned",
            "Lockbox (Full DACA/Control)": "lockbox_daca",
            "Cash / Securities / Deposit": "cash_securities",
        }
        selected_collateral = st.selectbox(
            "Collateral Type",
            options=list(collateral_options.keys()),
            index=0,
            help="Type and quality of security backing the loan"
        )
        collateral_type = collateral_options[selected_collateral]

    with col2:
        # Lien Position
        lien_options = {
            "1st Lien (Senior)": 0,
            "2nd Lien": 1,
            "3rd+ Lien (Junior/Unsecured)": 2,
        }
        selected_lien = st.selectbox(
            "Lien Position",
            options=list(lien_options.keys()),
            index=0,
            help="Priority position in capital structure"
        )
        ahead_positions = lien_options[selected_lien]

        # Communication Status
        comm_options = {
            "No Contact / Hostile": "none_hostile",
            "Sporadic (Only Under Pressure)": "sporadic",
            "Generally Responsive (Slow)": "slow_responsive",
            "Fully Engaged / Proactive": "engaged",
            "On Plan & Hitting Milestones": "plan_milestones",
        }
        selected_comm = st.selectbox(
            "Communication Status",
            options=list(comm_options.keys()),
            index=1,  # Default to sporadic
            help="Level of borrower engagement and communication"
        )
        communication_status = comm_options[selected_comm]

        # Exposure Amount
        exposure_base = st.number_input(
            "Exposure Amount ($)",
            min_value=0,
            max_value=50_000_000,
            value=50_000,
            step=5000,
            format="%d",
            help="Dollar amount at risk (e.g., outstanding balance, invested amount)"
        )

    # Calculate Score button
    st.markdown("---")
    if st.button("Calculate Recovery Score", type="primary", use_container_width=True):
        result = calculate_recovery_score(
            loan_status=loan_status,
            sector_code=sector_code,
            collateral_type=collateral_type,
            ahead_positions=ahead_positions,
            communication_status=communication_status,
            exposure_base=float(exposure_base),
        )

        # Display results
        st.markdown("---")
        st.subheader("Recovery Assessment")

        # Main score display
        score_color = get_recovery_color(result.total_recovery_score)
        st.markdown(f"""
        <div style="
            background: linear-gradient(135deg, {score_color}22, {score_color}44);
            border-left: 5px solid {score_color};
            padding: 20px;
            border-radius: 8px;
            margin-bottom: 20px;
        ">
            <h2 style="margin: 0; color: {score_color};">
                Recovery Score: {result.total_recovery_score:.1f} / 10
            </h2>
            <p style="margin: 5px 0 0 0; font-size: 1.2em;">
                Recovery Band: <strong>{result.recovery_band}</strong>
            </p>
        </div>
        """, unsafe_allow_html=True)

        # Metrics row
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Recovery % (Midpoint)", format_percentage(result.recovery_pct_midpoint))
        with col2:
            st.metric("Loss % (Midpoint)", format_percentage(result.loss_pct))
        with col3:
            st.metric("Bad Debt Expense", format_currency(result.bad_debt_expense))
        with col4:
            st.metric("Exposure Base", format_currency(result.exposure_base))

        # Range display
        st.markdown("##### Estimated Range")
        range_col1, range_col2, range_col3 = st.columns(3)
        with range_col1:
            st.markdown(f"**Recovery Range:** {format_percentage(result.recovery_pct_low)} - {format_percentage(result.recovery_pct_high)}")
        with range_col2:
            st.markdown(f"**Bad Debt Low:** {format_currency(result.bad_debt_low)}")
        with range_col3:
            st.markdown(f"**Bad Debt High:** {format_currency(result.bad_debt_high)}")

        # Component scores breakdown
        st.markdown("---")
        st.markdown("##### Score Components")

        # Create component breakdown data
        components = [
            {"Component": "Status", "Score": result.status_score, "Weight": f"{WEIGHTS['status']:.0%}"},
            {"Component": "Industry", "Score": result.industry_score, "Weight": f"{WEIGHTS['industry']:.0%}"},
            {"Component": "Collateral", "Score": result.collateral_score, "Weight": f"{WEIGHTS['collateral']:.0%}"},
            {"Component": "Lien Position", "Score": result.lien_score, "Weight": f"{WEIGHTS['lien']:.0%}"},
            {"Component": "Communication", "Score": result.communication_score, "Weight": f"{WEIGHTS['communication']:.0%}"},
        ]
        comp_df = pd.DataFrame(components)

        # Bar chart of components
        comp_chart = alt.Chart(comp_df).mark_bar().encode(
            x=alt.X("Score:Q", scale=alt.Scale(domain=[0, 10]), title="Score (0-10)"),
            y=alt.Y("Component:N", sort="-x", title=None),
            color=alt.Color(
                "Score:Q",
                scale=alt.Scale(domain=[0, 5, 10], range=["#ef4444", "#eab308", "#22c55e"]),
                legend=None
            ),
            tooltip=["Component", "Score", "Weight"]
        ).properties(height=200)

        st.altair_chart(comp_chart, use_container_width=True)

        # Show the formula
        with st.expander("Scoring Formula"):
            st.markdown(f"""
            **Total Recovery Score** =
            - {WEIGHTS['status']:.0%} x Status ({result.status_score}) +
            - {WEIGHTS['industry']:.0%} x Industry ({result.industry_score}) +
            - {WEIGHTS['collateral']:.0%} x Collateral ({result.collateral_score}) +
            - {WEIGHTS['lien']:.0%} x Lien ({result.lien_score}) +
            - {WEIGHTS['communication']:.0%} x Communication ({result.communication_score})
            - = **{result.total_recovery_score:.2f}**

            **Bad Debt Expense** = Exposure ({format_currency(result.exposure_base)}) x Loss % ({format_percentage(result.loss_pct)}) = **{format_currency(result.bad_debt_expense)}**
            """)


# =============================================================================
# TAB 2: PORTFOLIO ANALYSIS
# =============================================================================

with tab2:
    st.subheader("Portfolio Bad Debt Analysis")
    st.markdown("Score the entire portfolio to estimate aggregate bad debt exposure.")

    # Load data
    with st.spinner("Loading portfolio data..."):
        df = load_portfolio_data()

    if df.empty:
        st.error("Unable to load portfolio data. Please check database connection.")
        st.stop()

    # Filter options
    st.markdown("##### Filters")
    filter_col1, filter_col2, filter_col3 = st.columns(3)

    with filter_col1:
        # Status filter
        status_options = ["All Statuses"] + sorted(df["loan_status"].dropna().unique().tolist())
        selected_statuses = st.multiselect(
            "Loan Status",
            options=status_options[1:],
            default=[],
            help="Filter by loan status (empty = all)"
        )

    with filter_col2:
        # Partner filter
        if "partner_source" in df.columns:
            partner_options = sorted(df["partner_source"].dropna().unique().tolist())
            selected_partners = st.multiselect(
                "Partner Source",
                options=partner_options,
                default=[],
                help="Filter by partner (empty = all)"
            )
        else:
            selected_partners = []

    with filter_col3:
        # Include paid off loans?
        include_paid_off = st.checkbox(
            "Include Paid Off Loans",
            value=False,
            help="Include loans that have been fully paid off"
        )

    # Apply filters
    filtered_df = df.copy()

    if selected_statuses:
        filtered_df = filtered_df[filtered_df["loan_status"].isin(selected_statuses)]

    if selected_partners:
        filtered_df = filtered_df[filtered_df["partner_source"].isin(selected_partners)]

    if not include_paid_off:
        filtered_df = filtered_df[filtered_df["loan_status"] != "Paid Off"]

    # Score portfolio button
    st.markdown("---")
    if st.button("Score Portfolio", type="primary", use_container_width=True):
        with st.spinner("Scoring portfolio..."):
            # Score the filtered portfolio
            scored_df = score_portfolio(
                filtered_df,
                status_col="loan_status",
                sector_col="sector_code",
                exposure_col="net_balance",
                lien_col="ahead_positions",
            )

            # Get summary
            summary = get_portfolio_summary(scored_df)

        st.success(f"Scored {summary['deal_count']:,} deals")

        # Summary metrics
        st.markdown("---")
        st.subheader("Portfolio Summary")

        metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
        with metric_col1:
            st.metric("Total Exposure", format_currency(summary["total_exposure"]))
        with metric_col2:
            st.metric("Est. Bad Debt Expense", format_currency(summary["total_bad_debt_expense"]))
        with metric_col3:
            st.metric("Avg Recovery Score", f"{summary['avg_recovery_score']:.1f}")
        with metric_col4:
            st.metric("Avg Loss %", format_percentage(summary["avg_loss_pct"]))

        # Range
        range_col1, range_col2 = st.columns(2)
        with range_col1:
            st.metric("Bad Debt (Low Estimate)", format_currency(summary["total_bad_debt_low"]))
        with range_col2:
            st.metric("Bad Debt (High Estimate)", format_currency(summary["total_bad_debt_high"]))

        # Charts
        st.markdown("---")
        st.subheader("Analysis")

        chart_col1, chart_col2 = st.columns(2)

        with chart_col1:
            st.markdown("##### Bad Debt by Status")
            if "loan_status" in scored_df.columns and "bad_debt_expense" in scored_df.columns:
                status_summary = scored_df.groupby("loan_status").agg({
                    "bad_debt_expense": "sum",
                    "exposure_base": "sum",
                    "recovery_score": "mean",
                }).reset_index()
                status_summary.columns = ["Status", "Bad Debt Expense", "Exposure", "Avg Recovery Score"]
                status_summary = status_summary.sort_values("Bad Debt Expense", ascending=False)

                status_chart = alt.Chart(status_summary).mark_bar().encode(
                    x=alt.X("Bad Debt Expense:Q", title="Bad Debt Expense ($)"),
                    y=alt.Y("Status:N", sort="-x", title=None),
                    color=alt.Color(
                        "Avg Recovery Score:Q",
                        scale=alt.Scale(domain=[0, 5, 10], range=["#ef4444", "#eab308", "#22c55e"]),
                        title="Avg Recovery"
                    ),
                    tooltip=["Status", "Bad Debt Expense", "Exposure", "Avg Recovery Score"]
                ).properties(height=300)

                st.altair_chart(status_chart, use_container_width=True)

        with chart_col2:
            st.markdown("##### Bad Debt by Industry")
            if "sector_code" in scored_df.columns and "bad_debt_expense" in scored_df.columns:
                # Add sector names
                scored_df["Sector"] = scored_df["sector_code"].map(
                    lambda x: NAICS_SECTOR_NAMES.get(str(x), f"Sector {x}")
                )

                industry_summary = scored_df.groupby("Sector").agg({
                    "bad_debt_expense": "sum",
                    "exposure_base": "sum",
                    "recovery_score": "mean",
                }).reset_index()
                industry_summary.columns = ["Sector", "Bad Debt Expense", "Exposure", "Avg Recovery Score"]
                industry_summary = industry_summary.sort_values("Bad Debt Expense", ascending=False).head(10)

                industry_chart = alt.Chart(industry_summary).mark_bar().encode(
                    x=alt.X("Bad Debt Expense:Q", title="Bad Debt Expense ($)"),
                    y=alt.Y("Sector:N", sort="-x", title=None),
                    color=alt.Color(
                        "Avg Recovery Score:Q",
                        scale=alt.Scale(domain=[0, 5, 10], range=["#ef4444", "#eab308", "#22c55e"]),
                        title="Avg Recovery"
                    ),
                    tooltip=["Sector", "Bad Debt Expense", "Exposure", "Avg Recovery Score"]
                ).properties(height=300)

                st.altair_chart(industry_chart, use_container_width=True)

        # Recovery Score Distribution
        st.markdown("##### Recovery Score Distribution")
        if "recovery_score" in scored_df.columns:
            dist_chart = alt.Chart(scored_df).mark_bar().encode(
                x=alt.X("recovery_score:Q", bin=alt.Bin(maxbins=20), title="Recovery Score"),
                y=alt.Y("count()", title="Number of Deals"),
                color=alt.Color(
                    "recovery_score:Q",
                    scale=alt.Scale(domain=[0, 5, 10], range=["#ef4444", "#eab308", "#22c55e"]),
                    legend=None
                ),
            ).properties(height=200)

            st.altair_chart(dist_chart, use_container_width=True)

        # Detailed table
        st.markdown("---")
        st.subheader("Deal Details")

        # Select columns to display
        display_cols = [
            "loan_id",
            "deal_name",
            "loan_status",
            "sector_code",
            "net_balance",
            "recovery_score",
            "recovery_pct",
            "loss_pct",
            "bad_debt_expense",
        ]
        display_cols = [c for c in display_cols if c in scored_df.columns]

        # Sort by bad debt expense
        display_df = scored_df[display_cols].sort_values("bad_debt_expense", ascending=False)

        # Format columns
        if "net_balance" in display_df.columns:
            display_df["net_balance"] = display_df["net_balance"].apply(lambda x: f"${x:,.0f}" if pd.notna(x) else "$0")
        if "bad_debt_expense" in display_df.columns:
            display_df["bad_debt_expense"] = display_df["bad_debt_expense"].apply(lambda x: f"${x:,.0f}" if pd.notna(x) else "$0")
        if "recovery_pct" in display_df.columns:
            display_df["recovery_pct"] = display_df["recovery_pct"].apply(lambda x: f"{x:.1%}" if pd.notna(x) else "0%")
        if "loss_pct" in display_df.columns:
            display_df["loss_pct"] = display_df["loss_pct"].apply(lambda x: f"{x:.1%}" if pd.notna(x) else "0%")
        if "recovery_score" in display_df.columns:
            display_df["recovery_score"] = display_df["recovery_score"].apply(lambda x: f"{x:.1f}" if pd.notna(x) else "0")

        # Rename columns for display
        display_df = display_df.rename(columns={
            "loan_id": "Loan ID",
            "deal_name": "Deal Name",
            "loan_status": "Status",
            "sector_code": "Sector",
            "net_balance": "Net Balance",
            "recovery_score": "Recovery Score",
            "recovery_pct": "Recovery %",
            "loss_pct": "Loss %",
            "bad_debt_expense": "Bad Debt Expense",
        })

        st.dataframe(display_df, use_container_width=True, height=400)

        # Download button
        csv = scored_df.to_csv(index=False)
        st.download_button(
            label="Download Full Results (CSV)",
            data=csv,
            file_name="bad_debt_estimation.csv",
            mime="text/csv",
        )


# =============================================================================
# METHODOLOGY SECTION
# =============================================================================

st.markdown("---")
with st.expander("Methodology & Scoring Details"):
    st.markdown("""
    ### Recovery Score Methodology

    The recovery score estimates the probability of recovering exposure from a deal based on five weighted factors:

    | Factor | Weight | Description |
    |--------|--------|-------------|
    | Status | 10% | Current loan performance status |
    | Industry | 30% | NAICS sector risk profile |
    | Collateral | 30% | Type and quality of security |
    | Lien Position | 20% | Priority in capital structure |
    | Communication | 10% | Borrower engagement level |

    ### Recovery Bands

    | Score Range | Recovery % | Midpoint | Loss % |
    |-------------|------------|----------|--------|
    | 9.0 - 10.0 | 90% - 100% | 95% | 5% |
    | 7.0 - 8.9 | 70% - 89% | 79.5% | 20.5% |
    | 5.0 - 6.9 | 50% - 69% | 59.5% | 40.5% |
    | 3.0 - 4.9 | 30% - 49% | 39.5% | 60.5% |
    | 1.0 - 2.9 | 10% - 29% | 19.5% | 80.5% |
    | 0.0 - 1.0 | 0% - 9% | 4.5% | 95.5% |

    ### Bad Debt Expense Calculation

    ```
    Bad Debt Expense = Exposure Base x Loss %
    ```

    Where:
    - **Exposure Base** = Net balance (total invested - total paid)
    - **Loss %** = 1 - Recovery % (midpoint)

    ### Important Notes

    - This is a **pre-screening tool**, not a final valuation engine
    - Scores assume current market conditions and historical recovery patterns
    - Actual recovery may vary based on specific circumstances
    - Collateral and communication scores default to conservative values when not specified
    """)
