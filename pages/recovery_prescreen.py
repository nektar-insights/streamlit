# pages/recovery_prescreen.py
"""
Recovery Pre-Screen & Bad Debt Estimator

This page provides two tools:
1. PRE-SCREEN TOOL: Manual inputs for prospective deals (forward-looking)
2. BAD DEBT ESTIMATOR: Analyze estimated bad debt for existing portfolio

Uses a unified recovery scoring framework with fixed business rules.
"""

import streamlit as st
import pandas as pd
import altair as alt

from utils.config import setup_page, PRIMARY_COLOR
from utils.recovery_scoring import (
    RecoveryScoreResult,
    compute_recovery_prescreen,
    compute_recovery_batch,
    get_portfolio_bad_debt_summary,
    STATUS_CATEGORIES,
    INDUSTRY_CATEGORIES,
    COLLATERAL_CATEGORIES,
    LIEN_CATEGORIES,
    COMMUNICATION_CATEGORIES,
    WEIGHTS,
)

# Page setup
setup_page("CSL Capital | Recovery Pre-Screen")

# =============================================================================
# PAGE HEADER
# =============================================================================

st.title("Recovery Pre-Screen & Bad Debt Estimator")

st.markdown("""
This tool uses a **unified recovery scoring framework** to estimate potential losses.
The same scoring logic applies to both pre-screening prospective deals and estimating
bad debt on your existing portfolio.

**Scoring Components:**
- Status (10%): Loan status severity
- Industry (30%): NAICS sector risk
- Collateral (30%): Collateral type/quality
- Lien Position (20%): Lien seniority
- Communication (10%): Borrower responsiveness
""")

st.divider()

# =============================================================================
# TAB NAVIGATION
# =============================================================================

tab_prescreen, tab_portfolio = st.tabs([
    "Pre-Screen Tool",
    "Portfolio Bad Debt Estimator"
])

# =============================================================================
# TAB 1: PRE-SCREEN TOOL
# =============================================================================

with tab_prescreen:
    st.subheader("Pre-Screen a Prospective Deal")
    st.markdown("""
    Enter deal parameters below to estimate recovery prospects and potential bad debt.
    This is a **pre-screening tool** for deals NOT yet in your database.
    """)

    # Create two columns for inputs and outputs
    col_input, col_output = st.columns([1, 1])

    with col_input:
        st.markdown("#### Deal Parameters")

        # Exposure Amount
        exposure_amount = st.number_input(
            "Exposure / Principal Amount ($)",
            min_value=0.0,
            max_value=10_000_000.0,
            value=100_000.0,
            step=1000.0,
            format="%.2f",
            help="The total amount at risk for this deal"
        )

        # Status Category
        status_category = st.selectbox(
            "Current Status",
            options=STATUS_CATEGORIES,
            index=6,  # Default to "Active (Healthy)"
            help="The current or expected status of the borrower"
        )

        # Industry Category
        industry_category = st.selectbox(
            "Industry",
            options=INDUSTRY_CATEGORIES,
            index=6,  # Default to "Business / Professional Services"
            help="The borrower's industry sector"
        )

        # Collateral Type
        collateral_type = st.selectbox(
            "Collateral Type",
            options=COLLATERAL_CATEGORIES,
            index=0,  # Default to "None / Unsecured"
            help="The type and quality of collateral securing the loan"
        )

        # Lien Position
        lien_position = st.selectbox(
            "Lien Position",
            options=LIEN_CATEGORIES,
            index=0,  # Default to "First Lien"
            help="Your position in the capital stack"
        )

        # Communication Status
        communication_status = st.selectbox(
            "Communication Status",
            options=COMMUNICATION_CATEGORIES,
            index=1,  # Default to "Sporadic / only under pressure"
            help="How responsive and engaged is the borrower?"
        )

        # Calculate button
        calculate_btn = st.button("Calculate Recovery Score", type="primary", use_container_width=True)

    # Compute and display results
    with col_output:
        st.markdown("#### Recovery Analysis")

        # Always compute (for real-time updates)
        result = compute_recovery_prescreen(
            exposure_amount=exposure_amount,
            status_category=status_category,
            industry_category=industry_category,
            collateral_type=collateral_type,
            lien_position=lien_position,
            communication_status=communication_status,
        )

        # Display key metrics
        col_score, col_recovery, col_loss = st.columns(3)

        with col_score:
            st.metric(
                label="Total Recovery Score",
                value=f"{result.total_recovery_score:.1f}/10",
            )

        with col_recovery:
            st.metric(
                label="Est. Recovery %",
                value=f"{result.recovery_pct_midpoint:.0%}",
                delta=f"{result.recovery_band_label}",
                delta_color="off"
            )

        with col_loss:
            st.metric(
                label="Est. Loss %",
                value=f"{result.loss_pct_midpoint:.0%}",
            )

        st.divider()

        # Bad Debt Estimates
        st.markdown("#### Estimated Bad Debt Expense")

        col_low, col_mid, col_high = st.columns(3)

        with col_low:
            st.metric(
                label="Low Estimate",
                value=f"${result.estimated_bad_debt_expense_low:,.0f}",
                help="Best-case scenario (top of recovery band)"
            )

        with col_mid:
            st.metric(
                label="Midpoint Estimate",
                value=f"${result.estimated_bad_debt_expense_mid:,.0f}",
                help="Expected value (midpoint of recovery band)"
            )

        with col_high:
            st.metric(
                label="High Estimate",
                value=f"${result.estimated_bad_debt_expense_high:,.0f}",
                help="Worst-case scenario (bottom of recovery band)"
            )

        st.divider()

        # Component Scores Breakdown
        st.markdown("#### Score Breakdown")

        # Create score breakdown data
        score_data = pd.DataFrame([
            {"Component": "Status", "Score": result.status_score, "Weight": f"{WEIGHTS['status']:.0%}", "Contribution": result.status_score * WEIGHTS['status']},
            {"Component": "Industry", "Score": result.industry_score, "Weight": f"{WEIGHTS['industry']:.0%}", "Contribution": result.industry_score * WEIGHTS['industry']},
            {"Component": "Collateral", "Score": result.collateral_score, "Weight": f"{WEIGHTS['collateral']:.0%}", "Contribution": result.collateral_score * WEIGHTS['collateral']},
            {"Component": "Lien Position", "Score": result.lien_score, "Weight": f"{WEIGHTS['lien']:.0%}", "Contribution": result.lien_score * WEIGHTS['lien']},
            {"Component": "Communication", "Score": result.communication_score, "Weight": f"{WEIGHTS['communication']:.0%}", "Contribution": result.communication_score * WEIGHTS['communication']},
        ])

        # Create horizontal bar chart
        chart = alt.Chart(score_data).mark_bar().encode(
            x=alt.X("Score:Q", scale=alt.Scale(domain=[0, 10]), title="Score (0-10)"),
            y=alt.Y("Component:N", sort="-x", title=""),
            color=alt.condition(
                alt.datum.Score >= 7,
                alt.value("#2ca02c"),  # Green for good scores
                alt.condition(
                    alt.datum.Score >= 4,
                    alt.value("#ffbb78"),  # Orange for moderate
                    alt.value("#d62728")  # Red for poor
                )
            ),
            tooltip=[
                alt.Tooltip("Component:N", title="Component"),
                alt.Tooltip("Score:Q", title="Score", format=".1f"),
                alt.Tooltip("Weight:N", title="Weight"),
                alt.Tooltip("Contribution:Q", title="Weighted Contribution", format=".2f"),
            ]
        ).properties(
            height=200
        )

        st.altair_chart(chart, use_container_width=True)

        # Display weight explanation
        with st.expander("Understanding the Weights"):
            st.markdown(f"""
            | Component | Weight | Your Score | Contribution |
            |-----------|--------|------------|--------------|
            | Status | {WEIGHTS['status']:.0%} | {result.status_score:.1f} | {result.status_score * WEIGHTS['status']:.2f} |
            | Industry | {WEIGHTS['industry']:.0%} | {result.industry_score:.1f} | {result.industry_score * WEIGHTS['industry']:.2f} |
            | Collateral | {WEIGHTS['collateral']:.0%} | {result.collateral_score:.1f} | {result.collateral_score * WEIGHTS['collateral']:.2f} |
            | Lien Position | {WEIGHTS['lien']:.0%} | {result.lien_score:.1f} | {result.lien_score * WEIGHTS['lien']:.2f} |
            | Communication | {WEIGHTS['communication']:.0%} | {result.communication_score:.1f} | {result.communication_score * WEIGHTS['communication']:.2f} |
            | **Total** | **100%** | | **{result.total_recovery_score:.2f}** |
            """)

    # Disclaimer
    st.warning("""
    **Important Disclaimer:** This is a pre-screening tool only. Results are estimates based on
    general risk factors and should not be used as the sole basis for investment decisions.
    Always conduct thorough due diligence before funding any deal.
    """)


# =============================================================================
# TAB 2: PORTFOLIO BAD DEBT ESTIMATOR
# =============================================================================

with tab_portfolio:
    st.subheader("Portfolio Bad Debt Estimator")
    st.markdown("""
    Analyze estimated bad debt for your **existing portfolio** by loading deals from the database.
    The scoring framework uses conservative defaults for fields not yet captured (collateral, communication).
    """)

    # Load data button
    if st.button("Load Portfolio Data", type="primary"):
        with st.spinner("Loading portfolio data..."):
            try:
                # Load loan data using existing utilities
                from utils.data_loader import load_loan_summaries, load_deals
                from utils.loan_tape_data import prepare_loan_data

                loans_df = load_loan_summaries()
                deals_df = load_deals()

                if loans_df.empty:
                    st.error("No loan data found. Please check your database connection.")
                else:
                    # Prepare data
                    df = prepare_loan_data(loans_df, deals_df)

                    # Store in session state
                    st.session_state["portfolio_df"] = df
                    st.success(f"Loaded {len(df)} loans from the portfolio.")

            except Exception as e:
                st.error(f"Error loading data: {str(e)}")

    # Show analysis if data is loaded
    if "portfolio_df" in st.session_state:
        df = st.session_state["portfolio_df"]

        # Filter options
        st.markdown("#### Filter Options")

        col_filter1, col_filter2 = st.columns(2)

        with col_filter1:
            # Status filter
            status_options = ["All"] + sorted(df["loan_status"].dropna().unique().tolist())
            selected_status = st.selectbox("Filter by Status", options=status_options, index=0)

        with col_filter2:
            # Include paid off toggle
            include_paid_off = st.checkbox("Include Paid Off Loans", value=False)

        # Apply filters
        filtered_df = df.copy()

        if selected_status != "All":
            filtered_df = filtered_df[filtered_df["loan_status"] == selected_status]

        if not include_paid_off:
            filtered_df = filtered_df[filtered_df["loan_status"] != "Paid Off"]

        st.info(f"Analyzing {len(filtered_df)} loans after filters.")

        # Compute recovery scores for the portfolio
        if st.button("Calculate Bad Debt Estimates", type="secondary"):
            with st.spinner("Computing recovery scores for portfolio..."):
                try:
                    # Run batch computation
                    scored_df = compute_recovery_batch(filtered_df)

                    # Store results
                    st.session_state["scored_df"] = scored_df

                    # Get summary
                    summary = get_portfolio_bad_debt_summary(scored_df)

                    st.session_state["portfolio_summary"] = summary

                except Exception as e:
                    st.error(f"Error computing scores: {str(e)}")

        # Display results if available
        if "scored_df" in st.session_state and "portfolio_summary" in st.session_state:
            scored_df = st.session_state["scored_df"]
            summary = st.session_state["portfolio_summary"]

            st.divider()
            st.markdown("### Portfolio Bad Debt Summary")

            # Key metrics
            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.metric(
                    label="Total Exposure",
                    value=f"${summary['total_exposure']:,.0f}",
                    help="Sum of net_balance for all active loans"
                )

            with col2:
                st.metric(
                    label="Est. Bad Debt (Mid)",
                    value=f"${summary['total_bad_debt_mid']:,.0f}",
                    help="Midpoint estimate of bad debt expense"
                )

            with col3:
                st.metric(
                    label="Weighted Loss %",
                    value=f"{summary['weighted_loss_pct']:.1%}",
                    help="Exposure-weighted average loss percentage"
                )

            with col4:
                st.metric(
                    label="Weighted Recovery %",
                    value=f"{summary['weighted_recovery_pct']:.1%}",
                    help="Exposure-weighted average recovery percentage"
                )

            # Bad debt range
            st.markdown("#### Bad Debt Estimate Range")

            col_low, col_mid, col_high = st.columns(3)

            with col_low:
                st.metric(
                    label="Low Estimate",
                    value=f"${summary['total_bad_debt_low']:,.0f}",
                    delta=f"{summary['total_bad_debt_low']/summary['total_exposure']:.1%} of exposure" if summary['total_exposure'] > 0 else "N/A",
                    delta_color="off"
                )

            with col_mid:
                st.metric(
                    label="Midpoint Estimate",
                    value=f"${summary['total_bad_debt_mid']:,.0f}",
                    delta=f"{summary['total_bad_debt_mid']/summary['total_exposure']:.1%} of exposure" if summary['total_exposure'] > 0 else "N/A",
                    delta_color="off"
                )

            with col_high:
                st.metric(
                    label="High Estimate",
                    value=f"${summary['total_bad_debt_high']:,.0f}",
                    delta=f"{summary['total_bad_debt_high']/summary['total_exposure']:.1%} of exposure" if summary['total_exposure'] > 0 else "N/A",
                    delta_color="off"
                )

            st.divider()

            # Distribution by recovery band
            st.markdown("#### Distribution by Recovery Band")

            if summary['deals_by_band']:
                band_df = pd.DataFrame([
                    {"Recovery Band": band, "Count": count}
                    for band, count in summary['deals_by_band'].items()
                ])

                # Sort by band order
                band_order = ["90-100%", "70-89%", "50-69%", "30-49%", "10-29%", "0-9%"]
                band_df["sort_order"] = band_df["Recovery Band"].apply(
                    lambda x: band_order.index(x) if x in band_order else 99
                )
                band_df = band_df.sort_values("sort_order")

                chart = alt.Chart(band_df).mark_bar().encode(
                    x=alt.X("Recovery Band:N", sort=band_order, title="Recovery Band"),
                    y=alt.Y("Count:Q", title="Number of Loans"),
                    color=alt.Color("Recovery Band:N", scale=alt.Scale(
                        domain=band_order,
                        range=["#2ca02c", "#98df8a", "#ffbb78", "#ff7f0e", "#ff9896", "#d62728"]
                    ), legend=None),
                    tooltip=[
                        alt.Tooltip("Recovery Band:N"),
                        alt.Tooltip("Count:Q", title="Loan Count")
                    ]
                ).properties(
                    height=300
                )

                st.altair_chart(chart, use_container_width=True)

            st.divider()

            # Detailed loan table
            st.markdown("#### Detailed Loan Analysis")

            # Select columns to display
            display_cols = [
                "loan_id", "deal_name", "loan_status", "net_balance",
                "recovery_total_score", "recovery_band_label", "recovery_loss_pct",
                "recovery_bad_debt_mid"
            ]
            display_cols = [c for c in display_cols if c in scored_df.columns]

            # Rename for display
            rename_map = {
                "loan_id": "Loan ID",
                "deal_name": "Deal Name",
                "loan_status": "Status",
                "net_balance": "Net Balance",
                "recovery_total_score": "Recovery Score",
                "recovery_band_label": "Recovery Band",
                "recovery_loss_pct": "Est. Loss %",
                "recovery_bad_debt_mid": "Est. Bad Debt",
            }

            display_df = scored_df[display_cols].copy()
            display_df = display_df.rename(columns=rename_map)

            # Format columns
            if "Net Balance" in display_df.columns:
                display_df["Net Balance"] = display_df["Net Balance"].apply(
                    lambda x: f"${x:,.0f}" if pd.notna(x) else "N/A"
                )
            if "Recovery Score" in display_df.columns:
                display_df["Recovery Score"] = display_df["Recovery Score"].apply(
                    lambda x: f"{x:.1f}" if pd.notna(x) else "N/A"
                )
            if "Est. Loss %" in display_df.columns:
                display_df["Est. Loss %"] = display_df["Est. Loss %"].apply(
                    lambda x: f"{x:.1%}" if pd.notna(x) else "N/A"
                )
            if "Est. Bad Debt" in display_df.columns:
                display_df["Est. Bad Debt"] = display_df["Est. Bad Debt"].apply(
                    lambda x: f"${x:,.0f}" if pd.notna(x) else "N/A"
                )

            # Sort by bad debt (highest first)
            if "recovery_bad_debt_mid" in scored_df.columns:
                sort_idx = scored_df["recovery_bad_debt_mid"].fillna(0).sort_values(ascending=False).index
                display_df = display_df.loc[sort_idx]

            st.dataframe(display_df.head(50), use_container_width=True, hide_index=True)

            if len(scored_df) > 50:
                st.caption(f"Showing top 50 of {len(scored_df)} loans by estimated bad debt.")

            # Download option
            csv = scored_df.to_csv(index=False)
            st.download_button(
                label="Download Full Results (CSV)",
                data=csv,
                file_name="recovery_analysis.csv",
                mime="text/csv",
            )

            # Disclaimer
            st.warning("""
            **Important Note:** These estimates use conservative defaults:
            - **Collateral Score:** Defaulted to 1 (Unsecured) - actual collateral is not yet tracked
            - **Communication Score:** Defaulted to 3 (Sporadic) - borrower communication is not yet tracked

            The actual bad debt may be lower if loans have better collateral or communication than assumed.
            """)

    else:
        st.info("Click 'Load Portfolio Data' above to begin analysis.")


# =============================================================================
# SIDEBAR INFORMATION
# =============================================================================

with st.sidebar:
    st.markdown("### Recovery Score Guide")

    st.markdown("""
    **Score Ranges:**
    - 9-10: Excellent (90-100% recovery)
    - 7-8.9: Good (70-89% recovery)
    - 5-6.9: Moderate (50-69% recovery)
    - 3-4.9: Concerning (30-49% recovery)
    - 1-2.9: Poor (10-29% recovery)
    - 0-1: Severe (<10% recovery)
    """)

    st.divider()

    st.markdown("### Weight Allocation")
    st.markdown(f"""
    - Status: **{WEIGHTS['status']:.0%}**
    - Industry: **{WEIGHTS['industry']:.0%}**
    - Collateral: **{WEIGHTS['collateral']:.0%}**
    - Lien Position: **{WEIGHTS['lien']:.0%}**
    - Communication: **{WEIGHTS['communication']:.0%}**
    """)

    st.divider()

    st.markdown("### Data Sources")
    st.markdown("""
    **Pre-Screen:** Manual inputs

    **Portfolio Analysis:**
    - Status: `loan_status`
    - Industry: `sector_code` (NAICS)
    - Lien: `ahead_positions`
    - Exposure: `net_balance`
    - Collateral: Default (1)
    - Communication: Default (3)
    """)
