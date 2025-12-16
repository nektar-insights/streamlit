# pages/bad_debt_estimator.py
"""
Bad Debt Estimator

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
    WEIGHTS,
)
from utils.display_components import (
    create_status_filter,
    create_partner_source_filter,
    create_date_range_filter,
)

# Page setup
setup_page("CSL Capital | Bad Debt Estimator")

# =============================================================================
# PAGE HEADER
# =============================================================================

st.title("Bad Debt Estimator")

st.markdown("""
This tool uses a **unified recovery scoring framework** to estimate potential losses.
The same scoring logic applies to both pre-screening prospective deals and estimating
bad debt on your existing portfolio.
""")

with st.expander("Scoring Methodology", expanded=False):
    st.markdown("""
    **Pre-Screen Scoring (4 components):**

    | Component | Weight | Description |
    |-----------|--------|-------------|
    | Status | 11% | Loan status severity (Active, Delinquent, Default, etc.) |
    | Industry | 33% | NAICS sector risk based on historical performance |
    | Collateral | 33% | Collateral type and quality (Real Estate, Equipment, Unsecured) |
    | Lien Position | 22% | Position in capital stack (1st Lien, 2nd Lien, etc.) |

    *Note: Communication is excluded from pre-screening since borrower responsiveness is unknown at that stage.*

    **Portfolio Scoring (5 components):** Includes Communication (10% weight) when analyzing existing loans.

    **Recovery Score Bands:**
    - **9-10**: Excellent (90-100% expected recovery)
    - **7-8.9**: Good (70-89% expected recovery)
    - **5-6.9**: Moderate (50-69% expected recovery)
    - **3-4.9**: Concerning (30-49% expected recovery)
    - **1-2.9**: Poor (10-29% expected recovery)
    - **0-1**: Severe (<10% expected recovery)
    """)

st.divider()

# =============================================================================
# TAB NAVIGATION
# =============================================================================

tab_portfolio, tab_prescreen = st.tabs([
    "Portfolio Bad Debt Estimator",
    "Pre-Screen Tool"
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

        # Calculate button
        calculate_btn = st.button("Calculate Recovery Score", type="primary", width='stretch')

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

        # Create score breakdown data (excludes Communication which uses a default for pre-screen)
        score_data = pd.DataFrame([
            {"Component": "Status", "Score": result.status_score, "Weight": f"{WEIGHTS['status']:.0%}", "Contribution": result.status_score * WEIGHTS['status']},
            {"Component": "Industry", "Score": result.industry_score, "Weight": f"{WEIGHTS['industry']:.0%}", "Contribution": result.industry_score * WEIGHTS['industry']},
            {"Component": "Collateral", "Score": result.collateral_score, "Weight": f"{WEIGHTS['collateral']:.0%}", "Contribution": result.collateral_score * WEIGHTS['collateral']},
            {"Component": "Lien Position", "Score": result.lien_score, "Weight": f"{WEIGHTS['lien']:.0%}", "Contribution": result.lien_score * WEIGHTS['lien']},
        ])

        # Add color category for Altair v6 compatibility (nested conditions not supported)
        def score_color_category(score):
            if score >= 7:
                return "Good"
            elif score >= 4:
                return "Moderate"
            else:
                return "Poor"

        score_data["ColorCategory"] = score_data["Score"].apply(score_color_category)

        # Create horizontal bar chart
        chart = alt.Chart(score_data).mark_bar().encode(
            x=alt.X("Score:Q", scale=alt.Scale(domain=[0, 10]), title="Score (0-10)"),
            y=alt.Y("Component:N", sort="-x", title=""),
            color=alt.Color(
                "ColorCategory:N",
                scale=alt.Scale(
                    domain=["Good", "Moderate", "Poor"],
                    range=["#2ca02c", "#ffbb78", "#d62728"]
                ),
                legend=None
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

        st.altair_chart(chart, width='stretch')

        # Display weight explanation
        with st.expander("Understanding the Weights"):
            # Pre-screen weights (communication excluded, weights redistributed)
            ps_status = WEIGHTS['status'] / 0.90
            ps_industry = WEIGHTS['industry'] / 0.90
            ps_collateral = WEIGHTS['collateral'] / 0.90
            ps_lien = WEIGHTS['lien'] / 0.90

            st.markdown(f"""
            | Component | Weight | Your Score | Contribution |
            |-----------|--------|------------|--------------|
            | Status | {ps_status:.0%} | {result.status_score:.1f} | {result.status_score * ps_status:.2f} |
            | Industry | {ps_industry:.0%} | {result.industry_score:.1f} | {result.industry_score * ps_industry:.2f} |
            | Collateral | {ps_collateral:.0%} | {result.collateral_score:.1f} | {result.collateral_score * ps_collateral:.2f} |
            | Lien Position | {ps_lien:.0%} | {result.lien_score:.1f} | {result.lien_score * ps_lien:.2f} |
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

        # =================================================================
        # FILTER OPTIONS
        # =================================================================
        st.markdown("#### Filter Options")

        col_filter1, col_filter2 = st.columns(2)

        with col_filter1:
            # Status filter using centralized component
            filtered_df, selected_status = create_status_filter(
                df,
                status_col="loan_status",
                label="Filter by Status",
                include_all_option=True,
                key_prefix="bde_status"
            )

        with col_filter2:
            # Partner filter using centralized component
            filtered_df, selected_partners = create_partner_source_filter(
                filtered_df,
                partner_col="partner_source",
                label="Filter by Partner Source",
                default_all=True,
                key_prefix="bde_partner"
            )

        # Date/Vintage filter
        st.markdown("##### Vintage Filter")
        filtered_df, date_filter_active = create_date_range_filter(
            filtered_df,
            date_col="funding_date",
            label="Select Funding Date Range",
            checkbox_label="Filter by Vintage (Funding Date)",
            default_enabled=False,
            key_prefix="bde_vintage"
        )

        # Include paid off toggle
        include_paid_off = st.checkbox("Include Paid Off Loans", value=False, key="bde_include_paid_off")

        # Apply paid off filter
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

                st.altair_chart(chart, width='stretch')

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

            st.dataframe(display_df.head(50), width='stretch', hide_index=True)

            if len(scored_df) > 50:
                st.caption(f"Showing top 50 of {len(scored_df)} loans by estimated bad debt.")

            # Download option
            csv = scored_df.to_csv(index=False)
            st.download_button(
                label="Download Full Results (CSV)",
                data=csv,
                file_name="bad_debt_analysis.csv",
                mime="text/csv",
            )

            # Dynamic disclaimer based on data availability
            collateral_stats = summary.get("collateral_data_stats", {"with_data": 0, "using_default": len(scored_df)})
            communication_stats = summary.get("communication_data_stats", {"with_data": 0, "using_default": len(scored_df)})

            total_loans = len(scored_df)
            collateral_with_data = collateral_stats["with_data"]
            communication_with_data = communication_stats["with_data"]

            # Build dynamic disclaimer
            disclaimer_parts = []

            if collateral_stats["using_default"] > 0:
                if collateral_with_data > 0:
                    disclaimer_parts.append(
                        f"- **Collateral Score:** {collateral_with_data} loans have actual data; "
                        f"{collateral_stats['using_default']} loans use default (1 = Unsecured)"
                    )
                else:
                    disclaimer_parts.append(
                        "- **Collateral Score:** All loans use default (1 = Unsecured) - no collateral data tracked"
                    )

            if communication_stats["using_default"] > 0:
                if communication_with_data > 0:
                    disclaimer_parts.append(
                        f"- **Communication Score:** {communication_with_data} loans have actual data; "
                        f"{communication_stats['using_default']} loans use default (3 = Sporadic)"
                    )
                else:
                    disclaimer_parts.append(
                        "- **Communication Score:** All loans use default (3 = Sporadic) - no communication data tracked"
                    )

            # Show appropriate message based on data availability
            if disclaimer_parts:
                st.warning(f"""
                **Data Availability Note:** Some estimates use conservative defaults where HubSpot data is missing:
                {chr(10).join(disclaimer_parts)}

                Actual bad debt may be lower for loans with better collateral or communication than the defaults assume.
                """)
            else:
                st.success("""
                **Data Quality:** All loans have actual collateral and communication data from HubSpot.
                No conservative defaults were applied to this analysis.
                """)

    else:
        st.info("Click 'Load Portfolio Data' above to begin analysis.")


