# pages/tools.py
"""
Tools - Collection of analytical and forecasting tools

This page consolidates three key analytical capabilities:
1. Bad Debt Estimator: Pre-screen deals and estimate portfolio bad debt
2. Capital Forecast: Cash flow projections and forecasting
3. Deal Scorer: ML-based deal screening and model diagnostics
"""

import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
from datetime import date, timedelta
from dateutil.relativedelta import relativedelta

from utils.config import setup_page, PRIMARY_COLOR
from utils.data_loader import DataLoader, load_deals, load_qbo_data, load_loan_schedules
from utils.preprocessing import preprocess_dataframe
from utils.cash_flow_forecast import create_cash_flow_forecast
from utils.display_components import (
    create_status_filter,
    create_multi_status_filter,
    create_partner_source_filter,
    create_date_range_filter,
    create_loan_id_filter,
)

# Page setup
setup_page("CSL Capital | Tools")

st.title("Tools")
st.markdown("Analytical and forecasting tools for portfolio management and deal screening.")

# =============================================================================
# MAIN TAB NAVIGATION
# =============================================================================
tab_bad_debt, tab_forecast, tab_scorer, tab_maturity = st.tabs([
    "Bad Debt Estimator",
    "Capital Forecast",
    "Deal Scorer",
    "Loan Maturity Calculator"
])

# =============================================================================
# TAB 1: BAD DEBT ESTIMATOR
# =============================================================================
with tab_bad_debt:
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

    st.header("Bad Debt Estimator")

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

    # Sub-tabs for Bad Debt Estimator
    bde_tab_portfolio, bde_tab_prescreen = st.tabs([
        "Portfolio Bad Debt Estimator",
        "Pre-Screen Tool"
    ])

    # =========================================================================
    # BAD DEBT - PRE-SCREEN TOOL
    # =========================================================================
    with bde_tab_prescreen:
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
                help="The total amount at risk for this deal",
                key="bde_exposure"
            )

            # Status Category
            status_category = st.selectbox(
                "Current Status",
                options=STATUS_CATEGORIES,
                index=6,  # Default to "Active (Healthy)"
                help="The current or expected status of the borrower",
                key="bde_status_cat"
            )

            # Industry Category
            industry_category = st.selectbox(
                "Industry",
                options=INDUSTRY_CATEGORIES,
                index=6,  # Default to "Business / Professional Services"
                help="The borrower's industry sector",
                key="bde_industry"
            )

            # Collateral Type
            collateral_type = st.selectbox(
                "Collateral Type",
                options=COLLATERAL_CATEGORIES,
                index=0,  # Default to "None / Unsecured"
                help="The type and quality of collateral securing the loan",
                key="bde_collateral"
            )

            # Lien Position
            lien_position = st.selectbox(
                "Lien Position",
                options=LIEN_CATEGORIES,
                index=0,  # Default to "First Lien"
                help="Your position in the capital stack",
                key="bde_lien"
            )

            # Calculate button
            calculate_btn = st.button("Calculate Recovery Score", type="primary", use_container_width=True, key="bde_calc")

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

            # Create score breakdown data
            score_data = pd.DataFrame([
                {"Component": "Status", "Score": result.status_score, "Weight": f"{WEIGHTS['status']:.0%}", "Contribution": result.status_score * WEIGHTS['status']},
                {"Component": "Industry", "Score": result.industry_score, "Weight": f"{WEIGHTS['industry']:.0%}", "Contribution": result.industry_score * WEIGHTS['industry']},
                {"Component": "Collateral", "Score": result.collateral_score, "Weight": f"{WEIGHTS['collateral']:.0%}", "Contribution": result.collateral_score * WEIGHTS['collateral']},
                {"Component": "Lien Position", "Score": result.lien_score, "Weight": f"{WEIGHTS['lien']:.0%}", "Contribution": result.lien_score * WEIGHTS['lien']},
            ])

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

            st.altair_chart(chart, use_container_width=True)

            # Display weight explanation
            with st.expander("Understanding the Weights"):
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

    # =========================================================================
    # BAD DEBT - PORTFOLIO ESTIMATOR
    # =========================================================================
    with bde_tab_portfolio:
        from utils.data_loader import load_loan_summaries
        from utils.loan_tape_data import prepare_loan_data

        st.subheader("Portfolio Bad Debt Estimator")
        st.markdown("""
        Analyze estimated bad debt for your **existing portfolio** by loading deals from the database.
        The scoring framework uses conservative defaults for fields not yet captured (collateral, communication).
        """)

        # Load data button
        if st.button("Load Portfolio Data", type="primary", key="bde_load"):
            with st.spinner("Loading portfolio data..."):
                try:
                    loans_df = load_loan_summaries()
                    deals_df = load_deals()

                    if loans_df.empty:
                        st.error("No loan data found. Please check your database connection.")
                    else:
                        df = prepare_loan_data(loans_df, deals_df)
                        st.session_state["bde_portfolio_df"] = df
                        st.success(f"Loaded {len(df)} loans from the portfolio.")

                except Exception as e:
                    st.error(f"Error loading data: {str(e)}")

        # Show analysis if data is loaded
        if "bde_portfolio_df" in st.session_state:
            df = st.session_state["bde_portfolio_df"]

            st.markdown("#### Filter Options")

            col_filter1, col_filter2 = st.columns(2)

            with col_filter1:
                filtered_df, selected_statuses = create_multi_status_filter(
                    df,
                    status_col="loan_status",
                    label="Filter by Status (multi-select)",
                    default_all=True,
                    key_prefix="bde_port_status"
                )

            with col_filter2:
                filtered_df, selected_partners = create_partner_source_filter(
                    filtered_df,
                    partner_col="partner_source",
                    label="Filter by Partner Source",
                    default_all=True,
                    key_prefix="bde_port_partner"
                )

            # Loan ID filter
            st.markdown("##### Select Specific Loans")
            filtered_df, selected_loan_ids = create_loan_id_filter(
                filtered_df,
                loan_id_col="loan_id",
                label="Filter by Loan ID (leave empty for all)",
                key_prefix="bde_port_loan_id"
            )

            st.markdown("##### Vintage Filter")
            filtered_df, date_filter_active = create_date_range_filter(
                filtered_df,
                date_col="funding_date",
                label="Select Funding Date Range",
                checkbox_label="Filter by Vintage (Funding Date)",
                default_enabled=False,
                key_prefix="bde_port_vintage"
            )

            include_paid_off = st.checkbox("Include Paid Off Loans", value=False, key="bde_port_paid_off")

            if not include_paid_off and "Paid Off" in selected_statuses:
                filtered_df = filtered_df[filtered_df["loan_status"] != "Paid Off"]

            st.info(f"Analyzing {len(filtered_df)} loans after filters.")

            if st.button("Calculate Bad Debt Estimates", type="secondary", key="bde_calc_portfolio"):
                with st.spinner("Computing recovery scores for portfolio..."):
                    try:
                        scored_df = compute_recovery_batch(filtered_df)
                        st.session_state["bde_scored_df"] = scored_df
                        summary = get_portfolio_bad_debt_summary(scored_df)
                        st.session_state["bde_portfolio_summary"] = summary

                    except Exception as e:
                        st.error(f"Error computing scores: {str(e)}")

            if "bde_scored_df" in st.session_state and "bde_portfolio_summary" in st.session_state:
                scored_df = st.session_state["bde_scored_df"]
                summary = st.session_state["bde_portfolio_summary"]

                st.divider()
                st.markdown("### Portfolio Bad Debt Summary")

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

                st.markdown("#### Distribution by Recovery Band")

                if summary['deals_by_band']:
                    band_df = pd.DataFrame([
                        {"Recovery Band": band, "Count": count}
                        for band, count in summary['deals_by_band'].items()
                    ])

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

                st.markdown("#### Detailed Loan Analysis")

                display_cols = [
                    "loan_id", "deal_name", "loan_status", "net_balance",
                    "recovery_total_score", "recovery_band_label", "recovery_loss_pct",
                    "recovery_bad_debt_mid"
                ]
                display_cols = [c for c in display_cols if c in scored_df.columns]

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

                if "recovery_bad_debt_mid" in scored_df.columns:
                    sort_idx = scored_df["recovery_bad_debt_mid"].fillna(0).sort_values(ascending=False).index
                    display_df = display_df.loc[sort_idx]

                st.dataframe(display_df.head(50), use_container_width=True, hide_index=True)

                if len(scored_df) > 50:
                    st.caption(f"Showing top 50 of {len(scored_df)} loans by estimated bad debt.")

                csv = scored_df.to_csv(index=False)
                st.download_button(
                    label="Download Full Results (CSV)",
                    data=csv,
                    file_name="bad_debt_analysis.csv",
                    mime="text/csv",
                    key="bde_download"
                )

                collateral_stats = summary.get("collateral_data_stats", {"with_data": 0, "using_default": len(scored_df)})
                communication_stats = summary.get("communication_data_stats", {"with_data": 0, "using_default": len(scored_df)})

                disclaimer_parts = []

                if collateral_stats["using_default"] > 0:
                    if collateral_stats["with_data"] > 0:
                        disclaimer_parts.append(
                            f"- **Collateral Score:** {collateral_stats['with_data']} loans have actual data; "
                            f"{collateral_stats['using_default']} loans use default (1 = Unsecured)"
                        )
                    else:
                        disclaimer_parts.append(
                            "- **Collateral Score:** All loans use default (1 = Unsecured) - no collateral data tracked"
                        )

                if communication_stats["using_default"] > 0:
                    if communication_stats["with_data"] > 0:
                        disclaimer_parts.append(
                            f"- **Communication Score:** {communication_stats['with_data']} loans have actual data; "
                            f"{communication_stats['using_default']} loans use default (3 = Sporadic)"
                        )
                    else:
                        disclaimer_parts.append(
                            "- **Communication Score:** All loans use default (3 = Sporadic) - no communication data tracked"
                        )

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


# =============================================================================
# TAB 2: CAPITAL FORECAST
# =============================================================================
with tab_forecast:
    st.header("Capital Forecast")
    st.markdown("Cash flow projections and forecasting for forward-looking capital requirements analysis.")

    # Load QBO data
    qbo_df, gl_df = load_qbo_data()

    # Load and preprocess deals
    forecast_df = load_deals()

    # Preprocess using centralized utility
    forecast_df = preprocess_dataframe(
        forecast_df,
        numeric_cols=["amount", "total_funded_amount", "factor_rate", "loan_term", "commission"],
        date_cols=["date_created"]
    )

    # Filters in expander
    with st.expander("Filters", expanded=False):
        col1, col2 = st.columns(2)

        with col1:
            # Date range filter
            forecast_filtered_df, _ = create_date_range_filter(
                forecast_df,
                date_col="date_created",
                label="Select Date Range",
                checkbox_label="Filter by Date Created",
                default_enabled=False,
                key_prefix="forecast_date"
            )

        with col2:
            # Status filter
            status_col = "dealstage" if "dealstage" in forecast_filtered_df.columns else ("stage" if "stage" in forecast_filtered_df.columns else None)
            if status_col:
                forecast_filtered_df, selected_status = create_status_filter(
                    forecast_filtered_df,
                    status_col=status_col,
                    label="Filter by Deal Stage",
                    include_all_option=True,
                    key_prefix="forecast_status"
                )

        st.write(f"**Showing:** {len(forecast_filtered_df)} of {len(forecast_df)} deals")

    # Filter for closed won deals
    closed_won = forecast_filtered_df[forecast_filtered_df["is_closed_won"] == True] if "is_closed_won" in forecast_filtered_df.columns else forecast_filtered_df

    # Create forecast
    create_cash_flow_forecast(forecast_filtered_df, closed_won, qbo_df)


# =============================================================================
# TAB 3: DEAL SCORER
# =============================================================================
with tab_scorer:
    from utils.loan_tape_data import prepare_loan_data, consolidate_sector_code
    from utils.loan_tape_ml import (
        get_origination_model_coefficients,
        calculate_deal_risk_score,
        train_classification_small,
        train_regression_small,
        train_current_risk_model,
        create_coefficient_chart,
        render_ml_explainer,
        render_model_summary,
        render_executive_summary,
        render_factor_impact_examples,
        render_batch_scorer,
        NAICS_SECTOR_NAMES,
        RISK_LEVELS,
    )
    from utils.loan_tape_analytics import get_similar_deals_comparison

    st.header("Deal Scorer")
    st.markdown("ML-based deal screening and model diagnostics for risk assessment.")

    @st.cache_data(ttl=3600)
    def load_scorer_data():
        """Load and prepare loan data for model training."""
        loader = DataLoader()
        loans_df = loader.load_loan_summaries()
        deals_df = loader.load_deals()

        if loans_df.empty or deals_df.empty:
            return pd.DataFrame(), []

        df = prepare_loan_data(loans_df, deals_df)

        partners = []
        if "partner_source" in df.columns:
            partners = sorted(df["partner_source"].dropna().unique().tolist())

        return df, partners

    @st.cache_data(ttl=3600)
    def train_scorer_model(_df):
        """Train model and extract coefficients (cached)."""
        if _df.empty:
            return None
        return get_origination_model_coefficients(_df)

    # Load data
    with st.spinner("Loading portfolio data..."):
        scorer_df, partners = load_scorer_data()

    if scorer_df.empty:
        st.error("Unable to load loan data. Please check database connection.")
    else:
        # Train model and get coefficients
        with st.spinner("Training risk model on historical data..."):
            coefficients_data = train_scorer_model(scorer_df)

        if coefficients_data is None:
            st.error("Unable to train risk model. Insufficient data.")
        else:
            # Sub-tabs for Deal Scorer
            scorer_tab_new, scorer_tab_batch, scorer_tab_diag = st.tabs(["Score New Deal", "Batch Scoring", "Model Diagnostics"])

            # =================================================================
            # DEAL SCORER - SCORE NEW DEAL
            # =================================================================
            with scorer_tab_new:
                st.subheader("Screen a Prospective Deal")
                st.markdown("""
                Use this tool to **evaluate a new deal before committing capital**. Enter the deal
                characteristics below and the model will predict the likelihood of it becoming a
                problem loan based on historical portfolio patterns.
                """)

                model_quality = coefficients_data.get("model_quality", {})
                with st.expander("Model Information", expanded=False):
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        roc_auc = model_quality.get("roc_auc", 0)
                        st.metric("Model ROC AUC", f"{roc_auc:.3f}" if not np.isnan(roc_auc) else "N/A")
                    with col2:
                        st.metric("Training Samples", f"{model_quality.get('n_samples', 0):,}")
                    with col3:
                        st.metric("Historical Problem Rate", f"{model_quality.get('problem_rate', 0):.1%}")

                    st.caption("Model trained on historical loan outcomes to predict problem loan probability.")

                st.markdown("---")
                st.markdown("##### Deal Inputs")

                col1, col2 = st.columns(2)

                with col1:
                    partner_options = ["Select Partner..."] + partners
                    selected_partner = st.selectbox(
                        "Partner Source",
                        options=partner_options,
                        index=0,
                        help="The originating partner for this deal",
                        key="scorer_partner"
                    )
                    partner = selected_partner if selected_partner != "Select Partner..." else None

                    fico = st.number_input(
                        "FICO Score",
                        min_value=300,
                        max_value=850,
                        value=650,
                        step=5,
                        help="Borrower's credit score",
                        key="scorer_fico"
                    )

                    tib = st.number_input(
                        "Time in Business (Years)",
                        min_value=0.0,
                        max_value=100.0,
                        value=5.0,
                        step=0.5,
                        help="Years the business has been operating",
                        key="scorer_tib"
                    )

                    position_options = {
                        "1st (Best)": 0,
                        "2nd": 1,
                        "3rd+": 2
                    }
                    selected_position = st.selectbox(
                        "Lien Position",
                        options=list(position_options.keys()),
                        index=0,
                        help="Position in lien hierarchy (0=1st lien, best; 2=3rd+, worst)",
                        key="scorer_position"
                    )
                    position = position_options[selected_position]

                with col2:
                    sector_options = {"Select Industry...": None}
                    sector_options.update({f"{code} - {name}": code for code, name in sorted(NAICS_SECTOR_NAMES.items())})

                    selected_sector = st.selectbox(
                        "Industry (NAICS Sector)",
                        options=list(sector_options.keys()),
                        index=0,
                        help="2-digit NAICS sector code",
                        key="scorer_sector"
                    )
                    sector_code = sector_options[selected_sector]

                    deal_size = st.number_input(
                        "CSL Participation ($)",
                        min_value=0,
                        max_value=10_000_000,
                        value=50_000,
                        step=5000,
                        format="%d",
                        help="CSL's capital at risk in this deal (not total deal size)",
                        key="scorer_size"
                    )

                    factor_rate = st.number_input(
                        "Factor Rate",
                        min_value=1.0,
                        max_value=2.0,
                        value=1.30,
                        step=0.01,
                        format="%.2f",
                        help="Factor rate (e.g., 1.30 = 30% return)",
                        key="scorer_factor"
                    )

                    commission = st.number_input(
                        "Commission (%)",
                        min_value=0.0,
                        max_value=20.0,
                        value=5.0,
                        step=0.5,
                        help="Commission fee percentage",
                        key="scorer_commission"
                    ) / 100

                st.markdown("---")
                score_button = st.button("Score This Deal", type="primary", use_container_width=True, key="scorer_calc")

                if score_button:
                    if partner is None and sector_code is None:
                        st.warning("Please select at least a Partner or Industry to get a meaningful score.")

                    with st.spinner("Calculating risk score..."):
                        result = calculate_deal_risk_score(
                            partner=partner,
                            fico=fico,
                            tib=tib,
                            position=position,
                            sector_code=sector_code,
                            deal_size=deal_size,
                            factor_rate=factor_rate,
                            commission=commission,
                            coefficients_data=coefficients_data
                        )

                    st.markdown("---")
                    st.subheader("Risk Assessment")

                    col1, col2, col3 = st.columns([2, 1, 1])

                    with col1:
                        risk_color = result["risk_color"]
                        risk_emoji = result["risk_emoji"]
                        risk_level = result["risk_level"]
                        total_score = result["total_score"]

                        st.markdown(f"""
                        <div style="
                            background: linear-gradient(135deg, {risk_color}22, {risk_color}44);
                            border-left: 5px solid {risk_color};
                            padding: 20px;
                            border-radius: 8px;
                            margin-bottom: 20px;
                        ">
                            <h2 style="margin: 0; color: {risk_color};">
                                {risk_emoji} Overall Risk Score: {total_score:.0f} / 100
                            </h2>
                            <p style="margin: 5px 0 0 0; font-size: 1.2em; color: {risk_color};">
                                Risk Level: <strong>{risk_level}</strong>
                            </p>
                        </div>
                        """, unsafe_allow_html=True)

                    with col2:
                        st.metric("Raw Probability", f"{result['raw_probability']:.1%}")

                    with col3:
                        st.metric("Portfolio Avg", f"{model_quality.get('problem_rate', 0):.1%}")

                    st.markdown("##### Factor Breakdown")

                    factor_contributions = result["factor_contributions"]
                    if factor_contributions:
                        factor_df = pd.DataFrame(factor_contributions)

                        factor_df["color"] = factor_df["contribution"].apply(
                            lambda x: "#d62728" if x > 0 else "#2ca02c" if x < 0 else "#808080"
                        )

                        chart = alt.Chart(factor_df).mark_bar().encode(
                            x=alt.X("contribution:Q", title="Risk Contribution (points)",
                                    scale=alt.Scale(domain=[-30, 30])),
                            y=alt.Y("factor:N", title="Factor", sort="-x"),
                            color=alt.Color("color:N", scale=None),
                            tooltip=[
                                alt.Tooltip("factor:N", title="Factor"),
                                alt.Tooltip("contribution:Q", title="Contribution", format="+.1f")
                            ]
                        ).properties(
                            height=max(200, len(factor_df) * 35),
                            title="Risk Factor Contributions"
                        )

                        rule = alt.Chart(pd.DataFrame({"x": [0]})).mark_rule(
                            color="gray", strokeDash=[3, 3]
                        ).encode(x="x:Q")

                        st.altair_chart(chart + rule, use_container_width=True)

                        st.caption("Red bars increase risk, green bars decrease risk. Larger absolute values have more impact.")

                    st.markdown("---")
                    st.subheader("Similar Historical Deals")

                    fico_range = (fico - 25, fico + 25) if fico else None
                    tib_range = (max(0, tib - 2), tib + 2) if tib else None

                    comparison = get_similar_deals_comparison(
                        df=scorer_df,
                        partner=partner,
                        sector_code=sector_code,
                        fico_range=fico_range,
                        tib_range=tib_range,
                        position=position
                    )

                    similar_count = comparison["similar_count"]

                    if similar_count > 0:
                        st.markdown(f"**Found {similar_count} similar deals** ({comparison['match_criteria']})")

                        col1, col2, col3 = st.columns(3)

                        with col1:
                            similar_problem_rate = comparison["similar_problem_rate"]
                            portfolio_problem_rate = comparison["portfolio_problem_rate"]
                            delta = similar_problem_rate - portfolio_problem_rate
                            st.metric(
                                "Problem Rate",
                                f"{similar_problem_rate:.1%}",
                                delta=f"{delta:+.1%} vs portfolio",
                                delta_color="inverse"
                            )

                        with col2:
                            similar_perf = comparison["similar_avg_performance"]
                            portfolio_perf = comparison["portfolio_avg_performance"]
                            delta = similar_perf - portfolio_perf
                            st.metric(
                                "Avg Payment Performance",
                                f"{similar_perf:.1%}",
                                delta=f"{delta:+.1%} vs portfolio",
                                delta_color="normal"
                            )

                        with col3:
                            risk_mult = comparison["risk_multiplier"]
                            st.metric(
                                "Risk Multiplier",
                                f"{risk_mult:.1f}x",
                                delta="vs portfolio avg" if risk_mult > 1.1 else None,
                                delta_color="inverse" if risk_mult > 1.1 else "off"
                            )

                        for warning in comparison["warnings"]:
                            st.warning(f"Warning: {warning}")

                        with st.expander("View Similar Deals"):
                            similar_deals = comparison["similar_deals"]

                            display_cols = ["deal_name", "partner_source", "loan_status", "payment_performance",
                                          "fico", "tib", "total_invested", "is_problem", "problem_reason"]
                            display_cols = [c for c in display_cols if c in similar_deals.columns]

                            if not similar_deals.empty:
                                display_df = similar_deals[display_cols].head(20).copy()

                                problem_reasons = display_df.get("problem_reason", pd.Series("", index=display_df.index))

                                if "payment_performance" in display_df.columns:
                                    display_df["payment_performance"] = display_df["payment_performance"].apply(
                                        lambda x: f"{x:.1%}" if pd.notnull(x) else "N/A"
                                    )
                                if "total_invested" in display_df.columns:
                                    display_df["total_invested"] = display_df["total_invested"].apply(
                                        lambda x: f"${x:,.0f}" if pd.notnull(x) else "N/A"
                                    )
                                if "is_problem" in display_df.columns:
                                    display_df["is_problem"] = display_df["is_problem"].apply(
                                        lambda x: "Yes" if x else "No"
                                    )

                                if "problem_reason" in display_df.columns:
                                    display_df["problem_reason"] = problem_reasons.apply(
                                        lambda x: x if x else "-"
                                    )

                                rename_map = {
                                    "deal_name": "Deal Name",
                                    "partner_source": "Partner",
                                    "loan_status": "Status",
                                    "payment_performance": "Performance",
                                    "fico": "FICO",
                                    "tib": "TIB (yrs)",
                                    "total_invested": "Total Cost Basis",
                                    "is_problem": "Problem?",
                                    "problem_reason": "Reason"
                                }
                                display_df.rename(columns=rename_map, inplace=True)

                                st.dataframe(display_df, use_container_width=True, hide_index=True)

                                st.caption("""
                                **Problem Classification Logic:**
                                - **Status-based**: Loan status is Default, Bankruptcy, Charged Off, In Collections, Legal Action, NSF/Suspended, Non-Performing, Severe/Moderate Delinquency, or Active - Frequently Late
                                - **Paid off underperformance**: Loan paid off but recovered less than 90% of expected payments
                                - **Behind schedule**: Active loan is 15+ percentage points behind expected payment progress based on loan age
                                """)
                    else:
                        st.info("No similar historical deals found with current criteria. Try selecting fewer filters.")

                    st.markdown("---")
                    st.subheader("Recommendation")

                    recommendations = result["recommendations"]

                    if total_score < 40:
                        st.success(f"""
                        **LOW RISK - Recommended for approval**

                        This deal scores well on historical risk factors. Standard underwriting should apply.
                        """)
                    elif total_score < 60:
                        st.info(f"""
                        **MODERATE RISK - Proceed with caution**

                        This deal has some risk factors to consider. Review the factor breakdown above.
                        """)
                    elif total_score < 80:
                        st.warning(f"""
                        **ELEVATED RISK - Additional review recommended**

                        This deal shows elevated risk based on historical patterns. Consider the following:
                        """)
                        if recommendations:
                            for rec in recommendations:
                                st.markdown(f"- {rec}")
                    else:
                        st.error(f"""
                        **HIGH RISK - Consider declining or requiring:**
                        """)
                        if recommendations:
                            for rec in recommendations:
                                st.markdown(f"- {rec}")
                        else:
                            st.markdown("- Significant risk mitigation measures before approval")

                st.markdown("---")
                with st.expander("How This Works"):
                    st.markdown("""
                    ### Deal Risk Scoring Methodology

                    This tool uses a **logistic regression model** trained on historical loan outcomes to predict
                    the probability that a new deal will become a "problem loan."

                    #### Features Used:
                    - **FICO Score**: Borrower creditworthiness
                    - **Time in Business**: Business maturity/stability
                    - **Lien Position**: Priority in repayment hierarchy
                    - **Industry (NAICS)**: Sector-specific risk patterns
                    - **Partner Source**: Historical performance by originator
                    - **CSL Participation**: CSL's capital at risk in the deal

                    #### Risk Score Interpretation:
                    | Score Range | Risk Level | Recommendation |
                    |-------------|------------|----------------|
                    | 0-40 | LOW | Standard approval |
                    | 40-60 | MODERATE | Proceed with caution |
                    | 60-80 | ELEVATED | Additional review needed |
                    | 80-100 | HIGH | Consider declining |

                    #### Problem Loan Classification:
                    A loan is classified as a "problem" if ANY of the following apply:

                    1. **Status-based problems**: The loan status is one of:
                       - Default, Bankruptcy, Charged Off
                       - In Collections, Legal Action
                       - NSF / Suspended, Non-Performing
                       - Severe Delinquency, Moderate Delinquency
                       - Active - Frequently Late

                    2. **Paid-off underperformance**: The loan has been paid off but recovered
                       less than 90% of expected payments (indicating a settlement or loss).

                    3. **Behind schedule**: For active loans, payment performance is more than
                       15 percentage points behind where it should be based on loan age.

                    #### Similar Deals Comparison:
                    The tool also finds historical deals with similar characteristics to show how those deals
                    actually performed, providing context beyond the model's prediction.

                    #### Limitations:
                    - Model is only as good as the training data
                    - Does not capture recent market changes
                    - Should be used as one input among many in underwriting decisions
                    """)

            # =================================================================
            # DEAL SCORER - BATCH SCORING
            # =================================================================
            with scorer_tab_batch:
                render_batch_scorer(coefficients_data, partners)

            # =================================================================
            # DEAL SCORER - MODEL DIAGNOSTICS
            # =================================================================
            with scorer_tab_diag:
                st.subheader("ML Model Diagnostics")
                st.markdown("""
                This section provides **transparency into how the risk models work** and how well they
                perform. Use these diagnostics to understand which factors drive risk predictions,
                validate model quality, and identify patterns across partners and industries.
                """)

                # Train models first to get metrics for executive summary
                classification_metrics = None
                regression_metrics = None
                top_pos = pd.DataFrame()
                top_neg = pd.DataFrame()

                try:
                    model, metrics, top_pos, top_neg = train_classification_small(scorer_df)
                    classification_metrics = metrics
                except Exception:
                    pass

                try:
                    r_model, r_metrics = train_regression_small(scorer_df)
                    regression_metrics = r_metrics
                except Exception:
                    pass

                # Executive Summary - Plain English for non-technical users
                if classification_metrics and regression_metrics:
                    render_executive_summary(classification_metrics, regression_metrics, scorer_df)

                # Factor Impact Examples - Concrete examples of what changes mean
                render_factor_impact_examples(coefficients_data, scorer_df)

                st.markdown("""
                Three models are trained on your historical portfolio data:
                - **Problem Loan Classifier**: Predicts which loans will become problematic
                - **Payment Performance Regressor**: Predicts expected recovery percentage
                - **Current Risk Scorer**: Scores active loans using both origination and behavioral data
                """)

                st.markdown("---")

                # Problem Loan Prediction Model
                st.subheader("Problem Loan Prediction Model")
                st.markdown("""
                This model predicts which loans are likely to become "problem loans"
                (late payments, defaults, bankruptcies) based on observable characteristics at origination.
                """)

                with st.expander("How to interpret these metrics", expanded=False):
                    render_ml_explainer(metric_type="classification")

                # Use pre-trained classification model from above (avoid duplicate training)
                if classification_metrics is not None:
                    metrics = classification_metrics

                    col1, col2, col3, col4, col5 = st.columns([1.2, 1, 1, 1, 1.5])
                    with col1:
                        roc_val = metrics['ROC AUC'][0]
                        auc_tier = metrics.get('auc_tier', 'N/A')
                        st.metric("ROC AUC", f"{roc_val:.3f}" if pd.notnull(roc_val) else "N/A")
                        st.caption(f"Performance: **{auc_tier}**")
                    with col2:
                        prec_val = metrics['Precision'][0]
                        st.metric("Precision", f"{prec_val:.3f}")
                    with col3:
                        recall_val = metrics['Recall'][0]
                        st.metric("Recall", f"{recall_val:.3f}")
                    with col4:
                        st.metric("Baseline AUC", f"{metrics.get('baseline_auc', 0.5):.2f}")
                        st.caption("Random classifier")
                    with col5:
                        lift = metrics.get('auc_lift', 0)
                        st.metric("Lift vs Baseline", f"+{lift:.0%}" if lift > 0 else f"{lift:.0%}")
                        st.caption(f"n={metrics['n_samples']:,}, {metrics.get('n_splits', 'N/A')}-fold CV")

                    st.markdown("##### Key Risk Factors")
                    st.markdown("""
                    These coefficients show which features most strongly predict problem loans.
                    **Red flags** (positive coefficients) increase risk; **green flags** (negative coefficients) decrease risk.
                    """)

                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown("**Risk-Increasing Factors (Red Flags)**")
                        if not top_pos.empty:
                            display_df = top_pos.copy()
                            if "Feature" in display_df.columns:
                                chart_data = pd.DataFrame({
                                    "feature": display_df["Feature"],
                                    "coef": display_df["coef"]
                                })
                            else:
                                chart_data = display_df[["feature", "coef"]]
                            chart = create_coefficient_chart(chart_data, "Risk-Increasing Features", "#d62728")
                            st.altair_chart(chart, use_container_width=True)

                            table_data = display_df[["Feature", "coef"]].copy() if "Feature" in display_df.columns else display_df[["feature", "coef"]].copy()
                            table_data.columns = ["Feature", "Coefficient"]
                            table_data["Coefficient"] = table_data["Coefficient"].map(lambda x: f"{x:+.4f}")
                            st.dataframe(table_data, use_container_width=True, hide_index=True)

                    with col2:
                        st.markdown("**Risk-Decreasing Factors (Green Flags)**")
                        if not top_neg.empty:
                            display_df = top_neg.copy()
                            if "Feature" in display_df.columns:
                                chart_data = pd.DataFrame({
                                    "feature": display_df["Feature"],
                                    "coef": display_df["coef"]
                                })
                            else:
                                chart_data = display_df[["feature", "coef"]]
                            chart = create_coefficient_chart(chart_data, "Risk-Decreasing Features", "#2ca02c")
                            st.altair_chart(chart, use_container_width=True)

                            table_data = display_df[["Feature", "coef"]].copy() if "Feature" in display_df.columns else display_df[["feature", "coef"]].copy()
                            table_data.columns = ["Feature", "Coefficient"]
                            table_data["Coefficient"] = table_data["Coefficient"].map(lambda x: f"{x:+.4f}")
                            st.dataframe(table_data, use_container_width=True, hide_index=True)
                else:
                    st.warning("Classification model could not be trained. Check data quality or scikit-learn installation.")

                st.markdown("---")

                # Regression Model
                st.subheader("Payment Performance Prediction Model")
                st.markdown("""
                This model predicts the expected payment performance (% of invested capital recovered)
                for each loan based on its characteristics at origination.
                """)

                with st.expander("How to interpret these metrics", expanded=False):
                    render_ml_explainer(metric_type="regression")

                # Use pre-trained regression model from above (avoid duplicate training)
                if regression_metrics is not None:
                    r_metrics = regression_metrics

                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        r2_val = r_metrics['R2'][0]
                        r2_tier = r_metrics.get('r2_tier', 'N/A')
                        st.metric("R2 Score", f"{r2_val:.3f}" if pd.notnull(r2_val) else "N/A")
                        st.caption(f"Performance: **{r2_tier}**")
                    with col2:
                        rmse_val = r_metrics['RMSE'][0]
                        st.metric("RMSE", f"{rmse_val:.3f}")
                        st.caption("Avg prediction error")
                    with col3:
                        baseline_rmse = r_metrics.get('baseline_rmse', 0)
                        st.metric("Baseline RMSE", f"{baseline_rmse:.3f}")
                        st.caption("Mean predictor")
                    with col4:
                        improvement = r_metrics.get('rmse_improvement_pct', 0)
                        st.metric("RMSE Improvement", f"{improvement:.1%}")
                        st.caption(f"n={r_metrics['n_samples']:,}, {r_metrics.get('n_splits', 'N/A')}-fold CV")

                    st.markdown("##### Target Variable Context")
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Avg Payment Performance", f"{r_metrics.get('mean_target', 0):.1%}")
                    with col2:
                        st.metric("Std Dev", f"{r_metrics.get('std_target', 0):.1%}")
                else:
                    st.warning("Regression model could not be trained. Check data quality or scikit-learn installation.")

                st.markdown("---")

                # Current Risk Scores
                st.subheader("Current Risk Scores")
                st.markdown("""
                This model scores **active loans** on their current risk level using both
                origination characteristics and payment behavior. Higher scores indicate
                higher risk of becoming a problem loan.
                """)

                schedules_df = load_loan_schedules()

                try:
                    risk_model, risk_metrics, partner_rankings, industry_rankings, loan_predictions = train_current_risk_model(
                        scorer_df, schedules_df
                    )

                    col1, col2, col3, col4, col5 = st.columns([1.2, 1, 1, 1, 1.5])
                    with col1:
                        roc_val = risk_metrics['ROC AUC'][0]
                        auc_tier = risk_metrics.get('auc_tier', 'N/A')
                        st.metric("ROC AUC", f"{roc_val:.3f}" if pd.notnull(roc_val) else "N/A")
                        st.caption(f"Performance: **{auc_tier}**")
                    with col2:
                        prec_val = risk_metrics['Precision'][0]
                        st.metric("Precision", f"{prec_val:.3f}" if pd.notnull(prec_val) else "N/A")
                    with col3:
                        recall_val = risk_metrics['Recall'][0]
                        st.metric("Recall", f"{recall_val:.3f}" if pd.notnull(recall_val) else "N/A")
                    with col4:
                        st.metric("Problem Loans", f"{risk_metrics.get('n_problem_loans', 0)}")
                        st.caption(f"of {risk_metrics.get('n_samples', 0)} active")
                    with col5:
                        pos_rate = risk_metrics.get('pos_rate', 0)
                        st.metric("Problem Rate", f"{pos_rate:.1%}")
                        st.caption(f"{risk_metrics.get('n_splits', 'N/A')}-fold CV")

                    st.markdown("---")

                    st.markdown("##### Highest Risk Active Loans")
                    st.caption("Loans sorted by ML-predicted risk score (0-100). Higher = more likely to become a problem.")

                    top_risk_loans = loan_predictions.head(15).copy()
                    if not top_risk_loans.empty:
                        display_cols = ["loan_id", "deal_name", "loan_status", "partner_source",
                                       "risk_score", "payment_performance", "net_balance"]
                        display_df = top_risk_loans[[c for c in display_cols if c in top_risk_loans.columns]].copy()

                        if "risk_score" in display_df.columns:
                            display_df["risk_score"] = display_df["risk_score"].map(lambda x: f"{x:.1f}")
                        if "payment_performance" in display_df.columns:
                            display_df["payment_performance"] = display_df["payment_performance"].map(
                                lambda x: f"{x:.1%}" if pd.notnull(x) else "N/A"
                            )
                        if "net_balance" in display_df.columns:
                            display_df["net_balance"] = display_df["net_balance"].map(
                                lambda x: f"${x:,.0f}" if pd.notnull(x) else "N/A"
                            )

                        display_df.columns = ["Loan ID", "Deal Name", "Status", "Partner",
                                             "Risk Score", "Payment Perf", "Net Balance"][:len(display_df.columns)]
                        st.dataframe(display_df, use_container_width=True, hide_index=True)

                    st.markdown("---")

                    col1, col2 = st.columns(2)

                    with col1:
                        st.markdown("##### Partner Risk Leaderboard")
                        st.caption("Problem rate and avg risk score by partner source")

                        if not partner_rankings.empty:
                            partner_display = partner_rankings.copy()
                            partner_display["problem_rate"] = partner_display["problem_rate"].map(lambda x: f"{x:.1%}")
                            partner_display["avg_risk_score"] = partner_display["avg_risk_score"].map(lambda x: f"{x:.1f}")
                            partner_display["total_balance"] = partner_display["total_balance"].map(lambda x: f"${x:,.0f}")
                            partner_display = partner_display[["partner_source", "loan_count", "problem_rate", "avg_risk_score", "total_balance"]]
                            partner_display.columns = ["Partner", "Loans", "Problem Rate", "Avg Risk", "Balance"]
                            st.dataframe(partner_display, use_container_width=True, hide_index=True)

                            partner_chart_data = partner_rankings[partner_rankings["loan_count"] >= 2].head(10).copy()
                            if not partner_chart_data.empty:
                                chart = alt.Chart(partner_chart_data).mark_bar().encode(
                                    x=alt.X("partner_source:N", title="Partner", sort="-y"),
                                    y=alt.Y("problem_rate:Q", title="Problem Rate", axis=alt.Axis(format=".0%")),
                                    color=alt.Color(
                                        "problem_rate:Q",
                                        scale=alt.Scale(domain=[0, 0.2, 0.5], range=["#2ca02c", "#ffbb78", "#d62728"]),
                                        legend=None
                                    ),
                                    tooltip=[
                                        alt.Tooltip("partner_source:N", title="Partner"),
                                        alt.Tooltip("problem_rate:Q", title="Problem Rate", format=".1%"),
                                        alt.Tooltip("loan_count:Q", title="Loans"),
                                        alt.Tooltip("avg_risk_score:Q", title="Avg Risk Score", format=".1f"),
                                    ]
                                ).properties(height=250, title="Problem Rate by Partner (min 2 loans)")
                                st.altair_chart(chart, use_container_width=True)

                    with col2:
                        st.markdown("##### Industry Risk Heatmap")
                        st.caption("Problem rate by NAICS sector code")

                        if not industry_rankings.empty:
                            industry_display = industry_rankings.copy()
                            industry_display["problem_rate"] = industry_display["problem_rate"].map(lambda x: f"{x:.1%}")
                            industry_display["avg_risk_score"] = industry_display["avg_risk_score"].map(lambda x: f"{x:.1f}")
                            industry_display = industry_display[["sector_code", "sector_name", "loan_count", "problem_rate", "avg_risk_score"]]
                            industry_display.columns = ["Sector", "Name", "Loans", "Problem Rate", "Avg Risk"]
                            st.dataframe(industry_display.head(10), use_container_width=True, hide_index=True)

                            industry_chart_data = industry_rankings[industry_rankings["loan_count"] >= 2].head(10).copy()
                            if not industry_chart_data.empty:
                                industry_chart_data["display_label"] = industry_chart_data["sector_code"] + " - " + industry_chart_data["sector_name"].astype(str)
                                chart = alt.Chart(industry_chart_data).mark_bar().encode(
                                    x=alt.X("display_label:N", title="Industry", sort="-y"),
                                    y=alt.Y("problem_rate:Q", title="Problem Rate", axis=alt.Axis(format=".0%")),
                                    color=alt.Color(
                                        "problem_rate:Q",
                                        scale=alt.Scale(domain=[0, 0.2, 0.5], range=["#2ca02c", "#ffbb78", "#d62728"]),
                                        legend=None
                                    ),
                                    tooltip=[
                                        alt.Tooltip("display_label:N", title="Industry"),
                                        alt.Tooltip("problem_rate:Q", title="Problem Rate", format=".1%"),
                                        alt.Tooltip("loan_count:Q", title="Loans"),
                                        alt.Tooltip("avg_risk_score:Q", title="Avg Risk Score", format=".1f"),
                                    ]
                                ).properties(height=250, title="Problem Rate by Industry (min 2 loans)")
                                st.altair_chart(chart, use_container_width=True)

                    st.markdown("---")

                    st.markdown("##### Top Risk Factors (Model Coefficients)")
                    st.caption("Features that most strongly predict problem loan status. Positive = increases risk, Negative = decreases risk.")

                    coef_df = risk_metrics.get("coef_df", pd.DataFrame())
                    if not coef_df.empty:
                        col1, col2 = st.columns(2)

                        with col1:
                            st.markdown("**Risk-Increasing Factors**")
                            top_pos = coef_df.head(10).copy()
                            if not top_pos.empty:
                                chart_data = pd.DataFrame({
                                    "feature": top_pos["display_name"],
                                    "coef": top_pos["coef"]
                                })
                                chart = create_coefficient_chart(chart_data, "Risk-Increasing Features", "#d62728")
                                st.altair_chart(chart, use_container_width=True)

                        with col2:
                            st.markdown("**Risk-Decreasing Factors**")
                            top_neg = coef_df.tail(10).iloc[::-1].copy()
                            if not top_neg.empty:
                                chart_data = pd.DataFrame({
                                    "feature": top_neg["display_name"],
                                    "coef": top_neg["coef"]
                                })
                                chart = create_coefficient_chart(chart_data, "Risk-Decreasing Features", "#2ca02c")
                                st.altair_chart(chart, use_container_width=True)

                    st.markdown("---")
                    csv_data = loan_predictions.to_csv(index=False).encode("utf-8")
                    st.download_button(
                        label="Download Risk Scores (CSV)",
                        data=csv_data,
                        file_name=f"current_risk_scores_{pd.Timestamp.today().strftime('%Y%m%d')}.csv",
                        mime="text/csv",
                        key="scorer_download"
                    )

                except ValueError as e:
                    st.warning(f"Current risk model could not run: {e}")
                except Exception as e:
                    st.warning(f"Current risk model error: {e}")

                st.markdown("---")

                if classification_metrics and regression_metrics:
                    render_model_summary(classification_metrics, regression_metrics)


# =============================================================================
# TAB 4: LOAN MATURITY CALCULATOR
# =============================================================================
with tab_maturity:
    st.header("Loan Maturity Calculator")
    st.markdown("""
    Calculate the expected maturity date of a loan based on funding date, payment schedule,
    and payment frequency. This tool accounts for business days, weekends, and federal holidays
    for accurate maturity projections.
    """)

    # US Federal Holidays for business day calculations
    def get_us_federal_holidays(year: int) -> set:
        """
        Returns a set of US federal holiday dates for a given year.
        Includes: New Year's, MLK Day, Presidents Day, Memorial Day, Juneteenth,
        Independence Day, Labor Day, Columbus Day, Veterans Day, Thanksgiving, Christmas.
        """
        holidays = set()

        # New Year's Day - January 1
        holidays.add(date(year, 1, 1))

        # MLK Day - Third Monday of January
        jan_first = date(year, 1, 1)
        days_until_monday = (7 - jan_first.weekday()) % 7
        first_monday = jan_first + timedelta(days=days_until_monday)
        mlk_day = first_monday + timedelta(weeks=2)
        holidays.add(mlk_day)

        # Presidents Day - Third Monday of February
        feb_first = date(year, 2, 1)
        days_until_monday = (7 - feb_first.weekday()) % 7
        first_monday = feb_first + timedelta(days=days_until_monday)
        presidents_day = first_monday + timedelta(weeks=2)
        holidays.add(presidents_day)

        # Memorial Day - Last Monday of May
        may_last = date(year, 5, 31)
        days_since_monday = may_last.weekday()
        memorial_day = may_last - timedelta(days=days_since_monday)
        holidays.add(memorial_day)

        # Juneteenth - June 19
        holidays.add(date(year, 6, 19))

        # Independence Day - July 4
        holidays.add(date(year, 7, 4))

        # Labor Day - First Monday of September
        sep_first = date(year, 9, 1)
        days_until_monday = (7 - sep_first.weekday()) % 7
        labor_day = sep_first + timedelta(days=days_until_monday)
        holidays.add(labor_day)

        # Columbus Day - Second Monday of October
        oct_first = date(year, 10, 1)
        days_until_monday = (7 - oct_first.weekday()) % 7
        first_monday = oct_first + timedelta(days=days_until_monday)
        columbus_day = first_monday + timedelta(weeks=1)
        holidays.add(columbus_day)

        # Veterans Day - November 11
        holidays.add(date(year, 11, 11))

        # Thanksgiving - Fourth Thursday of November
        nov_first = date(year, 11, 1)
        days_until_thursday = (3 - nov_first.weekday()) % 7
        first_thursday = nov_first + timedelta(days=days_until_thursday)
        thanksgiving = first_thursday + timedelta(weeks=3)
        holidays.add(thanksgiving)

        # Christmas Day - December 25
        holidays.add(date(year, 12, 25))

        return holidays

    def get_holidays_in_range(start_date: date, end_date: date) -> set:
        """Get all US federal holidays between two dates."""
        holidays = set()
        for year in range(start_date.year, end_date.year + 2):
            holidays.update(get_us_federal_holidays(year))
        return holidays

    def is_business_day(check_date: date, holidays: set, skip_holidays: bool) -> bool:
        """Check if a date is a business day (not weekend, optionally not holiday)."""
        if check_date.weekday() >= 5:  # Saturday = 5, Sunday = 6
            return False
        if skip_holidays and check_date in holidays:
            return False
        return True

    def add_business_days(start_date: date, num_days: int, holidays: set, skip_holidays: bool) -> date:
        """Add a number of business days to a start date."""
        current_date = start_date
        days_added = 0

        while days_added < num_days:
            current_date += timedelta(days=1)
            if is_business_day(current_date, holidays, skip_holidays):
                days_added += 1

        return current_date

    def calculate_maturity_date(
        funding_date: date,
        num_payments: int,
        payment_interval: str,
        first_payment_delay: int = 0,
        business_days_only: bool = True,
        skip_holidays: bool = True
    ) -> dict:
        """
        Calculate the loan maturity date and payment schedule details.

        Args:
            funding_date: The date the loan is funded
            num_payments: Total number of payments
            payment_interval: 'daily', 'weekly', 'bi-weekly', or 'monthly'
            first_payment_delay: Days between funding and first payment
            business_days_only: For daily payments, skip weekends
            skip_holidays: For business day calculations, skip US holidays

        Returns:
            Dictionary with maturity date and schedule details
        """
        # Get holidays for the range we might need
        estimated_end = funding_date + timedelta(days=num_payments * 35)  # Conservative estimate
        holidays = get_holidays_in_range(funding_date, estimated_end) if skip_holidays else set()

        # Calculate first payment date
        if first_payment_delay > 0:
            if business_days_only and payment_interval == "daily":
                first_payment_date = add_business_days(funding_date, first_payment_delay, holidays, skip_holidays)
            else:
                first_payment_date = funding_date + timedelta(days=first_payment_delay)
        else:
            first_payment_date = funding_date

        # Calculate all payment dates based on interval
        payment_dates = []
        current_date = first_payment_date

        if payment_interval == "daily":
            if business_days_only:
                # First payment
                if not is_business_day(current_date, holidays, skip_holidays):
                    current_date = add_business_days(current_date, 0, holidays, skip_holidays)
                payment_dates.append(current_date)

                # Subsequent payments
                for _ in range(1, num_payments):
                    current_date = add_business_days(current_date, 1, holidays, skip_holidays)
                    payment_dates.append(current_date)
            else:
                for i in range(num_payments):
                    payment_dates.append(current_date + timedelta(days=i))

        elif payment_interval == "weekly":
            for i in range(num_payments):
                payment_dates.append(current_date + timedelta(weeks=i))

        elif payment_interval == "bi-weekly":
            for i in range(num_payments):
                payment_dates.append(current_date + timedelta(weeks=i * 2))

        elif payment_interval == "monthly":
            for i in range(num_payments):
                payment_dates.append(current_date + relativedelta(months=i))

        maturity_date = payment_dates[-1] if payment_dates else funding_date

        # Calculate duration
        total_days = (maturity_date - funding_date).days
        total_weeks = total_days / 7
        total_months = total_days / 30.44  # Average days per month

        # Count business days and holidays in the range
        business_days_count = 0
        holidays_in_range = 0
        weekends_in_range = 0

        check_date = funding_date
        while check_date <= maturity_date:
            if check_date.weekday() >= 5:
                weekends_in_range += 1
            elif check_date in holidays:
                holidays_in_range += 1
            else:
                business_days_count += 1
            check_date += timedelta(days=1)

        return {
            "funding_date": funding_date,
            "first_payment_date": first_payment_date,
            "maturity_date": maturity_date,
            "payment_dates": payment_dates,
            "num_payments": num_payments,
            "payment_interval": payment_interval,
            "total_calendar_days": total_days,
            "total_weeks": total_weeks,
            "total_months": total_months,
            "business_days_count": business_days_count,
            "weekends_in_range": weekends_in_range,
            "holidays_in_range": holidays_in_range,
            "business_days_only": business_days_only,
            "skip_holidays": skip_holidays,
        }

    # Input section
    st.markdown("---")
    st.markdown("### Loan Parameters")

    col_input, col_output = st.columns([1, 1])

    with col_input:
        # Funding Date
        funding_date_input = st.date_input(
            "Funding Date",
            value=date.today(),
            help="The date the loan funds are disbursed",
            key="maturity_funding_date"
        )

        # Number of Payments
        num_payments_input = st.number_input(
            "Number of Payments",
            min_value=1,
            max_value=1000,
            value=60,
            step=1,
            help="Total number of scheduled payments",
            key="maturity_num_payments"
        )

        # Payment Interval
        interval_options = {
            "Daily": "daily",
            "Weekly": "weekly",
            "Bi-Weekly (Every 2 Weeks)": "bi-weekly",
            "Monthly": "monthly"
        }
        selected_interval = st.selectbox(
            "Payment Interval",
            options=list(interval_options.keys()),
            index=0,
            help="How often payments are made",
            key="maturity_interval"
        )
        payment_interval = interval_options[selected_interval]

        st.markdown("---")
        st.markdown("### Advanced Options")

        # First Payment Delay
        first_payment_delay = st.number_input(
            "First Payment Delay (Days)",
            min_value=0,
            max_value=90,
            value=0,
            step=1,
            help="Number of days between funding and the first payment (grace period)",
            key="maturity_delay"
        )

        # Business days only (for daily payments)
        if payment_interval == "daily":
            business_days_only = st.checkbox(
                "Business Days Only (Skip Weekends)",
                value=True,
                help="For daily payments, only count Monday-Friday",
                key="maturity_business_days"
            )

            skip_holidays = st.checkbox(
                "Skip Federal Holidays",
                value=True,
                help="Exclude US federal holidays from payment schedule",
                key="maturity_skip_holidays"
            )
        else:
            business_days_only = False
            skip_holidays = False
            st.info("Business day options apply only to daily payment schedules.")

    # Calculate and display results
    with col_output:
        st.markdown("### Maturity Analysis")

        # Calculate maturity
        result = calculate_maturity_date(
            funding_date=funding_date_input,
            num_payments=num_payments_input,
            payment_interval=payment_interval,
            first_payment_delay=first_payment_delay,
            business_days_only=business_days_only,
            skip_holidays=skip_holidays
        )

        # Key metrics
        col1, col2 = st.columns(2)

        with col1:
            st.metric(
                label="Maturity Date",
                value=result["maturity_date"].strftime("%B %d, %Y"),
                help="The date of the final payment"
            )

        with col2:
            st.metric(
                label="First Payment Date",
                value=result["first_payment_date"].strftime("%B %d, %Y"),
                help="The date of the first payment"
            )

        st.divider()

        # Duration breakdown
        st.markdown("#### Loan Duration")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric(
                label="Calendar Days",
                value=f"{result['total_calendar_days']:,}",
                help="Total days from funding to maturity"
            )

        with col2:
            st.metric(
                label="Weeks",
                value=f"{result['total_weeks']:.1f}",
                help="Duration in weeks"
            )

        with col3:
            st.metric(
                label="Months",
                value=f"{result['total_months']:.1f}",
                help="Duration in months (approximate)"
            )

        st.divider()

        # Business day breakdown
        st.markdown("#### Calendar Breakdown")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric(
                label="Business Days",
                value=f"{result['business_days_count']:,}",
                help="Monday-Friday, excluding holidays"
            )

        with col2:
            st.metric(
                label="Weekend Days",
                value=f"{result['weekends_in_range']:,}",
                help="Saturdays and Sundays"
            )

        with col3:
            st.metric(
                label="Federal Holidays",
                value=f"{result['holidays_in_range']}",
                help="US federal holidays in the period"
            )

        # Payment schedule summary
        st.divider()
        st.markdown("#### Payment Schedule Summary")

        schedule_info = f"""
        | Parameter | Value |
        |-----------|-------|
        | Funding Date | {result['funding_date'].strftime('%Y-%m-%d')} |
        | First Payment | {result['first_payment_date'].strftime('%Y-%m-%d')} |
        | Last Payment (Maturity) | {result['maturity_date'].strftime('%Y-%m-%d')} |
        | Total Payments | {result['num_payments']} |
        | Payment Frequency | {selected_interval} |
        | Business Days Only | {'Yes' if result['business_days_only'] else 'No'} |
        | Skip Holidays | {'Yes' if result['skip_holidays'] else 'No'} |
        """
        st.markdown(schedule_info)

    # Payment Schedule Table (expandable)
    st.markdown("---")
    with st.expander("View Full Payment Schedule", expanded=False):
        payment_dates = result["payment_dates"]

        # Create schedule dataframe
        schedule_df = pd.DataFrame({
            "Payment #": range(1, len(payment_dates) + 1),
            "Payment Date": payment_dates,
            "Day of Week": [d.strftime("%A") for d in payment_dates],
        })

        # Format date column
        schedule_df["Payment Date"] = schedule_df["Payment Date"].apply(
            lambda x: x.strftime("%Y-%m-%d")
        )

        # Show first and last payments with option to see all
        st.markdown("**First 10 Payments:**")
        st.dataframe(schedule_df.head(10), use_container_width=True, hide_index=True)

        if len(payment_dates) > 20:
            st.markdown("**Last 10 Payments:**")
            st.dataframe(schedule_df.tail(10), use_container_width=True, hide_index=True)

        # Download full schedule
        csv_data = schedule_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="Download Full Schedule (CSV)",
            data=csv_data,
            file_name=f"payment_schedule_{result['funding_date'].strftime('%Y%m%d')}.csv",
            mime="text/csv",
            key="maturity_download"
        )

    # Methodology explanation
    st.markdown("---")
    with st.expander("Calculation Methodology", expanded=False):
        st.markdown("""
        ### How the Maturity Date is Calculated

        #### Payment Interval Logic

        | Interval | Calculation Method |
        |----------|-------------------|
        | **Daily** | Adds 1 day per payment. If "Business Days Only" is enabled, skips Saturdays, Sundays, and optionally federal holidays. |
        | **Weekly** | Adds 7 calendar days per payment from the first payment date. |
        | **Bi-Weekly** | Adds 14 calendar days per payment from the first payment date. |
        | **Monthly** | Adds 1 calendar month per payment, handling month-end dates appropriately (e.g., Jan 31 → Feb 28). |

        #### Business Day Handling (Daily Payments)

        When "Business Days Only" is enabled:
        - **Weekends**: Saturdays and Sundays are skipped
        - **Holidays**: If "Skip Federal Holidays" is enabled, the following US federal holidays are excluded:
          - New Year's Day (Jan 1)
          - Martin Luther King Jr. Day (3rd Monday of January)
          - Presidents Day (3rd Monday of February)
          - Memorial Day (Last Monday of May)
          - Juneteenth (June 19)
          - Independence Day (July 4)
          - Labor Day (1st Monday of September)
          - Columbus Day (2nd Monday of October)
          - Veterans Day (November 11)
          - Thanksgiving (4th Thursday of November)
          - Christmas Day (December 25)

        #### First Payment Delay

        The "First Payment Delay" adds a grace period between the funding date and the first payment.
        This is common in MCA (Merchant Cash Advance) loans where there's a waiting period before
        payments begin.

        #### Important Notes

        - This calculator provides an **estimated** maturity date based on a regular payment schedule
        - Actual maturity may vary due to payment holidays, ACH processing delays, or payment failures
        - For loans with irregular payment schedules, the actual maturity may differ
        - Holiday calculations use US federal holidays and may not include state-specific holidays
        """)

    # Quick reference
    st.markdown("---")
    st.markdown("### Quick Reference: Common Loan Terms")

    reference_data = {
        "Term": ["3 months", "6 months", "9 months", "12 months", "18 months", "24 months"],
        "Daily (Bus. Days)": ["~63 payments", "~126 payments", "~189 payments", "~252 payments", "~378 payments", "~504 payments"],
        "Weekly": ["13 payments", "26 payments", "39 payments", "52 payments", "78 payments", "104 payments"],
        "Bi-Weekly": ["6-7 payments", "13 payments", "19-20 payments", "26 payments", "39 payments", "52 payments"],
        "Monthly": ["3 payments", "6 payments", "9 payments", "12 payments", "18 payments", "24 payments"],
    }

    st.dataframe(pd.DataFrame(reference_data), use_container_width=True, hide_index=True)
    st.caption("Note: Daily business day counts assume approximately 21 business days per month.")
