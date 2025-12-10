# pages/deal_scorer.py
"""
Deal Risk Scorer - Screen prospective deals based on historical portfolio performance.

Uses ML model coefficients trained on historical loan data to score new deals
and show factor-by-factor risk contribution breakdown.
"""

import streamlit as st
import pandas as pd
import numpy as np
import altair as alt

from utils.config import setup_page, PRIMARY_COLOR
from utils.data_loader import DataLoader
from utils.loan_tape_data import prepare_loan_data, consolidate_sector_code
from utils.loan_tape_ml import (
    get_origination_model_coefficients,
    calculate_deal_risk_score,
    NAICS_SECTOR_NAMES,
    RISK_LEVELS,
)
from utils.loan_tape_analytics import get_similar_deals_comparison

# Page setup
setup_page("CSL Capital | Deal Risk Scorer")

st.title("Deal Risk Scorer")
st.markdown("*Screen prospective deals based on historical portfolio performance*")
st.markdown("---")


@st.cache_data(ttl=3600)
def load_and_prepare_data():
    """Load and prepare loan data for model training."""
    loader = DataLoader()
    loans_df = loader.load_loan_summaries()
    deals_df = loader.load_deals()

    if loans_df.empty or deals_df.empty:
        return pd.DataFrame(), []

    # Prepare loan data with calculations
    df = prepare_loan_data(loans_df, deals_df)

    # Get unique partners for dropdown
    partners = []
    if "partner_source" in df.columns:
        partners = sorted(df["partner_source"].dropna().unique().tolist())

    return df, partners


@st.cache_data(ttl=3600)
def train_model_and_get_coefficients(_df):
    """Train model and extract coefficients (cached)."""
    if _df.empty:
        return None
    return get_origination_model_coefficients(_df)


# Load data
with st.spinner("Loading portfolio data..."):
    df, partners = load_and_prepare_data()

if df.empty:
    st.error("Unable to load loan data. Please check database connection.")
    st.stop()

# Train model and get coefficients
with st.spinner("Training risk model on historical data..."):
    coefficients_data = train_model_and_get_coefficients(df)

if coefficients_data is None:
    st.error("Unable to train risk model. Insufficient data.")
    st.stop()

# Display model quality metrics
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

# =============================================================================
# DEAL INPUT SECTION
# =============================================================================
st.subheader("Deal Inputs")

col1, col2 = st.columns(2)

with col1:
    # Partner Source
    partner_options = ["Select Partner..."] + partners
    selected_partner = st.selectbox(
        "Partner Source",
        options=partner_options,
        index=0,
        help="The originating partner for this deal"
    )
    partner = selected_partner if selected_partner != "Select Partner..." else None

    # FICO Score
    fico = st.number_input(
        "FICO Score",
        min_value=300,
        max_value=850,
        value=650,
        step=5,
        help="Borrower's credit score"
    )

    # Time in Business
    tib = st.number_input(
        "Time in Business (Years)",
        min_value=0.0,
        max_value=100.0,
        value=5.0,
        step=0.5,
        help="Years the business has been operating"
    )

    # Lien Position
    position_options = {
        "1st (Best)": 0,
        "2nd": 1,
        "3rd+": 2
    }
    selected_position = st.selectbox(
        "Lien Position",
        options=list(position_options.keys()),
        index=0,
        help="Position in lien hierarchy (0=1st lien, best; 2=3rd+, worst)"
    )
    position = position_options[selected_position]

with col2:
    # Industry (NAICS Sector)
    sector_options = {"Select Industry...": None}
    sector_options.update({f"{code} - {name}": code for code, name in sorted(NAICS_SECTOR_NAMES.items())})

    selected_sector = st.selectbox(
        "Industry (NAICS Sector)",
        options=list(sector_options.keys()),
        index=0,
        help="2-digit NAICS sector code"
    )
    sector_code = sector_options[selected_sector]

    # Deal Size
    deal_size = st.number_input(
        "Deal Size ($)",
        min_value=0,
        max_value=10_000_000,
        value=50_000,
        step=5000,
        format="%d",
        help="Total loan/participation amount"
    )

    # Factor Rate
    factor_rate = st.number_input(
        "Factor Rate",
        min_value=1.0,
        max_value=2.0,
        value=1.30,
        step=0.01,
        format="%.2f",
        help="Factor rate (e.g., 1.30 = 30% return)"
    )

    # Commission Rate
    commission = st.number_input(
        "Commission (%)",
        min_value=0.0,
        max_value=20.0,
        value=5.0,
        step=0.5,
        help="Commission fee percentage"
    ) / 100  # Convert to decimal

# Score button
st.markdown("---")
score_button = st.button("Score This Deal", type="primary", width='stretch')

# =============================================================================
# RISK ASSESSMENT RESULTS
# =============================================================================
if score_button:
    # Validate inputs
    if partner is None and sector_code is None:
        st.warning("Please select at least a Partner or Industry to get a meaningful score.")

    # Calculate risk score
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

    # Main score display
    col1, col2, col3 = st.columns([2, 1, 1])

    with col1:
        # Large score display with color
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

    # Factor breakdown
    st.markdown("##### Factor Breakdown")

    factor_contributions = result["factor_contributions"]
    if factor_contributions:
        # Create data for visualization
        factor_df = pd.DataFrame(factor_contributions)

        # Color based on direction
        factor_df["color"] = factor_df["contribution"].apply(
            lambda x: "#d62728" if x > 0 else "#2ca02c" if x < 0 else "#808080"
        )

        # Create horizontal bar chart
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

        # Add zero line
        rule = alt.Chart(pd.DataFrame({"x": [0]})).mark_rule(
            color="gray", strokeDash=[3, 3]
        ).encode(x="x:Q")

        st.altair_chart(chart + rule, width='stretch')

        # Legend
        st.caption("Red bars increase risk, green bars decrease risk. Larger absolute values have more impact.")

    # =============================================================================
    # SIMILAR HISTORICAL DEALS
    # =============================================================================
    st.markdown("---")
    st.subheader("Similar Historical Deals")

    # Calculate FICO range for similar deals lookup
    fico_range = (fico - 25, fico + 25) if fico else None
    tib_range = (max(0, tib - 2), tib + 2) if tib else None

    comparison = get_similar_deals_comparison(
        df=df,
        partner=partner,
        sector_code=sector_code,
        fico_range=fico_range,
        tib_range=tib_range,
        position=position
    )

    similar_count = comparison["similar_count"]

    if similar_count > 0:
        st.markdown(f"**Found {similar_count} similar deals** ({comparison['match_criteria']})")

        # Comparison metrics
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

        # Warnings
        for warning in comparison["warnings"]:
            st.warning(f"Warning: {warning}")

        # Show similar deals table (expandable)
        with st.expander("View Similar Deals"):
            similar_deals = comparison["similar_deals"]

            # Format for display
            display_cols = ["deal_name", "partner_source", "loan_status", "payment_performance",
                          "fico", "tib", "total_invested", "is_problem"]
            display_cols = [c for c in display_cols if c in similar_deals.columns]

            if not similar_deals.empty:
                display_df = similar_deals[display_cols].head(20).copy()

                # Format columns
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

                # Rename columns for display
                rename_map = {
                    "deal_name": "Deal Name",
                    "partner_source": "Partner",
                    "loan_status": "Status",
                    "payment_performance": "Performance",
                    "fico": "FICO",
                    "tib": "TIB (yrs)",
                    "total_invested": "Amount",
                    "is_problem": "Problem?"
                }
                display_df.rename(columns=rename_map, inplace=True)

                st.dataframe(display_df, width='stretch', hide_index=True)
    else:
        st.info("No similar historical deals found with current criteria. Try selecting fewer filters.")

    # =============================================================================
    # RECOMMENDATION
    # =============================================================================
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

# =============================================================================
# FOOTER / HELP
# =============================================================================
st.markdown("---")
with st.expander("How This Works"):
    st.markdown("""
    ### Deal Risk Scoring Methodology

    This tool uses a **logistic regression model** trained on historical loan outcomes to predict
    the probability that a new deal will become a "problem loan" (late payments, defaults, etc.).

    #### Features Used:
    - **FICO Score**: Borrower creditworthiness
    - **Time in Business**: Business maturity/stability
    - **Lien Position**: Priority in repayment hierarchy
    - **Industry (NAICS)**: Sector-specific risk patterns
    - **Partner Source**: Historical performance by originator
    - **Deal Size**: Loan amount

    #### Risk Score Interpretation:
    | Score Range | Risk Level | Recommendation |
    |-------------|------------|----------------|
    | 0-40 | LOW | Standard approval |
    | 40-60 | MODERATE | Proceed with caution |
    | 60-80 | ELEVATED | Additional review needed |
    | 80-100 | HIGH | Consider declining |

    #### Similar Deals Comparison:
    The tool also finds historical deals with similar characteristics to show how those deals
    actually performed, providing context beyond the model's prediction.

    #### Limitations:
    - Model is only as good as the training data
    - Does not capture recent market changes
    - Should be used as one input among many in underwriting decisions
    """)
