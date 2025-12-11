# pages/deal_scorer.py
"""
Deal Scorer & ML Diagnostics - Screen new deals and analyze model performance.

This page provides two key capabilities:
1. Score New Deal: Screen prospective deals using ML-based risk scoring
2. Model Diagnostics: Understand model performance and key risk factors
"""

import streamlit as st
import pandas as pd
import numpy as np
import altair as alt

from utils.config import setup_page, PRIMARY_COLOR
from utils.data_loader import DataLoader, load_loan_schedules
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
    NAICS_SECTOR_NAMES,
    RISK_LEVELS,
)
from utils.loan_tape_analytics import get_similar_deals_comparison

# Page setup
setup_page("CSL Capital | Deal Scorer")

st.title("Deal Scorer")
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

# =============================================================================
# TABS
# =============================================================================
tabs = st.tabs(["Score New Deal", "Model Diagnostics"])

# =============================================================================
# TAB 1: SCORE NEW DEAL
# =============================================================================
with tabs[0]:
    st.subheader("Screen a Prospective Deal")
    st.markdown("""
    Use this tool to **evaluate a new deal before committing capital**. Enter the deal
    characteristics below and the model will predict the likelihood of it becoming a
    problem loan based on historical portfolio patterns.

    The risk score combines multiple factors including partner track record, borrower
    creditworthiness, industry risk, and deal structure to give you an at-a-glance
    assessment of deal quality.
    """)

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
    st.markdown("##### Deal Inputs")

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
    score_button = st.button("Score This Deal", type="primary", use_container_width=True)

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

            st.altair_chart(chart + rule, use_container_width=True)

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

                # Format for display - include problem_reason for tooltip
                display_cols = ["deal_name", "partner_source", "loan_status", "payment_performance",
                              "fico", "tib", "total_invested", "is_problem", "problem_reason"]
                display_cols = [c for c in display_cols if c in similar_deals.columns]

                if not similar_deals.empty:
                    display_df = similar_deals[display_cols].head(20).copy()

                    # Store original problem_reason for tooltip before formatting
                    problem_reasons = display_df.get("problem_reason", pd.Series("", index=display_df.index))

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
                        # Create Problem column with indicator
                        display_df["is_problem"] = display_df["is_problem"].apply(
                            lambda x: "Yes" if x else "No"
                        )

                    # Add problem reason column (show dash for non-problems)
                    if "problem_reason" in display_df.columns:
                        display_df["problem_reason"] = problem_reasons.apply(
                            lambda x: x if x else "-"
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
                        "is_problem": "Problem?",
                        "problem_reason": "Reason"
                    }
                    display_df.rename(columns=rename_map, inplace=True)

                    st.dataframe(display_df, use_container_width=True, hide_index=True)

                    # Add legend explaining problem classification
                    st.caption("""
                    **Problem Classification Logic:**
                    - **Status-based**: Loan status is Default, Bankruptcy, Charged Off, In Collections, Legal Action, NSF/Suspended, Non-Performing, Severe/Moderate Delinquency, or Active - Frequently Late
                    - **Paid off underperformance**: Loan paid off but recovered less than 90% of expected payments
                    - **Behind schedule**: Active loan is 15+ percentage points behind expected payment progress based on loan age
                    """)
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
    # HELP SECTION
    # =============================================================================
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
        - **Deal Size**: Loan amount

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
           (e.g., a loan 50% through its term should be ~50% paid; if only 30% paid,
           it's 20pp behind and flagged as a problem)

        #### Similar Deals Comparison:
        The tool also finds historical deals with similar characteristics to show how those deals
        actually performed, providing context beyond the model's prediction.

        #### Limitations:
        - Model is only as good as the training data
        - Does not capture recent market changes
        - Should be used as one input among many in underwriting decisions
        """)

# =============================================================================
# TAB 2: MODEL DIAGNOSTICS
# =============================================================================
with tabs[1]:
    st.subheader("ML Model Diagnostics")
    st.markdown("""
    This section provides **transparency into how the risk models work** and how well they
    perform. Use these diagnostics to understand which factors drive risk predictions,
    validate model quality, and identify patterns across partners and industries.

    Three models are trained on your historical portfolio data:
    - **Problem Loan Classifier**: Predicts which loans will become problematic
    - **Payment Performance Regressor**: Predicts expected recovery percentage
    - **Current Risk Scorer**: Scores active loans using both origination and behavioral data
    """)

    st.markdown("---")

    # =====================================================================
    # Problem Loan Prediction Model
    # =====================================================================
    st.subheader("Problem Loan Prediction Model")
    st.markdown("""
    This model predicts which loans are likely to become "problem loans"
    (late payments, defaults, bankruptcies) based on observable characteristics at origination.
    """)

    # Add explainer in expandable section
    with st.expander("How to interpret these metrics", expanded=False):
        render_ml_explainer(metric_type="classification")

    classification_metrics = None
    regression_metrics = None

    try:
        model, metrics, top_pos, top_neg = train_classification_small(df)
        classification_metrics = metrics

        # Metrics row with performance tier badges
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

        # Coefficient visualizations with better labeling
        st.markdown("##### Key Risk Factors")
        st.markdown("""
        These coefficients show which features most strongly predict problem loans.
        **Red flags** (positive coefficients) increase risk; **green flags** (negative coefficients) decrease risk.
        """)

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Risk-Increasing Factors (Red Flags)**")
            if not top_pos.empty:
                # Use Feature column (human-readable) for display
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

                # Display table with interpretation
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

    except ImportError:
        st.warning("scikit-learn or scipy not installed. Run `pip install scikit-learn scipy` to enable modeling.")
    except Exception as e:
        st.warning(f"Classification model could not run: {e}")

    st.markdown("---")

    # =====================================================================
    # Regression Model
    # =====================================================================
    st.subheader("Payment Performance Prediction Model")
    st.markdown("""
    This model predicts the expected payment performance (% of invested capital recovered)
    for each loan based on its characteristics at origination.
    """)

    with st.expander("How to interpret these metrics", expanded=False):
        render_ml_explainer(metric_type="regression")

    try:
        r_model, r_metrics = train_regression_small(df)
        regression_metrics = r_metrics

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            r2_val = r_metrics['R2'][0]
            r2_tier = r_metrics.get('r2_tier', 'N/A')
            st.metric("R² Score", f"{r2_val:.3f}" if pd.notnull(r2_val) else "N/A")
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

        # Context about the target variable
        st.markdown("##### Target Variable Context")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Avg Payment Performance", f"{r_metrics.get('mean_target', 0):.1%}")
        with col2:
            st.metric("Std Dev", f"{r_metrics.get('std_target', 0):.1%}")

    except ImportError:
        st.warning("scikit-learn not installed. Run `pip install scikit-learn` to enable modeling.")
    except Exception as e:
        st.warning(f"Regression model could not run: {e}")

    st.markdown("---")

    # =====================================================================
    # Current Risk Scores (Active Loans)
    # =====================================================================
    st.subheader("Current Risk Scores")
    st.markdown("""
    This model scores **active loans** on their current risk level using both
    origination characteristics and payment behavior. Higher scores indicate
    higher risk of becoming a problem loan.
    """)

    # Load loan schedules for behavioral features
    schedules_df = load_loan_schedules()

    try:
        risk_model, risk_metrics, partner_rankings, industry_rankings, loan_predictions = train_current_risk_model(
            df, schedules_df
        )

        # Metrics row
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

        # Show top risk loans
        st.markdown("##### Highest Risk Active Loans")
        st.caption("Loans sorted by ML-predicted risk score (0-100). Higher = more likely to become a problem.")

        top_risk_loans = loan_predictions.head(15).copy()
        if not top_risk_loans.empty:
            display_cols = ["loan_id", "deal_name", "loan_status", "partner_source",
                           "risk_score", "payment_performance", "net_balance"]
            display_df = top_risk_loans[[c for c in display_cols if c in top_risk_loans.columns]].copy()

            # Format columns
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

        # Partner and Industry rankings side by side
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

                # Partner problem rate chart
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

                # Industry problem rate chart
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

        # Top Risk Factors from the model
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

        # Download predictions
        st.markdown("---")
        csv_data = loan_predictions.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="Download Risk Scores (CSV)",
            data=csv_data,
            file_name=f"current_risk_scores_{pd.Timestamp.today().strftime('%Y%m%d')}.csv",
            mime="text/csv",
        )

    except ValueError as e:
        st.warning(f"Current risk model could not run: {e}")
    except Exception as e:
        st.warning(f"Current risk model error: {e}")

    st.markdown("---")

    # =====================================================================
    # Executive Summary
    # =====================================================================
    if classification_metrics and regression_metrics:
        render_model_summary(classification_metrics, regression_metrics)
