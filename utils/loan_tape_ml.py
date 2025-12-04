# utils/loan_tape_ml.py
"""
Machine learning utilities for loan tape dashboard.
Handles model training, evaluation, and visualization of ML results.
"""

import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
from typing import Tuple, Dict, List
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import roc_auc_score, precision_score, recall_score

from utils.loan_tape_analytics import (
    make_problem_label,
    build_feature_matrix,
    safe_kfold,
    get_display_name,
    get_metric_tier,
    assess_data_quality,
    FEATURE_DISPLAY_NAMES,
    METRIC_THRESHOLDS,
)


def create_coefficient_chart(coef_df: pd.DataFrame, title: str, color: str = "#1f77b4") -> alt.Chart:
    """
    Create a horizontal bar chart for model coefficients.

    Args:
        coef_df: DataFrame with 'feature' and 'coef' columns
        title: Chart title
        color: Bar color

    Returns:
        Altair chart object
    """
    chart = alt.Chart(coef_df).mark_bar(color=color).encode(
        x=alt.X("coef:Q", title="Coefficient Value"),
        y=alt.Y("feature:N", title="Feature", sort="-x"),
        tooltip=[
            alt.Tooltip("feature:N", title="Feature"),
            alt.Tooltip("coef:Q", title="Coefficient", format=".4f")
        ]
    ).properties(
        title=title,
        width=700,
        height=300
    )
    return chart


def render_ml_explainer(metric_type: str = "classification"):
    """
    Render an explainer box for ML metrics and their directionality.

    Args:
        metric_type: Type of model ("classification" or "regression")
    """
    if metric_type == "classification":
        st.markdown("""
        #### Understanding Classification Metrics

        **What is this model doing?**
        This model predicts which loans are likely to become "problem loans" (late payments, defaults, etc.)
        based on observable characteristics at origination.

        ---

        **Key Metrics Explained:**

        | Metric | What It Measures | Good Value | Interpretation |
        |--------|------------------|------------|----------------|
        | **ROC AUC** | Overall ability to rank risky loans higher | > 0.70 | Model is better than random guessing |
        | **Precision** | Accuracy of problem predictions | > 0.50 | Half of flagged loans are actual problems |
        | **Recall** | Coverage of actual problems | > 0.50 | Model catches half of all problem loans |

        ---

        **Performance Tiers:**
        - **Excellent (> 0.90)**: Production-ready predictive power
        - **Good (0.80-0.90)**: Useful for prioritization and screening
        - **Fair (0.70-0.80)**: Directionally correct, use with caution
        - **Poor (< 0.70)**: Limited predictive value

        ---

        **Understanding the Coefficients:**

        - **Red Flags (Positive Coefficients)**: Features that *increase* the probability of a loan becoming a problem.
          - *Example*: If "Partner: XYZ" has a large positive coefficient, loans from that partner have higher default risk.

        - **Green Flags (Negative Coefficients)**: Features that *decrease* the probability of a loan becoming a problem.
          - *Example*: If "FICO Score" has a negative coefficient, higher FICO scores mean lower default risk.

        ---

        **What to do with these results:**
        1. Review the top risk-increasing factors - are there partners or industries to monitor more closely?
        2. Consider adjusting underwriting criteria based on the strongest signals
        3. Use the model to prioritize collections efforts on high-risk active loans
        """)
    else:  # regression
        st.markdown("""
        #### Understanding Regression Metrics

        **What is this model doing?**
        This model predicts the expected "payment performance" (% of invested capital recovered) for each loan.

        ---

        **Key Metrics Explained:**

        | Metric | What It Measures | Good Value | Interpretation |
        |--------|------------------|------------|----------------|
        | **R²** | Variance explained by model | > 0.30 | Model explains 30%+ of performance variation |
        | **RMSE** | Average prediction error | < 0.15 | Predictions off by ~15 percentage points on average |

        ---

        **Performance Tiers:**
        - **Excellent (R² > 0.70)**: Strong predictive model
        - **Good (R² 0.50-0.70)**: Useful predictions with moderate uncertainty
        - **Fair (R² 0.30-0.50)**: Captures major patterns, limited precision
        - **Poor (R² < 0.30)**: High uncertainty, use directionally only

        ---

        **Interpreting R²:**
        - R² = 1.0 means perfect predictions (unrealistic)
        - R² = 0.0 means the model is no better than predicting the average
        - R² < 0.0 means the model is actually worse than the average (poor fit)

        ---

        **What to do with these results:**
        1. If R² is low, performance may be driven by factors not in the model (external events, etc.)
        2. Compare RMSE to your portfolio's actual performance variance
        3. Use predictions to estimate expected recovery for active loans
        """)

    st.markdown("---")


def train_classification_small(df: pd.DataFrame) -> Tuple[Pipeline, Dict, pd.DataFrame, pd.DataFrame]:
    """
    Train a logistic regression model to predict problem loans.

    Args:
        df: DataFrame with loan data

    Returns:
        Tuple containing:
        - Fitted pipeline model
        - Dictionary of cross-validation metrics (includes baselines)
        - DataFrame of top positive coefficients (with human-readable names)
        - DataFrame of top negative coefficients (with human-readable names)
    """
    y = make_problem_label(df)
    X = build_feature_matrix(df)

    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = [c for c in X.columns if c not in num_cols]

    # Build preprocessing pipeline
    pre = ColumnTransformer([
        ("num", Pipeline([
            ("imp", SimpleImputer(strategy="median")),
            ("sc", StandardScaler())
        ]), num_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore", min_frequency=0.05), cat_cols)
    ])

    # Build full model pipeline
    model = Pipeline([
        ("pre", pre),
        ("clf", LogisticRegression(max_iter=2000, class_weight="balanced", C=1.0, solver="lbfgs"))
    ])

    # Determine safe number of CV folds
    min_class = int(pd.Series(y).value_counts().min()) if y.nunique() == 2 else len(df)
    n_splits = max(2, min(safe_kfold(len(df)), min_class))
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    # Cross-validation
    aucs, precs, recs = [], [], []
    for tr, te in skf.split(X, y):
        model.fit(X.iloc[tr], y.iloc[tr])
        p = model.predict_proba(X.iloc[te])[:, 1]
        yhat = (p >= 0.5).astype(int)

        try:
            aucs.append(roc_auc_score(y.iloc[te], p))
        except:
            pass

        precs.append(precision_score(y.iloc[te], yhat, zero_division=0))
        recs.append(recall_score(y.iloc[te], yhat, zero_division=0))

    # Final fit on all data for coefficients
    model.fit(X, y)

    # Extract top coefficients with human-readable names
    try:
        ohe = model.named_steps["pre"].named_transformers_["cat"]
        num_names = num_cols
        cat_names = ohe.get_feature_names_out(cat_cols).tolist() if cat_cols else []
        feat_names = num_names + cat_names
        coefs = model.named_steps["clf"].coef_.ravel()

        coef_df = pd.DataFrame({"feature": feat_names, "coef": coefs})
        # Add human-readable display names
        coef_df["display_name"] = coef_df["feature"].apply(get_display_name)
        coef_df = coef_df.sort_values("coef", ascending=False)

        top_pos = coef_df.head(10)[["display_name", "feature", "coef"]].rename(columns={"display_name": "Feature"})
        top_neg = coef_df.tail(10).iloc[::-1][["display_name", "feature", "coef"]].rename(columns={"display_name": "Feature"})
    except Exception as e:
        top_pos = pd.DataFrame(columns=["Feature", "feature", "coef"])
        top_neg = pd.DataFrame(columns=["Feature", "feature", "coef"])

    # Calculate baseline metrics for comparison
    pos_rate = float(y.mean())
    baseline_auc = 0.5  # Random classifier
    baseline_precision = pos_rate  # Always predicting positive
    baseline_recall = 1.0  # Always predicting positive catches all positives

    # Get metric performance tiers
    mean_auc = np.mean(aucs) if aucs else np.nan
    mean_prec = np.mean(precs)
    mean_rec = np.mean(recs)

    auc_tier, auc_color = get_metric_tier("roc_auc", mean_auc)
    prec_tier, prec_color = get_metric_tier("precision", mean_prec)
    rec_tier, rec_color = get_metric_tier("recall", mean_rec)

    metrics = {
        "ROC AUC": (mean_auc, np.std(aucs) if aucs else np.nan),
        "Precision": (mean_prec, np.std(precs)),
        "Recall": (mean_rec, np.std(recs)),
        "n_samples": len(df),
        "pos_rate": pos_rate,
        "n_splits": n_splits,
        # Baselines for comparison
        "baseline_auc": baseline_auc,
        "baseline_precision": baseline_precision,
        # Performance tiers
        "auc_tier": auc_tier,
        "prec_tier": prec_tier,
        "rec_tier": rec_tier,
        # Lift over baseline
        "auc_lift": (mean_auc - baseline_auc) / baseline_auc if baseline_auc > 0 and not np.isnan(mean_auc) else 0,
    }

    return model, metrics, top_pos, top_neg


def train_regression_small(df: pd.DataFrame) -> Tuple[Pipeline, Dict]:
    """
    Train a ridge regression model to predict payment performance.

    Args:
        df: DataFrame with loan data

    Returns:
        Tuple containing:
        - Fitted pipeline model
        - Dictionary of cross-validation metrics (includes baselines and tiers)
    """
    y = pd.to_numeric(df.get("payment_performance"), errors="coerce")
    mask = y.notna()
    X = build_feature_matrix(df.loc[mask])

    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = [c for c in X.columns if c not in num_cols]

    # Build preprocessing pipeline
    pre = ColumnTransformer([
        ("num", Pipeline([
            ("imp", SimpleImputer(strategy="median")),
            ("sc", StandardScaler())
        ]), num_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore", min_frequency=0.05), cat_cols)
    ])

    # Build full model pipeline
    model = Pipeline([
        ("pre", pre),
        ("reg", Ridge(alpha=1.0))
    ])

    # Cross-validation
    n_splits = safe_kfold(mask.sum())
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    r2s, rmses = [], []

    # Calculate baseline (mean predictor) RMSE
    y_actual = y.loc[mask]
    baseline_rmse = np.sqrt(np.mean((y_actual - y_actual.mean()) ** 2))

    for tr, te in kf.split(X):
        model.fit(X.iloc[tr], y.iloc[mask].iloc[tr])
        yhat = model.predict(X.iloc[te])
        ytrue = y.iloc[mask].iloc[te]

        if ytrue.nunique() > 1:
            r2 = 1 - np.sum((ytrue - yhat) ** 2) / np.sum((ytrue - ytrue.mean()) ** 2)
        else:
            r2 = np.nan

        r2s.append(r2)
        rmses.append(np.sqrt(np.mean((ytrue - yhat) ** 2)))

    # Final fit on all data
    model.fit(X, y.loc[mask])

    mean_r2 = np.nanmean(r2s)
    mean_rmse = np.mean(rmses)

    # Get performance tier
    r2_tier, r2_color = get_metric_tier("r2", mean_r2)

    # Calculate improvement over baseline
    rmse_improvement = (baseline_rmse - mean_rmse) / baseline_rmse if baseline_rmse > 0 else 0

    return model, {
        "R2": (mean_r2, np.nanstd(r2s)),
        "RMSE": (mean_rmse, np.std(rmses)),
        "n_samples": int(mask.sum()),
        "n_splits": n_splits,
        # Baselines
        "baseline_rmse": baseline_rmse,
        "mean_target": float(y_actual.mean()),
        "std_target": float(y_actual.std()),
        # Performance tier
        "r2_tier": r2_tier,
        # Improvement metrics
        "rmse_improvement_pct": rmse_improvement,
    }


def render_corr_outputs(df: pd.DataFrame):
    """
    Render correlation analysis outputs with visualizations.

    Args:
        df: DataFrame with loan data
    """
    from utils.loan_tape_analytics import compute_correlations

    st.subheader("Correlation Snapshot")
    corr_df = compute_correlations(df)

    if corr_df.empty:
        st.info("Not enough numeric data for correlations.")
        return

    # Display correlation table
    disp = corr_df.copy()
    for c in ["pearson_r_vs_performance", "spearman_rho_vs_performance", "pointbiserial_vs_problem"]:
        if c in disp.columns:
            disp[c] = disp[c].map(lambda x: f"{x:.3f}" if pd.notnull(x) else "")

    st.dataframe(disp, use_container_width=True, hide_index=True)

    # Heatmap visualization (Spearman correlation)
    hm = corr_df[["feature", "spearman_rho_vs_performance"]].dropna()
    if not hm.empty:
        chart = alt.Chart(hm).mark_rect().encode(
            x=alt.X("feature:N", sort=None, title="Feature"),
            y=alt.Y("var:N", sort=None, title="Target", axis=alt.Axis(labels=True)),
            color=alt.Color(
                "value:Q",
                scale=alt.Scale(scheme="redblue", domain=[-1, 1]),
                title="Spearman ρ"
            ),
            tooltip=[
                alt.Tooltip("feature:N"),
                alt.Tooltip("value:Q", title="ρ", format=".3f")
            ]
        ).transform_calculate(
            var="'payment_performance'"
        ).transform_calculate(
            value="datum.spearman_rho_vs_performance"
        ).properties(
            width=700,
            height=150,
            title="Correlation with Payment Performance"
        )
        st.altair_chart(chart, use_container_width=True)

    # Download option
    st.download_button(
        "Download correlations (CSV)",
        corr_df.to_csv(index=False).encode("utf-8"),
        file_name="correlations.csv",
        mime="text/csv"
    )


def render_fico_tib_heatmap(df: pd.DataFrame):
    """
    Render FICO × TIB heatmap showing average payment performance.

    Args:
        df: DataFrame with loan data including fico and tib columns
    """
    st.subheader("FICO × TIB: Avg Payment Performance")

    if "fico" not in df.columns or "tib" not in df.columns:
        st.info("Need both FICO and TIB for this view.")
        return

    d = df.copy()
    d["fico"] = pd.to_numeric(d["fico"], errors="coerce")
    d["tib"] = pd.to_numeric(d["tib"], errors="coerce")

    # Create bins
    fico_bins = [0, 580, 620, 660, 700, 740, 850]
    tib_bins = [0, 5, 10, 15, 20, 100]

    d["fico_band"] = pd.cut(
        d["fico"],
        bins=fico_bins,
        labels=["<580", "580-619", "620-659", "660-699", "700-739", "740+"],
        right=False
    )
    d["tib_band"] = pd.cut(
        d["tib"],
        bins=tib_bins,
        labels=["≤5", "5-10", "10-15", "15-20", "20+"],
        right=False
    )

    # Aggregate by bands
    m = d.groupby(["fico_band", "tib_band"], observed=True).agg(
        avg_perf=("payment_performance", "mean"),
        n=("loan_id", "count")
    ).reset_index().dropna()

    if m.empty:
        st.info("Insufficient data to render heatmap.")
        return

    # Create heatmap
    heat = alt.Chart(m).mark_rect().encode(
        x=alt.X("fico_band:N", title="FICO Band"),
        y=alt.Y("tib_band:N", title="TIB Band"),
        color=alt.Color(
            "avg_perf:Q",
            scale=alt.Scale(scheme="viridis", domain=[0, 1]),
            title="Avg Perf"
        ),
        tooltip=[
            alt.Tooltip("fico_band:N", title="FICO"),
            alt.Tooltip("tib_band:N", title="TIB"),
            alt.Tooltip("avg_perf:Q", title="Avg Performance", format=".2%"),
            alt.Tooltip("n:Q", title="Count")
        ]
    ).properties(
        width=500,
        height=300,
        title="FICO × TIB Heatmap"
    )

    st.altair_chart(heat, use_container_width=True)


def render_data_quality_summary(df: pd.DataFrame):
    """
    Render a data quality summary for the ML section.

    Shows sample size, feature completeness, class balance, and any warnings
    that may affect model reliability.

    Args:
        df: DataFrame with loan data
    """
    st.subheader("Data Quality Overview")

    quality = assess_data_quality(df)

    # Top-level metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Sample Size", f"{quality['sample_size']:,}")
    with col2:
        problem_rate = quality['class_balance']['problem_rate']
        st.metric("Problem Loan Rate", f"{problem_rate:.1%}")
    with col3:
        st.metric("Problem Loans", quality['class_balance']['problem_loans'])
    with col4:
        st.metric("Good Loans", quality['class_balance']['good_loans'])

    # Feature completeness table
    if quality['feature_completeness']:
        st.markdown("##### Feature Completeness")

        completeness_data = []
        for feat, info in quality['feature_completeness'].items():
            pct = info['pct']
            status = "Complete" if pct >= 0.95 else ("Mostly Complete" if pct >= 0.7 else ("Partial" if pct >= 0.5 else "Limited"))
            completeness_data.append({
                "Feature": get_display_name(feat),
                "Valid Records": f"{info['valid']:,}",
                "Completeness": f"{pct:.0%}",
                "Status": status
            })

        completeness_df = pd.DataFrame(completeness_data)
        st.dataframe(completeness_df, use_container_width=True, hide_index=True)

    # Warnings
    if quality['warnings']:
        st.markdown("##### Data Warnings")
        for warning in quality['warnings']:
            st.warning(warning)

    # Overall assessment
    if quality['is_adequate']:
        st.success("Data quality is sufficient for ML modeling. Results should be interpretable.")
    else:
        st.error("Data quality is limited. ML results should be treated as exploratory only.")

    st.markdown("---")


def render_model_summary(classification_metrics: Dict, regression_metrics: Dict):
    """
    Render an executive summary of model performance.

    Provides a high-level interpretation of what the models found
    and actionable recommendations.

    Args:
        classification_metrics: Dict from train_classification_small
        regression_metrics: Dict from train_regression_small
    """
    st.subheader("Model Summary & Recommendations")

    # Classification model assessment
    st.markdown("##### Problem Loan Prediction")

    auc = classification_metrics.get('ROC AUC', (np.nan, np.nan))[0]
    auc_tier = classification_metrics.get('auc_tier', 'N/A')
    pos_rate = classification_metrics.get('pos_rate', 0)

    if not np.isnan(auc):
        if auc >= 0.80:
            st.success(f"""
            **Model Performance: {auc_tier}** (ROC AUC = {auc:.3f})

            The model shows strong ability to identify problem loans. Key insights:
            - Review the top risk-increasing factors for underwriting adjustments
            - Consider using model scores to prioritize collections efforts
            - Monitor partner and industry concentrations in high-risk segments
            """)
        elif auc >= 0.70:
            st.info(f"""
            **Model Performance: {auc_tier}** (ROC AUC = {auc:.3f})

            The model shows moderate predictive power. Recommendations:
            - Use predictions as one input among several for decision-making
            - Focus on the strongest coefficient signals
            - Consider gathering additional data features to improve predictions
            """)
        else:
            st.warning(f"""
            **Model Performance: {auc_tier}** (ROC AUC = {auc:.3f})

            The model has limited predictive power. This could mean:
            - Problem loans are driven by external factors not captured in the data
            - Sample size may be insufficient for pattern detection
            - Consider this analysis as directional only
            """)
    else:
        st.warning("Classification model could not be evaluated (insufficient data or class imbalance).")

    # Regression model assessment
    st.markdown("##### Payment Performance Prediction")

    r2 = regression_metrics.get('R2', (np.nan, np.nan))[0]
    r2_tier = regression_metrics.get('r2_tier', 'N/A')
    rmse_improvement = regression_metrics.get('rmse_improvement_pct', 0)

    if not np.isnan(r2):
        if r2 >= 0.50:
            st.success(f"""
            **Model Performance: {r2_tier}** (R² = {r2:.3f})

            The model explains {r2:.0%} of payment performance variation.
            - Predictions can be used for portfolio-level forecasting
            - RMSE improvement over baseline: {rmse_improvement:.1%}
            """)
        elif r2 >= 0.20:
            st.info(f"""
            **Model Performance: {r2_tier}** (R² = {r2:.3f})

            The model captures some patterns in payment performance.
            - Use predictions for directional insights
            - Significant unexplained variance remains
            """)
        else:
            st.warning(f"""
            **Model Performance: {r2_tier}** (R² = {r2:.3f})

            Payment performance appears difficult to predict from available features.
            - External factors may dominate performance outcomes
            - Consider this analysis as exploratory
            """)
    else:
        st.warning("Regression model could not be evaluated.")

    st.markdown("---")
