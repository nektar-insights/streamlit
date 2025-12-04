# utils/loan_tape_ml.py
"""
Machine learning utilities for loan tape dashboard.
Handles model training, evaluation, and visualization of ML results.
"""

import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
from typing import Tuple, Dict
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import roc_auc_score, precision_score, recall_score

from utils.loan_tape_analytics import make_problem_label, build_feature_matrix, safe_kfold


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
        st.info("""
        **📊 Understanding Classification Metrics:**

        - **ROC AUC (Area Under ROC Curve)**: Measures model's ability to distinguish problem vs. non-problem loans
          - **Higher is better** (Range: 0.5-1.0, where 0.5 = random guess, 1.0 = perfect)
          - **Direction: ↑** Good performance is closer to 1.0

        - **Precision**: Of all loans predicted as problems, what % were actually problems?
          - **Higher is better** (Range: 0-1)
          - **Direction: ↑** Minimizes false alarms

        - **Recall**: Of all actual problem loans, what % did we correctly identify?
          - **Higher is better** (Range: 0-1)
          - **Direction: ↑** Minimizes missed problems

        - **Positive Coefficients (Risk-Increasing)**: Features that increase probability of loan problems
          - **Red flags** - Higher values mean higher risk

        - **Negative Coefficients (Risk-Decreasing)**: Features that decrease probability of loan problems
          - **Green flags** - Higher values mean lower risk
        """)
    else:  # regression
        st.info("""
        **📊 Understanding Regression Metrics:**

        - **R² (R-Squared)**: Proportion of variance in payment performance explained by the model
          - **Higher is better** (Range: -∞ to 1.0, where 1.0 = perfect prediction)
          - **Direction: ↑** Good performance is closer to 1.0
          - Values below 0 mean model performs worse than simply predicting the mean

        - **RMSE (Root Mean Squared Error)**: Average prediction error in payment performance
          - **Lower is better** (Range: 0 to ∞)
          - **Direction: ↓** Smaller values indicate more accurate predictions
          - Measured in same units as payment_performance (percentage points)

        - **Coefficients**: Show relationship between features and payment performance
          - **Positive**: Feature increases payment performance (better for us)
          - **Negative**: Feature decreases payment performance (worse for us)
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
        - Dictionary of cross-validation metrics
        - DataFrame of top positive coefficients
        - DataFrame of top negative coefficients
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

    # Extract top coefficients
    try:
        ohe = model.named_steps["pre"].named_transformers_["cat"]
        num_names = num_cols
        cat_names = ohe.get_feature_names_out(cat_cols).tolist() if cat_cols else []
        feat_names = num_names + cat_names
        coefs = model.named_steps["clf"].coef_.ravel()
        coef_df = pd.DataFrame({"feature": feat_names, "coef": coefs}).sort_values("coef", ascending=False)
        top_pos = coef_df.head(10)
        top_neg = coef_df.tail(10).iloc[::-1]
    except Exception as e:
        top_pos = pd.DataFrame(columns=["feature", "coef"])
        top_neg = pd.DataFrame(columns=["feature", "coef"])

    metrics = {
        "ROC AUC": (np.mean(aucs) if aucs else np.nan, np.std(aucs) if aucs else np.nan),
        "Precision": (np.mean(precs), np.std(precs)),
        "Recall": (np.mean(recs), np.std(recs)),
        "n_samples": len(df),
        "pos_rate": float(y.mean())
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
        - Dictionary of cross-validation metrics
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
    kf = KFold(n_splits=safe_kfold(mask.sum()), shuffle=True, random_state=42)
    r2s, rmses = [], []

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

    return model, {
        "R2": (np.nanmean(r2s), np.nanstd(r2s)),
        "RMSE": (np.mean(rmses), np.std(rmses)),
        "n_samples": int(mask.sum())
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
            alt.Tooltip("avg_perf:Q", title="Avg Performance", format=".1%"),
            alt.Tooltip("n:Q", title="Count")
        ]
    ).properties(
        width=500,
        height=300,
        title="FICO × TIB Heatmap"
    )

    st.altair_chart(heat, use_container_width=True)
