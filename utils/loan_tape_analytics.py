# utils/loan_tape_analytics.py
"""
Analytics and correlation utilities for loan tape dashboard.
Handles feature engineering, correlation analysis, and statistical calculations.
"""

import pandas as pd
import numpy as np
from scipy.stats import pearsonr, spearmanr, pointbiserialr
from typing import Tuple, Optional
from utils.data_loader import load_naics_sector_risk
from utils.loan_tape_data import consolidate_sector_code

# Problem loan statuses
PROBLEM_STATUSES = {
    "Late", "Default", "Bankrupt", "Severe", "Severe Delinquency",
    "Moderate Delinquency", "Active - Frequently Late"
}


def make_problem_label(df: pd.DataFrame, perf_cutoff: float = 0.90) -> pd.Series:
    """
    Create binary problem label based on loan status and payment performance.

    Args:
        df: DataFrame with loan data
        perf_cutoff: Performance threshold below which a loan is considered problematic

    Returns:
        pd.Series: Binary labels (1 = problem loan, 0 = good loan)
    """
    status_bad = df.get("loan_status", "").isin(PROBLEM_STATUSES)
    perf_bad = pd.to_numeric(df.get("payment_performance", np.nan), errors="coerce") < perf_cutoff
    return (status_bad | perf_bad).astype(int)


def safe_kfold(n_items: int, preferred: int = 5) -> int:
    """
    Determine safe number of cross-validation folds based on sample size.

    Args:
        n_items: Number of samples
        preferred: Preferred number of folds

    Returns:
        int: Safe number of folds (minimum 2, ensures at least 2 items per fold)
    """
    return max(2, min(preferred, n_items if n_items >= preferred else max(2, n_items // 2)))


def compute_correlations(base_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute correlations between features and loan performance metrics.

    Calculates Pearson, Spearman, and point-biserial correlations between
    numeric features and payment performance/problem status.

    Args:
        base_df: DataFrame with loan data

    Returns:
        pd.DataFrame: Correlation statistics for each feature
    """
    df = base_df.copy()

    # Attach risk score from sector table if available
    try:
        sr = load_naics_sector_risk()
        if "sector_code" not in df.columns and "industry" in df.columns:
            # Extract 2-digit sector code and apply consolidation (e.g., 32, 33 -> 31 for Manufacturing)
            df["sector_code"] = df["industry"].astype(str).str[:2].str.zfill(2).apply(consolidate_sector_code)
        if not sr.empty and "sector_code" in df.columns:
            # Apply consolidation to risk table as well
            sr = sr.copy()
            sr["sector_code"] = sr["sector_code"].astype(str).str.zfill(2).apply(consolidate_sector_code)
            sr = sr.drop_duplicates(subset=["sector_code"], keep="first")
            df = df.merge(sr[["sector_code", "risk_score"]], on="sector_code", how="left")
    except:
        pass

    # Select numeric columns for correlation analysis
    num_cols = []
    candidate_cols = [
        "fico", "tib", "risk_score", "total_invested", "total_paid",
        "net_balance", "commission_fee", "remaining_maturity_months",
        "days_since_funding"
    ]
    for c in candidate_cols:
        if c in df.columns:
            num_cols.append(c)

    # Target variables
    y_perf = pd.to_numeric(df.get("payment_performance", np.nan), errors="coerce")
    y_clf = make_problem_label(df)

    # Calculate correlations for each feature
    rows = []
    for col in num_cols:
        x = pd.to_numeric(df[col], errors="coerce")
        mask_perf = x.notna() & y_perf.notna()
        mask_clf = x.notna() & y_clf.notna()

        # Pearson and Spearman correlation with payment performance
        if mask_perf.sum() >= 8:
            try:
                pr, pp = pearsonr(x[mask_perf], y_perf[mask_perf])
            except:
                pr, pp = np.nan, np.nan
            try:
                srho, sp = spearmanr(x[mask_perf], y_perf[mask_perf])
            except:
                srho, sp = np.nan, np.nan
        else:
            pr = pp = srho = sp = np.nan

        # Point-biserial correlation with problem status
        if mask_clf.sum() >= 8 and y_clf[mask_clf].nunique() > 1:
            try:
                pbr, pbr_p = pointbiserialr(y_clf[mask_clf], x[mask_clf])
            except:
                pbr, pbr_p = np.nan, np.nan
        else:
            pbr = pbr_p = np.nan

        rows.append({
            "feature": col,
            "pearson_r_vs_performance": pr,
            "pearson_p": pp,
            "spearman_rho_vs_performance": srho,
            "spearman_p": sp,
            "pointbiserial_vs_problem": pbr,
            "pointbiserial_p": pbr_p
        })

    out = pd.DataFrame(rows).sort_values("spearman_rho_vs_performance", ascending=False)
    return out


def build_feature_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build feature matrix for machine learning models.

    Creates a standardized feature set including numeric fields,
    sector risk scores, and categorical variables.

    Args:
        df: DataFrame with loan data

    Returns:
        pd.DataFrame: Feature matrix ready for ML models
    """
    # Start with core numeric features
    X = pd.DataFrame({
        "fico": pd.to_numeric(df.get("fico"), errors="coerce"),
        "tib": pd.to_numeric(df.get("tib"), errors="coerce"),
        "total_invested": pd.to_numeric(df.get("total_invested"), errors="coerce"),
        "net_balance": pd.to_numeric(df.get("net_balance"), errors="coerce"),
    })

    # Add sector risk score
    try:
        sr = load_naics_sector_risk()
        if "sector_code" not in df.columns and "industry" in df.columns:
            # Extract 2-digit sector code and apply consolidation (e.g., 32, 33 -> 31 for Manufacturing)
            sec = df["industry"].astype(str).str[:2].str.zfill(2).apply(consolidate_sector_code)
        else:
            sec = df.get("sector_code")

        if sr is not None and not sr.empty and sec is not None:
            # Apply consolidation to risk table as well
            sr = sr.copy()
            sr["sector_code"] = sr["sector_code"].astype(str).str.zfill(2).apply(consolidate_sector_code)
            sr = sr.drop_duplicates(subset=["sector_code"], keep="first")
            risk_map = dict(zip(sr["sector_code"], sr["risk_score"]))
            X["sector_risk"] = sec.map(risk_map)
    except:
        pass

    # Add categorical features
    cat_df = pd.DataFrame()
    for c in ["industry", "partner_source"]:
        if c in df.columns:
            cat_df[c] = df[c].astype(str)

    if not cat_df.empty:
        X = pd.concat([X, cat_df], axis=1)

    return X


def calculate_cohort_performance(df: pd.DataFrame, cohort_col: str = "cohort") -> pd.DataFrame:
    """
    Calculate performance metrics by cohort.

    Args:
        df: DataFrame with loan data
        cohort_col: Column name containing cohort labels

    Returns:
        pd.DataFrame: Aggregated metrics by cohort
    """
    if cohort_col not in df.columns:
        return pd.DataFrame()

    cohort_metrics = df.groupby(cohort_col, observed=True).agg({
        "loan_id": "count",
        "total_invested": "sum",
        "total_paid": "sum",
        "net_balance": "sum",
        "payment_performance": "mean",
        "current_roi": "mean"
    }).reset_index()

    cohort_metrics.columns = [
        cohort_col, "loan_count", "total_invested", "total_paid",
        "net_balance", "avg_payment_performance", "avg_roi"
    ]

    # Calculate realization rate
    cohort_metrics["realization_rate"] = (
        cohort_metrics["total_paid"] / cohort_metrics["total_invested"]
    ).fillna(0)

    return cohort_metrics


def calculate_portfolio_metrics(df: pd.DataFrame) -> dict:
    """
    Calculate overall portfolio-level metrics.

    Args:
        df: DataFrame with loan data

    Returns:
        dict: Portfolio-level metrics
    """
    total_invested = df["total_invested"].sum() if "total_invested" in df.columns else 0
    total_paid = df["total_paid"].sum() if "total_paid" in df.columns else 0
    net_balance = df["net_balance"].sum() if "net_balance" in df.columns else 0

    # Active loans
    active_df = df[df.get("loan_status", "") != "Paid Off"]
    active_count = len(active_df)
    active_invested = active_df["total_invested"].sum() if "total_invested" in active_df.columns else 0

    # Problem loans
    problem_label = make_problem_label(df)
    problem_count = problem_label.sum()
    problem_rate = problem_count / len(df) if len(df) > 0 else 0

    # Payment performance
    avg_performance = df["payment_performance"].mean() if "payment_performance" in df.columns else 0
    median_performance = df["payment_performance"].median() if "payment_performance" in df.columns else 0

    return {
        "total_loans": len(df),
        "total_invested": total_invested,
        "total_paid": total_paid,
        "net_balance": net_balance,
        "realization_rate": total_paid / total_invested if total_invested > 0 else 0,
        "active_loans": active_count,
        "active_invested": active_invested,
        "problem_loans": problem_count,
        "problem_rate": problem_rate,
        "avg_payment_performance": avg_performance,
        "median_payment_performance": median_performance,
    }


def calculate_partner_performance(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate performance metrics by partner source.

    Args:
        df: DataFrame with loan data

    Returns:
        pd.DataFrame: Aggregated metrics by partner
    """
    if "partner_source" not in df.columns:
        return pd.DataFrame()

    partner_metrics = df.groupby("partner_source").agg({
        "loan_id": "count",
        "total_invested": "sum",
        "total_paid": "sum",
        "net_balance": "sum",
        "payment_performance": ["mean", "median"],
        "current_roi": ["mean", "median"]
    }).reset_index()

    # Flatten column names
    partner_metrics.columns = [
        "partner_source", "loan_count", "total_invested", "total_paid",
        "net_balance", "avg_payment_performance", "median_payment_performance",
        "avg_roi", "median_roi"
    ]

    # Calculate realization rate
    partner_metrics["realization_rate"] = (
        partner_metrics["total_paid"] / partner_metrics["total_invested"]
    ).fillna(0)

    # Sort by total invested
    partner_metrics = partner_metrics.sort_values("total_invested", ascending=False)

    return partner_metrics


def calculate_industry_performance(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate performance metrics by industry/sector.

    Args:
        df: DataFrame with loan data

    Returns:
        pd.DataFrame: Aggregated metrics by industry
    """
    if "industry" not in df.columns:
        return pd.DataFrame()

    industry_metrics = df.groupby("industry").agg({
        "loan_id": "count",
        "total_invested": "sum",
        "total_paid": "sum",
        "net_balance": "sum",
        "payment_performance": "mean",
        "current_roi": "mean"
    }).reset_index()

    industry_metrics.columns = [
        "industry", "loan_count", "total_invested", "total_paid",
        "net_balance", "avg_payment_performance", "avg_roi"
    ]

    # Calculate problem rate
    problem_by_industry = df.groupby("industry").apply(
        lambda x: make_problem_label(x).mean()
    ).reset_index()
    problem_by_industry.columns = ["industry", "problem_rate"]

    industry_metrics = industry_metrics.merge(problem_by_industry, on="industry", how="left")

    # Sort by total invested
    industry_metrics = industry_metrics.sort_values("total_invested", ascending=False)

    return industry_metrics
