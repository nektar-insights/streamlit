# utils/loan_tape_analytics.py
"""
Analytics and correlation utilities for loan tape dashboard.
Handles feature engineering, correlation analysis, and statistical calculations.
"""

import pandas as pd
import numpy as np
from scipy.stats import pearsonr, spearmanr, pointbiserialr
from typing import Tuple, Optional, Dict, List
from utils.data_loader import load_naics_sector_risk
from utils.loan_tape_data import consolidate_sector_code

# =============================================================================
# CONSTANTS - Single Source of Truth
# =============================================================================

# Import problem statuses from centralized definition
from utils.status_constants import PROBLEM_STATUSES

# Feature name mapping for human-readable display
FEATURE_DISPLAY_NAMES = {
    # Core numeric features
    "fico": "FICO Score",
    "tib": "Time in Business (Years)",
    "total_invested": "CSL Participation Amount",
    "net_balance": "Net Outstanding Balance",
    "sector_risk": "Industry Risk Score",
    "total_paid": "Total Payments Received",
    "commission_fee": "Commission Fee Rate",
    "remaining_maturity_months": "Months Until Maturity",
    "days_since_funding": "Days Since Funding",
    "risk_score": "Calculated Risk Score",
    # New features for Current Risk Model
    "ahead_positions": "Positions Ahead (0=1st Lien)",
    "position": "Lien Position",
    "factor_rate": "Factor Rate",
    "effective_yield": "Effective Yield (Factor - Fees)",
    "payment_performance": "Payment Performance %",
    "pct_term_elapsed": "% of Term Elapsed",
    # Payment behavior features (from loan_schedules)
    "total_due_payments": "Total Due Payments",
    "pct_on_time": "On-Time Payment Rate",
    "pct_late": "Late Payment Rate",
    "pct_missed": "Missed Payment Rate",
    "pct_partial": "Partial Payment Rate",
    "consecutive_missed": "Consecutive Missed Payments",
    "days_since_last_payment": "Days Since Last Payment",
    "avg_payment_variance": "Avg Payment Variance",
    # Categorical features (one-hot encoded names)
    "industry": "Industry (NAICS)",
    "partner_source": "Partner Source",
}

# Performance tier thresholds for interpretation
METRIC_THRESHOLDS = {
    "roc_auc": {"excellent": 0.90, "good": 0.80, "fair": 0.70, "poor": 0.60},
    "precision": {"excellent": 0.80, "good": 0.65, "fair": 0.50, "poor": 0.35},
    "recall": {"excellent": 0.80, "good": 0.65, "fair": 0.50, "poor": 0.35},
    "r2": {"excellent": 0.70, "good": 0.50, "fair": 0.30, "poor": 0.10},
}

# Default performance cutoff for problem loan definition
DEFAULT_PERFORMANCE_CUTOFF = 0.90

# Shortfall threshold - how much behind expected schedule to be considered a problem
# 0.15 means a loan is flagged if it's 15+ percentage points behind expected progress
PERFORMANCE_SHORTFALL_THRESHOLD = 0.15


def make_problem_label(
    df: pd.DataFrame,
    perf_cutoff: float = DEFAULT_PERFORMANCE_CUTOFF,
    model_type: str = "default",
    include_reason: bool = False
) -> pd.Series:
    """
    Create binary problem label based on loan status and payment performance.

    For model_type="default":
        A loan is considered a "problem loan" if:
        1. Its status is in PROBLEM_STATUSES (Default, Bankruptcy, Charged Off, etc.), OR
        2. Its payment_performance is below the cutoff threshold (default 90%)

    For model_type="origination":
        A loan is considered a "problem loan" if:
        1. Its status is in PROBLEM_STATUSES (Default, Bankruptcy, Charged Off, etc.), OR
        2. For Paid Off loans: payment_performance < cutoff (didn't fully pay back), OR
        3. For active loans: significantly behind expected payment schedule based on loan age
           (payment_performance is 15+ percentage points below expected progress)

        This avoids incorrectly labeling healthy active loans as "problems" just because
        they haven't paid off yet (e.g., a loan 6 months into a 12-month term at 50% paid
        is performing normally, not a problem).

    Args:
        df: DataFrame with loan data
        perf_cutoff: Performance threshold for completed loans (default 90%)
        model_type: "default" for legacy behavior, "origination" for smarter handling
        include_reason: If True, also returns a Series with problem reasons

    Returns:
        pd.Series: Binary labels (1 = problem loan, 0 = good loan)
        If include_reason=True, returns tuple of (labels, reasons)
    """
    # Initialize reason tracking
    reasons = pd.Series("", index=df.index)

    # Status-based problems (always applies)
    loan_status = df.get("loan_status", pd.Series(dtype=str))
    status_bad = loan_status.isin(PROBLEM_STATUSES)

    # Set reasons for status-based problems
    if status_bad.any():
        reasons.loc[status_bad] = "Status: " + loan_status.loc[status_bad].astype(str)

    if model_type == "origination":
        # For origination models, be smarter about performance-based problems
        # Only flag loans that are actually underperforming relative to expectations

        # Initialize as not a performance problem
        perf_bad = pd.Series(False, index=df.index)

        # Get required data
        payment_perf = pd.to_numeric(df.get("payment_performance", np.nan), errors="coerce")

        # 1. Paid Off loans: flag if they didn't pay back enough
        paid_off_mask = loan_status == "Paid Off"
        if paid_off_mask.any():
            paid_off_problem = payment_perf.loc[paid_off_mask] < perf_cutoff
            perf_bad.loc[paid_off_mask] = paid_off_problem

            # Set reason for paid-off underperformers (only if not already status-flagged)
            paid_off_problem_idx = paid_off_mask & perf_bad & (reasons == "")
            if paid_off_problem_idx.any():
                reasons.loc[paid_off_problem_idx] = payment_perf.loc[paid_off_problem_idx].apply(
                    lambda x: f"Paid off at {x:.0%} (below {perf_cutoff:.0%} threshold)"
                )

        # 2. Active loans: calculate expected progress based on loan age
        active_mask = (loan_status != "Paid Off") & ~status_bad
        if active_mask.any() and "funding_date" in df.columns:
            today = pd.Timestamp.today().tz_localize(None)

            # Calculate expected progress based on time elapsed vs term
            funding_dates = pd.to_datetime(df.loc[active_mask, "funding_date"], errors="coerce")
            if funding_dates.dt.tz is not None:
                funding_dates = funding_dates.dt.tz_localize(None)

            # Calculate maturity dates
            if "maturity_date" in df.columns:
                maturity_dates = pd.to_datetime(df.loc[active_mask, "maturity_date"], errors="coerce")
                if maturity_dates.dt.tz is not None:
                    maturity_dates = maturity_dates.dt.tz_localize(None)
            else:
                # Default to 12 months if no maturity date
                maturity_dates = funding_dates + pd.Timedelta(days=365)

            # Calculate expected progress (0 to 1, capped at 1 for past-maturity loans)
            total_term_days = (maturity_dates - funding_dates).dt.days.replace(0, 365)
            elapsed_days = (today - funding_dates).dt.days.clip(lower=0)
            expected_progress = (elapsed_days / total_term_days).clip(upper=1.0)

            # Calculate shortfall (how far behind expected)
            actual_perf = payment_perf.loc[active_mask].fillna(0)
            shortfall = expected_progress - actual_perf

            # Flag as problem if significantly behind expected (by more than threshold)
            behind_schedule = shortfall > PERFORMANCE_SHORTFALL_THRESHOLD
            perf_bad.loc[active_mask] = behind_schedule

            # Set reason for behind-schedule active loans (only if not already flagged)
            behind_idx = active_mask & perf_bad & (reasons == "")
            if behind_idx.any():
                # Build reason string showing expected vs actual
                for idx in df.index[behind_idx]:
                    if idx in expected_progress.index:
                        exp = expected_progress.loc[idx]
                        act = actual_perf.loc[idx] if idx in actual_perf.index else 0
                        shortfall_val = exp - act
                        reasons.loc[idx] = f"Behind schedule: {act:.0%} paid vs {exp:.0%} expected ({shortfall_val:.0%} shortfall)"

    else:
        # Default behavior: simple threshold on payment_performance
        payment_perf = pd.to_numeric(df.get("payment_performance", np.nan), errors="coerce")
        perf_bad = payment_perf < perf_cutoff

        # Set reason for performance-based problems (only if not already status-flagged)
        perf_problem_idx = perf_bad & (reasons == "")
        if perf_problem_idx.any():
            reasons.loc[perf_problem_idx] = payment_perf.loc[perf_problem_idx].apply(
                lambda x: f"Performance: {x:.0%} (below {perf_cutoff:.0%} threshold)" if pd.notnull(x) else "Performance: Unknown"
            )

    # Combine into final label
    is_problem = (status_bad | perf_bad).astype(int)

    if include_reason:
        return is_problem, reasons
    return is_problem


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


def get_display_name(feature_name: str) -> str:
    """
    Convert a raw feature name to a human-readable display name.

    Handles both direct mappings and one-hot encoded feature names
    (e.g., 'partner_source_ABC' -> 'Partner: ABC')

    Args:
        feature_name: Raw feature/column name

    Returns:
        str: Human-readable display name
    """
    # Direct mapping
    if feature_name in FEATURE_DISPLAY_NAMES:
        return FEATURE_DISPLAY_NAMES[feature_name]

    # Handle one-hot encoded categorical features (e.g., 'industry_44' or 'partner_source_ABC')
    for prefix in ["industry_", "partner_source_"]:
        if feature_name.startswith(prefix):
            category = feature_name[len(prefix):]
            if prefix == "industry_":
                return f"Industry: {category}"
            else:
                return f"Partner: {category}"

    # Default: capitalize and replace underscores
    return feature_name.replace("_", " ").title()


def get_metric_tier(metric_name: str, value: float) -> Tuple[str, str]:
    """
    Get performance tier and color for a metric value.

    Args:
        metric_name: Name of the metric ('roc_auc', 'precision', 'recall', 'r2')
        value: The metric value

    Returns:
        Tuple[str, str]: (tier_label, color_code)
    """
    if metric_name not in METRIC_THRESHOLDS:
        return ("N/A", "#808080")

    thresholds = METRIC_THRESHOLDS[metric_name]

    if pd.isna(value):
        return ("N/A", "#808080")
    elif value >= thresholds["excellent"]:
        return ("Excellent", "#2ca02c")  # green
    elif value >= thresholds["good"]:
        return ("Good", "#98df8a")  # light green
    elif value >= thresholds["fair"]:
        return ("Fair", "#ffbb78")  # orange
    elif value >= thresholds["poor"]:
        return ("Poor", "#ff7f0e")  # dark orange
    else:
        return ("Very Poor", "#d62728")  # red


def assess_data_quality(df: pd.DataFrame) -> Dict:
    """
    Assess data quality for ML modeling.

    Checks:
    - Sample size adequacy
    - Feature completeness (missing value rates)
    - Class balance for target variable
    - Feature variance

    Args:
        df: DataFrame with loan data

    Returns:
        Dict containing:
        - 'sample_size': total rows
        - 'feature_completeness': dict of {column: completeness_pct}
        - 'class_balance': dict with positive/negative counts and rate
        - 'warnings': list of warning messages
        - 'is_adequate': bool indicating if data is sufficient for ML
    """
    warnings = []
    feature_cols = ["fico", "tib", "total_invested", "net_balance", "industry", "partner_source"]

    # Sample size check
    n_samples = len(df)
    if n_samples < 50:
        warnings.append(f"Small sample size ({n_samples} loans). Results may be unreliable.")
    elif n_samples < 100:
        warnings.append(f"Moderate sample size ({n_samples} loans). Consider results as directional.")

    # Feature completeness
    completeness = {}
    for col in feature_cols:
        if col in df.columns:
            valid_count = df[col].notna().sum()
            pct = valid_count / n_samples if n_samples > 0 else 0
            completeness[col] = {"valid": valid_count, "pct": pct}
            if pct < 0.5:
                warnings.append(f"'{get_display_name(col)}' is <50% complete ({pct:.0%})")
        else:
            completeness[col] = {"valid": 0, "pct": 0.0}
            warnings.append(f"'{get_display_name(col)}' column is missing")

    # Class balance
    y = make_problem_label(df)
    pos_count = int(y.sum())
    neg_count = int(len(y) - pos_count)
    pos_rate = pos_count / len(y) if len(y) > 0 else 0

    class_balance = {
        "problem_loans": pos_count,
        "good_loans": neg_count,
        "problem_rate": pos_rate
    }

    if pos_rate < 0.05:
        warnings.append(f"Very few problem loans ({pos_rate:.1%}). Model may struggle to learn patterns.")
    elif pos_rate > 0.50:
        warnings.append(f"High problem loan rate ({pos_rate:.1%}). This is unusual - verify data quality.")

    # Minimum class count for stratified CV
    min_class = min(pos_count, neg_count)
    if min_class < 5:
        warnings.append(f"Minority class has only {min_class} samples. Cross-validation will be limited.")

    is_adequate = (n_samples >= 30 and min_class >= 3 and
                   any(v["pct"] > 0.5 for v in completeness.values()))

    return {
        "sample_size": n_samples,
        "feature_completeness": completeness,
        "class_balance": class_balance,
        "warnings": warnings,
        "is_adequate": is_adequate
    }


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


def get_payment_behavior_features(schedules_df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate loan_schedules to loan-level behavioral features.

    Only considers DUE payments (payment_date <= today), not future scheduled.

    Args:
        schedules_df: DataFrame with loan payment schedules containing columns:
            - loan_id, payment_date, expected_payment, actual_payment
            - status: "Paid", "Paid Late", "Partial", "Missed", "Scheduled"
            - payment_difference (actual - expected)

    Returns:
        DataFrame with columns:
        - loan_id
        - total_due_payments: count of payments that were due
        - pct_on_time: % of due payments with status "Paid"
        - pct_late: % with status "Paid Late"
        - pct_missed: % with status "Missed"
        - pct_partial: % with status "Partial"
        - consecutive_missed: current streak of missed payments (0 if last was paid)
        - days_since_last_payment: days since most recent actual payment
        - avg_payment_variance: mean of (actual - expected) / expected
    """
    if schedules_df.empty:
        return pd.DataFrame(columns=[
            "loan_id", "total_due_payments", "pct_on_time", "pct_late",
            "pct_missed", "pct_partial", "consecutive_missed",
            "days_since_last_payment", "avg_payment_variance"
        ])

    df = schedules_df.copy()

    # Normalize loan_id
    df["loan_id"] = df["loan_id"].astype(str).str.strip().str.replace(r'\.0$', '', regex=True)

    # Parse dates and filter to due payments only (payment_date <= today)
    df["payment_date"] = pd.to_datetime(df["payment_date"], errors="coerce")
    today = pd.Timestamp.today().tz_localize(None)

    # Normalize payment_date to tz-naive if needed
    if df["payment_date"].dt.tz is not None:
        df["payment_date"] = df["payment_date"].dt.tz_localize(None)

    # Filter to payments that are due (not future scheduled)
    due_mask = df["payment_date"].notna() & (df["payment_date"] <= today)
    due_df = df[due_mask].copy()

    if due_df.empty:
        return pd.DataFrame(columns=[
            "loan_id", "total_due_payments", "pct_on_time", "pct_late",
            "pct_missed", "pct_partial", "consecutive_missed",
            "days_since_last_payment", "avg_payment_variance"
        ])

    # Normalize numeric columns
    due_df["expected_payment"] = pd.to_numeric(due_df.get("expected_payment"), errors="coerce")
    due_df["actual_payment"] = pd.to_numeric(due_df.get("actual_payment"), errors="coerce")

    # Normalize status column
    if "status" in due_df.columns:
        due_df["status"] = due_df["status"].astype(str).str.strip()
    else:
        # Create status from actual vs expected if not available
        due_df["status"] = "Unknown"

    # Calculate status percentages per loan
    def calc_loan_features(group):
        total = len(group)
        if total == 0:
            return pd.Series({
                "total_due_payments": 0,
                "pct_on_time": 0.0,
                "pct_late": 0.0,
                "pct_missed": 0.0,
                "pct_partial": 0.0,
                "consecutive_missed": 0,
                "days_since_last_payment": np.nan,
                "avg_payment_variance": 0.0,
            })

        # Calculate percentage of each status
        status_counts = group["status"].value_counts()
        pct_on_time = status_counts.get("Paid", 0) / total
        pct_late = status_counts.get("Paid Late", 0) / total
        pct_missed = status_counts.get("Missed", 0) / total
        pct_partial = status_counts.get("Partial", 0) / total

        # Calculate consecutive missed payments (from most recent)
        sorted_group = group.sort_values("payment_date", ascending=False)
        consecutive_missed = 0
        for _, row in sorted_group.iterrows():
            if row["status"] == "Missed":
                consecutive_missed += 1
            else:
                break

        # Days since last actual payment
        paid_payments = group[group["actual_payment"].notna() & (group["actual_payment"] > 0)]
        if not paid_payments.empty:
            last_payment_date = paid_payments["payment_date"].max()
            days_since_last_payment = (today - last_payment_date).days
        else:
            days_since_last_payment = np.nan

        # Average payment variance: (actual - expected) / expected
        variance_mask = (group["expected_payment"].notna() &
                        (group["expected_payment"] > 0) &
                        group["actual_payment"].notna())
        variance_group = group[variance_mask]
        if not variance_group.empty:
            variance = (variance_group["actual_payment"] - variance_group["expected_payment"]) / variance_group["expected_payment"]
            avg_payment_variance = variance.mean()
        else:
            avg_payment_variance = 0.0

        return pd.Series({
            "total_due_payments": total,
            "pct_on_time": pct_on_time,
            "pct_late": pct_late,
            "pct_missed": pct_missed,
            "pct_partial": pct_partial,
            "consecutive_missed": consecutive_missed,
            "days_since_last_payment": days_since_last_payment,
            "avg_payment_variance": avg_payment_variance,
        })

    result = due_df.groupby("loan_id").apply(calc_loan_features).reset_index()

    return result


def build_feature_matrix(
    df: pd.DataFrame,
    schedules_df: Optional[pd.DataFrame] = None,
    model_type: str = "origination"
) -> pd.DataFrame:
    """
    Build feature matrix for machine learning models.

    Creates a standardized feature set including numeric fields,
    sector risk scores, and categorical variables. For current_risk models,
    also includes payment behavior features from loan schedules.

    Args:
        df: DataFrame with loan data
        schedules_df: Optional DataFrame with payment schedules (required for model_type="current_risk")
        model_type: Type of model - "origination" (default) or "current_risk"
            - "origination": Only uses features available at loan origination
            - "current_risk": Includes payment behavior features for active loan risk assessment

    Returns:
        pd.DataFrame: Feature matrix ready for ML models
    """
    # Start with core numeric features (origination features)
    # Note: ahead_positions from HubSpot = number of positions ahead (0=1st lien, 1=2nd, etc.)
    X = pd.DataFrame({
        "fico": pd.to_numeric(df.get("fico"), errors="coerce"),
        "tib": pd.to_numeric(df.get("tib"), errors="coerce"),
        "total_invested": pd.to_numeric(df.get("total_invested"), errors="coerce"),
        "ahead_positions": pd.to_numeric(df.get("ahead_positions"), errors="coerce"),
    })

    # Add sector code (2-digit NAICS)
    if "sector_code" not in df.columns and "industry" in df.columns:
        sec = df["industry"].astype(str).str[:2].str.zfill(2).apply(consolidate_sector_code)
    else:
        sec = df.get("sector_code")

    # Add sector risk score
    try:
        sr = load_naics_sector_risk()
        if sr is not None and not sr.empty and sec is not None:
            # Apply consolidation to risk table as well
            sr = sr.copy()
            sr["sector_code"] = sr["sector_code"].astype(str).str.zfill(2).apply(consolidate_sector_code)
            sr = sr.drop_duplicates(subset=["sector_code"], keep="first")
            risk_map = dict(zip(sr["sector_code"], sr["risk_score"]))
            X["sector_risk"] = sec.map(risk_map)
    except:
        pass

    # Add current_risk specific features (behavioral + temporal)
    if model_type == "current_risk":
        # Add payment behavior features from schedules
        if schedules_df is not None and not schedules_df.empty:
            behavior_features = get_payment_behavior_features(schedules_df)

            if not behavior_features.empty:
                # Normalize loan_id in df for merge
                df_loan_ids = df["loan_id"].astype(str).str.strip().str.replace(r'\.0$', '', regex=True)

                # Create mapping from behavior features
                behavior_features = behavior_features.set_index("loan_id")
                for col in ["total_due_payments", "pct_on_time", "pct_late", "pct_missed",
                           "pct_partial", "consecutive_missed", "days_since_last_payment",
                           "avg_payment_variance"]:
                    if col in behavior_features.columns:
                        X[col] = df_loan_ids.map(behavior_features[col])

        # Add temporal features
        # pct_term_elapsed: (today - funding_date) / (maturity_date - funding_date)
        today = pd.Timestamp.today().tz_localize(None)

        if "funding_date" in df.columns and "maturity_date" in df.columns:
            funding_dates = pd.to_datetime(df["funding_date"], errors="coerce")
            maturity_dates = pd.to_datetime(df["maturity_date"], errors="coerce")

            # Make tz-naive if needed
            if funding_dates.dt.tz is not None:
                funding_dates = funding_dates.dt.tz_localize(None)
            if maturity_dates.dt.tz is not None:
                maturity_dates = maturity_dates.dt.tz_localize(None)

            term_length = (maturity_dates - funding_dates).dt.days
            elapsed = (today - funding_dates).dt.days

            # Calculate pct_term_elapsed (clipped to 0-2 to handle past-maturity loans)
            X["pct_term_elapsed"] = np.where(
                term_length > 0,
                (elapsed / term_length).clip(0, 2),
                np.nan
            )

        # Add payment_performance from loan data
        if "payment_performance" in df.columns:
            X["payment_performance"] = pd.to_numeric(df["payment_performance"], errors="coerce")

        # Add net_balance for current_risk (useful for scale)
        X["net_balance"] = pd.to_numeric(df.get("net_balance"), errors="coerce")

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


def find_similar_deals(
    df: pd.DataFrame,
    partner: str = None,
    sector_code: str = None,
    fico_range: tuple = None,
    tib_range: tuple = None,
    position: int = None,
    min_matches: int = 5
) -> pd.DataFrame:
    """
    Find historical deals with similar characteristics.

    Filters the loan portfolio to find deals matching the specified criteria.
    Used for comparing new deal parameters against historical performance.

    Args:
        df: DataFrame with loan data (must include payment_performance, loan_status)
        partner: Partner source name to match (exact match)
        sector_code: 2-digit NAICS sector code to match
        fico_range: Tuple of (min_fico, max_fico) to filter by
        tib_range: Tuple of (min_tib, max_tib) to filter by
        position: Lien position to match (0=1st, 1=2nd, 2=3rd+)
        min_matches: Minimum number of matches required; if fewer, relaxes filters

    Returns:
        pd.DataFrame with columns:
        - loan_id, deal_name, partner_source, sector_code, industry_name
        - fico, tib, ahead_positions
        - loan_status, payment_performance
        - total_invested, total_paid, net_balance
        - is_problem (boolean)
        Also includes summary statistics in DataFrame attributes:
        - attrs['match_count']: Number of matching deals
        - attrs['problem_rate']: Problem rate among matches
        - attrs['avg_performance']: Average payment performance
        - attrs['match_criteria']: Description of criteria used
    """
    if df.empty:
        result = pd.DataFrame()
        result.attrs['match_count'] = 0
        result.attrs['problem_rate'] = 0.0
        result.attrs['avg_performance'] = 0.0
        result.attrs['match_criteria'] = "No data available"
        return result

    # Start with all loans
    similar = df.copy()
    criteria_used = []

    # Apply filters progressively
    # Partner filter
    if partner is not None and "partner_source" in similar.columns:
        partner_match = similar["partner_source"] == partner
        if partner_match.sum() >= min_matches:
            similar = similar[partner_match]
            criteria_used.append(f"Partner: {partner}")

    # Sector/Industry filter
    if sector_code is not None:
        # Check for sector_code column, or derive from industry
        if "sector_code" not in similar.columns and "industry" in similar.columns:
            similar["sector_code"] = similar["industry"].astype(str).str[:2].str.zfill(2).apply(consolidate_sector_code)

        if "sector_code" in similar.columns:
            sector_match = similar["sector_code"] == sector_code
            if sector_match.sum() >= min_matches:
                similar = similar[sector_match]
                criteria_used.append(f"Sector: {sector_code}")

    # FICO range filter
    if fico_range is not None and "fico" in similar.columns:
        min_fico, max_fico = fico_range
        similar["fico_num"] = pd.to_numeric(similar["fico"], errors="coerce")
        fico_match = (similar["fico_num"] >= min_fico) & (similar["fico_num"] <= max_fico)
        if fico_match.sum() >= min_matches:
            similar = similar[fico_match]
            criteria_used.append(f"FICO: {min_fico}-{max_fico}")

    # TIB range filter
    if tib_range is not None and "tib" in similar.columns:
        min_tib, max_tib = tib_range
        similar["tib_num"] = pd.to_numeric(similar["tib"], errors="coerce")
        tib_match = (similar["tib_num"] >= min_tib) & (similar["tib_num"] <= max_tib)
        if tib_match.sum() >= min_matches:
            similar = similar[tib_match]
            criteria_used.append(f"TIB: {min_tib}-{max_tib} years")

    # Position filter
    if position is not None and "ahead_positions" in similar.columns:
        similar["position_num"] = pd.to_numeric(similar["ahead_positions"], errors="coerce")
        position_match = similar["position_num"] == position
        if position_match.sum() >= min_matches:
            similar = similar[position_match]
            pos_labels = {0: "1st", 1: "2nd", 2: "3rd+"}
            criteria_used.append(f"Position: {pos_labels.get(position, str(position))}")

    # Add problem label with reasons
    is_problem, problem_reason = make_problem_label(similar, model_type="origination", include_reason=True)
    similar["is_problem"] = is_problem
    similar["problem_reason"] = problem_reason

    # Calculate summary statistics
    match_count = len(similar)
    problem_rate = float(similar["is_problem"].mean()) if match_count > 0 else 0.0

    if "payment_performance" in similar.columns:
        avg_performance = float(pd.to_numeric(similar["payment_performance"], errors="coerce").mean())
    else:
        avg_performance = 0.0

    # Select relevant columns for output
    output_cols = [
        "loan_id", "deal_name", "partner_source",
        "fico", "tib", "ahead_positions",
        "loan_status", "payment_performance",
        "total_invested", "total_paid", "net_balance",
        "is_problem", "problem_reason"
    ]

    # Add sector columns if available
    if "sector_code" in similar.columns:
        output_cols.insert(3, "sector_code")
    if "industry_name" in similar.columns:
        output_cols.insert(4, "industry_name")

    # Filter to available columns
    output_cols = [c for c in output_cols if c in similar.columns]
    result = similar[output_cols].copy()

    # Sort by payment_performance (worst first)
    if "payment_performance" in result.columns:
        result = result.sort_values("payment_performance", ascending=True)

    # Attach metadata
    result.attrs['match_count'] = match_count
    result.attrs['problem_rate'] = problem_rate
    result.attrs['avg_performance'] = avg_performance
    result.attrs['match_criteria'] = " + ".join(criteria_used) if criteria_used else "All deals"

    return result


def get_similar_deals_comparison(
    df: pd.DataFrame,
    partner: str = None,
    sector_code: str = None,
    fico_range: tuple = None,
    tib_range: tuple = None,
    position: int = None
) -> Dict:
    """
    Get comparison metrics between similar deals and portfolio average.

    Args:
        df: DataFrame with loan data
        partner: Partner source name to match
        sector_code: 2-digit NAICS sector code to match
        fico_range: Tuple of (min_fico, max_fico)
        tib_range: Tuple of (min_tib, max_tib)
        position: Lien position (0=1st, 1=2nd, 2=3rd+)

    Returns:
        Dict containing:
        - 'similar_deals': DataFrame of matching deals
        - 'similar_count': Number of similar deals
        - 'similar_problem_rate': Problem rate for similar deals
        - 'similar_avg_performance': Avg payment performance for similar deals
        - 'portfolio_problem_rate': Problem rate for entire portfolio
        - 'portfolio_avg_performance': Avg payment performance for portfolio
        - 'risk_multiplier': How much riskier similar deals are vs portfolio
        - 'match_criteria': Description of matching criteria
        - 'warnings': List of risk warnings
    """
    # Get similar deals
    similar = find_similar_deals(
        df, partner, sector_code, fico_range, tib_range, position
    )

    # Calculate portfolio metrics
    portfolio_problem = make_problem_label(df)
    portfolio_problem_rate = float(portfolio_problem.mean()) if len(df) > 0 else 0.0

    if "payment_performance" in df.columns:
        portfolio_avg_performance = float(pd.to_numeric(df["payment_performance"], errors="coerce").mean())
    else:
        portfolio_avg_performance = 0.0

    # Extract similar deal metrics from attrs
    similar_count = similar.attrs.get('match_count', 0)
    similar_problem_rate = similar.attrs.get('problem_rate', 0.0)
    similar_avg_performance = similar.attrs.get('avg_performance', 0.0)
    match_criteria = similar.attrs.get('match_criteria', '')

    # Calculate risk multiplier
    if portfolio_problem_rate > 0:
        risk_multiplier = similar_problem_rate / portfolio_problem_rate
    else:
        risk_multiplier = 1.0

    # Generate warnings
    warnings = []
    if similar_count > 0:
        if risk_multiplier >= 2.0:
            warnings.append(f"This combination has {risk_multiplier:.1f}x higher problem rate than portfolio average")
        elif risk_multiplier >= 1.5:
            warnings.append(f"This combination has {risk_multiplier:.1f}x higher problem rate than portfolio average")

        if similar_avg_performance < portfolio_avg_performance * 0.9:
            perf_diff = (portfolio_avg_performance - similar_avg_performance) * 100
            warnings.append(f"Similar deals have {perf_diff:.1f}pp lower payment performance than portfolio average")

    return {
        'similar_deals': similar,
        'similar_count': similar_count,
        'similar_problem_rate': similar_problem_rate,
        'similar_avg_performance': similar_avg_performance,
        'portfolio_problem_rate': portfolio_problem_rate,
        'portfolio_avg_performance': portfolio_avg_performance,
        'risk_multiplier': risk_multiplier,
        'match_criteria': match_criteria,
        'warnings': warnings,
    }


def calculate_industry_performance(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate performance metrics by industry/sector.

    Groups by consolidated 2-digit NAICS sector code. NAICS codes 31, 32, and 33
    (Manufacturing subsectors) are consolidated into a single "31 - Manufacturing" category.

    Args:
        df: DataFrame with loan data

    Returns:
        pd.DataFrame: Aggregated metrics by consolidated sector code
    """
    if "industry" not in df.columns:
        return pd.DataFrame()

    # Create working copy with consolidated sector codes
    work_df = df.copy()
    work_df["sector_code"] = work_df["industry"].astype(str).str[:2].str.zfill(2).apply(consolidate_sector_code)

    industry_metrics = work_df.groupby("sector_code").agg({
        "loan_id": "count",
        "total_invested": "sum",
        "total_paid": "sum",
        "net_balance": "sum",
        "payment_performance": "mean",
        "current_roi": "mean"
    }).reset_index()

    industry_metrics.columns = [
        "sector_code", "loan_count", "total_invested", "total_paid",
        "net_balance", "avg_payment_performance", "avg_roi"
    ]

    # Calculate problem rate by consolidated sector
    problem_by_sector = work_df.groupby("sector_code").apply(
        lambda x: make_problem_label(x).mean()
    ).reset_index()
    problem_by_sector.columns = ["sector_code", "problem_rate"]

    industry_metrics = industry_metrics.merge(problem_by_sector, on="sector_code", how="left")

    # Sort by total invested
    industry_metrics = industry_metrics.sort_values("total_invested", ascending=False)

    return industry_metrics
