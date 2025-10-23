# utils/loan_tape_data.py
"""
Data preparation and transformation utilities for loan tape dashboard.
Handles data loading, merging, IRR calculations, and risk scoring.
"""

import pandas as pd
import numpy as np
import numpy_financial as npf
from typing import Optional

# Import constants from main config
from utils.config import PLATFORM_FEE_RATE

# Platform fee for calculations
PLATFORM_FEE = PLATFORM_FEE_RATE

# Status risk multipliers for risk scoring
STATUS_RISK_MULTIPLIERS = {
    "Active": 1.0,
    "Active - Frequently Late": 1.3,
    "Minor Delinquency": 1.5,
    "Past Delinquency": 1.2,
    "Moderate Delinquency": 2.0,
    "Late": 2.5,
    "Severe Delinquency": 3.0,
    "Default": 4.0,
    "Bankrupt": 5.0,
    "Severe": 5.0,
    "Paid Off": 0.0,
}


def prepare_loan_data(loans_df: pd.DataFrame, deals_df: pd.DataFrame) -> pd.DataFrame:
    """
    Merge loan summaries with deal data and derive calculated fields.

    Args:
        loans_df: Loan summaries dataframe
        deals_df: Deals dataframe with commission_fee, fico, tib, industry, etc.

    Returns:
        pd.DataFrame: Enriched loan data with calculated metrics
    """
    if not loans_df.empty and not deals_df.empty:
        # Merge with deals data
        merge_cols = ["loan_id", "deal_name", "partner_source", "industry", "commission_fee", "fico", "tib"]
        merge_cols = [c for c in merge_cols if c in deals_df.columns]
        df = loans_df.merge(deals_df[merge_cols], on="loan_id", how="left")
    else:
        df = loans_df.copy()

    # Normalize date fields
    for date_col in ["funding_date", "maturity_date", "payoff_date"]:
        if date_col in df.columns:
            df[date_col] = pd.to_datetime(df[date_col], errors="coerce")

    # Convert numeric fields
    df["commission_fee"] = pd.to_numeric(df.get("commission_fee", 0), errors="coerce").fillna(0.0)
    df["csl_participation_amount"] = pd.to_numeric(df.get("csl_participation_amount", 0), errors="coerce").fillna(0.0)
    df["total_paid"] = pd.to_numeric(df.get("total_paid", 0), errors="coerce").fillna(0.0)

    # Calculate fees and totals
    df["commission_fees"] = df["csl_participation_amount"] * df["commission_fee"]
    df["platform_fees"] = df["csl_participation_amount"] * PLATFORM_FEE
    df["total_invested"] = df["csl_participation_amount"] + df["platform_fees"] + df["commission_fees"]
    df["net_balance"] = df["total_invested"] - df["total_paid"]

    # Calculate ROI
    df["current_roi"] = np.where(
        df["total_invested"] > 0,
        (df["total_paid"] / df["total_invested"]) - 1,
        0.0
    )

    # Flags
    df["is_unpaid"] = df.get("loan_status", "").ne("Paid Off")

    # Days since funding
    try:
        today = pd.Timestamp.today().tz_localize(None)
        df["days_since_funding"] = df["funding_date"].apply(
            lambda x: (today - pd.to_datetime(x).tz_localize(None)).days if pd.notnull(x) else 0
        )
    except:
        df["days_since_funding"] = 0

    # Months left to maturity
    df["remaining_maturity_months"] = 0.0
    try:
        if "maturity_date" in df.columns:
            today = pd.Timestamp.today().tz_localize(None)
            active_mask = (df.get("loan_status", "") != "Paid Off") & (df["maturity_date"] > today)
            df.loc[active_mask, "remaining_maturity_months"] = df.loc[active_mask, "maturity_date"].apply(
                lambda x: (pd.to_datetime(x).tz_localize(None) - today).days / 30 if pd.notnull(x) else 0
            )
    except:
        pass

    # Cohort analysis fields
    try:
        df["cohort"] = df["funding_date"].dt.to_period("Q").astype(str)
        df["funding_month"] = df["funding_date"].dt.to_period("M")
    except:
        df["cohort"] = "Unknown"
        df["funding_month"] = pd.NaT

    # Sector code (first two digits of NAICS)
    if "industry" in df.columns:
        df["sector_code"] = df["industry"].astype(str).str[:2]

    # Payment performance calculation
    if "payment_performance" not in df.columns or df["payment_performance"].isna().all():
        with np.errstate(divide="ignore", invalid="ignore"):
            perf = np.where(df["total_invested"] > 0, df["total_paid"] / df["total_invested"], np.nan)
            df["payment_performance"] = np.clip(perf, 0, 1)

    return df


def calculate_irr(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate realized and expected Internal Rate of Return (IRR) for loans.

    Args:
        df: Dataframe with loan data including funding/payoff dates and amounts

    Returns:
        pd.DataFrame: Dataframe with added IRR columns
    """
    result_df = df.copy()

    def calc_realized_irr(row):
        """Calculate IRR for paid-off loans"""
        if pd.isna(row.get("funding_date")) or pd.isna(row.get("payoff_date")) or row.get("total_invested", 0) <= 0:
            return None
        try:
            funding_date = pd.to_datetime(row["funding_date"]).tz_localize(None)
            payoff_date = pd.to_datetime(row["payoff_date"]).tz_localize(None)
            if payoff_date <= funding_date:
                return None

            days = (payoff_date - funding_date).days
            years = days / 365.0

            if years < 0.01:
                simple = (row["total_paid"] / row["total_invested"]) - 1
                return (1 + simple) ** (1 / max(years, 1e-6)) - 1

            try:
                irr = npf.irr([-row["total_invested"], row["total_paid"]])
                if irr is None or irr < -1 or irr > 10:
                    simple = (row["total_paid"] / row["total_invested"]) - 1
                    return (1 + simple) ** (1 / years) - 1
                return irr
            except:
                simple = (row["total_paid"] / row["total_invested"]) - 1
                return (1 + simple) ** (1 / years) - 1
        except:
            return None

    def calc_expected_irr(row):
        """Calculate expected IRR for active loans"""
        if pd.isna(row.get("funding_date")) or pd.isna(row.get("maturity_date")) or row.get("total_invested", 0) <= 0:
            return None
        try:
            funding_date = pd.to_datetime(row["funding_date"]).tz_localize(None)
            maturity_date = pd.to_datetime(row["maturity_date"]).tz_localize(None)
            if maturity_date <= funding_date:
                return None

            # Determine expected payment
            if "our_rtr" in row and pd.notnull(row["our_rtr"]):
                expected_payment = row["our_rtr"]
            elif "roi" in row and pd.notnull(row["roi"]):
                expected_payment = row["total_invested"] * (1 + row["roi"])
            else:
                # Fallback: assume 1.2x return
                expected_payment = row["total_invested"] * 1.2

            days = (maturity_date - funding_date).days
            years = days / 365.0

            if years < 0.01:
                simple = (expected_payment / row["total_invested"]) - 1
                return (1 + simple) ** (1 / max(years, 1e-6)) - 1

            try:
                irr = npf.irr([-row["total_invested"], expected_payment])
                if irr is None or irr < -1 or irr > 10:
                    simple = (expected_payment / row["total_invested"]) - 1
                    return (1 + simple) ** (1 / years) - 1
                return irr
            except:
                simple = (expected_payment / row["total_invested"]) - 1
                return (1 + simple) ** (1 / years) - 1
        except:
            return None

    try:
        result_df["realized_irr"] = result_df.apply(calc_realized_irr, axis=1)
        result_df["expected_irr"] = result_df.apply(calc_expected_irr, axis=1)
        result_df["realized_irr_pct"] = result_df["realized_irr"].apply(
            lambda x: f"{x:.2%}" if pd.notnull(x) else "N/A"
        )
        result_df["expected_irr_pct"] = result_df["expected_irr"].apply(
            lambda x: f"{x:.2%}" if pd.notnull(x) else "N/A"
        )
    except:
        result_df["realized_irr"] = None
        result_df["expected_irr"] = None
        result_df["realized_irr_pct"] = "N/A"
        result_df["expected_irr_pct"] = "N/A"

    return result_df


def calculate_risk_scores(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate risk scores for active loans based on payment performance and status.

    Args:
        df: Dataframe with loan data

    Returns:
        pd.DataFrame: Filtered dataframe of active loans with risk scores and bands
    """
    risk_df = df[df.get("loan_status", "") != "Paid Off"].copy()
    if risk_df.empty:
        return risk_df

    # Normalize payment performance
    risk_df["payment_performance"] = pd.to_numeric(
        risk_df["payment_performance"], errors="coerce"
    ).clip(upper=1.0)
    risk_df["performance_gap"] = 1 - risk_df["payment_performance"]

    # Apply status multipliers
    risk_df["status_multiplier"] = risk_df["loan_status"].map(STATUS_RISK_MULTIPLIERS).fillna(1.0)

    # Calculate overdue factor
    today = pd.Timestamp.today().tz_localize(None)
    risk_df["days_past_maturity"] = risk_df["maturity_date"].apply(
        lambda x: max(0, (today - pd.to_datetime(x)).days) if pd.notnull(x) else 0
    )
    # Clamp at 12 months past due
    risk_df["overdue_factor"] = (risk_df["days_past_maturity"] / 30).clip(upper=12) / 12

    # Calculate final risk score
    risk_df["risk_score"] = (
        risk_df["performance_gap"] *
        risk_df["status_multiplier"] *
        (1 + risk_df["overdue_factor"])
    ).clip(upper=5.0)

    # Categorize into risk bands
    risk_bins = [0, 0.5, 1.0, 1.5, 2.0, 5.0]
    risk_labels = ["Low (0-0.5)", "Moderate (0.5-1.0)", "Elevated (1.0-1.5)",
                   "High (1.5-2.0)", "Severe (2.0+)"]
    risk_df["risk_band"] = pd.cut(risk_df["risk_score"], bins=risk_bins, labels=risk_labels)

    return risk_df


def calculate_expected_payment_to_date(row) -> float:
    """
    Calculate expected payment amount to date based on loan progression.

    Args:
        row: DataFrame row with funding_date, maturity_date, and our_rtr

    Returns:
        float: Expected payment amount to current date
    """
    if pd.isna(row.get("funding_date")) or pd.isna(row.get("maturity_date")) or pd.isna(row.get("our_rtr")):
        return 0.0
    try:
        funding_date = pd.to_datetime(row["funding_date"]).tz_localize(None)
        maturity_date = pd.to_datetime(row["maturity_date"]).tz_localize(None)
        today = pd.Timestamp.today().tz_localize(None)

        # If matured, expect full payment
        if today >= maturity_date:
            return float(row["our_rtr"])

        # Calculate pro-rated expected payment
        total_days = (maturity_date - funding_date).days
        days_elapsed = (today - funding_date).days

        if total_days <= 0:
            return 0.0

        expected_pct = min(1.0, max(0.0, days_elapsed / total_days))
        return float(row["our_rtr"]) * expected_pct
    except:
        return 0.0


def format_dataframe_for_display(
    df: pd.DataFrame,
    columns: Optional[list] = None,
    rename_map: Optional[dict] = None
) -> pd.DataFrame:
    """
    Format dataframe for display with proper number and date formatting.

    Args:
        df: Dataframe to format
        columns: List of columns to include (all if None)
        rename_map: Dictionary mapping old column names to new display names

    Returns:
        pd.DataFrame: Formatted dataframe ready for display
    """
    if columns:
        display_columns = [c for c in columns if c in df.columns]
        display_df = df[display_columns].copy()
    else:
        display_df = df.copy()

    if rename_map:
        display_df.rename(
            columns={k: v for k, v in rename_map.items() if k in display_df.columns},
            inplace=True
        )

    # Format numeric columns based on column name patterns
    for col in display_df.select_dtypes(include=["float64", "float32"]).columns:
        col_upper = col.upper()
        if any(term in col_upper for term in ["ROI", "RATE", "PERCENTAGE", "PERFORMANCE"]):
            display_df[col] = display_df[col].map(lambda x: f"{x:.2%}" if pd.notnull(x) else "")
        elif any(term in col_upper for term in ["MATURITY", "MONTHS"]):
            display_df[col] = display_df[col].map(lambda x: f"{x:.1f}" if pd.notnull(x) else "")
        elif any(term in col_upper for term in ["CAPITAL", "INVESTED", "PAID", "BALANCE", "FEES"]):
            display_df[col] = display_df[col].map(lambda x: f"${x:,.2f}" if pd.notnull(x) else "")

    # Format date columns
    try:
        reverse_map = {v: k for k, v in rename_map.items()} if rename_map else {}
        for col in display_df.columns:
            if any(term in col for term in ["Date", "Funding", "Maturity", "Funded"]):
                original_col = reverse_map.get(col, col.replace(" ", "_").lower())
                if original_col in df.columns and pd.api.types.is_datetime64_any_dtype(df[original_col]):
                    display_df[col] = pd.to_datetime(df[original_col]).dt.strftime("%Y-%m-%d")
    except:
        pass

    return display_df
