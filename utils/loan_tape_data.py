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

# NAICS Sector Consolidation Mapping
# Maps sector codes to consolidated categories to avoid duplicate classifications
# NAICS codes 31, 32, 33 are all Manufacturing subsectors:
#   31: Food, Beverage, Tobacco, Textile
#   32: Wood, Paper, Printing, Petroleum, Chemical, Plastics
#   33: Primary Metal, Machinery, Computer, Electrical, Transportation Equipment
# We consolidate all to "31" with unified "Manufacturing" label
NAICS_SECTOR_CONSOLIDATION = {
    "32": "31",  # Manufacturing subsector -> unified Manufacturing
    "33": "31",  # Manufacturing subsector -> unified Manufacturing
}

# Override sector names for consolidated codes
NAICS_CONSOLIDATED_NAMES = {
    "31": "Manufacturing",
}


def consolidate_sector_code(sector_code: str) -> str:
    """
    Consolidate NAICS sector codes to unified categories.

    Specifically consolidates Manufacturing subsectors (31, 32, 33) into a single
    "31 - Manufacturing" classification.

    Args:
        sector_code: 2-digit NAICS sector code

    Returns:
        str: Consolidated sector code (e.g., "32" -> "31")
    """
    if pd.isna(sector_code):
        return sector_code
    sector_str = str(sector_code).strip().zfill(2)
    return NAICS_SECTOR_CONSOLIDATION.get(sector_str, sector_str)


def _normalize_loan_id(series: pd.Series) -> pd.Series:
    """
    Normalize loan_id values for consistent matching.
    Handles: float->int conversion, string trimming, and type conversion.
    """
    if series.empty:
        return series

    # Convert to string and strip whitespace
    result = series.astype(str).str.strip()

    # Remove ".0" suffix from float-converted strings (e.g., "123.0" -> "123")
    result = result.str.replace(r'\.0$', '', regex=True)

    # Remove "nan" and "None" values
    result = result.replace(['nan', 'None', 'NaN', ''], pd.NA)

    return result


def calculate_payment_velocity(row) -> dict:
    """
    Calculate payment velocity metrics for a loan.

    Returns dict with:
        - daily_velocity: Average daily payment amount
        - projected_payoff_days: Estimated days to full payoff
        - projected_payoff_date: Estimated payoff date
        - is_stalled: True if no payments in 30+ days
    """
    result = {
        "daily_velocity": 0.0,
        "projected_payoff_days": None,
        "projected_payoff_date": None,
        "is_stalled": False,
    }

    total_paid = float(row.get("total_paid", 0) or 0)
    our_rtr = float(row.get("our_rtr", 0) or 0)
    funding_date = row.get("funding_date")

    if pd.isna(funding_date) or our_rtr <= 0:
        return result

    try:
        funding_date = pd.to_datetime(funding_date).tz_localize(None)
        today = pd.Timestamp.today().tz_localize(None)
        days_since_funding = (today - funding_date).days

        if days_since_funding <= 0:
            return result

        # Calculate daily payment velocity
        daily_velocity = total_paid / days_since_funding if days_since_funding > 0 else 0
        result["daily_velocity"] = daily_velocity

        # Calculate remaining balance and projected payoff
        remaining = our_rtr - total_paid

        if remaining <= 0:
            # Already paid off
            result["projected_payoff_days"] = 0
            result["projected_payoff_date"] = today
        elif daily_velocity > 0:
            days_to_payoff = remaining / daily_velocity
            result["projected_payoff_days"] = days_to_payoff
            result["projected_payoff_date"] = today + pd.Timedelta(days=days_to_payoff)
        else:
            # No payments = stalled
            result["is_stalled"] = True

        return result

    except Exception:
        return result


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
        # Normalize loan_id in both dataframes for consistent matching
        loans_df = loans_df.copy()
        deals_df = deals_df.copy()
        loans_df["loan_id"] = _normalize_loan_id(loans_df["loan_id"])
        deals_df["loan_id"] = _normalize_loan_id(deals_df["loan_id"])

        # Merge with deals data
        # Note: ahead_positions is the lien position (0=1st, 1=2nd, 2+=junior) - needed for recovery scoring
        # collateral_type and communication_status are used for recovery scoring (defaults applied if missing)
        merge_cols = ["loan_id", "deal_name", "partner_source", "industry", "commission_fee", "fico", "tib", "factor_rate", "loan_term", "ahead_positions", "collateral_type", "communication_status"]
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

    # =========================================================================
    # ENHANCED: Remaining maturity calculation with payment velocity
    # =========================================================================
    df["remaining_maturity_months"] = 0.0
    df["projected_payoff_date"] = pd.NaT
    df["is_past_maturity"] = False
    df["is_stalled"] = False

    try:
        if "maturity_date" in df.columns:
            today = pd.Timestamp.today().tz_localize(None)

            # Identify loans past maturity
            df["is_past_maturity"] = (
                (df.get("loan_status", "") != "Paid Off") &
                (df["maturity_date"].notna()) &
                (df["maturity_date"] < today)
            )

            # For loans NOT past maturity: use original calculation
            active_not_past = (
                (df.get("loan_status", "") != "Paid Off") &
                (df["maturity_date"].notna()) &
                (df["maturity_date"] > today)
            )
            df.loc[active_not_past, "remaining_maturity_months"] = df.loc[active_not_past, "maturity_date"].apply(
                lambda x: (pd.to_datetime(x).tz_localize(None) - today).days / 30 if pd.notnull(x) else 0
            )
            df.loc[active_not_past, "projected_payoff_date"] = df.loc[active_not_past, "maturity_date"]

            # For loans PAST maturity: calculate based on payment velocity
            past_maturity_mask = df["is_past_maturity"]
            if past_maturity_mask.any():
                velocity_results = df.loc[past_maturity_mask].apply(calculate_payment_velocity, axis=1)
                velocity_df = pd.DataFrame(velocity_results.tolist(), index=velocity_results.index)

                # Update projected payoff date
                df.loc[past_maturity_mask, "projected_payoff_date"] = velocity_df["projected_payoff_date"]
                df.loc[past_maturity_mask, "is_stalled"] = velocity_df["is_stalled"]

                # Calculate remaining months from projected payoff
                df.loc[past_maturity_mask, "remaining_maturity_months"] = velocity_df["projected_payoff_days"].apply(
                    lambda x: x / 30 if pd.notnull(x) and x > 0 else 0
                )
    except Exception as e:
        print(f"Error calculating remaining maturity: {e}")
        pass

    # Cohort analysis fields
    try:
        df["cohort"] = df["funding_date"].dt.to_period("Q").astype(str)
        df["funding_month"] = df["funding_date"].dt.to_period("M")
    except:
        df["cohort"] = "Unknown"
        df["funding_month"] = pd.NaT

    # Sector code (first two digits of NAICS) and load sector names
    if "industry" in df.columns:
        df["naics_code"] = df["industry"].astype(str).str.strip()
        df["sector_code"] = df["naics_code"].str[:2].str.zfill(2)

        # Apply NAICS sector consolidation (e.g., 32, 33 -> 31 for Manufacturing)
        df["sector_code"] = df["sector_code"].apply(consolidate_sector_code)

        # Load NAICS sector risk data to get sector names
        try:
            from utils.data_loader import load_naics_sector_risk
            naics_df = load_naics_sector_risk()
            if not naics_df.empty and "sector_code" in naics_df.columns and "sector_name" in naics_df.columns:
                # Ensure sector_code is string and zero-padded
                naics_df["sector_code"] = naics_df["sector_code"].astype(str).str.zfill(2)

                # Apply consolidation to NAICS lookup table as well
                naics_df["sector_code"] = naics_df["sector_code"].apply(consolidate_sector_code)

                # For consolidated codes, override the sector name
                for code, name in NAICS_CONSOLIDATED_NAMES.items():
                    naics_df.loc[naics_df["sector_code"] == code, "sector_name"] = name

                # Merge to get sector names (use first row for each consolidated code)
                df = df.merge(
                    naics_df[["sector_code", "sector_name"]].drop_duplicates(subset=["sector_code"], keep="first"),
                    on="sector_code",
                    how="left"
                )

                # Use sector_name as industry_name, fallback to NAICS code if not available
                # Apply consolidated names override for any remaining unmatched codes
                df["industry_name"] = df["sector_name"].fillna(df["naics_code"])
                for code, name in NAICS_CONSOLIDATED_NAMES.items():
                    df.loc[df["sector_code"] == code, "industry_name"] = name
            else:
                # Fallback: use NAICS code as industry name, with consolidated names
                df["industry_name"] = df["naics_code"]
                for code, name in NAICS_CONSOLIDATED_NAMES.items():
                    df.loc[df["sector_code"] == code, "industry_name"] = name
        except Exception as e:
            # Fallback: use NAICS code as industry name
            df["industry_name"] = df.get("naics_code", "Unknown")

    # Payment performance calculation
    if "payment_performance" not in df.columns or df["payment_performance"].isna().all():
        with np.errstate(divide="ignore", invalid="ignore"):
            perf = np.where(df["total_invested"] > 0, df["total_paid"] / df["total_invested"], np.nan)
            df["payment_performance"] = np.clip(perf, 0, 1)

    return df


def calculate_irr(df: pd.DataFrame, schedules_df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """
    Calculate realized and expected Internal Rate of Return (IRR) for loans using actual payment schedules.

    This is a TRUE IRR calculation that uses all intermediate cash flows with their actual dates,
    not just a simple CAGR calculation.

    Args:
        df: Dataframe with loan data including funding/payoff dates and amounts
        schedules_df: Optional dataframe with loan payment schedules (loan_id, payment_date, actual_payment)

    Returns:
        pd.DataFrame: Dataframe with added IRR columns
    """
    result_df = df.copy()

    # Load schedules if not provided
    if schedules_df is None:
        try:
            from utils.data_loader import load_loan_schedules
            schedules_df = load_loan_schedules()
        except:
            schedules_df = pd.DataFrame()

    # Preprocess schedules
    if not schedules_df.empty and "loan_id" in schedules_df.columns:
        schedules_df = schedules_df.copy()
        schedules_df["payment_date"] = pd.to_datetime(schedules_df["payment_date"], errors="coerce").dt.tz_localize(None)
        schedules_df["actual_payment"] = pd.to_numeric(schedules_df["actual_payment"], errors="coerce")
        # Filter to valid payments only
        schedules_df = schedules_df[
            schedules_df["actual_payment"].notna() &
            (schedules_df["actual_payment"] > 0) &
            schedules_df["payment_date"].notna()
        ]
    else:
        schedules_df = pd.DataFrame()

    def calc_realized_irr(row):
        """Calculate TRUE IRR for paid-off loans using all actual payment cash flows"""
        if pd.isna(row.get("funding_date")) or row.get("total_invested", 0) <= 0:
            return None

        try:
            funding_date = pd.to_datetime(row["funding_date"]).tz_localize(None)
            loan_id = row.get("loan_id")

            # Get all actual payments for this loan from schedules
            if not schedules_df.empty and loan_id:
                loan_payments = schedules_df[schedules_df["loan_id"] == loan_id].copy()

                if not loan_payments.empty:
                    # Sort by payment date
                    loan_payments = loan_payments.sort_values("payment_date")

                    # Build cash flow series with dates
                    cash_flows = []
                    dates = []

                    # Initial investment (negative cash flow)
                    cash_flows.append(-row["total_invested"])
                    dates.append(funding_date)

                    # All actual payments (positive cash flows)
                    for _, payment in loan_payments.iterrows():
                        cash_flows.append(payment["actual_payment"])
                        dates.append(payment["payment_date"])

                    # Check we have at least 2 cash flows
                    if len(cash_flows) < 2:
                        return None

                    # Calculate time periods in years from funding date
                    years = [(d - funding_date).days / 365.0 for d in dates]

                    # Use XIRR-style calculation (IRR with irregular periods)
                    # For numpy-financial, we need to convert to periodic cash flows
                    # We'll use daily periods and place cash flows on the correct days

                    if len(set(dates)) == len(dates):  # All unique dates
                        # Calculate using Newton's method for XIRR
                        irr = calculate_xirr(cash_flows, dates)
                        if irr is not None and -0.99 <= irr <= 10:
                            return irr

            # Fallback: Use simple 2-cash-flow method if schedules not available
            if pd.isna(row.get("payoff_date")) or row.get("total_paid", 0) <= 0:
                return None

            payoff_date = pd.to_datetime(row["payoff_date"]).tz_localize(None)
            if payoff_date <= funding_date:
                return None

            years = (payoff_date - funding_date).days / 365.0
            if years < 0.01:
                return None

            # CAGR fallback (annualized return)
            simple = (row["total_paid"] / row["total_invested"]) - 1
            return (1 + simple) ** (1 / years) - 1

        except Exception as e:
            return None

    def calc_expected_irr(row):
        """Calculate expected IRR for active loans using actual payments + expected remaining"""
        if pd.isna(row.get("funding_date")) or pd.isna(row.get("maturity_date")) or row.get("total_invested", 0) <= 0:
            return None

        try:
            funding_date = pd.to_datetime(row["funding_date"]).tz_localize(None)
            maturity_date = pd.to_datetime(row["maturity_date"]).tz_localize(None)
            loan_id = row.get("loan_id")

            if maturity_date <= funding_date:
                return None

            # Determine expected total payment
            if "our_rtr" in row and pd.notnull(row["our_rtr"]):
                expected_total_payment = row["our_rtr"]
            elif "roi" in row and pd.notnull(row["roi"]):
                expected_total_payment = row["total_invested"] * (1 + row["roi"])
            else:
                expected_total_payment = row["total_invested"] * 1.2

            # Get actual payments received so far
            total_paid_so_far = row.get("total_paid", 0)
            expected_remaining = max(0, expected_total_payment - total_paid_so_far)

            # Build cash flow series
            cash_flows = []
            dates = []

            # Initial investment
            cash_flows.append(-row["total_invested"])
            dates.append(funding_date)

            # Add actual payments from schedules if available
            if not schedules_df.empty and loan_id:
                loan_payments = schedules_df[schedules_df["loan_id"] == loan_id].copy()

                if not loan_payments.empty:
                    loan_payments = loan_payments.sort_values("payment_date")
                    for _, payment in loan_payments.iterrows():
                        cash_flows.append(payment["actual_payment"])
                        dates.append(payment["payment_date"])

            # Add expected remaining payment at maturity
            if expected_remaining > 0:
                cash_flows.append(expected_remaining)
                dates.append(maturity_date)

            if len(cash_flows) < 2:
                return None

            # Calculate XIRR
            if len(set(dates)) == len(dates):
                irr = calculate_xirr(cash_flows, dates)
                if irr is not None and -0.99 <= irr <= 10:
                    return irr

            # Fallback to simple CAGR
            years = (maturity_date - funding_date).days / 365.0
            if years < 0.01:
                return None
            simple = (expected_total_payment / row["total_invested"]) - 1
            return (1 + simple) ** (1 / years) - 1

        except Exception as e:
            return None

    try:
        result_df["realized_irr"] = result_df.apply(calc_realized_irr, axis=1)
        result_df["expected_irr"] = result_df.apply(calc_expected_irr, axis=1)
        result_df["realized_irr_pct"] = result_df["realized_irr"].apply(
            lambda x: f"{x:.1%}" if pd.notnull(x) else "N/A"
        )
        result_df["expected_irr_pct"] = result_df["expected_irr"].apply(
            lambda x: f"{x:.1%}" if pd.notnull(x) else "N/A"
        )
    except Exception as e:
        result_df["realized_irr"] = None
        result_df["expected_irr"] = None
        result_df["realized_irr_pct"] = "N/A"
        result_df["expected_irr_pct"] = "N/A"

    return result_df


def calculate_xirr(cash_flows: list, dates: list, guess: float = 0.1) -> Optional[float]:
    """
    Calculate IRR for irregular cash flows (XIRR).

    Uses Newton's method to find the rate that makes NPV = 0.

    Args:
        cash_flows: List of cash flow amounts
        dates: List of dates corresponding to each cash flow
        guess: Initial guess for IRR

    Returns:
        float: Annualized IRR, or None if calculation fails
    """
    if len(cash_flows) != len(dates) or len(cash_flows) < 2:
        return None

    try:
        # Convert dates to years from first date
        start_date = dates[0]
        years = [(d - start_date).days / 365.0 for d in dates]

        # Newton's method to find IRR
        rate = guess
        max_iterations = 100
        tolerance = 1e-6

        for i in range(max_iterations):
            # Calculate NPV and its derivative
            npv = sum(cf / ((1 + rate) ** y) for cf, y in zip(cash_flows, years))
            dnpv = sum(-y * cf / ((1 + rate) ** (y + 1)) for cf, y in zip(cash_flows, years))

            if abs(dnpv) < 1e-10:
                break

            # Newton's method update
            new_rate = rate - npv / dnpv

            if abs(new_rate - rate) < tolerance:
                return new_rate

            rate = new_rate

            # Keep rate in reasonable bounds
            if rate < -0.99:
                rate = -0.99
            elif rate > 10:
                rate = 10

        # Check if we converged to a reasonable solution
        npv = sum(cf / ((1 + rate) ** y) for cf, y in zip(cash_flows, years))
        if abs(npv) < 0.01:  # NPV close enough to zero
            return rate

        return None

    except Exception as e:
        return None


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

    # Calculate overdue factor - ADJUSTED for payment progress
    today = pd.Timestamp.today().tz_localize(None)
    risk_df["days_past_maturity"] = risk_df["maturity_date"].apply(
        lambda x: max(0, (today - pd.to_datetime(x)).days) if pd.notnull(x) else 0
    )

    # Calculate days_since_funding for all active loans (needed for display)
    risk_df["days_since_funding"] = risk_df["funding_date"].apply(
        lambda x: (today - pd.to_datetime(x).tz_localize(None)).days if pd.notnull(x) else 0
    )

    # NEW: Only penalize if payment performance is lagging
    # If loan is 90%+ paid, don't penalize for being past maturity
    risk_df["overdue_factor"] = 0.0

    # Loans performing well (>=90% paid) get no overdue penalty
    performing_well = risk_df["payment_performance"] >= 0.90

    # Loans underperforming get scaled penalty based on shortfall
    underperforming = ~performing_well & (risk_df["days_past_maturity"] > 0)

    if underperforming.any():
        # Calculate expected progress based on time elapsed
        risk_df.loc[underperforming, "days_since_funding"] = risk_df.loc[underperforming, "funding_date"].apply(
            lambda x: (today - pd.to_datetime(x).tz_localize(None)).days if pd.notnull(x) else 0
        )

        # Original term in days (funding to maturity)
        risk_df.loc[underperforming, "original_term_days"] = risk_df.loc[underperforming].apply(
            lambda row: (pd.to_datetime(row["maturity_date"]).tz_localize(None) -
                        pd.to_datetime(row["funding_date"]).tz_localize(None)).days
            if pd.notnull(row["maturity_date"]) and pd.notnull(row["funding_date"]) else 1,
            axis=1
        )

        # Expected progress (capped at 100%)
        risk_df.loc[underperforming, "expected_progress"] = (
            risk_df.loc[underperforming, "days_since_funding"] /
            risk_df.loc[underperforming, "original_term_days"].replace(0, 1)
        ).clip(upper=1.0)

        # Shortfall = how far behind they are vs expected
        risk_df.loc[underperforming, "payment_shortfall"] = (
            risk_df.loc[underperforming, "expected_progress"] -
            risk_df.loc[underperforming, "payment_performance"]
        ).clip(lower=0)

        # Overdue factor scales with shortfall and time past maturity
        risk_df.loc[underperforming, "overdue_factor"] = (
            risk_df.loc[underperforming, "payment_shortfall"] *
            (1 + risk_df.loc[underperforming, "days_past_maturity"] / 365)
        ).clip(upper=1.0)

    # Clean up temp columns (keep days_since_funding for display)
    temp_cols = ["original_term_days", "expected_progress", "payment_shortfall"]
    risk_df = risk_df.drop(columns=[c for c in temp_cols if c in risk_df.columns], errors="ignore")

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

    ENHANCED: For loans past maturity that are still paying, uses payment
    velocity to determine if they're on track rather than expecting 100%.

    Args:
        row: DataFrame row with funding_date, maturity_date, our_rtr, total_paid

    Returns:
        float: Expected payment amount to current date
    """
    if pd.isna(row.get("funding_date")) or pd.isna(row.get("our_rtr")):
        return 0.0

    try:
        funding_date = pd.to_datetime(row["funding_date"]).tz_localize(None)
        maturity_date = pd.to_datetime(row.get("maturity_date")).tz_localize(None) if pd.notnull(row.get("maturity_date")) else None
        today = pd.Timestamp.today().tz_localize(None)
        our_rtr = float(row["our_rtr"])
        total_paid = float(row.get("total_paid", 0) or 0)

        # Case 1: No maturity date - use simple time-based pro-rating
        if maturity_date is None:
            return our_rtr  # Expect full payment if no maturity info

        # Case 2: Before maturity - standard pro-rating
        if today < maturity_date:
            total_days = (maturity_date - funding_date).days
            days_elapsed = (today - funding_date).days

            if total_days <= 0:
                return 0.0

            expected_pct = min(1.0, max(0.0, days_elapsed / total_days))
            return our_rtr * expected_pct

        # Case 3: Past maturity - check if loan is performing well
        payment_performance = total_paid / our_rtr if our_rtr > 0 else 0

        # If 90%+ paid, they're almost done - expect full amount
        if payment_performance >= 0.90:
            return our_rtr

        # If actively paying (has velocity), calculate expected based on trajectory
        days_since_funding = (today - funding_date).days
        if days_since_funding > 0 and total_paid > 0:
            daily_velocity = total_paid / days_since_funding

            # Project where they "should" be based on original term
            original_term_days = (maturity_date - funding_date).days
            if original_term_days > 0:
                # Expected daily rate to hit target on time
                expected_daily = our_rtr / original_term_days

                # If paying at >= 80% of expected rate, they're on track (extended)
                if daily_velocity >= expected_daily * 0.80:
                    # Pro-rate based on their actual velocity
                    return min(our_rtr, total_paid * 1.1)  # Expect 10% more than current

        # Default: expect full payment for severely past-due loans
        return our_rtr

    except Exception:
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
        if any(term in col_upper for term in ["ROI", "RATE", "PERCENTAGE", "PERFORMANCE", "IRR"]):
            display_df[col] = display_df[col].map(lambda x: f"{x:.1%}" if pd.notnull(x) else "")
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
