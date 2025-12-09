# utils/recovery_scoring.py
"""
Recovery scoring and bad debt expense estimation module.

Estimates recovery probability and bad debt expense per deal based on:
- Status: Current loan status (performing, delinquent, default, etc.)
- Industry: NAICS sector classification
- Collateral: Type and quality of collateral
- Lien Position: Priority in capital structure
- Communication: Borrower engagement level

This is a pre-screening tool, not a final valuation engine.

IMPORTANT: Uses canonical status constants from utils/status_constants.py
and NAICS helpers from utils/loan_tape_data.py for consistency.
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import Optional, Dict, Tuple, List

# Import canonical status constants
from utils.status_constants import (
    ALL_VALID_STATUSES,
    PROBLEM_STATUSES,
    TERMINAL_STATUSES,
    PROTECTED_STATUSES,
    is_problem_status,
    is_terminal_status,
)

# Import NAICS helpers
from utils.loan_tape_data import consolidate_sector_code

# Import NAICS display names
from utils.loan_tape_ml import NAICS_SECTOR_NAMES

# =============================================================================
# SCORING CONSTANTS
# =============================================================================

# Status → Score (0-10)
# Based on historical recovery rates by loan status
# IMPORTANT: Only includes statuses from ALL_VALID_STATUSES (no legacy values)
STATUS_SCORES: Dict[str, float] = {
    # Terminal statuses - very low recovery
    "Bankruptcy": 1.0,
    "Charged Off": 1.0,
    # Severe problem statuses
    "Non-Performing": 2.0,
    "Default": 2.0,
    "Severe Delinquency": 3.0,
    "NSF / Suspended": 3.0,
    # Active collection/legal
    "In Collections": 5.0,
    "Legal Action": 5.0,
    # Moderate delinquency
    "Moderate Delinquency": 6.0,
    # Minor issues
    "Active - Frequently Late": 7.0,
    "Past Delinquency": 8.0,
    "Minor Delinquency": 9.0,
    # Performing statuses
    "Active": 10.0,
    "Paid Off": 10.0,
}

# Validate that all STATUS_SCORES keys are in ALL_VALID_STATUSES
_invalid_statuses = set(STATUS_SCORES.keys()) - set(ALL_VALID_STATUSES)
if _invalid_statuses:
    raise ValueError(
        f"STATUS_SCORES contains invalid statuses not in ALL_VALID_STATUSES: {_invalid_statuses}"
    )

DEFAULT_STATUS_SCORE = 5.0  # Unknown status - moderate risk

# Industry/Sector → Score (1-10)
# Higher score = higher expected recovery
# Based on industry volatility, asset liquidity, and historical defaults
# Keys are 2-digit NAICS sector codes matching NAICS_SECTOR_NAMES
INDUSTRY_SCORES: Dict[str, float] = {
    # High Risk (1-3) - Cyclical, high failure rates
    "23": 1.0,   # Construction - highest default rates, cyclical
    "11": 2.0,   # Agriculture, Forestry, Fishing - weather/commodity dependent
    "48": 2.0,   # Transportation & Warehousing - high fixed costs
    "44": 3.0,   # Retail Trade - competitive, thin margins
    "42": 3.0,   # Wholesale Trade - similar to retail
    # Moderate Risk (4-6) - Discretionary, capital intensive
    "72": 4.0,   # Accommodation & Food Services - high failure rate
    "71": 4.0,   # Arts, Entertainment, Recreation - discretionary
    "31": 5.0,   # Manufacturing - capital intensive (includes 32, 33 via consolidation)
    "81": 6.0,   # Other Services - mixed
    "56": 6.0,   # Administrative & Waste Services
    "62": 6.0,   # Health Care & Social Assistance
    # Lower Risk (7-9) - Stable, asset-backed
    "54": 7.0,   # Professional Services - low capital needs
    "52": 7.0,   # Finance & Insurance
    "53": 7.0,   # Real Estate - asset-backed
    "55": 7.0,   # Management of Companies
    "51": 8.0,   # Information/Tech - high margins
    "22": 9.0,   # Utilities - stable, regulated
    "21": 9.0,   # Mining, Quarrying, Oil/Gas - asset-rich
    # Lowest Risk (10) - Government-backed, nonprofit
    "61": 10.0,  # Educational Services - often nonprofit/gov-backed
    "92": 10.0,  # Public Administration - government
}

# Validate that all INDUSTRY_SCORES keys are in NAICS_SECTOR_NAMES
_invalid_sectors = set(INDUSTRY_SCORES.keys()) - set(NAICS_SECTOR_NAMES.keys())
if _invalid_sectors:
    raise ValueError(
        f"INDUSTRY_SCORES contains invalid sector codes not in NAICS_SECTOR_NAMES: {_invalid_sectors}"
    )

DEFAULT_INDUSTRY_SCORE = 5.0  # Unknown industry

# Collateral Type → Score (1-10)
# Higher score = more liquid/valuable collateral
# NOTE: This field does NOT currently exist in the codebase schema.
# These values are for future use or manual input scenarios.
COLLATERAL_SCORES: Dict[str, float] = {
    "none": 1.0,
    "unsecured": 1.0,
    "intangible": 2.0,           # IP, licenses, brand value
    "inventory": 3.0,            # Consumer goods, perishables
    "receivables_unverified": 4.0,
    "receivables_verified": 5.0,  # Verified/insured AR
    "equipment_specialized": 6.0, # Industry-specific equipment
    "equipment_generic": 7.0,     # Liquid/generic equipment
    "property_encumbered": 8.0,   # Leased or with liens
    "lockbox_no_daca": 8.0,       # Lockbox without control agreement
    "property_owned": 9.0,        # Owned outright real estate
    "lockbox_daca": 10.0,         # Lockbox with full DACA/control
    "cash_securities": 10.0,      # Cash, marketable securities, deposit pledge
}
DEFAULT_COLLATERAL_SCORE = 1.0  # Assume unsecured if unknown (conservative)

# Lien Position → Score
# Uses existing ahead_positions field: 0=1st lien, 1=2nd lien, 2+=junior
LIEN_SCORES: Dict[int, float] = {
    0: 10.0,  # 1st Lien - Senior position
    1: 5.0,   # 2nd Lien - Subordinate
    2: 0.0,   # 3rd+ Lien - Junior/Unsecured
}
DEFAULT_LIEN_SCORE = 0.0  # Assume junior if unknown (conservative)

# Communication Status → Score (1-10)
# NOTE: This field does NOT currently exist in the codebase schema.
# These values are for future use or manual input scenarios.
COMMUNICATION_SCORES: Dict[str, float] = {
    "none_hostile": 1.0,      # No contact or hostile
    "sporadic": 3.0,          # Only under pressure
    "slow_responsive": 6.0,   # Generally responsive but slow
    "engaged": 9.0,           # Fully engaged and proactive
    "plan_milestones": 10.0,  # On a plan and hitting milestones
}
DEFAULT_COMMUNICATION_SCORE = 3.0  # Assume sporadic if unknown

# Component weights (must sum to 1.0)
WEIGHTS = {
    "status": 0.10,
    "industry": 0.30,
    "collateral": 0.30,
    "lien": 0.20,
    "communication": 0.10,
}

# Recovery bands: (min_score, max_score) -> (recovery_low, recovery_high)
RECOVERY_BANDS: List[Tuple[float, float, float, float]] = [
    (9.0, 10.0, 0.90, 1.00),   # Score 9-10 → 90-100% recovery
    (7.0, 9.0, 0.70, 0.89),    # Score 7-8.9 → 70-89% recovery
    (5.0, 7.0, 0.50, 0.69),    # Score 5-6.9 → 50-69% recovery
    (3.0, 5.0, 0.30, 0.49),    # Score 3-4.9 → 30-49% recovery
    (1.0, 3.0, 0.10, 0.29),    # Score 1-2.9 → 10-29% recovery
    (0.0, 1.0, 0.00, 0.09),    # Score 0-1 → 0-9% recovery
]


# =============================================================================
# RESULT DATACLASS
# =============================================================================

@dataclass
class RecoveryScoreResult:
    """Result of recovery score calculation."""

    # Component scores (0-10 scale)
    status_score: float
    industry_score: float
    collateral_score: float
    lien_score: float
    communication_score: float

    # Weighted total recovery score (0-10 scale)
    total_recovery_score: float

    # Recovery metrics
    recovery_band: str           # e.g., "7.0 - 8.9"
    recovery_pct_low: float      # e.g., 0.70
    recovery_pct_high: float     # e.g., 0.89
    recovery_pct_midpoint: float # e.g., 0.795

    # Loss metrics
    loss_pct: float              # 1 - recovery_pct_midpoint

    # Bad debt expense calculation
    exposure_base: float
    bad_debt_expense: float      # exposure_base * loss_pct (midpoint)
    bad_debt_low: float          # exposure_base * (1 - recovery_pct_high)
    bad_debt_high: float         # exposure_base * (1 - recovery_pct_low)

    # Additional context
    is_problem_loan: bool        # True if status is in PROBLEM_STATUSES
    is_terminal: bool            # True if status is in TERMINAL_STATUSES

    def to_dict(self) -> Dict:
        """Convert to dictionary for DataFrame operations."""
        return {
            "status_score": self.status_score,
            "industry_score": self.industry_score,
            "collateral_score": self.collateral_score,
            "lien_score": self.lien_score,
            "communication_score": self.communication_score,
            "total_recovery_score": self.total_recovery_score,
            "recovery_band": self.recovery_band,
            "recovery_pct_low": self.recovery_pct_low,
            "recovery_pct_high": self.recovery_pct_high,
            "recovery_pct_midpoint": self.recovery_pct_midpoint,
            "loss_pct": self.loss_pct,
            "exposure_base": self.exposure_base,
            "bad_debt_expense": self.bad_debt_expense,
            "bad_debt_low": self.bad_debt_low,
            "bad_debt_high": self.bad_debt_high,
            "is_problem_loan": self.is_problem_loan,
            "is_terminal": self.is_terminal,
        }


# =============================================================================
# SCORING FUNCTIONS
# =============================================================================

def get_status_score(loan_status: Optional[str]) -> float:
    """
    Get recovery score based on loan status.

    Uses canonical ALL_VALID_STATUSES from utils/status_constants.py.
    """
    if pd.isna(loan_status) or loan_status is None:
        return DEFAULT_STATUS_SCORE
    status_str = str(loan_status).strip()
    return STATUS_SCORES.get(status_str, DEFAULT_STATUS_SCORE)


def get_industry_score(sector_code: Optional[str]) -> float:
    """
    Get recovery score based on NAICS sector code (2-digit).

    Applies consolidate_sector_code() to handle manufacturing subsectors (31, 32, 33).
    """
    if pd.isna(sector_code) or sector_code is None:
        return DEFAULT_INDUSTRY_SCORE
    # Normalize to 2-digit string and apply consolidation
    sector_str = str(sector_code).strip()[:2].zfill(2)
    consolidated = consolidate_sector_code(sector_str)
    return INDUSTRY_SCORES.get(consolidated, DEFAULT_INDUSTRY_SCORE)


def get_collateral_score(collateral_type: Optional[str]) -> float:
    """
    Get recovery score based on collateral type.

    NOTE: collateral_type field does not currently exist in the schema.
    This function is for future use or manual input scenarios.
    """
    if pd.isna(collateral_type) or collateral_type is None:
        return DEFAULT_COLLATERAL_SCORE
    return COLLATERAL_SCORES.get(str(collateral_type).strip().lower(), DEFAULT_COLLATERAL_SCORE)


def get_lien_score(ahead_positions: Optional[int]) -> float:
    """
    Get recovery score based on lien position (ahead_positions field).

    Uses existing field from deals table:
    - 0 = 1st lien (best, senior position)
    - 1 = 2nd lien (subordinate)
    - 2+ = 3rd+ lien (junior/unsecured)
    """
    if pd.isna(ahead_positions) or ahead_positions is None:
        return DEFAULT_LIEN_SCORE
    position = int(ahead_positions)
    if position <= 0:
        return LIEN_SCORES[0]  # 1st lien
    elif position == 1:
        return LIEN_SCORES[1]  # 2nd lien
    else:
        return LIEN_SCORES[2]  # 3rd+ lien


def get_communication_score(communication_status: Optional[str]) -> float:
    """
    Get recovery score based on communication status.

    NOTE: communication_status field does not currently exist in the schema.
    This function is for future use or manual input scenarios.
    """
    if pd.isna(communication_status) or communication_status is None:
        return DEFAULT_COMMUNICATION_SCORE
    return COMMUNICATION_SCORES.get(str(communication_status).strip().lower(), DEFAULT_COMMUNICATION_SCORE)


def get_recovery_band(score: float) -> Tuple[str, float, float, float]:
    """
    Get recovery band details for a given recovery score.

    Returns:
        Tuple of (band_label, recovery_low, recovery_high, midpoint)
    """
    for min_score, max_score, rec_low, rec_high in RECOVERY_BANDS:
        if min_score <= score < max_score or (score == 10.0 and max_score == 10.0):
            midpoint = (rec_low + rec_high) / 2
            band_label = f"{min_score:.1f} - {max_score:.1f}"
            return band_label, rec_low, rec_high, midpoint

    # Fallback for edge cases
    return "0.0 - 1.0", 0.0, 0.09, 0.045


def calculate_recovery_score(
    loan_status: Optional[str] = None,
    sector_code: Optional[str] = None,
    collateral_type: Optional[str] = "none",
    ahead_positions: Optional[int] = 2,
    communication_status: Optional[str] = "sporadic",
    exposure_base: float = 0.0,
) -> RecoveryScoreResult:
    """
    Calculate recovery score and bad debt expense for a deal.

    Args:
        loan_status: Current loan status (must be in ALL_VALID_STATUSES)
        sector_code: 2-digit NAICS sector code or full NAICS code
        collateral_type: Type of collateral (see COLLATERAL_SCORES keys)
        ahead_positions: Lien position (0=1st, 1=2nd, 2+=junior)
        communication_status: Borrower communication level (see COMMUNICATION_SCORES keys)
        exposure_base: Dollar amount at risk (e.g., net_balance, total_invested)

    Returns:
        RecoveryScoreResult with all scoring components and bad debt calculation
    """
    # Calculate component scores
    status_score = get_status_score(loan_status)
    industry_score = get_industry_score(sector_code)
    collateral_score = get_collateral_score(collateral_type)
    lien_score = get_lien_score(ahead_positions)
    communication_score = get_communication_score(communication_status)

    # Calculate weighted total
    total_score = (
        WEIGHTS["status"] * status_score +
        WEIGHTS["industry"] * industry_score +
        WEIGHTS["collateral"] * collateral_score +
        WEIGHTS["lien"] * lien_score +
        WEIGHTS["communication"] * communication_score
    )

    # Clamp to 0-10 range
    total_score = max(0.0, min(10.0, total_score))

    # Get recovery band
    band_label, rec_low, rec_high, rec_midpoint = get_recovery_band(total_score)

    # Calculate loss percentage
    loss_pct = 1.0 - rec_midpoint

    # Calculate bad debt expense
    exposure = max(0.0, float(exposure_base)) if not pd.isna(exposure_base) else 0.0
    bad_debt_expense = exposure * loss_pct
    bad_debt_low = exposure * (1.0 - rec_high)
    bad_debt_high = exposure * (1.0 - rec_low)

    # Determine problem/terminal status using canonical helpers
    is_problem = is_problem_status(loan_status) if loan_status else False
    is_terminal = is_terminal_status(loan_status) if loan_status else False

    return RecoveryScoreResult(
        status_score=round(status_score, 2),
        industry_score=round(industry_score, 2),
        collateral_score=round(collateral_score, 2),
        lien_score=round(lien_score, 2),
        communication_score=round(communication_score, 2),
        total_recovery_score=round(total_score, 2),
        recovery_band=band_label,
        recovery_pct_low=round(rec_low, 4),
        recovery_pct_high=round(rec_high, 4),
        recovery_pct_midpoint=round(rec_midpoint, 4),
        loss_pct=round(loss_pct, 4),
        exposure_base=round(exposure, 2),
        bad_debt_expense=round(bad_debt_expense, 2),
        bad_debt_low=round(bad_debt_low, 2),
        bad_debt_high=round(bad_debt_high, 2),
        is_problem_loan=is_problem,
        is_terminal=is_terminal,
    )


def calculate_bad_debt_expense(
    loan_status: Optional[str] = None,
    sector_code: Optional[str] = None,
    exposure_base: float = 0.0,
    collateral_type: Optional[str] = "none",
    ahead_positions: Optional[int] = 2,
    communication_status: Optional[str] = "sporadic",
) -> float:
    """
    Convenience function to calculate just the bad debt expense.

    Returns:
        Bad debt expense (midpoint estimate)
    """
    result = calculate_recovery_score(
        loan_status=loan_status,
        sector_code=sector_code,
        collateral_type=collateral_type,
        ahead_positions=ahead_positions,
        communication_status=communication_status,
        exposure_base=exposure_base,
    )
    return result.bad_debt_expense


def score_portfolio(
    df: pd.DataFrame,
    status_col: str = "loan_status",
    sector_col: str = "sector_code",
    exposure_col: str = "net_balance",
    collateral_col: Optional[str] = None,
    lien_col: str = "ahead_positions",
    communication_col: Optional[str] = None,
) -> pd.DataFrame:
    """
    Score an entire portfolio DataFrame and add recovery/bad debt columns.

    Args:
        df: DataFrame with loan data
        status_col: Column name for loan status (default: loan_status)
        sector_col: Column name for sector code (will derive from 'industry' if needed)
        exposure_col: Column name for exposure base (default: net_balance)
        collateral_col: Column name for collateral type (optional - field doesn't exist in schema)
        lien_col: Column name for lien position (default: ahead_positions)
        communication_col: Column name for communication status (optional - field doesn't exist in schema)

    Returns:
        DataFrame with added columns:
        - recovery_score: Total weighted recovery score (0-10)
        - recovery_pct: Recovery percentage midpoint
        - loss_pct: Loss percentage (1 - recovery_pct)
        - bad_debt_expense: Estimated bad debt expense
        - bad_debt_low: Low estimate of bad debt
        - bad_debt_high: High estimate of bad debt
        - is_problem_loan: Boolean flag from canonical PROBLEM_STATUSES
        - is_terminal: Boolean flag from canonical TERMINAL_STATUSES
    """
    if df.empty:
        return df

    result_df = df.copy()

    # Derive sector_code from industry if needed, using consolidate_sector_code
    if sector_col not in result_df.columns and "industry" in result_df.columns:
        result_df["sector_code"] = (
            result_df["industry"]
            .astype(str)
            .str[:2]
            .str.zfill(2)
            .apply(consolidate_sector_code)
        )
        sector_col = "sector_code"

    # Initialize result columns
    scores = []

    for idx, row in result_df.iterrows():
        # Extract values with defaults
        status = row.get(status_col) if status_col in result_df.columns else None
        sector = row.get(sector_col) if sector_col in result_df.columns else None
        exposure = row.get(exposure_col, 0) if exposure_col in result_df.columns else 0

        # Optional fields (these don't exist in current schema)
        collateral = row.get(collateral_col) if collateral_col and collateral_col in result_df.columns else "none"
        lien = row.get(lien_col) if lien_col in result_df.columns else 2
        communication = row.get(communication_col) if communication_col and communication_col in result_df.columns else "sporadic"

        # Calculate score
        score_result = calculate_recovery_score(
            loan_status=status,
            sector_code=sector,
            collateral_type=collateral,
            ahead_positions=lien,
            communication_status=communication,
            exposure_base=exposure,
        )
        scores.append(score_result.to_dict())

    # Add all score columns to DataFrame
    score_df = pd.DataFrame(scores)

    # Rename to avoid conflicts and add prefix for clarity
    rename_map = {
        "total_recovery_score": "recovery_score",
        "recovery_pct_midpoint": "recovery_pct",
    }
    score_df = score_df.rename(columns=rename_map)

    # Select columns to add
    cols_to_add = [
        "recovery_score",
        "recovery_pct",
        "loss_pct",
        "bad_debt_expense",
        "bad_debt_low",
        "bad_debt_high",
        "status_score",
        "industry_score",
        "collateral_score",
        "lien_score",
        "communication_score",
        "is_problem_loan",
        "is_terminal",
    ]

    for col in cols_to_add:
        if col in score_df.columns:
            result_df[col] = score_df[col].values

    return result_df


def get_portfolio_summary(df: pd.DataFrame) -> Dict:
    """
    Generate summary statistics for a scored portfolio.

    Args:
        df: DataFrame that has been scored with score_portfolio()

    Returns:
        Dictionary with portfolio-level bad debt metrics
    """
    if df.empty or "bad_debt_expense" not in df.columns:
        return {
            "total_exposure": 0,
            "total_bad_debt_expense": 0,
            "total_bad_debt_low": 0,
            "total_bad_debt_high": 0,
            "avg_recovery_score": 0,
            "avg_recovery_pct": 0,
            "avg_loss_pct": 0,
            "deal_count": 0,
            "problem_loan_count": 0,
            "terminal_loan_count": 0,
        }

    return {
        "total_exposure": df["exposure_base"].sum() if "exposure_base" in df.columns else df.get("net_balance", pd.Series([0])).sum(),
        "total_bad_debt_expense": df["bad_debt_expense"].sum(),
        "total_bad_debt_low": df["bad_debt_low"].sum(),
        "total_bad_debt_high": df["bad_debt_high"].sum(),
        "avg_recovery_score": df["recovery_score"].mean(),
        "avg_recovery_pct": df["recovery_pct"].mean(),
        "avg_loss_pct": df["loss_pct"].mean(),
        "deal_count": len(df),
        "problem_loan_count": df["is_problem_loan"].sum() if "is_problem_loan" in df.columns else 0,
        "terminal_loan_count": df["is_terminal"].sum() if "is_terminal" in df.columns else 0,
    }


# =============================================================================
# DISPLAY HELPERS
# =============================================================================

def get_recovery_color(score: float) -> str:
    """Get a color for recovery score visualization (green=good, red=bad)."""
    if score >= 8:
        return "#22c55e"  # Green
    elif score >= 6:
        return "#84cc16"  # Lime
    elif score >= 4:
        return "#eab308"  # Yellow
    elif score >= 2:
        return "#f97316"  # Orange
    else:
        return "#ef4444"  # Red


def format_currency(value: float) -> str:
    """Format a number as currency."""
    if pd.isna(value):
        return "$0"
    return f"${value:,.0f}"


def format_percentage(value: float) -> str:
    """Format a decimal as percentage."""
    if pd.isna(value):
        return "0%"
    return f"{value:.1%}"


def get_status_score_table() -> pd.DataFrame:
    """Return STATUS_SCORES as a DataFrame for display."""
    return pd.DataFrame([
        {"Status": status, "Score": score, "Is Problem": status in PROBLEM_STATUSES}
        for status, score in sorted(STATUS_SCORES.items(), key=lambda x: x[1])
    ])


def get_industry_score_table() -> pd.DataFrame:
    """Return INDUSTRY_SCORES as a DataFrame with sector names for display."""
    return pd.DataFrame([
        {
            "Sector Code": code,
            "Sector Name": NAICS_SECTOR_NAMES.get(code, f"Unknown ({code})"),
            "Score": score
        }
        for code, score in sorted(INDUSTRY_SCORES.items(), key=lambda x: x[1])
    ])
