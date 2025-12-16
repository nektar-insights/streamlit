# utils/recovery_scoring.py
"""
Recovery / Pre-Screen Scoring Engine

A unified scoring framework for:
1. PRE-SCREEN TOOL: Manual inputs for prospective deals (forward-looking)
2. BAD DEBT ESTIMATOR: Data-driven scoring for existing deals in our portfolio

The same scoring framework and weights are used in both contexts; the difference
is only where the inputs come from (manual vs database).

Scoring Components (0-10 scale):
- Status Score (10% weight): Based on loan status severity
- Industry Score (30% weight): Based on NAICS sector risk
- Collateral Score (30% weight): Based on collateral type/quality
- Lien Score (20% weight): Based on lien position (ahead_positions)
- Communication Score (10% weight): Based on borrower responsiveness

Total Recovery Score = weighted sum of all components
Recovery % is mapped from Total Score via bands
Loss % = 1 - Recovery %
Bad Debt Expense = Exposure Base * Loss %
"""

from dataclasses import dataclass
from typing import Dict, Optional, Tuple, List, Union
import pandas as pd
import numpy as np

# Import existing constants from the codebase
from utils.status_constants import ALL_VALID_STATUSES, TERMINAL_STATUSES
from utils.loan_tape_data import consolidate_sector_code


# =============================================================================
# SCORING CONSTANTS - Business Rules (Fixed)
# =============================================================================

# Weight configuration for the total recovery score
WEIGHTS = {
    "status": 0.10,
    "industry": 0.30,
    "collateral": 0.30,
    "lien": 0.20,
    "communication": 0.10,
}

# =============================================================================
# STATUS SCORE MAPPING (0-10)
# Maps ALL_VALID_STATUSES to score based on severity
# Higher score = better recovery prospects
# =============================================================================

STATUS_SCORE_MAP: Dict[str, float] = {
    # Terminal - complete loss
    "Bankruptcy": 1.0,
    # Non-performing
    "Non-Performing": 2.0,
    "Charged Off": 2.0,
    # Severe issues (>90 days)
    "Severe Delinquency": 3.0,
    "Default": 3.0,
    # Legal involvement
    "Legal Action": 5.0,
    "In Collections": 5.0,
    # Moderate issues (30-90 days)
    "Moderate Delinquency": 6.0,
    "NSF / Suspended": 6.0,  # Default to moderate unless we have days info
    # Minor issues (<30 days)
    "Minor Delinquency": 9.0,
    "Active - Frequently Late": 9.0,
    "Past Delinquency": 9.0,
    # Healthy
    "Active": 10.0,
    "Paid Off": 10.0,  # Special handling: loss_pct = 0 regardless
}

# Legacy status names that may appear in older data
# Maps legacy name -> canonical ALL_VALID_STATUSES name
LEGACY_STATUS_MAPPING: Dict[str, str] = {
    "Bankrupt": "Bankruptcy",
    "bankrupt": "Bankruptcy",
    "BANKRUPTCY": "Bankruptcy",
    "Late": "Minor Delinquency",  # Generic late -> Minor
    "Severe": "Severe Delinquency",
    "NSF": "NSF / Suspended",
    "Suspended": "NSF / Suspended",
    "Collections": "In Collections",
    "Legal": "Legal Action",
    "Defaulted": "Default",
    "ChargedOff": "Charged Off",
    "Charged-Off": "Charged Off",
    "PaidOff": "Paid Off",
    "Paid-Off": "Paid Off",
    "paid off": "Paid Off",
}

# Pre-screen status categories (user-facing labels)
STATUS_CATEGORIES = [
    "Bankruptcy",
    "Non-Performing",
    "NSF / Suspended >90 days",
    "Legal",
    "NSF / Suspended 30-90 days",
    "NSF / Suspended <30 days",
    "Active (Healthy)",
]

# Map pre-screen category labels to scores
PRESCREEN_STATUS_SCORE_MAP: Dict[str, float] = {
    "Bankruptcy": 1.0,
    "Non-Performing": 2.0,
    "NSF / Suspended >90 days": 3.0,
    "Legal": 5.0,
    "NSF / Suspended 30-90 days": 6.0,
    "NSF / Suspended <30 days": 9.0,
    "Active (Healthy)": 10.0,
}

# =============================================================================
# INDUSTRY SCORE MAPPING (1-10)
# Based on NAICS 2-digit sector codes
# Lower score = higher risk industries
# =============================================================================

# Industry buckets for pre-screen tool (user-facing labels)
INDUSTRY_CATEGORIES = [
    "Construction / Trades",
    "Transportation / Logistics",
    "Retail / E-commerce",
    "Food Service / Restaurants",
    "Manufacturing",
    "Healthcare Services",
    "Business / Professional Services",
    "SaaS / Tech Services",
    "Energy / Utilities",
    "Government / Education Contractors",
]

# Map industry categories to scores
PRESCREEN_INDUSTRY_SCORE_MAP: Dict[str, float] = {
    "Construction / Trades": 1.0,
    "Transportation / Logistics": 2.0,
    "Retail / E-commerce": 3.0,
    "Food Service / Restaurants": 4.0,
    "Manufacturing": 5.0,
    "Healthcare Services": 6.0,
    "Business / Professional Services": 7.0,
    "SaaS / Tech Services": 8.0,
    "Energy / Utilities": 9.0,
    "Government / Education Contractors": 10.0,
}

# Map NAICS 2-digit sector codes to industry categories
# Uses consolidated codes (32, 33 -> 31 for Manufacturing)
NAICS_TO_INDUSTRY_CATEGORY: Dict[str, str] = {
    "23": "Construction / Trades",
    "48": "Transportation / Logistics",
    "49": "Transportation / Logistics",  # Warehousing
    "44": "Retail / E-commerce",
    "45": "Retail / E-commerce",
    "72": "Food Service / Restaurants",
    "31": "Manufacturing",  # Consolidated (31, 32, 33)
    "62": "Healthcare Services",
    "54": "Business / Professional Services",
    "56": "Business / Professional Services",  # Admin & Waste
    "51": "SaaS / Tech Services",
    "22": "Energy / Utilities",
    "92": "Government / Education Contractors",
    "61": "Government / Education Contractors",
    # Additional sectors - map to closest category based on risk
    "11": "Construction / Trades",  # Agriculture - similar risk to construction
    "21": "Energy / Utilities",  # Mining/Oil/Gas
    "42": "Retail / E-commerce",  # Wholesale Trade
    "52": "Business / Professional Services",  # Finance/Insurance
    "53": "Business / Professional Services",  # Real Estate
    "55": "Business / Professional Services",  # Management
    "71": "Food Service / Restaurants",  # Arts/Entertainment
    "81": "Business / Professional Services",  # Other Services
}

# Direct NAICS sector code to score (for batch processing)
NAICS_SECTOR_SCORE_MAP: Dict[str, float] = {
    code: PRESCREEN_INDUSTRY_SCORE_MAP.get(category, 5.0)
    for code, category in NAICS_TO_INDUSTRY_CATEGORY.items()
}

# =============================================================================
# COLLATERAL SCORE MAPPING (1-10)
# Higher score = better collateral quality
# =============================================================================

COLLATERAL_CATEGORIES = [
    "None / Unsecured",
    "Intangible (IP, Licenses, Brand Value)",
    "Inventory / Consumer Goods",
    "Accounts Receivable (Unverified)",
    "Accounts Receivable (Verified / Insured)",
    "Equipment (Specialized)",
    "Equipment (Generic / Liquid)",
    "Property (Leased / Encumbered)",
    "Lockbox without control agreement",
    "Property (Owned Outright)",
    "Lockbox with full DACA / control",
    "Cash / Marketable Securities / Deposit Pledge",
]

COLLATERAL_SCORE_MAP: Dict[str, float] = {
    "None / Unsecured": 1.0,
    "Intangible (IP, Licenses, Brand Value)": 2.0,
    "Inventory / Consumer Goods": 3.0,
    "Accounts Receivable (Unverified)": 4.0,
    "Accounts Receivable (Verified / Insured)": 5.0,
    "Equipment (Specialized)": 6.0,
    "Equipment (Generic / Liquid)": 7.0,
    "Property (Leased / Encumbered)": 8.0,
    "Lockbox without control agreement": 8.0,
    "Property (Owned Outright)": 9.0,
    "Lockbox with full DACA / control": 10.0,
    "Cash / Marketable Securities / Deposit Pledge": 10.0,
}

# Default collateral score when data is not available
COLLATERAL_SCORE_DEFAULT = 1.0  # Conservative: assume unsecured

# HubSpot field name variations for collateral
# HubSpot may use different property names for the same data
COLLATERAL_FIELD_NAMES = ["collateral_type", "collateral", "collateral_score", "collateral_category"]

# =============================================================================
# LIEN POSITION SCORE MAPPING (0-10)
# Based on ahead_positions field (0 = 1st lien, 1 = 2nd, 2+ = junior)
# =============================================================================

LIEN_CATEGORIES = [
    "First Lien",
    "Second Lien",
    "Junior or Unsecured",
]

LIEN_SCORE_MAP: Dict[str, float] = {
    "First Lien": 10.0,
    "Second Lien": 5.0,
    "Junior or Unsecured": 0.0,
}

# Default lien score when data is not available
LIEN_SCORE_DEFAULT = 0.0  # Conservative: assume junior/unsecured

# =============================================================================
# COMMUNICATION SCORE MAPPING (1-10)
# Based on borrower responsiveness
# =============================================================================

COMMUNICATION_CATEGORIES = [
    "No contact / hostile",
    "Sporadic / only under pressure",
    "Generally responsive but slow",
    "Fully engaged and proactive",
    "On a plan and hitting milestones",
]

COMMUNICATION_SCORE_MAP: Dict[str, float] = {
    "No contact / hostile": 1.0,
    "Sporadic / only under pressure": 3.0,
    "Generally responsive but slow": 6.0,
    "Fully engaged and proactive": 9.0,
    "On a plan and hitting milestones": 10.0,
}

# Default communication score when data is not available
COMMUNICATION_SCORE_DEFAULT = 3.0  # Conservative: assume sporadic

# HubSpot field name variations for communication
# HubSpot may use different property names for the same data
COMMUNICATION_FIELD_NAMES = ["communication_status", "communication", "communication_score", "borrower_communication"]

# =============================================================================
# RECOVERY BAND MAPPING
# Maps total recovery score to recovery percentage bands
# =============================================================================

RECOVERY_BANDS: List[Tuple[Tuple[float, float], str, float, float, float]] = [
    # (score_range, label, midpoint, low, high)
    ((9.0, 10.01), "90-100%", 0.95, 0.90, 1.00),
    ((7.0, 9.0), "70-89%", 0.795, 0.70, 0.89),
    ((5.0, 7.0), "50-69%", 0.595, 0.50, 0.69),
    ((3.0, 5.0), "30-49%", 0.395, 0.30, 0.49),
    ((1.0, 3.0), "10-29%", 0.195, 0.10, 0.29),
    ((0.0, 1.0), "0-9%", 0.045, 0.00, 0.09),
]


# =============================================================================
# RESULT DATACLASS
# =============================================================================

@dataclass
class RecoveryScoreResult:
    """Container for all recovery score computation results."""

    # Individual component scores (0-10)
    status_score: float
    industry_score: float
    collateral_score: float
    lien_score: float
    communication_score: float

    # Weighted total recovery score (0-10)
    total_recovery_score: float

    # Recovery band information
    recovery_band_label: str
    recovery_pct_midpoint: float
    recovery_pct_low: float
    recovery_pct_high: float

    # Loss percentage (1 - recovery_pct)
    loss_pct_midpoint: float

    # Exposure and bad debt estimates
    exposure_base: float
    estimated_bad_debt_expense_mid: float
    estimated_bad_debt_expense_low: float
    estimated_bad_debt_expense_high: float

    # Metadata
    is_paid_off: bool = False
    data_source: str = "unknown"  # "prescreen" or "database"

    # Track whether defaults were used (for transparency)
    collateral_used_default: bool = False
    communication_used_default: bool = False

    def to_dict(self) -> Dict:
        """Convert to dictionary for DataFrame construction."""
        return {
            "status_score": self.status_score,
            "industry_score": self.industry_score,
            "collateral_score": self.collateral_score,
            "lien_score": self.lien_score,
            "communication_score": self.communication_score,
            "total_recovery_score": self.total_recovery_score,
            "recovery_band_label": self.recovery_band_label,
            "recovery_pct_midpoint": self.recovery_pct_midpoint,
            "recovery_pct_low": self.recovery_pct_low,
            "recovery_pct_high": self.recovery_pct_high,
            "loss_pct_midpoint": self.loss_pct_midpoint,
            "exposure_base": self.exposure_base,
            "estimated_bad_debt_expense_mid": self.estimated_bad_debt_expense_mid,
            "estimated_bad_debt_expense_low": self.estimated_bad_debt_expense_low,
            "estimated_bad_debt_expense_high": self.estimated_bad_debt_expense_high,
            "is_paid_off": self.is_paid_off,
            "data_source": self.data_source,
            "collateral_used_default": self.collateral_used_default,
            "communication_used_default": self.communication_used_default,
        }


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def normalize_status(status: str) -> str:
    """
    Normalize a status value to a canonical ALL_VALID_STATUSES name.

    Args:
        status: Raw status string

    Returns:
        Canonical status name from ALL_VALID_STATUSES, or original if not found
    """
    if pd.isna(status) or status is None:
        return "Active"  # Default assumption

    status_str = str(status).strip()

    # Check if already canonical
    if status_str in ALL_VALID_STATUSES:
        return status_str

    # Check legacy mapping
    if status_str in LEGACY_STATUS_MAPPING:
        return LEGACY_STATUS_MAPPING[status_str]

    # Case-insensitive search in canonical list
    status_lower = status_str.lower()
    for canonical in ALL_VALID_STATUSES:
        if canonical.lower() == status_lower:
            return canonical

    # Default to Active if truly unknown
    return "Active"


def get_status_score(status: str) -> float:
    """
    Get the status score (0-10) for a given status.

    Args:
        status: Loan status (will be normalized)

    Returns:
        Status score (0-10)
    """
    normalized = normalize_status(status)
    return STATUS_SCORE_MAP.get(normalized, 5.0)  # Default to middle score


def get_industry_score_from_naics(sector_code: Union[str, int, None]) -> float:
    """
    Get the industry score (1-10) from a NAICS sector code.

    Args:
        sector_code: 2-digit NAICS sector code (may be full code, will extract first 2 digits)

    Returns:
        Industry score (1-10)
    """
    if pd.isna(sector_code) or sector_code is None:
        return 5.0  # Default to middle score

    # Extract first 2 digits and normalize
    code_str = str(sector_code).strip()[:2].zfill(2)

    # Apply consolidation (e.g., 32, 33 -> 31 for Manufacturing)
    consolidated = consolidate_sector_code(code_str)

    return NAICS_SECTOR_SCORE_MAP.get(consolidated, 5.0)


def get_lien_score_from_ahead_positions(ahead_positions: Union[int, float, None]) -> float:
    """
    Get the lien score (0-10) from ahead_positions value.

    Args:
        ahead_positions: Number of positions ahead (0=1st lien, 1=2nd, 2+=junior)

    Returns:
        Lien score (0-10)
    """
    if pd.isna(ahead_positions) or ahead_positions is None:
        return LIEN_SCORE_DEFAULT

    try:
        pos = int(ahead_positions)
        if pos == 0:
            return 10.0  # First lien
        elif pos == 1:
            return 5.0   # Second lien
        else:
            return 0.0   # Junior or unsecured
    except (ValueError, TypeError):
        return LIEN_SCORE_DEFAULT


def get_collateral_score_from_deal(deal_data: Dict) -> Tuple[float, bool]:
    """
    Get the collateral score from deal data, checking multiple possible field names.

    Handles both category labels (e.g., "Accounts Receivable (Verified / Insured)")
    and numeric scores (1-10).

    Args:
        deal_data: Dictionary with deal fields

    Returns:
        Tuple of (score, used_default) where used_default is True if we fell back to default
    """
    # Try each possible field name
    for field_name in COLLATERAL_FIELD_NAMES:
        value = deal_data.get(field_name)

        if pd.isna(value) or value is None or value == "":
            continue

        # Check if it's a category label
        if isinstance(value, str):
            value_str = str(value).strip()

            # Direct match in score map
            if value_str in COLLATERAL_SCORE_MAP:
                return COLLATERAL_SCORE_MAP[value_str], False

            # Case-insensitive search
            value_lower = value_str.lower()
            for category, score in COLLATERAL_SCORE_MAP.items():
                if category.lower() == value_lower:
                    return score, False

            # Partial match (for flexible HubSpot labels)
            for category, score in COLLATERAL_SCORE_MAP.items():
                if value_lower in category.lower() or category.lower() in value_lower:
                    return score, False

        # Check if it's a numeric score (1-10)
        try:
            numeric_value = float(value)
            if 1.0 <= numeric_value <= 10.0:
                return numeric_value, False
        except (ValueError, TypeError):
            pass

    # No valid value found, use default
    return COLLATERAL_SCORE_DEFAULT, True


def get_communication_score_from_deal(deal_data: Dict) -> Tuple[float, bool]:
    """
    Get the communication score from deal data, checking multiple possible field names.

    Handles both category labels (e.g., "Fully engaged and proactive")
    and numeric scores (1-10).

    Args:
        deal_data: Dictionary with deal fields

    Returns:
        Tuple of (score, used_default) where used_default is True if we fell back to default
    """
    # Try each possible field name
    for field_name in COMMUNICATION_FIELD_NAMES:
        value = deal_data.get(field_name)

        if pd.isna(value) or value is None or value == "":
            continue

        # Check if it's a category label
        if isinstance(value, str):
            value_str = str(value).strip()

            # Direct match in score map
            if value_str in COMMUNICATION_SCORE_MAP:
                return COMMUNICATION_SCORE_MAP[value_str], False

            # Case-insensitive search
            value_lower = value_str.lower()
            for category, score in COMMUNICATION_SCORE_MAP.items():
                if category.lower() == value_lower:
                    return score, False

            # Partial match (for flexible HubSpot labels)
            for category, score in COMMUNICATION_SCORE_MAP.items():
                if value_lower in category.lower() or category.lower() in value_lower:
                    return score, False

        # Check if it's a numeric score (1-10)
        try:
            numeric_value = float(value)
            if 1.0 <= numeric_value <= 10.0:
                return numeric_value, False
        except (ValueError, TypeError):
            pass

    # No valid value found, use default
    return COMMUNICATION_SCORE_DEFAULT, True


def get_recovery_band(total_score: float) -> Tuple[str, float, float, float]:
    """
    Map total recovery score to recovery band.

    Args:
        total_score: Weighted total recovery score (0-10)

    Returns:
        Tuple of (band_label, midpoint, low, high)
    """
    for (low, high), label, midpoint, pct_low, pct_high in RECOVERY_BANDS:
        if low <= total_score < high:
            return label, midpoint, pct_low, pct_high

    # Default to worst band
    return "0-9%", 0.045, 0.00, 0.09


def compute_total_recovery_score(
    status_score: float,
    industry_score: float,
    collateral_score: float,
    lien_score: float,
    communication_score: float
) -> float:
    """
    Compute the weighted total recovery score.

    Args:
        status_score: Status score (0-10)
        industry_score: Industry score (1-10)
        collateral_score: Collateral score (1-10)
        lien_score: Lien score (0-10)
        communication_score: Communication score (1-10)

    Returns:
        Weighted total score (0-10)
    """
    return (
        WEIGHTS["status"] * status_score +
        WEIGHTS["industry"] * industry_score +
        WEIGHTS["collateral"] * collateral_score +
        WEIGHTS["lien"] * lien_score +
        WEIGHTS["communication"] * communication_score
    )


# =============================================================================
# MAIN SCORING FUNCTIONS
# =============================================================================

def compute_recovery_prescreen(
    exposure_amount: float,
    status_category: str,
    industry_category: str,
    collateral_type: str,
    lien_position: str,
) -> RecoveryScoreResult:
    """
    Compute recovery score for a PRE-SCREEN scenario (manual inputs).

    This is used for forward-looking assessment of prospective deals
    that are NOT yet in our database.

    Note: Communication is excluded from pre-screen scoring since borrower
    responsiveness is unknown at this stage. The 10% weight is redistributed
    proportionally to the other 4 components.

    Args:
        exposure_amount: Principal/exposure amount to assess
        status_category: One of STATUS_CATEGORIES
        industry_category: One of INDUSTRY_CATEGORIES
        collateral_type: One of COLLATERAL_CATEGORIES
        lien_position: One of LIEN_CATEGORIES

    Returns:
        RecoveryScoreResult with all computed values
    """
    # Get individual scores from user-facing categories
    status_score = PRESCREEN_STATUS_SCORE_MAP.get(status_category, 5.0)
    industry_score = PRESCREEN_INDUSTRY_SCORE_MAP.get(industry_category, 5.0)
    collateral_score = COLLATERAL_SCORE_MAP.get(collateral_type, COLLATERAL_SCORE_DEFAULT)
    lien_score = LIEN_SCORE_MAP.get(lien_position, LIEN_SCORE_DEFAULT)

    # Pre-screen weights: redistribute communication's 10% to other components
    # Original: status=10%, industry=30%, collateral=30%, lien=20%, communication=10%
    # Pre-screen: scale the 4 components to sum to 100% (divide by 0.90)
    prescreen_weights = {
        "status": WEIGHTS["status"] / 0.90,      # 10% -> 11.1%
        "industry": WEIGHTS["industry"] / 0.90,  # 30% -> 33.3%
        "collateral": WEIGHTS["collateral"] / 0.90,  # 30% -> 33.3%
        "lien": WEIGHTS["lien"] / 0.90,          # 20% -> 22.2%
    }

    # Compute total weighted score (without communication)
    total_score = (
        prescreen_weights["status"] * status_score +
        prescreen_weights["industry"] * industry_score +
        prescreen_weights["collateral"] * collateral_score +
        prescreen_weights["lien"] * lien_score
    )

    # Get recovery band
    band_label, recovery_mid, recovery_low, recovery_high = get_recovery_band(total_score)

    # Calculate loss and bad debt
    loss_mid = 1.0 - recovery_mid

    # Clamp exposure base
    exposure_base = max(0.0, exposure_amount)

    # Calculate bad debt estimates
    bad_debt_mid = exposure_base * loss_mid
    bad_debt_low = exposure_base * (1.0 - recovery_high)  # Low bad debt when recovery is high
    bad_debt_high = exposure_base * (1.0 - recovery_low)  # High bad debt when recovery is low

    return RecoveryScoreResult(
        status_score=status_score,
        industry_score=industry_score,
        collateral_score=collateral_score,
        lien_score=lien_score,
        communication_score=0.0,  # Not used in pre-screen
        total_recovery_score=total_score,
        recovery_band_label=band_label,
        recovery_pct_midpoint=recovery_mid,
        recovery_pct_low=recovery_low,
        recovery_pct_high=recovery_high,
        loss_pct_midpoint=loss_mid,
        exposure_base=exposure_base,
        estimated_bad_debt_expense_mid=bad_debt_mid,
        estimated_bad_debt_expense_low=bad_debt_low,
        estimated_bad_debt_expense_high=bad_debt_high,
        is_paid_off=False,
        data_source="prescreen",
    )


def compute_recovery_for_deal(deal_data: Dict) -> RecoveryScoreResult:
    """
    Compute recovery score for an EXISTING DEAL from the database.

    Uses existing fields from the loan tape / deals pipeline and applies
    conservative defaults where data is missing (collateral, communication).

    Args:
        deal_data: Dictionary with deal/loan fields:
            - loan_status: Loan status (from ALL_VALID_STATUSES)
            - industry or sector_code: NAICS code (full or 2-digit)
            - ahead_positions: Lien position (0=1st, 1=2nd, 2+=junior)
            - net_balance: Outstanding balance (exposure base)
            - collateral_type/collateral/collateral_score (optional): Collateral info from HubSpot
            - communication_status/communication/communication_score (optional): Communication info from HubSpot

    Returns:
        RecoveryScoreResult with all computed values
    """
    # Extract and normalize loan status
    loan_status = deal_data.get("loan_status", "Active")
    normalized_status = normalize_status(loan_status)

    # Check for Paid Off - special case with 0% loss
    is_paid_off = normalized_status == "Paid Off"

    # Get status score
    status_score = get_status_score(normalized_status)

    # Get industry score from NAICS
    sector_code = deal_data.get("sector_code") or deal_data.get("industry")
    industry_score = get_industry_score_from_naics(sector_code)

    # Get collateral score using enhanced helper (checks multiple field names)
    collateral_score, collateral_used_default = get_collateral_score_from_deal(deal_data)

    # Get lien score from ahead_positions
    ahead_positions = deal_data.get("ahead_positions")
    lien_score = get_lien_score_from_ahead_positions(ahead_positions)

    # Get communication score using enhanced helper (checks multiple field names)
    communication_score, communication_used_default = get_communication_score_from_deal(deal_data)

    # Compute total weighted score
    total_score = compute_total_recovery_score(
        status_score, industry_score, collateral_score, lien_score, communication_score
    )

    # Get recovery band
    band_label, recovery_mid, recovery_low, recovery_high = get_recovery_band(total_score)

    # Calculate loss percentage
    # EDGE CASE: Paid Off loans have 0% loss regardless of score
    if is_paid_off:
        loss_mid = 0.0
    else:
        loss_mid = 1.0 - recovery_mid

    # Get exposure base (net_balance)
    # EDGE CASE: Clamp to 0 if net_balance is negative
    net_balance = deal_data.get("net_balance", 0.0)
    if pd.isna(net_balance) or net_balance is None:
        exposure_base = 0.0
    else:
        exposure_base = max(0.0, float(net_balance))

    # Calculate bad debt estimates
    if is_paid_off:
        bad_debt_mid = 0.0
        bad_debt_low = 0.0
        bad_debt_high = 0.0
    else:
        bad_debt_mid = exposure_base * loss_mid
        bad_debt_low = exposure_base * (1.0 - recovery_high)
        bad_debt_high = exposure_base * (1.0 - recovery_low)

    return RecoveryScoreResult(
        status_score=status_score,
        industry_score=industry_score,
        collateral_score=collateral_score,
        lien_score=lien_score,
        communication_score=communication_score,
        total_recovery_score=total_score,
        recovery_band_label=band_label,
        recovery_pct_midpoint=recovery_mid,
        recovery_pct_low=recovery_low,
        recovery_pct_high=recovery_high,
        loss_pct_midpoint=loss_mid,
        exposure_base=exposure_base,
        estimated_bad_debt_expense_mid=bad_debt_mid,
        estimated_bad_debt_expense_low=bad_debt_low,
        estimated_bad_debt_expense_high=bad_debt_high,
        is_paid_off=is_paid_off,
        data_source="database",
        collateral_used_default=collateral_used_default,
        communication_used_default=communication_used_default,
    )


def compute_recovery_batch(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute recovery scores for a batch of existing deals (DataFrame).

    Applies compute_recovery_for_deal() to each row and returns the original
    DataFrame with recovery scoring columns added.

    Args:
        df: DataFrame with loan/deal data containing:
            - loan_status
            - industry or sector_code
            - ahead_positions
            - net_balance

    Returns:
        DataFrame with additional columns:
            - recovery_status_score
            - recovery_industry_score
            - recovery_collateral_score
            - recovery_lien_score
            - recovery_communication_score
            - recovery_total_score
            - recovery_band_label
            - recovery_pct_midpoint
            - recovery_loss_pct
            - recovery_bad_debt_mid
            - recovery_bad_debt_low
            - recovery_bad_debt_high
    """
    if df.empty:
        return df

    result_df = df.copy()

    # Process each row
    results = []
    for idx, row in df.iterrows():
        deal_data = row.to_dict()
        score_result = compute_recovery_for_deal(deal_data)
        results.append(score_result.to_dict())

    # Create results DataFrame and merge
    results_df = pd.DataFrame(results)

    # Rename columns to avoid conflicts
    column_rename = {
        "status_score": "recovery_status_score",
        "industry_score": "recovery_industry_score",
        "collateral_score": "recovery_collateral_score",
        "lien_score": "recovery_lien_score",
        "communication_score": "recovery_communication_score",
        "total_recovery_score": "recovery_total_score",
        "loss_pct_midpoint": "recovery_loss_pct",
        "estimated_bad_debt_expense_mid": "recovery_bad_debt_mid",
        "estimated_bad_debt_expense_low": "recovery_bad_debt_low",
        "estimated_bad_debt_expense_high": "recovery_bad_debt_high",
    }
    results_df = results_df.rename(columns=column_rename)

    # Reset index to ensure alignment
    results_df.index = result_df.index

    # Select columns to add
    cols_to_add = [
        "recovery_status_score",
        "recovery_industry_score",
        "recovery_collateral_score",
        "recovery_lien_score",
        "recovery_communication_score",
        "recovery_total_score",
        "recovery_band_label",
        "recovery_pct_midpoint",
        "recovery_pct_low",
        "recovery_pct_high",
        "recovery_loss_pct",
        "exposure_base",
        "recovery_bad_debt_mid",
        "recovery_bad_debt_low",
        "recovery_bad_debt_high",
        "is_paid_off",
        "collateral_used_default",
        "communication_used_default",
    ]

    for col in cols_to_add:
        result_df[col] = results_df[col]

    return result_df


def get_portfolio_bad_debt_summary(df: pd.DataFrame) -> Dict:
    """
    Calculate portfolio-level bad debt summary from batch results.

    Args:
        df: DataFrame that has been processed by compute_recovery_batch()

    Returns:
        Dictionary with portfolio summary:
            - total_exposure: Sum of all exposure_base
            - total_bad_debt_mid: Sum of recovery_bad_debt_mid
            - total_bad_debt_low: Sum of recovery_bad_debt_low
            - total_bad_debt_high: Sum of recovery_bad_debt_high
            - weighted_recovery_pct: Exposure-weighted average recovery %
            - weighted_loss_pct: Exposure-weighted average loss %
            - deals_by_band: Count of deals in each recovery band
            - collateral_data_stats: Stats on collateral data availability
            - communication_data_stats: Stats on communication data availability
    """
    if df.empty or "recovery_bad_debt_mid" not in df.columns:
        return {
            "total_exposure": 0.0,
            "total_bad_debt_mid": 0.0,
            "total_bad_debt_low": 0.0,
            "total_bad_debt_high": 0.0,
            "weighted_recovery_pct": 0.0,
            "weighted_loss_pct": 0.0,
            "deals_by_band": {},
            "collateral_data_stats": {"with_data": 0, "using_default": 0},
            "communication_data_stats": {"with_data": 0, "using_default": 0},
        }

    # Exclude paid-off deals from bad debt calculation
    active_df = df[~df["is_paid_off"]].copy() if "is_paid_off" in df.columns else df.copy()

    total_exposure = active_df["exposure_base"].sum()
    total_bad_debt_mid = active_df["recovery_bad_debt_mid"].sum()
    total_bad_debt_low = active_df["recovery_bad_debt_low"].sum()
    total_bad_debt_high = active_df["recovery_bad_debt_high"].sum()

    # Weighted averages
    if total_exposure > 0:
        weighted_recovery = (active_df["recovery_pct_midpoint"] * active_df["exposure_base"]).sum() / total_exposure
        weighted_loss = (active_df["recovery_loss_pct"] * active_df["exposure_base"]).sum() / total_exposure
    else:
        weighted_recovery = 0.0
        weighted_loss = 0.0

    # Deals by band
    deals_by_band = df["recovery_band_label"].value_counts().to_dict()

    # Calculate data availability statistics
    total_loans = len(df)

    # Collateral data stats
    if "collateral_used_default" in df.columns:
        collateral_using_default = df["collateral_used_default"].sum()
        collateral_with_data = total_loans - collateral_using_default
    else:
        collateral_using_default = total_loans
        collateral_with_data = 0

    # Communication data stats
    if "communication_used_default" in df.columns:
        communication_using_default = df["communication_used_default"].sum()
        communication_with_data = total_loans - communication_using_default
    else:
        communication_using_default = total_loans
        communication_with_data = 0

    return {
        "total_exposure": total_exposure,
        "total_bad_debt_mid": total_bad_debt_mid,
        "total_bad_debt_low": total_bad_debt_low,
        "total_bad_debt_high": total_bad_debt_high,
        "weighted_recovery_pct": weighted_recovery,
        "weighted_loss_pct": weighted_loss,
        "deals_by_band": deals_by_band,
        "collateral_data_stats": {
            "with_data": int(collateral_with_data),
            "using_default": int(collateral_using_default),
        },
        "communication_data_stats": {
            "with_data": int(communication_with_data),
            "using_default": int(communication_using_default),
        },
    }


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Result class
    "RecoveryScoreResult",
    # Main functions
    "compute_recovery_prescreen",
    "compute_recovery_for_deal",
    "compute_recovery_batch",
    "get_portfolio_bad_debt_summary",
    # Helper functions
    "normalize_status",
    "get_status_score",
    "get_industry_score_from_naics",
    "get_lien_score_from_ahead_positions",
    "get_collateral_score_from_deal",
    "get_communication_score_from_deal",
    "get_recovery_band",
    # Constants for UI
    "STATUS_CATEGORIES",
    "INDUSTRY_CATEGORIES",
    "COLLATERAL_CATEGORIES",
    "LIEN_CATEGORIES",
    "COMMUNICATION_CATEGORIES",
    "WEIGHTS",
    # Field name constants (for debugging/reference)
    "COLLATERAL_FIELD_NAMES",
    "COMMUNICATION_FIELD_NAMES",
]


# =============================================================================
# MAIN - Example Usage
# =============================================================================

if __name__ == "__main__":
    # Example 1: Pre-screen scenario
    print("=" * 60)
    print("EXAMPLE 1: Pre-Screen Scenario")
    print("=" * 60)

    result = compute_recovery_prescreen(
        exposure_amount=100000.0,
        status_category="NSF / Suspended 30-90 days",
        industry_category="Construction / Trades",
        collateral_type="Accounts Receivable (Unverified)",
        lien_position="Second Lien",
        communication_status="Sporadic / only under pressure",
    )

    print(f"Exposure: ${result.exposure_base:,.2f}")
    print(f"\nComponent Scores (0-10):")
    print(f"  Status Score:        {result.status_score:.1f}")
    print(f"  Industry Score:      {result.industry_score:.1f}")
    print(f"  Collateral Score:    {result.collateral_score:.1f}")
    print(f"  Lien Score:          {result.lien_score:.1f}")
    print(f"  Communication Score: {result.communication_score:.1f}")
    print(f"\nTotal Recovery Score:  {result.total_recovery_score:.2f}")
    print(f"Recovery Band:         {result.recovery_band_label}")
    print(f"Recovery % (midpoint): {result.recovery_pct_midpoint:.1%}")
    print(f"Loss % (midpoint):     {result.loss_pct_midpoint:.1%}")
    print(f"\nEstimated Bad Debt Expense:")
    print(f"  Low:    ${result.estimated_bad_debt_expense_low:,.2f}")
    print(f"  Mid:    ${result.estimated_bad_debt_expense_mid:,.2f}")
    print(f"  High:   ${result.estimated_bad_debt_expense_high:,.2f}")

    # Example 2: Existing deal from database
    print("\n" + "=" * 60)
    print("EXAMPLE 2: Existing Deal (Database)")
    print("=" * 60)

    deal = {
        "loan_status": "Moderate Delinquency",
        "industry": "238110",  # Construction - Electrical
        "ahead_positions": 1,  # Second lien
        "net_balance": 75000.0,
    }

    result = compute_recovery_for_deal(deal)

    print(f"Deal Status: {deal['loan_status']}")
    print(f"Industry (NAICS): {deal['industry']}")
    print(f"Lien Position: {deal['ahead_positions']} (Second)")
    print(f"Net Balance: ${deal['net_balance']:,.2f}")
    print(f"\nComponent Scores (0-10):")
    print(f"  Status Score:        {result.status_score:.1f}")
    print(f"  Industry Score:      {result.industry_score:.1f}")
    print(f"  Collateral Score:    {result.collateral_score:.1f} (default - unsecured)")
    print(f"  Lien Score:          {result.lien_score:.1f}")
    print(f"  Communication Score: {result.communication_score:.1f} (default - sporadic)")
    print(f"\nTotal Recovery Score:  {result.total_recovery_score:.2f}")
    print(f"Recovery Band:         {result.recovery_band_label}")
    print(f"Loss % (midpoint):     {result.loss_pct_midpoint:.1%}")
    print(f"\nEstimated Bad Debt Expense: ${result.estimated_bad_debt_expense_mid:,.2f}")

    # Example 3: Paid off deal (edge case)
    print("\n" + "=" * 60)
    print("EXAMPLE 3: Paid Off Deal (Edge Case)")
    print("=" * 60)

    paid_deal = {
        "loan_status": "Paid Off",
        "industry": "541110",  # Legal Services
        "ahead_positions": 0,
        "net_balance": -500.0,  # Overpaid (negative balance)
    }

    result = compute_recovery_for_deal(paid_deal)

    print(f"Deal Status: {paid_deal['loan_status']}")
    print(f"Net Balance: ${paid_deal['net_balance']:,.2f}")
    print(f"\nTotal Recovery Score:  {result.total_recovery_score:.2f}")
    print(f"Loss % (midpoint):     {result.loss_pct_midpoint:.1%} (forced to 0 for Paid Off)")
    print(f"Exposure Base:         ${result.exposure_base:,.2f} (clamped to 0)")
    print(f"Bad Debt Expense:      ${result.estimated_bad_debt_expense_mid:,.2f}")
