# utils/status_constants.py
"""
Loan Status Constants - Single Source of Truth

This module provides centralized definitions for all loan status constants
and utility functions for status determination across the application.
"""

from datetime import datetime, timedelta
from typing import Optional, Union
import pandas as pd
import numpy as np

# =============================================================================
# STATUS CONSTANTS - Single Source of Truth
# =============================================================================

# Terminal statuses - NEVER change once set (loan lifecycle is complete)
TERMINAL_STATUSES = ["Paid Off", "Charged Off", "Bankruptcy"]

# Protected statuses - Require manual intervention to change
PROTECTED_STATUSES = [
    "Charged Off", "In Collections", "Legal Action",
    "Default", "Bankruptcy", "NSF / Suspended", "Non-Performing"
]

# Problem statuses - Used for ML classification and risk flagging
PROBLEM_STATUSES = {
    "Default", "Bankruptcy", "Charged Off", "In Collections",
    "Legal Action", "NSF / Suspended", "Non-Performing",
    "Severe Delinquency", "Moderate Delinquency",
    "Active - Frequently Late"
}

# Delinquency ladder (in order of severity)
DELINQUENCY_STATUSES = [
    "Minor Delinquency",
    "Moderate Delinquency",
    "Severe Delinquency"
]

# All valid statuses for validation
ALL_VALID_STATUSES = [
    "Active",
    "Active - Frequently Late",
    "Minor Delinquency",
    "Moderate Delinquency",
    "Severe Delinquency",
    "Past Delinquency",
    "Default",
    "Bankruptcy",
    "Charged Off",
    "In Collections",
    "Legal Action",
    "NSF / Suspended",
    "Non-Performing",
    "Paid Off"
]

# Status Colors - Unified color mapping for all valid statuses
# Organized by category for visual consistency across all pages
STATUS_COLORS = {
    # Active statuses - green tones
    "Active": "#2ca02c",                    # Green
    "Active - Frequently Late": "#98df8a",  # Light green
    # Delinquency statuses - yellow/orange progression
    "Minor Delinquency": "#ffdd57",         # Yellow
    "Moderate Delinquency": "#ffbb78",      # Light orange
    "Severe Delinquency": "#ff9800",        # Orange
    "Past Delinquency": "#aec7e8",          # Light blue (recovered)
    # Problem statuses - red/orange tones
    "Default": "#ff7f0e",                   # Dark orange
    "NSF / Suspended": "#e377c2",           # Pink
    "Non-Performing": "#d62728",            # Red
    "In Collections": "#9467bd",            # Purple
    "Legal Action": "#8c564b",              # Brown
    # Terminal statuses
    "Paid Off": "#1f77b4",                  # Blue
    "Charged Off": "#d62728",               # Red
    "Bankruptcy": "#7f0000",                # Dark red
}

# Status groupings for summary views
STATUS_GROUPS = {
    "Active": ["Active", "Active - Frequently Late"],
    "Delinquent": ["Minor Delinquency", "Moderate Delinquency", "Severe Delinquency", "Past Delinquency"],
    "Problem": ["Default", "NSF / Suspended", "Non-Performing", "In Collections", "Legal Action"],
    "Terminal": ["Paid Off", "Charged Off", "Bankruptcy"],
}

# Group colors for summary charts
STATUS_GROUP_COLORS = {
    "Active": "#2ca02c",      # Green
    "Delinquent": "#ffbb78",  # Orange
    "Problem": "#d62728",     # Red
    "Terminal": "#1f77b4",    # Blue
}


# =============================================================================
# CONFIGURATION CONSTANTS
# =============================================================================

# Threshold for preserving manual status (days)
MANUAL_STATUS_THRESHOLD_DAYS = 30

# Tolerance for considering a loan paid off (percentage, e.g., 0.98 = 98%)
PAID_OFF_TOLERANCE = 0.98

# Threshold for considering a loan ahead on payments (percentage)
PAYMENT_AHEAD_THRESHOLD = 1.05


# =============================================================================
# VALIDATION FUNCTIONS
# =============================================================================

def validate_status(status: str) -> bool:
    """Validate if a status string is valid.

    Args:
        status: The status string to validate

    Returns:
        True if status is in ALL_VALID_STATUSES, False otherwise
    """
    return status in ALL_VALID_STATUSES


def is_terminal_status(status: str) -> bool:
    """Check if status is terminal (loan lifecycle complete).

    Terminal statuses indicate the loan has reached its final state
    and should NEVER be automatically changed.

    Args:
        status: The status string to check

    Returns:
        True if status is terminal, False otherwise
    """
    return status in TERMINAL_STATUSES


def is_problem_status(status: str) -> bool:
    """Check if status indicates a problem loan.

    Problem statuses are used for ML classification and risk flagging.

    Args:
        status: The status string to check

    Returns:
        True if status indicates a problem, False otherwise
    """
    return status in PROBLEM_STATUSES


def is_protected_status(status: str) -> bool:
    """Check if status is protected (requires manual intervention to change).

    Args:
        status: The status string to check

    Returns:
        True if status is protected, False otherwise
    """
    return status in PROTECTED_STATUSES


# =============================================================================
# DATE UTILITIES
# =============================================================================

def safe_date_parse(date_val: Union[str, datetime, pd.Timestamp, None]) -> Optional[datetime]:
    """Safely parse a date value to datetime.

    Args:
        date_val: Date value in various formats

    Returns:
        datetime object or None if parsing fails
    """
    if date_val is None or (isinstance(date_val, float) and np.isnan(date_val)):
        return None
    if isinstance(date_val, datetime):
        return date_val
    if isinstance(date_val, pd.Timestamp):
        return date_val.to_pydatetime()
    try:
        return pd.to_datetime(date_val).to_pydatetime()
    except (ValueError, TypeError):
        return None


def calculate_payoff_date(
    schedules_df: pd.DataFrame,
    loan_id: str
) -> Optional[datetime]:
    """Calculate the expected payoff date for a loan from its schedule.

    Args:
        schedules_df: DataFrame containing payment schedules
        loan_id: The loan ID to look up

    Returns:
        Expected payoff date or None if not found
    """
    if schedules_df is None or schedules_df.empty:
        return None

    # Normalize loan_id for comparison
    loan_id_str = str(loan_id).strip().replace('.0', '')

    loan_schedule = schedules_df[
        schedules_df['loan_id'].astype(str).str.strip().str.replace(r'\.0$', '', regex=True) == loan_id_str
    ]

    if loan_schedule.empty:
        return None

    # Get the last scheduled payment date
    date_col = 'payment_date' if 'payment_date' in loan_schedule.columns else 'due_date'
    if date_col not in loan_schedule.columns:
        return None

    last_date = loan_schedule[date_col].max()
    return safe_date_parse(last_date)


def detect_early_payoff(
    schedules_df: pd.DataFrame,
    loan_id: str,
    current_date: Optional[datetime] = None
) -> bool:
    """Detect if a loan shows an early payoff in the schedule.

    An early payoff is indicated when the schedule shows the loan
    is fully paid before the originally scheduled end date.

    Args:
        schedules_df: DataFrame containing payment schedules
        loan_id: The loan ID to check
        current_date: Reference date (defaults to today)

    Returns:
        True if early payoff detected, False otherwise
    """
    if schedules_df is None or schedules_df.empty:
        return False

    if current_date is None:
        current_date = datetime.now()

    # Normalize loan_id for comparison
    loan_id_str = str(loan_id).strip().replace('.0', '')

    loan_schedule = schedules_df[
        schedules_df['loan_id'].astype(str).str.strip().str.replace(r'\.0$', '', regex=True) == loan_id_str
    ].copy()

    if loan_schedule.empty:
        return False

    # Check for early payoff indicator columns
    if 'is_paid_off' in loan_schedule.columns:
        return loan_schedule['is_paid_off'].any()

    if 'early_payoff' in loan_schedule.columns:
        return loan_schedule['early_payoff'].any()

    return False


# =============================================================================
# MANUAL STATUS HANDLING
# =============================================================================

def should_preserve_manual_status(
    manual_status_flag: bool,
    last_status_update: Optional[datetime],
    threshold_days: int = MANUAL_STATUS_THRESHOLD_DAYS
) -> bool:
    """Determine if a manually set status should be preserved.

    Manual statuses are preserved if:
    1. The manual_status flag is True, AND
    2. The status was updated within the threshold period

    Args:
        manual_status_flag: Whether the status was manually set
        last_status_update: When the status was last updated
        threshold_days: Days after which manual status can be overridden

    Returns:
        True if manual status should be preserved, False otherwise
    """
    if not manual_status_flag:
        return False

    if last_status_update is None:
        # If no update date, preserve by default
        return True

    update_date = safe_date_parse(last_status_update)
    if update_date is None:
        return True

    cutoff_date = datetime.now() - timedelta(days=threshold_days)
    return update_date > cutoff_date


# =============================================================================
# MAIN STATUS DETERMINATION FUNCTION
# =============================================================================

def determine_loan_status(
    loan_id: str,
    current_status: str,
    payment_progress: float,
    days_past_due: int,
    late_payment_count: int,
    manual_status: bool = False,
    last_status_update: Optional[datetime] = None,
    schedules_df: Optional[pd.DataFrame] = None,
    total_paid: Optional[float] = None,
    total_expected: Optional[float] = None
) -> str:
    """Determine the appropriate loan status based on multiple factors.

    Priority Order:
    1. Early Payoff in schedule -> Paid Off
    2. payment_progress >= 98% -> Paid Off
    3. No scheduled payments + has paid -> Paid Off
    4. current_status in TERMINAL_STATUSES -> preserve (lifecycle complete)
    5. current_status in PROTECTED_STATUSES -> preserve (needs manual review)
    6. manual_status = True (< 30 days) -> preserve
    7. Delinquency calculation -> Severe/Moderate/Minor Delinquency
    8. Past missed payments -> Past Delinquency
    9. 3+ late payments -> Active - Frequently Late
    10. Default -> Active

    Args:
        loan_id: Unique identifier for the loan
        current_status: Current status of the loan
        payment_progress: Percentage of expected payments made (0-1 or 0-100)
        days_past_due: Number of days past due on current payment
        late_payment_count: Total count of late payments
        manual_status: Whether status was manually set
        last_status_update: When status was last updated
        schedules_df: Optional DataFrame with payment schedules
        total_paid: Optional total amount paid
        total_expected: Optional total amount expected

    Returns:
        Determined loan status string
    """
    # Normalize payment_progress to 0-1 scale if needed
    if payment_progress > 1:
        payment_progress = payment_progress / 100

    # === PRIORITY 1: Early Payoff in schedule ===
    if detect_early_payoff(schedules_df, loan_id):
        return "Paid Off"

    # === PRIORITY 2: Payment progress indicates paid off ===
    if payment_progress >= PAID_OFF_TOLERANCE:
        return "Paid Off"

    # === PRIORITY 3: No scheduled payments remaining + has paid ===
    if total_paid is not None and total_expected is not None:
        if total_expected > 0 and total_paid >= total_expected * PAID_OFF_TOLERANCE:
            return "Paid Off"

    # === PRIORITY 4: Check Terminal Statuses ===
    # Terminal statuses should NEVER change - loan lifecycle is complete
    if current_status in TERMINAL_STATUSES:
        return current_status

    # === PRIORITY 5: Check Protected Statuses ===
    # Protected statuses require manual intervention to change
    if current_status in PROTECTED_STATUSES:
        return current_status

    # === PRIORITY 6: Preserve recent manual status ===
    if should_preserve_manual_status(manual_status, last_status_update):
        return current_status

    # === PRIORITY 7: Delinquency based on days past due ===
    if days_past_due >= 90:
        return "Severe Delinquency"
    elif days_past_due >= 60:
        return "Moderate Delinquency"
    elif days_past_due >= 30:
        return "Minor Delinquency"

    # === PRIORITY 8: Past delinquency (was late but now current) ===
    if days_past_due == 0 and late_payment_count > 0:
        # Check if there's a history of delinquency
        if current_status in DELINQUENCY_STATUSES:
            return "Past Delinquency"

    # === PRIORITY 9: Frequently late ===
    if late_payment_count >= 3:
        return "Active - Frequently Late"

    # === PRIORITY 10: Default to Active ===
    return "Active"


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Constants
    "TERMINAL_STATUSES",
    "PROTECTED_STATUSES",
    "PROBLEM_STATUSES",
    "DELINQUENCY_STATUSES",
    "ALL_VALID_STATUSES",
    "STATUS_COLORS",
    "STATUS_GROUPS",
    "STATUS_GROUP_COLORS",
    "MANUAL_STATUS_THRESHOLD_DAYS",
    "PAID_OFF_TOLERANCE",
    "PAYMENT_AHEAD_THRESHOLD",
    # Validation Functions
    "validate_status",
    "is_terminal_status",
    "is_problem_status",
    "is_protected_status",
    # Date Utilities
    "safe_date_parse",
    "calculate_payoff_date",
    "detect_early_payoff",
    # Status Functions
    "should_preserve_manual_status",
    "determine_loan_status",
]
