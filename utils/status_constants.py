# utils/status_constants.py
"""
Loan Status Constants - Re-exports from central source of truth.

This module provides convenient access to loan status constants
for use in the Streamlit dashboard utilities.
"""

# Re-export all status constants from the central source
from scripts.loan_status_utils import (
    # Status Lists
    TERMINAL_STATUSES,
    PROTECTED_STATUSES,
    PROBLEM_STATUSES,
    DELINQUENCY_STATUSES,
    ALL_VALID_STATUSES,
    # Configuration
    MANUAL_STATUS_THRESHOLD_DAYS,
    PAID_OFF_TOLERANCE,
    PAYMENT_AHEAD_THRESHOLD,
    # Validation Functions
    validate_status,
    is_terminal_status,
    is_problem_status,
    is_protected_status,
    # Status Functions
    should_preserve_manual_status,
    determine_loan_status,
)

__all__ = [
    "TERMINAL_STATUSES",
    "PROTECTED_STATUSES",
    "PROBLEM_STATUSES",
    "DELINQUENCY_STATUSES",
    "ALL_VALID_STATUSES",
    "MANUAL_STATUS_THRESHOLD_DAYS",
    "PAID_OFF_TOLERANCE",
    "PAYMENT_AHEAD_THRESHOLD",
    "validate_status",
    "is_terminal_status",
    "is_problem_status",
    "is_protected_status",
    "should_preserve_manual_status",
    "determine_loan_status",
]
