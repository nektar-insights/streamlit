"""
Status Constants - Mirrored from hubspot/scripts/loan_status_utils.py

IMPORTANT: Keep in sync with hubspot repo. Source of truth is hubspot.
Last synced: 2025-12-05
"""

# Terminal statuses - NEVER change once set
TERMINAL_STATUSES = ["Paid Off", "Charged Off", "Bankruptcy"]

# Protected statuses - Require manual intervention
PROTECTED_STATUSES = [
    "Charged Off", "In Collections", "Legal Action",
    "Default", "Bankruptcy", "NSF / Suspended", "Non-Performing"
]

# Problem statuses - For ML and risk flagging
PROBLEM_STATUSES = {
    "Default", "Bankruptcy", "Charged Off", "In Collections",
    "Legal Action", "NSF / Suspended", "Non-Performing",
    "Severe Delinquency", "Moderate Delinquency",
    "Active - Frequently Late"
}

# All valid statuses
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


def is_problem_status(status: str) -> bool:
    return status in PROBLEM_STATUSES
