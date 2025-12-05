# utils/manual_status_editor.py
"""
Manual Status Override Component

Allows users to manually set loan status, which will be preserved
by backend scripts (manual_status flag = True).
"""
import streamlit as st
from datetime import datetime
from typing import Optional
from utils.status_constants import ALL_VALID_STATUSES, TERMINAL_STATUSES, PROTECTED_STATUSES
from utils.config import get_supabase_client


def update_loan_status_manual(loan_id: str, new_status: str) -> bool:
    """
    Update loan status manually and set manual_status flag.

    Args:
        loan_id: The loan ID to update
        new_status: The new status to set

    Returns:
        True if successful, False otherwise
    """
    try:
        supabase = get_supabase_client()
        supabase.table("loan_summaries").update({
            "loan_status": new_status,
            "manual_status": True,
            "status_last_manual_update": datetime.now().isoformat(),
            "status_changed_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat()
        }).eq("loan_id", loan_id).execute()
        return True
    except Exception as e:
        st.error(f"Failed to update status: {e}")
        return False


def render_status_badge(status: str) -> str:
    """Return colored badge HTML for status."""
    if status in TERMINAL_STATUSES:
        if status == "Paid Off":
            return f'<span style="background-color:#1f77b4;color:white;padding:2px 8px;border-radius:4px;">{status}</span>'
        else:
            return f'<span style="background-color:#d62728;color:white;padding:2px 8px;border-radius:4px;">{status}</span>'
    elif status in PROTECTED_STATUSES:
        return f'<span style="background-color:#ff7f0e;color:white;padding:2px 8px;border-radius:4px;">{status}</span>'
    elif "Delinquency" in status or "Late" in status:
        return f'<span style="background-color:#ffbb78;color:black;padding:2px 8px;border-radius:4px;">{status}</span>'
    else:
        return f'<span style="background-color:#2ca02c;color:white;padding:2px 8px;border-radius:4px;">{status}</span>'


def render_manual_status_editor(
    loan_id: str,
    current_status: str,
    is_manual: bool = False,
    compact: bool = False
):
    """
    Render manual status override UI for a single loan.

    Args:
        loan_id: The loan ID to edit
        current_status: Current loan status
        is_manual: Whether current status was manually set
        compact: Use compact layout (for tables)
    """
    # Status indicator
    if is_manual:
        st.caption("Manual status (preserved)")

    if current_status in TERMINAL_STATUSES:
        st.warning(f"Terminal status '{current_status}' - cannot be changed")
        return

    if compact:
        col1, col2 = st.columns([3, 1])
    else:
        col1, col2 = st.columns([2, 1])

    with col1:
        # Organize statuses by category for better UX
        status_options = []

        # Active statuses first
        status_options.append("Active")
        status_options.append("Active - Frequently Late")

        # Delinquency ladder
        status_options.extend([
            "Minor Delinquency",
            "Moderate Delinquency",
            "Severe Delinquency",
            "Past Delinquency"
        ])

        # Problem/Protected statuses
        status_options.extend([
            "Default",
            "NSF / Suspended",
            "Non-Performing",
            "In Collections",
            "Legal Action",
            "Bankruptcy",
            "Charged Off"
        ])

        # Terminal
        status_options.append("Paid Off")

        try:
            default_index = status_options.index(current_status)
        except ValueError:
            default_index = 0

        new_status = st.selectbox(
            "Status" if compact else "Select New Status",
            options=status_options,
            index=default_index,
            key=f"status_select_{loan_id}",
            label_visibility="collapsed" if compact else "visible"
        )

    with col2:
        if st.button("Update", key=f"update_btn_{loan_id}", type="primary"):
            if new_status != current_status:
                with st.spinner("Updating..."):
                    success = update_loan_status_manual(loan_id, new_status)
                if success:
                    st.success("Updated")
                    st.rerun()
            else:
                st.info("No change")

    if not compact:
        st.caption("Manual status will be preserved by automated jobs for 30 days")


def render_bulk_status_editor(loan_ids: list, current_statuses: dict):
    """
    Render bulk status update UI for multiple loans.

    Args:
        loan_ids: List of loan IDs to update
        current_statuses: Dict mapping loan_id to current status
    """
    st.subheader(f"Bulk Status Update ({len(loan_ids)} loans)")

    # Filter out terminal statuses
    editable_loans = [lid for lid in loan_ids if current_statuses.get(lid) not in TERMINAL_STATUSES]
    terminal_loans = [lid for lid in loan_ids if current_statuses.get(lid) in TERMINAL_STATUSES]

    if terminal_loans:
        st.warning(f"{len(terminal_loans)} loans have terminal status and cannot be changed")

    if not editable_loans:
        st.info("No editable loans selected")
        return

    new_status = st.selectbox(
        "Set all selected loans to:",
        options=[
            "Default",
            "NSF / Suspended",
            "Non-Performing",
            "In Collections",
            "Legal Action",
            "Charged Off"
        ],
        key="bulk_status_select"
    )

    if st.button(f"Update {len(editable_loans)} Loans", type="primary"):
        success_count = 0
        for loan_id in editable_loans:
            if update_loan_status_manual(loan_id, new_status):
                success_count += 1

        st.success(f"Updated {success_count}/{len(editable_loans)} loans to '{new_status}'")
        st.rerun()
