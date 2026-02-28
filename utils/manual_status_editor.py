# utils/manual_status_editor.py
"""
Manual Status Override Component

Allows users to manually set loan status, which will be preserved
by backend scripts (manual_status flag = True).
"""
import streamlit as st
from datetime import datetime
from typing import Optional, Tuple
from google.cloud import bigquery
from utils.status_constants import ALL_VALID_STATUSES, TERMINAL_STATUSES, PROTECTED_STATUSES
from utils.config import get_bq_client, _TABLE_MAP


def clear_loan_summaries_cache():
    """Clear the loan_summaries cache to force a fresh load after updates."""
    try:
        # Clear the DataLoader's cached load_loan_summaries method
        from utils.data_loader import DataLoader
        DataLoader.load_loan_summaries.clear()
        print("Cache cleared for load_loan_summaries")
    except Exception as e:
        print(f"Warning: Could not clear cache: {e}")


def update_loan_status_manual(loan_id: str, new_status: str) -> Tuple[bool, str]:
    """
    Update loan status manually and set manual_status flag.

    This function updates the loan_summaries table with:
    - loan_status: The new status value
    - manual_status: True (to flag this as a manual override)
    - status_last_manual_update: Timestamp of this update
    - status_changed_at: Timestamp of status change
    - updated_at: General update timestamp

    Args:
        loan_id: The loan ID to update
        new_status: The new status to set

    Returns:
        Tuple of (success: bool, message: str)
    """
    try:
        bq = get_bq_client()
        timestamp = datetime.now().isoformat()
        table = _TABLE_MAP["loan_summaries"]

        print(f"Attempting to update loan {loan_id} to status '{new_status}'")

        job_config = bigquery.QueryJobConfig(
            query_parameters=[
                bigquery.ScalarQueryParameter("new_status", "STRING", new_status),
                bigquery.ScalarQueryParameter("loan_id", "STRING", loan_id),
                bigquery.ScalarQueryParameter("timestamp", "STRING", timestamp),
            ]
        )
        bq.query(
            f"""
            UPDATE `{table}`
            SET
                loan_status = @new_status,
                manual_status = TRUE,
                status_last_manual_update = @timestamp,
                status_changed_at = @timestamp,
                updated_at = @timestamp
            WHERE loan_id = @loan_id
            """,
            job_config=job_config
        ).result()

        print(f"Update successful for loan {loan_id}")
        clear_loan_summaries_cache()
        return True, f"Status updated to '{new_status}'"

    except Exception as e:
        error_msg = str(e)
        print(f"Update failed with error: {error_msg}")
        if "permission" in error_msg.lower() or "denied" in error_msg.lower():
            return False, f"Permission denied. Check BigQuery service account permissions."
        return False, f"Update failed: {error_msg}"


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

        # Callback to keep user on Loan Tape tab when changing status
        def on_status_change():
            st.session_state.stay_on_loan_tape_tab = True

        new_status = st.selectbox(
            "Status" if compact else "Select New Status",
            options=status_options,
            index=default_index,
            key=f"status_select_{loan_id}",
            label_visibility="collapsed" if compact else "visible",
            on_change=on_status_change
        )

    with col2:
        if st.button("Update", key=f"update_btn_{loan_id}", type="primary"):
            if new_status != current_status:
                with st.spinner("Updating status in database..."):
                    success, message = update_loan_status_manual(loan_id, new_status)
                if success:
                    st.success(message)
                    # Set flag to stay on Loan Tape tab after rerun
                    st.session_state.stay_on_loan_tape_tab = True
                    st.rerun()
                else:
                    st.error(message)
            else:
                st.info("No change - select a different status")

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
        errors = []
        progress_bar = st.progress(0)

        for i, loan_id in enumerate(editable_loans):
            success, message = update_loan_status_manual(loan_id, new_status)
            if success:
                success_count += 1
            else:
                errors.append(f"Loan {loan_id}: {message}")
            progress_bar.progress((i + 1) / len(editable_loans))

        progress_bar.empty()

        if success_count == len(editable_loans):
            st.success(f"Successfully updated all {success_count} loans to '{new_status}'")
        elif success_count > 0:
            st.warning(f"Updated {success_count}/{len(editable_loans)} loans. Some failed.")
            with st.expander("View errors"):
                for error in errors:
                    st.error(error)
        else:
            st.error(f"Failed to update any loans")
            with st.expander("View errors"):
                for error in errors:
                    st.error(error)
        st.rerun()
