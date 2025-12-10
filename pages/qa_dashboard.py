"""
Data Quality Dashboard - Consolidated QA Checks
================================================
Unified data quality dashboard that consolidates checks from:
- Deal data integrity (missing loan IDs, duplicates)
- Loan date validation (maturity vs funding dates)
- Data freshness monitoring
- Portfolio health indicators

Prioritizes critical issues at the top with clear action items.
"""

import pandas as pd
import streamlit as st
import altair as alt
from utils.config import setup_page, PRIMARY_COLOR
from utils.data_loader import (
    load_deals,
    load_loan_summaries,
    load_qbo_data,
)
from utils.preprocessing import preprocess_dataframe
from utils.loan_tape_data import prepare_loan_data

# ----------------------------
# Page Configuration
# ----------------------------
setup_page("CSL Capital | Data Quality")

# ----------------------------
# Data Loading
# ----------------------------
@st.cache_data(ttl=3600)
def load_all_qa_data():
    """Load all required data for QA checks"""
    deals_df = load_deals()
    loans_df = load_loan_summaries()
    qbo_txn_df, qbo_gl_df = load_qbo_data()

    # Preprocess all datasets
    deals_df = preprocess_dataframe(deals_df)
    loans_df = preprocess_dataframe(loans_df)
    qbo_txn_df = preprocess_dataframe(qbo_txn_df)
    qbo_gl_df = preprocess_dataframe(qbo_gl_df)

    # Prepare combined loan data for loan-specific checks
    prepared_loans_df = pd.DataFrame()
    if not loans_df.empty and not deals_df.empty:
        try:
            prepared_loans_df = prepare_loan_data(loans_df, deals_df)
        except Exception as e:
            st.warning(f"Could not prepare loan data: {e}")

    return deals_df, loans_df, prepared_loans_df, qbo_txn_df, qbo_gl_df


# ----------------------------
# QA Check Functions
# ----------------------------

def check_missing_loan_ids(deals_df):
    """
    Check for won deals missing loan IDs
    Returns: dict with 'status', 'count', 'data', 'action'
    """
    if deals_df.empty or "is_closed_won" not in deals_df.columns:
        return {
            "status": "pass",
            "count": 0,
            "total": 0,
            "data": pd.DataFrame(),
            "action": "No deals data available"
        }

    won_deals = deals_df[deals_df["is_closed_won"] == True]

    if "loan_id" not in deals_df.columns:
        return {
            "status": "critical",
            "count": len(won_deals),
            "total": len(won_deals),
            "data": won_deals,
            "action": "loan_id column missing from deals data"
        }

    missing = won_deals[
        (won_deals["loan_id"].isna()) |
        (won_deals["loan_id"].astype(str).str.strip() == "") |
        (won_deals["loan_id"].astype(str).str.strip() == "nan")
    ]

    return {
        "status": "critical" if len(missing) > 0 else "pass",
        "count": len(missing),
        "total": len(won_deals),
        "data": missing,
        "action": "Add loan_id to these deals in HubSpot"
    }


def check_duplicate_loan_ids(deals_df):
    """Check for duplicate loan_ids in won deals"""
    if deals_df.empty or "loan_id" not in deals_df.columns:
        return {
            "status": "pass",
            "count": 0,
            "data": pd.DataFrame(),
            "action": "No loan_id data available"
        }

    # Filter to valid loan IDs only
    valid = deals_df[
        (deals_df["loan_id"].notna()) &
        (deals_df["loan_id"].astype(str).str.strip() != "") &
        (deals_df["loan_id"].astype(str).str.strip() != "nan")
    ].copy()

    if valid.empty:
        return {
            "status": "pass",
            "count": 0,
            "data": pd.DataFrame(),
            "action": "No valid loan IDs to check"
        }

    # Normalize loan_id for comparison
    valid["loan_id_normalized"] = valid["loan_id"].astype(str).str.strip().str.replace(r'\.0$', '', regex=True)

    duplicates = valid[valid["loan_id_normalized"].duplicated(keep=False)]

    return {
        "status": "critical" if len(duplicates) > 0 else "pass",
        "count": len(duplicates),
        "data": duplicates.sort_values("loan_id_normalized"),
        "action": "Resolve duplicate loan_id assignments in HubSpot"
    }


def check_invalid_dates(loans_df):
    """Maturity date should be after funding date for active loans"""
    if loans_df.empty:
        return {
            "status": "pass",
            "count": 0,
            "data": pd.DataFrame(),
            "action": "No loan data available"
        }

    # Check required columns
    if "maturity_date" not in loans_df.columns or "funding_date" not in loans_df.columns:
        return {
            "status": "pass",
            "count": 0,
            "data": pd.DataFrame(),
            "action": "Date columns not available"
        }

    # Filter to active loans (not paid off)
    status_col = "loan_status" if "loan_status" in loans_df.columns else None
    if status_col:
        active = loans_df[loans_df[status_col] != "Paid Off"].copy()
    else:
        active = loans_df.copy()

    invalid = active[
        (active["maturity_date"].notna()) &
        (active["funding_date"].notna()) &
        (active["maturity_date"] <= active["funding_date"])
    ]

    return {
        "status": "critical" if len(invalid) > 0 else "pass",
        "count": len(invalid),
        "data": invalid,
        "action": "Correct maturity dates in loan records"
    }


def check_expiring_loans(loans_df, days=30):
    """Loans maturing in next X days"""
    if loans_df.empty or "maturity_date" not in loans_df.columns:
        return {
            "status": "pass",
            "count": 0,
            "data": pd.DataFrame(),
            "action": "No maturity date data available"
        }

    today = pd.Timestamp.now().normalize()
    cutoff = today + pd.Timedelta(days=days)

    # Filter to active loans
    status_col = "loan_status" if "loan_status" in loans_df.columns else None
    if status_col:
        active = loans_df[loans_df[status_col] != "Paid Off"].copy()
    else:
        active = loans_df.copy()

    expiring = active[
        (active["maturity_date"].notna()) &
        (active["maturity_date"] >= today) &
        (active["maturity_date"] <= cutoff)
    ].copy()

    if not expiring.empty:
        expiring["days_until_maturity"] = (expiring["maturity_date"] - today).dt.days
        expiring = expiring.sort_values("days_until_maturity")

    return {
        "status": "warning" if len(expiring) > 0 else "pass",
        "count": len(expiring),
        "data": expiring,
        "action": "Review upcoming maturities for renewal/payoff"
    }


def check_past_maturity(loans_df):
    """Loans past maturity date but not paid off"""
    if loans_df.empty or "maturity_date" not in loans_df.columns:
        return {
            "status": "pass",
            "count": 0,
            "data": pd.DataFrame(),
            "action": "No maturity date data available"
        }

    today = pd.Timestamp.now().normalize()

    # Filter to active loans
    status_col = "loan_status" if "loan_status" in loans_df.columns else None
    if status_col:
        active = loans_df[loans_df[status_col] != "Paid Off"].copy()
    else:
        active = loans_df.copy()

    overdue = active[
        (active["maturity_date"].notna()) &
        (active["maturity_date"] < today)
    ].copy()

    if not overdue.empty:
        overdue["days_overdue"] = (today - overdue["maturity_date"]).dt.days
        overdue = overdue.sort_values("days_overdue", ascending=False)

    return {
        "status": "warning" if len(overdue) > 0 else "pass",
        "count": len(overdue),
        "data": overdue,
        "action": "Review loan status, update payment status, or mark as paid off"
    }


def check_missing_fields(deals_df):
    """Check for missing amount, factor_rate, loan_term in won deals"""
    if deals_df.empty or "is_closed_won" not in deals_df.columns:
        return {
            "status": "pass",
            "issues": {},
            "action": "No deals data available"
        }

    won = deals_df[deals_df["is_closed_won"] == True]
    fields = ["amount", "factor_rate", "loan_term"]
    issues = {}

    for field in fields:
        if field in won.columns:
            missing = won[field].isna().sum()
            if missing > 0:
                issues[field] = missing

    return {
        "status": "warning" if issues else "pass",
        "issues": issues,
        "total": len(won),
        "action": "Complete missing deal fields in HubSpot"
    }


def check_data_freshness(deals_df):
    """Warn if no new deals in 7+ days"""
    if deals_df.empty or "date_created" not in deals_df.columns:
        return {"status": "pass", "days": None, "action": "No date data available"}

    latest = deals_df["date_created"].max()
    if pd.isna(latest):
        return {"status": "pass", "days": None, "action": "No valid dates found"}

    days_since = (pd.Timestamp.now() - latest).days

    return {
        "status": "warning" if days_since > 7 else "pass",
        "days": days_since,
        "latest_date": latest,
        "action": "Verify HubSpot sync is working"
    }


def get_record_counts(deals_df, loans_df, qbo_txn_df, qbo_gl_df):
    """Get record counts for all data sources"""
    won_deals = 0
    if not deals_df.empty and "is_closed_won" in deals_df.columns:
        won_deals = len(deals_df[deals_df["is_closed_won"] == True])

    return {
        "Deals": len(deals_df),
        "Won Deals": won_deals,
        "Loans": len(loans_df),
        "QBO Transactions": len(qbo_txn_df),
        "QBO GL Entries": len(qbo_gl_df),
    }


def check_qbo_balance(qbo_txn_df):
    """Check for null/zero amounts"""
    if qbo_txn_df.empty or "total_amount" not in qbo_txn_df.columns:
        return {"status": "info", "null_count": 0, "zero_count": 0}

    null_count = qbo_txn_df["total_amount"].isna().sum()
    zero_count = (qbo_txn_df["total_amount"] == 0).sum()

    return {
        "status": "info",
        "null_count": null_count,
        "zero_count": zero_count
    }


# ----------------------------
# Display Components
# ----------------------------

def render_issue_card(title, check_result, show_data=True):
    """Render a consistent issue card"""
    status = check_result.get("status", "pass")

    if status == "critical":
        icon = "🔴"
    elif status == "warning":
        icon = "🟡"
    else:
        icon = "✅"

    count = check_result.get("count", 0)
    expanded = (status == "critical")

    with st.expander(f"{icon} {title} ({count})", expanded=expanded):
        if "action" in check_result and check_result["action"]:
            st.info(f"**Action Required:** {check_result['action']}")

        # Show additional context
        if "total" in check_result and check_result["total"] > 0:
            pct = (count / check_result["total"] * 100) if check_result["total"] > 0 else 0
            st.write(f"**Affected:** {count} of {check_result['total']} ({pct:.1f}%)")

        if "issues" in check_result and check_result["issues"]:
            st.write("**Missing fields breakdown:**")
            for field, missing_count in check_result["issues"].items():
                st.write(f"- {field.replace('_', ' ').title()}: {missing_count}")

        if "days" in check_result and check_result["days"] is not None:
            st.write(f"**Days since last update:** {check_result['days']}")
            if "latest_date" in check_result:
                st.write(f"**Latest record:** {check_result['latest_date'].strftime('%Y-%m-%d')}")

        if show_data and "data" in check_result and not check_result["data"].empty:
            # Show relevant columns only
            display_cols = []
            priority_cols = ["loan_id", "deal_name", "name", "partner_source", "amount",
                           "funding_date", "maturity_date", "loan_status",
                           "days_until_maturity", "days_overdue"]

            for col in priority_cols:
                if col in check_result["data"].columns:
                    display_cols.append(col)

            # Limit to first 10 columns if many available
            if len(display_cols) == 0:
                display_cols = list(check_result["data"].columns)[:10]

            display_df = check_result["data"][display_cols].copy()

            # Format date columns
            for col in display_df.columns:
                if "date" in col.lower() and display_df[col].dtype == 'datetime64[ns]':
                    display_df[col] = display_df[col].dt.strftime('%Y-%m-%d')

            st.dataframe(display_df, width='stretch', hide_index=True)

            # Download button
            csv = check_result["data"].to_csv(index=False)
            st.download_button(
                label=f"📥 Download {title} Data",
                data=csv,
                file_name=f"{title.lower().replace(' ', '_')}_{pd.Timestamp.now().strftime('%Y%m%d')}.csv",
                mime="text/csv"
            )


def render_summary_metrics(all_checks):
    """Render summary metrics row"""
    critical = sum(1 for c in all_checks if c.get("status") == "critical")
    warnings = sum(1 for c in all_checks if c.get("status") == "warning")
    passing = sum(1 for c in all_checks if c.get("status") == "pass")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("🔴 Critical", critical)
    with col2:
        st.metric("🟡 Warnings", warnings)
    with col3:
        st.metric("✅ Passing", passing)
    with col4:
        total = critical + warnings + passing
        health_score = ((passing * 100 + warnings * 50) / total) if total > 0 else 100
        st.metric("Health Score", f"{health_score:.0f}%")


# ----------------------------
# Main Dashboard
# ----------------------------

st.title("Data Quality Dashboard")
st.markdown("Comprehensive data quality checks across all data sources")

# Load data
with st.spinner("Loading data..."):
    deals_df, loans_df, prepared_loans_df, qbo_txn_df, qbo_gl_df = load_all_qa_data()

# Use prepared_loans_df for loan checks if available, otherwise use raw loans_df
loan_check_df = prepared_loans_df if not prepared_loans_df.empty else loans_df

# Run all QA checks
missing_loan_ids = check_missing_loan_ids(deals_df)
duplicate_loan_ids = check_duplicate_loan_ids(deals_df)
invalid_dates = check_invalid_dates(loan_check_df)
expiring_loans = check_expiring_loans(loan_check_df, days=30)
past_maturity = check_past_maturity(loan_check_df)
missing_fields = check_missing_fields(deals_df)
data_freshness = check_data_freshness(deals_df)

# Compile all checks for summary
all_checks = [
    missing_loan_ids,
    duplicate_loan_ids,
    invalid_dates,
    expiring_loans,
    past_maturity,
    missing_fields,
    data_freshness
]

# ----------------------------
# Executive Summary
# ----------------------------
st.header("Executive Summary")
render_summary_metrics(all_checks)

st.markdown("---")

# ----------------------------
# Critical Issues Section
# ----------------------------
st.header("🔴 Critical Issues")
st.caption("These issues require immediate attention")

critical_checks = [
    ("Won Deals Missing Loan ID", missing_loan_ids),
    ("Duplicate Loan IDs", duplicate_loan_ids),
    ("Invalid Date Logic (Maturity <= Funding)", invalid_dates),
]

has_critical = False
for title, check in critical_checks:
    if check.get("status") == "critical":
        has_critical = True
        render_issue_card(title, check)

if not has_critical:
    st.success("✅ No critical issues found!")

st.markdown("---")

# ----------------------------
# Warnings Section
# ----------------------------
st.header("🟡 Warnings")
st.caption("Issues that should be monitored")

warning_checks = [
    ("Loans Expiring Soon (30 Days)", expiring_loans),
    ("Loans Past Maturity", past_maturity),
    ("Missing Critical Fields in Won Deals", missing_fields),
    ("Data Freshness", data_freshness),
]

has_warnings = False
for title, check in warning_checks:
    if check.get("status") == "warning":
        has_warnings = True
        render_issue_card(title, check)

if not has_warnings:
    st.success("✅ No warnings found!")

st.markdown("---")

# ----------------------------
# Passing Checks Section
# ----------------------------
with st.expander("✅ Passing Checks", expanded=False):
    st.caption("All checks that passed validation")

    passing_items = []
    all_named_checks = critical_checks + warning_checks

    for title, check in all_named_checks:
        if check.get("status") == "pass":
            passing_items.append(title)

    if passing_items:
        for item in passing_items:
            st.write(f"✅ {item}")
    else:
        st.info("No passing checks to display")

st.markdown("---")

# ----------------------------
# Developer Tools Section
# ----------------------------
with st.expander("🔧 Developer Tools", expanded=False):
    st.subheader("Cache Management")
    st.caption("Use these buttons to refresh cached data")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        if st.button("Refresh Deals Cache", help="Clear deals data cache"):
            st.cache_data.clear()
            st.success("Cache cleared!")
            st.rerun()

    with col2:
        if st.button("Refresh Loans Cache", help="Clear loans data cache"):
            st.cache_data.clear()
            st.success("Cache cleared!")
            st.rerun()

    with col3:
        if st.button("Refresh QBO Cache", help="Clear QBO data cache"):
            st.cache_data.clear()
            st.success("Cache cleared!")
            st.rerun()

    with col4:
        if st.button("Refresh All Caches", type="primary", help="Clear all cached data"):
            st.cache_data.clear()
            st.success("All caches cleared!")
            st.rerun()

    st.markdown("---")

    # Raw Data Counts
    st.subheader("Raw Data Counts")
    record_counts = get_record_counts(deals_df, loans_df, qbo_txn_df, qbo_gl_df)

    counts_df = pd.DataFrame([
        {"Data Source": k, "Record Count": f"{v:,}"}
        for k, v in record_counts.items()
    ])
    st.dataframe(counts_df, width='stretch', hide_index=True)

    st.markdown("---")

    # Column Listings
    st.subheader("Column Listings")

    col1, col2 = st.columns(2)

    with col1:
        st.write("**Deals Columns:**")
        if not deals_df.empty:
            st.code("\n".join(deals_df.columns.tolist()[:20]))
            if len(deals_df.columns) > 20:
                st.caption(f"... and {len(deals_df.columns) - 20} more columns")
        else:
            st.write("No data")

        st.write("**Loans Columns:**")
        if not loans_df.empty:
            st.code("\n".join(loans_df.columns.tolist()[:20]))
            if len(loans_df.columns) > 20:
                st.caption(f"... and {len(loans_df.columns) - 20} more columns")
        else:
            st.write("No data")

    with col2:
        st.write("**QBO Transaction Columns:**")
        if not qbo_txn_df.empty:
            st.code("\n".join(qbo_txn_df.columns.tolist()[:20]))
            if len(qbo_txn_df.columns) > 20:
                st.caption(f"... and {len(qbo_txn_df.columns) - 20} more columns")
        else:
            st.write("No data")

        st.write("**QBO GL Columns:**")
        if not qbo_gl_df.empty:
            st.code("\n".join(qbo_gl_df.columns.tolist()[:20]))
            if len(qbo_gl_df.columns) > 20:
                st.caption(f"... and {len(qbo_gl_df.columns) - 20} more columns")
        else:
            st.write("No data")

    st.markdown("---")

    # QBO Balance Check
    st.subheader("QBO Data Quality")
    qbo_balance = check_qbo_balance(qbo_txn_df)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Null Amounts", qbo_balance["null_count"])
    with col2:
        st.metric("Zero Amounts", qbo_balance["zero_count"])
    with col3:
        total_txn = len(qbo_txn_df)
        valid_txn = total_txn - qbo_balance["null_count"] - qbo_balance["zero_count"]
        st.metric("Valid Transactions", valid_txn)

    st.markdown("---")

    # Last Updated Timestamps
    st.subheader("Data Freshness Details")

    freshness_data = []

    if not deals_df.empty and "date_created" in deals_df.columns:
        latest = deals_df["date_created"].max()
        if pd.notna(latest):
            days_ago = (pd.Timestamp.now() - latest).days
            freshness_data.append({
                "Data Source": "Deals",
                "Latest Record": latest.strftime("%Y-%m-%d"),
                "Days Ago": days_ago
            })

    if not loans_df.empty and "funding_date" in loans_df.columns:
        latest = loans_df["funding_date"].max()
        if pd.notna(latest):
            days_ago = (pd.Timestamp.now() - latest).days
            freshness_data.append({
                "Data Source": "Loans (Funding)",
                "Latest Record": latest.strftime("%Y-%m-%d"),
                "Days Ago": days_ago
            })

    if not qbo_txn_df.empty and "txn_date" in qbo_txn_df.columns:
        latest = qbo_txn_df["txn_date"].max()
        if pd.notna(latest):
            days_ago = (pd.Timestamp.now() - latest).days
            freshness_data.append({
                "Data Source": "QBO Transactions",
                "Latest Record": latest.strftime("%Y-%m-%d"),
                "Days Ago": days_ago
            })

    if not qbo_gl_df.empty and "txn_date" in qbo_gl_df.columns:
        latest = qbo_gl_df["txn_date"].max()
        if pd.notna(latest):
            days_ago = (pd.Timestamp.now() - latest).days
            freshness_data.append({
                "Data Source": "QBO General Ledger",
                "Latest Record": latest.strftime("%Y-%m-%d"),
                "Days Ago": days_ago
            })

    if freshness_data:
        st.dataframe(pd.DataFrame(freshness_data), width='stretch', hide_index=True)
    else:
        st.info("No freshness data available")

# ----------------------------
# Footer
# ----------------------------
st.divider()
st.caption(f"Dashboard last updated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')} | Data sources: Supabase (deals, loan_summaries, qbo_invoice_payments, qbo_general_ledger)")
