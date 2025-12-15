#!/usr/bin/env python3
"""
Analyze mismatch between loan_schedules.actual_payment and loan_summaries.total_paid.

This script investigates:
1. Loans where sum(actual_payment) differs from total_paid by >5%
2. Schedule records with actual_payment = 0 or NULL while status is "Scheduled"
3. Potential root causes for the gap

Run: streamlit run scripts/analyze_payment_mismatch.py
  OR in Python environment with database access:
     python scripts/analyze_payment_mismatch.py
"""

import pandas as pd
import numpy as np
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Try to import DataLoader, fall back to direct Supabase if available
try:
    from utils.data_loader import DataLoader
    HAS_DATA_LOADER = True
except ImportError:
    HAS_DATA_LOADER = False
    print("Warning: Could not import DataLoader. Install required dependencies.")
    print("Run: pip install -r requirements.txt")


def normalize_loan_id(series: pd.Series) -> pd.Series:
    """Normalize loan_id for consistent matching."""
    if series.empty:
        return series
    result = series.astype(str).str.strip()
    result = result.str.replace(r'\.0$', '', regex=True)
    result = result.replace(['nan', 'None', 'NaN', ''], pd.NA)
    return result


def analyze_payment_mismatch():
    """Main analysis function."""
    print("=" * 80)
    print("LOAN PAYMENT MISMATCH ANALYSIS")
    print("=" * 80)

    if not HAS_DATA_LOADER:
        print("\nERROR: Cannot run analysis without database access.")
        print("Please ensure you have .streamlit/secrets.toml configured.")
        return None

    # Load data
    loader = DataLoader()

    print("\n[1] Loading data...")
    summaries_df = loader.load_loan_summaries()
    schedules_df = loader.load_loan_schedules()

    print(f"    Loaded {len(summaries_df)} loan summaries")
    print(f"    Loaded {len(schedules_df)} loan schedules")

    if summaries_df.empty or schedules_df.empty:
        print("ERROR: Could not load data. Check database connection.")
        return

    # Normalize loan_ids
    summaries_df["loan_id"] = normalize_loan_id(summaries_df["loan_id"])
    schedules_df["loan_id"] = normalize_loan_id(schedules_df["loan_id"])

    # Convert numeric columns
    summaries_df["total_paid"] = pd.to_numeric(summaries_df.get("total_paid"), errors="coerce").fillna(0)
    schedules_df["actual_payment"] = pd.to_numeric(schedules_df.get("actual_payment"), errors="coerce").fillna(0)
    schedules_df["expected_payment"] = pd.to_numeric(schedules_df.get("expected_payment"), errors="coerce").fillna(0)

    # =========================================================================
    # ANALYSIS 1: Compare total_paid vs sum(actual_payment) by loan
    # =========================================================================
    print("\n" + "=" * 80)
    print("[2] COMPARING total_paid (summaries) vs sum(actual_payment) (schedules)")
    print("=" * 80)

    # Aggregate actual_payment by loan
    schedule_totals = schedules_df.groupby("loan_id").agg(
        schedule_sum=("actual_payment", "sum"),
        schedule_count=("actual_payment", "count"),
        paid_count=("actual_payment", lambda x: (x > 0).sum()),
        zero_or_null_count=("actual_payment", lambda x: (x == 0).sum() + x.isna().sum()),
    ).reset_index()

    # Merge with summaries
    comparison_df = summaries_df[["loan_id", "total_paid", "loan_status"]].merge(
        schedule_totals,
        on="loan_id",
        how="left"
    )

    # Calculate mismatch
    comparison_df["schedule_sum"] = comparison_df["schedule_sum"].fillna(0)
    comparison_df["difference"] = comparison_df["total_paid"] - comparison_df["schedule_sum"]
    comparison_df["pct_diff"] = np.where(
        comparison_df["total_paid"] > 0,
        abs(comparison_df["difference"]) / comparison_df["total_paid"],
        np.where(comparison_df["schedule_sum"] > 0, 1, 0)
    )

    # Summary stats
    total_loans = len(comparison_df)
    loans_with_schedules = comparison_df["schedule_count"].notna().sum()
    loans_without_schedules = total_loans - loans_with_schedules

    print(f"\n    Total loans in summaries: {total_loans}")
    print(f"    Loans with schedules: {loans_with_schedules}")
    print(f"    Loans without schedules: {loans_without_schedules}")

    # Identify mismatched loans (>5% difference)
    mismatch_threshold = 0.05
    mismatched = comparison_df[
        (comparison_df["pct_diff"] > mismatch_threshold) &
        (comparison_df["total_paid"] > 0)  # Only count loans with payments
    ].copy()

    print(f"\n    Loans with >5% mismatch: {len(mismatched)} ({len(mismatched)/total_loans*100:.1f}%)")

    if not mismatched.empty:
        total_discrepancy = mismatched["difference"].sum()
        print(f"    Total $ discrepancy: ${total_discrepancy:,.2f}")

        # Breakdown by status
        print("\n    Mismatch by loan status:")
        status_breakdown = mismatched.groupby("loan_status").agg(
            count=("loan_id", "count"),
            total_diff=("difference", "sum")
        ).sort_values("count", ascending=False)
        for status, row in status_breakdown.iterrows():
            print(f"      - {status}: {row['count']} loans, ${row['total_diff']:,.2f} diff")

    # =========================================================================
    # ANALYSIS 2: Schedule records with actual_payment = 0/NULL but status = "Scheduled"
    # =========================================================================
    print("\n" + "=" * 80)
    print("[3] SCHEDULE RECORDS WITH MISSING actual_payment")
    print("=" * 80)

    # Parse payment_date
    schedules_df["payment_date"] = pd.to_datetime(schedules_df["payment_date"], errors="coerce")
    today = pd.Timestamp.today()
    if schedules_df["payment_date"].dt.tz is not None:
        schedules_df["payment_date"] = schedules_df["payment_date"].dt.tz_localize(None)

    # Filter to past-due schedules
    past_due_schedules = schedules_df[schedules_df["payment_date"] <= today].copy()

    # Identify records with actual_payment = 0 or NULL
    missing_actual = past_due_schedules[
        (past_due_schedules["actual_payment"] == 0) |
        (past_due_schedules["actual_payment"].isna())
    ]

    print(f"\n    Total past-due schedule records: {len(past_due_schedules)}")
    print(f"    Records with actual_payment = 0 or NULL: {len(missing_actual)} ({len(missing_actual)/len(past_due_schedules)*100:.1f}%)")

    if "status" in schedules_df.columns:
        print("\n    Status distribution for missing actual_payment:")
        status_dist = missing_actual["status"].value_counts()
        for status, count in status_dist.items():
            print(f"      - {status}: {count}")

    # How much expected payment is missing actual?
    missing_expected_total = missing_actual["expected_payment"].sum()
    print(f"\n    Expected payment total for missing records: ${missing_expected_total:,.2f}")

    # Unique loans affected
    affected_loans = missing_actual["loan_id"].nunique()
    print(f"    Unique loans affected: {affected_loans}")

    # =========================================================================
    # ANALYSIS 3: Deep dive on mismatched loans
    # =========================================================================
    print("\n" + "=" * 80)
    print("[4] DEEP DIVE: TOP 20 MISMATCHED LOANS")
    print("=" * 80)

    top_mismatched = mismatched.nlargest(20, "difference").copy()

    print("\n    Loan ID              | Status         | total_paid  | schedule_sum | Difference   | % Diff")
    print("    " + "-" * 95)

    for _, row in top_mismatched.iterrows():
        print(f"    {str(row['loan_id'])[:20]:<20} | {str(row['loan_status'])[:14]:<14} | ${row['total_paid']:>10,.2f} | ${row['schedule_sum']:>10,.2f} | ${row['difference']:>10,.2f} | {row['pct_diff']*100:>5.1f}%")

    # =========================================================================
    # ANALYSIS 4: Check for loans with total_paid but NO schedules
    # =========================================================================
    print("\n" + "=" * 80)
    print("[5] LOANS WITH PAYMENTS BUT NO SCHEDULE RECORDS")
    print("=" * 80)

    loans_with_payments = comparison_df[comparison_df["total_paid"] > 0]
    loans_no_schedules = loans_with_payments[loans_with_payments["schedule_count"].isna()]

    print(f"\n    Loans with total_paid > 0 but no schedules: {len(loans_no_schedules)}")
    if not loans_no_schedules.empty:
        print(f"    Total $ in these loans: ${loans_no_schedules['total_paid'].sum():,.2f}")

        print("\n    By status:")
        status_dist = loans_no_schedules.groupby("loan_status")["total_paid"].agg(["count", "sum"])
        for status, row in status_dist.iterrows():
            print(f"      - {status}: {row['count']} loans, ${row['sum']:,.2f}")

    # =========================================================================
    # ANALYSIS 5: Pattern analysis - WHERE is the gap?
    # =========================================================================
    print("\n" + "=" * 80)
    print("[6] GAP PATTERN ANALYSIS")
    print("=" * 80)

    # For mismatched loans, check: is schedule_sum < total_paid or vice versa?
    undercounted = mismatched[mismatched["difference"] > 0]  # total_paid > schedule_sum
    overcounted = mismatched[mismatched["difference"] < 0]   # schedule_sum > total_paid

    print(f"\n    Schedules UNDERCOUNTING (total_paid > schedule_sum): {len(undercounted)} loans")
    print(f"    Total undercounted: ${undercounted['difference'].sum():,.2f}")

    print(f"\n    Schedules OVERCOUNTING (schedule_sum > total_paid): {len(overcounted)} loans")
    print(f"    Total overcounted: ${abs(overcounted['difference'].sum()):,.2f}")

    # =========================================================================
    # ANALYSIS 6: Check QBO data for unmatched payments
    # =========================================================================
    print("\n" + "=" * 80)
    print("[7] QBO PAYMENTS ANALYSIS")
    print("=" * 80)

    try:
        qbo_payments, _ = loader.load_qbo_data()
        qbo_payments["loan_id"] = normalize_loan_id(qbo_payments["loan_id"])
        qbo_payments["total_amount"] = pd.to_numeric(qbo_payments["total_amount"], errors="coerce")

        # Filter to payment types
        payment_types = ["Payment", "Deposit", "Receipt"]
        if "transaction_type" in qbo_payments.columns:
            qbo_filtered = qbo_payments[qbo_payments["transaction_type"].isin(payment_types)]
        else:
            qbo_filtered = qbo_payments

        # Aggregate by loan_id
        qbo_by_loan = qbo_filtered.groupby("loan_id")["total_amount"].sum().reset_index()
        qbo_by_loan.columns = ["loan_id", "qbo_total"]

        # Compare with summaries
        three_way = summaries_df[["loan_id", "total_paid"]].merge(
            schedule_totals[["loan_id", "schedule_sum"]],
            on="loan_id",
            how="left"
        ).merge(
            qbo_by_loan,
            on="loan_id",
            how="left"
        )
        three_way = three_way.fillna(0)

        # Check QBO vs total_paid
        qbo_diff = (three_way["total_paid"] - three_way["qbo_total"]).abs()
        qbo_close = (qbo_diff / three_way["total_paid"].replace(0, 1)) < 0.05

        print(f"\n    Total loans with QBO data: {(three_way['qbo_total'] > 0).sum()}")
        print(f"    Loans where QBO ≈ total_paid (within 5%): {qbo_close.sum()}")
        print(f"    Loans where QBO differs from total_paid: {(~qbo_close & (three_way['total_paid'] > 0)).sum()}")

    except Exception as e:
        print(f"    Could not analyze QBO data: {e}")

    # =========================================================================
    # SUMMARY & RECOMMENDATIONS
    # =========================================================================
    print("\n" + "=" * 80)
    print("[8] SUMMARY & RECOMMENDATIONS")
    print("=" * 80)

    print("\n    ROOT CAUSE HYPOTHESIS:")
    print("    " + "-" * 60)

    if len(undercounted) > len(overcounted):
        print("""
    The primary issue is UNDERCOUNTING in loan_schedules.actual_payment.

    This supports the hypothesis that:
    - total_paid in loan_summaries reflects ALL QBO payments received
    - actual_payment in schedules only reflects payments that matched
      to a specific schedule record during reconciliation
    - Unmatched payments (early payoffs, lump sums, irregular amounts)
      are NOT being populated into schedule records
    """)
    else:
        print("""
    The data shows mixed issues - some overcounting, some undercounting.
    Further investigation needed to determine root cause.
    """)

    print("\n    RECOMMENDATIONS:")
    print("    " + "-" * 60)
    print("""
    1. RE-RUN RECONCILIATION: Run reconcile_loan_payments.py to update
       actual_payment values for all loans.

    2. FIX MATCHING LOGIC: Review the payment matching algorithm to handle:
       - Early payoff lump sums
       - Irregular payment amounts
       - Payments that don't align with expected schedule dates

    3. BACKFILL MISSING DATA: For loans where total_paid > schedule_sum,
       create a script to backfill the "missing" payments into schedules.

    4. VALIDATION CHECK: Add a validation step to reconciliation that
       flags loans where sum(actual_payment) != total_paid.

    5. IRR CALCULATION: Consider using total_paid from summaries as the
       authoritative source for IRR calculation, rather than schedule sum.
    """)

    # Export affected loans
    print("\n    Exporting affected loans to CSV...")
    output_path = os.path.join(os.path.dirname(__file__), "payment_mismatch_analysis.csv")
    mismatched.to_csv(output_path, index=False)
    print(f"    Saved: {output_path}")

    return {
        "total_loans": total_loans,
        "mismatched_count": len(mismatched),
        "total_discrepancy": mismatched["difference"].sum() if not mismatched.empty else 0,
        "undercounted": len(undercounted),
        "overcounted": len(overcounted),
        "loans_without_schedules": loans_without_schedules,
        "missing_actual_records": len(missing_actual),
    }


if __name__ == "__main__":
    results = analyze_payment_mismatch()
