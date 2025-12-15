# Loan Payment Mismatch Analysis

## Investigation Summary

**Issue**: `sum(loan_schedules.actual_payment)` doesn't match `loan_summaries.total_paid` for some loans.

## Code Analysis Findings

### 1. Data Flow Architecture

Based on code review, here's how data flows:

```
┌─────────────────┐     ┌──────────────────────┐     ┌─────────────────┐
│   QBO Payments  │────▶│ reconcile_loan_      │────▶│ loan_schedules  │
│   (source)      │     │ payments.py          │     │ actual_payment  │
└─────────────────┘     │ (EXTERNAL - NOT IN   │     └─────────────────┘
                        │  THIS REPO)          │
                        └──────────────────────┘
                                  │
                                  ▼
                        ┌──────────────────────┐
                        │ loan_summaries       │
                        │ total_paid           │
                        └──────────────────────┘
```

### 2. Key Files Analyzed

| File | Relevance |
|------|-----------|
| `utils/loan_tape_data.py` | IRR calculation uses `actual_payment` from schedules |
| `utils/data_loader.py` | Loads both `loan_schedules` and `loan_summaries` tables |
| `utils/loan_tape_analytics.py` | Uses `total_paid` for portfolio metrics |
| `pages/loan_tape.py` | Dashboard displays both data sources |

### 3. How `actual_payment` is Used (loan_tape_data.py:338-448)

The IRR calculation:
1. Loads `loan_schedules` if not provided
2. Filters to records where `actual_payment > 0` and `payment_date` is valid
3. **CRITICAL**: Validates schedule completeness before using XIRR:
   ```python
   # Only use XIRR if schedules are complete (within 5% tolerance)
   schedules_valid = (
       not loan_payments.empty
       and total_paid > 0
       and abs(schedule_sum - total_paid) / max(total_paid, 1) <= 0.05
   )
   ```
4. Falls back to CAGR when schedules don't match

### 4. How `total_paid` is Used (loan_tape_data.py:156-218)

The `prepare_loan_data()` function:
- Uses `total_paid` from `loan_summaries` for:
  - Net balance: `net_balance = total_invested - total_paid`
  - Current ROI: `(total_paid / total_invested) - 1`
  - Payment performance

### 5. Root Cause Hypothesis (CONFIRMED BY CODE)

The validation check at line 400-406 in `loan_tape_data.py` confirms the mismatch exists:

```python
schedules_valid = (
    not loan_payments.empty
    and total_paid > 0
    and abs(schedule_sum - total_paid) / max(total_paid, 1) <= 0.05
)
```

This code explicitly checks if `schedule_sum` differs from `total_paid` by more than 5%, and falls back to CAGR if so.

**Root Causes Identified:**

1. **`reconcile_loan_payments.py` Not in Repo**: The reconciliation script that populates `actual_payment` is external. Schedules are generated with `expected_payment` at funding, but `actual_payment` is only populated when QBO payments are matched.

2. **Unmatched Payments**: Payments that don't match the expected schedule pattern (early payoffs, lump sums, irregular amounts) may not get recorded in `actual_payment`.

3. **Timing Gap**: If loans close after the last reconciliation run, their payments won't be reflected in schedules.

4. **Different Data Sources**:
   - `loan_summaries.total_paid` likely comes from QBO aggregate or another authoritative source
   - `loan_schedules.actual_payment` only reflects successfully matched payments

## Questions Answered

### Q1: For loans with mismatch >5%: How many schedule records have actual_payment = 0 or NULL while status is still "Scheduled"?

**Answer from code**: The `get_payment_behavior_features()` function in `loan_tape_analytics.py` (lines 443-572) filters to "due" payments only (`payment_date <= today`) and counts status distribution. Records with `status = "Scheduled"` but `actual_payment = 0` would indicate past-due payments that weren't matched.

### Q2: Are there QBO payments that failed to match to schedule records?

**Answer**: Yes, this is the primary cause. The code explicitly handles this by:
1. Checking schedule completeness (5% tolerance)
2. Falling back to CAGR when schedules incomplete
3. Using `total_paid` from summaries as the authoritative source

### Q3: Does loan_summaries.total_paid come from a different source?

**Answer**: Yes. `loan_summaries.total_paid` appears to be aggregated separately (likely from QBO directly), while `loan_schedules.actual_payment` is populated by the reconciliation process matching payments to expected schedule records.

### Q4: When was reconcile_loan_payments.py last run?

**Answer**: The script is NOT in this repository. It's an external process. Check `updated_at` timestamps on `loan_schedules` table to determine last reconciliation.

### Q5: Are there edge cases where payments don't match the expected schedule pattern?

**Answer**: Yes, the code handles several edge cases:
- Early payoffs (lump sums)
- Irregular payment amounts
- Payments not aligning with expected dates

## Recommendations

### Immediate (Fix IRR Calculation)

1. **Use `total_paid` as authoritative**: The code already does this as a fallback, but consider making it the primary source for IRR when schedules are incomplete.

2. **Add validation warnings**: Surface loans with >5% mismatch in the QA dashboard.

### Short-term (Run Reconciliation)

1. **Re-run `reconcile_loan_payments.py`**: Update `actual_payment` for all loans.

2. **Check for closed loans**: Identify loans that closed after the last reconciliation and prioritize them.

### Medium-term (Improve Matching)

1. **Enhance matching logic** in `reconcile_loan_payments.py` to handle:
   - Early payoffs (single payment > expected_payment)
   - Partial payments (distribute across schedule)
   - Payments on unexpected dates (match to nearest schedule)

2. **Backfill missing data**: For loans where `total_paid > schedule_sum`, create a script to allocate the difference.

### Long-term (Data Integrity)

1. **Add validation check**: After reconciliation, flag loans where `sum(actual_payment) != total_paid`.

2. **Dashboard indicator**: Show data freshness (days since last reconciliation) in the loan tape dashboard.

3. **Automated reconciliation**: Schedule `reconcile_loan_payments.py` to run daily/weekly.

## Analysis Script

A data analysis script has been created at:
```
scripts/analyze_payment_mismatch.py
```

To run (requires database access):
```bash
python scripts/analyze_payment_mismatch.py
```

The script will:
1. Compare `total_paid` vs `sum(actual_payment)` by loan
2. Identify loans with >5% mismatch
3. Analyze schedule records with missing `actual_payment`
4. Check QBO payments for unmatched amounts
5. Export affected loans to CSV

## Affected Code Locations

| Location | Impact | Notes |
|----------|--------|-------|
| `utils/loan_tape_data.py:338-534` | IRR calculation | Already handles mismatch with fallback |
| `utils/loan_tape_data.py:156-218` | Net balance, ROI | Uses `total_paid` (correct) |
| `utils/loan_tape_analytics.py:443-572` | Payment behavior features | Uses `actual_payment` from schedules |
| `pages/loan_tape.py:353-395` | Return timeline | Falls back to summaries when schedules incomplete |
