# Testing Checklist for Streamlit Refactoring

## Overview
This document provides a comprehensive testing plan for the Phase 1 and Phase 2 refactoring changes.

---

## 🔴 Phase 1 Changes - CRITICAL TESTS

### 1. Theme Consistency Test
**What Changed:** Fixed x_QA_audit.py theme issues, created setup_page() utility

**Test Steps:**
1. Start the Streamlit app: `streamlit run streamlit_app.py`
2. Navigate to each page and verify:
   - ✅ Logo appears in top-left corner
   - ✅ Green theme colors are consistent
   - ✅ Page title appears in browser tab
   - ✅ No duplicate page config errors

**Pages to Test:**
- [ ] Main Dashboard (`streamlit_app.py`)
- [ ] Loan Tape (`pages/loan_tape.py`)
- [ ] Capital Forecast (`pages/capital_forecast.py`)
- [ ] QBO Dashboard (`pages/qbo_dashboard.py`)
- [ ] QA Audit (`pages/x_QA_audit.py`) - **CRITICAL** (had the bug)
- [ ] QA Debugger (`pages/x_qa_debugger.py`)

**Expected Result:** All pages should have identical branding and no errors.

---

### 2. Data Loading Test
**What Changed:** Consolidated all data loading into `utils/data_loader.py`

**Test Steps:**
1. Open each page and verify data loads correctly
2. Check browser console for errors (F12)
3. Verify data freshness indicators work
4. Test cache refresh buttons (if present)

**Data Sources to Verify:**
- [ ] Deals data loads (`load_deals()`)
- [ ] MCA deals load (`load_mca_deals()`)
- [ ] Combined MCA deals load (`load_combined_mca_deals()`)
- [ ] QBO data loads (`load_qbo_data()`)
- [ ] Loan summaries load (`load_loan_summaries()`)
- [ ] Loan schedules load (`load_loan_schedules()`)
- [ ] NAICS risk data loads (`load_naics_sector_risk()`)

**Expected Result:** All data loads without errors, same data as before refactoring.

---

## 🟢 Phase 2 Changes - FUNCTIONALITY TESTS

### 3. Preprocessing Test (x_QA_audit.py)
**What Changed:** Replaced custom `preprocess_data()` with `preprocess_dataframe()`

**Test Steps:**
1. Navigate to QA Audit page
2. Verify all sections load:
   - [ ] Executive Summary metrics display
   - [ ] MCA Deals Audit section works
   - [ ] Deal Data Audit section works
   - [ ] QBO Financial Data Analysis works
   - [ ] Cross-Dataset Analysis works
   - [ ] System Health Summary shows

**Data to Verify:**
- [ ] Numeric columns are converted correctly (amounts, balances)
- [ ] Date columns display properly (formatting)
- [ ] No new null values introduced
- [ ] Calculations match previous results

**Expected Result:** Page works identically to before, no data corruption.

---

### 4. New Utility Modules - Import Test

**Test Steps:**
Create a test Python file to verify imports work:

```python
# test_imports.py
import sys
sys.path.append('/home/user/streamlit')

# Test all new modules can be imported
from utils.loan_tape_data import prepare_loan_data, calculate_irr
from utils.loan_tape_analytics import compute_correlations
from utils.loan_tape_ml import train_classification_small
from utils.preprocessing import preprocess_dataframe
from utils.display_components import display_metric_row

print("✅ All imports successful!")
```

Run: `python test_imports.py`

**Expected Result:** No import errors.

---

## 📊 Functional Testing Matrix

### Page-by-Page Testing

#### Main Dashboard (streamlit_app.py)
- [ ] Page loads without errors
- [ ] Data displays correctly
- [ ] Metrics show proper values
- [ ] Charts render
- [ ] No console errors

#### Loan Tape (pages/loan_tape.py)
- [ ] Page loads without errors
- [ ] Loan data displays
- [ ] Risk calculations work
- [ ] IRR calculations display
- [ ] Charts render correctly
- [ ] ML models train (if applicable)

#### Capital Forecast (pages/capital_forecast.py)
- [ ] Page loads without errors
- [ ] Forecast calculations work
- [ ] QBO data integrates properly
- [ ] Charts display
- [ ] Export functions work

#### QBO Dashboard (pages/qbo_dashboard.py)
- [ ] Page loads without errors
- [ ] Transaction data displays
- [ ] General ledger shows
- [ ] Analysis tabs work
- [ ] Filters function correctly

#### QA Audit (pages/x_QA_audit.py)
- [ ] Page loads without errors
- [ ] All data sections populate
- [ ] Preprocessing works correctly
- [ ] No duplicate page config error
- [ ] Download buttons work
- [ ] Cache refresh works

#### QA Debugger (pages/x_qa_debugger.py)
- [ ] Page loads without errors
- [ ] Debug info displays
- [ ] Data import test works

---

## 🔍 Regression Testing

### Data Integrity Checks

**Create a validation script:**

```python
# validate_data.py
import pandas as pd
from utils.data_loader import load_deals, load_qbo_data, load_mca_deals

# Load data
deals = load_deals()
qbo_txn, qbo_gl = load_qbo_data()
mca = load_mca_deals()

# Validation checks
print("Deals count:", len(deals))
print("QBO transactions:", len(qbo_txn))
print("MCA deals:", len(mca))

# Check for expected columns
assert 'loan_id' in deals.columns, "Missing loan_id in deals"
assert 'total_amount' in qbo_txn.columns, "Missing total_amount in QBO"
assert 'deal_number' in mca.columns, "Missing deal_number in MCA"

print("✅ All validation checks passed!")
```

Run: `cd /home/user/streamlit && python validate_data.py`

---

## 🚨 Error Scenarios to Test

### 1. Missing Data Scenarios
- [ ] What happens if Supabase is unreachable?
- [ ] Are error messages user-friendly?
- [ ] Does caching handle failures gracefully?

### 2. Edge Cases
- [ ] Empty datasets (no records returned)
- [ ] Very large datasets (performance)
- [ ] Null values in key columns
- [ ] Invalid date formats
- [ ] Missing expected columns

### 3. Browser Compatibility
- [ ] Chrome
- [ ] Firefox
- [ ] Safari
- [ ] Edge

---

## 📝 Performance Testing

### Load Time Comparison

**Before Refactoring:**
1. Record page load times for each page
2. Note any lag or delays

**After Refactoring:**
1. Compare load times
2. Check if caching works properly
3. Monitor memory usage

**Metrics to Track:**
- Initial page load time
- Data refresh time
- Chart rendering time
- Memory usage (check browser dev tools)

---

## ✅ Acceptance Criteria

Mark complete when ALL of the following are true:

- [ ] All 6 pages load without errors
- [ ] Theme is consistent across all pages
- [ ] No duplicate page config errors
- [ ] All data loads correctly
- [ ] No data corruption or missing values
- [ ] Charts and visualizations render
- [ ] Download buttons work
- [ ] No new console errors
- [ ] Performance is same or better
- [ ] All interactive features work
- [ ] Cache refresh functions work

---

## 🐛 Bug Reporting Template

If you find issues, document them with this format:

```markdown
**Bug:** [Brief description]
**Page:** [Which page]
**Steps to Reproduce:**
1. [Step 1]
2. [Step 2]

**Expected:** [What should happen]
**Actual:** [What actually happens]
**Error Message:** [Any error messages]
**Console Logs:** [Browser console errors]
```

---

## 🔧 Quick Fixes

### If you encounter import errors:
```bash
cd /home/user/streamlit
python -c "import sys; sys.path.append('.'); from utils.preprocessing import preprocess_dataframe; print('OK')"
```

### If you encounter Streamlit errors:
```bash
# Clear Streamlit cache
rm -rf ~/.streamlit/cache
streamlit cache clear
```

### If data doesn't load:
1. Check Supabase connection
2. Verify tables exist
3. Check cache TTL settings
4. Try manual cache clear

---

## 📞 Support

If you encounter issues during testing:

1. Check browser console (F12) for JavaScript errors
2. Check terminal for Python errors
3. Review the error messages carefully
4. Check if the issue existed before refactoring
5. Document the bug using the template above

---

## Next Steps After Testing

Once testing is complete:

✅ **If all tests pass:**
- Mark this checklist complete
- Proceed with Phase 3 (optional)
- Consider merging to main branch

❌ **If issues found:**
- Document all bugs
- Prioritize by severity
- Fix critical issues first
- Re-test after fixes

---

**Testing Started:** _____________
**Testing Completed:** _____________
**Tested By:** _____________
**Overall Result:** [ ] PASS / [ ] FAIL
**Notes:**
