# Maturity Date Implementation Plan
**Date**: 2025-11-08
**Status**: In Progress

---

## Overview

This plan implements canonical maturity date handling with HubSpot as the single source of truth.
All changes maintain backward compatibility and follow an incremental rollout strategy.

---

## Phase 1: Foundation (COMPLETED ✅)

### 1.1 Create Canonical Maturity API ✅
**File**: `services/maturity.py`
**Status**: Complete

**Deliverables**:
- [x] `MaturityInfo` dataclass with full metadata
- [x] `MaturityService` class with canonical methods:
  - `resolve_maturity()` - Extract maturity from HubSpot record
  - `select_amendment()` - Deterministic amendment selection
  - `validate_maturity()` - Quality validation
  - `days_to_maturity()` - Calculate days to/from maturity
  - `bucket_maturity()` - Bucket into time ranges
  - `calculate_days_past_maturity()` - For risk scoring
  - `calculate_remaining_maturity_months()` - For portfolio metrics

**Tests**:
- [ ] Unit tests for each method (5 test cases each)
- [ ] Property tests for deterministic amendment selection
- [ ] Edge case tests (None values, leap days, extreme dates)

---

### 1.2 Define Storage Schema ✅
**File**: `SCHEMA_MIGRATION.sql`
**Status**: Complete (needs to be run in Supabase)

**Deliverables**:
- [x] Schema extensions for `loan_summaries`:
  - `maturity_basis` VARCHAR(20)
  - `maturity_version` INT
  - `maturity_last_updated_at` TIMESTAMPTZ
  - `maturity_quality` VARCHAR(20)
  - `maturity_source` VARCHAR(100)
  - `amendment_id` VARCHAR(50)
- [x] New `maturity_dq_log` table for issue tracking
- [x] `maturity_dq_summary` view for reporting
- [x] Indexes for performance
- [x] Backfill script for existing data validation

**Action Required**:
Run `SCHEMA_MIGRATION.sql` in Supabase SQL editor

---

## Phase 2: Refactor Existing Code (IN PROGRESS ⏳)

### 2.1 Refactor Data Preparation Layer
**File**: `utils/loan_tape_data.py`
**Status**: Pending
**Impact**: Core data processing logic

**Changes**:
1. Import `MaturityService` from `services.maturity`
2. Update `prepare_loan_data()` to use canonical API:
   - Replace raw `pd.to_datetime(maturity_date)` with `MaturityService.resolve_maturity()`
   - Use `MaturityService.calculate_remaining_maturity_months()` instead of manual calculation
   - Use `MaturityService.calculate_days_past_maturity()` for risk scoring
3. Add maturity quality validation and logging

**Acceptance Criteria**:
- [ ] All maturity operations use `MaturityService`
- [ ] No raw `pd.to_datetime()` on maturity_date
- [ ] DQ issues logged to `maturity_dq_log`
- [ ] Backward compatible (no breaking changes)

---

### 2.2 Refactor QA Dashboard
**File**: `pages/loan_qa.py`
**Status**: Pending
**Impact**: QA checks and validation

**Changes**:
1. Import `MaturityService` from `services.maturity`
2. Update QA functions to use canonical API:
   - `identify_date_issues()` → use `MaturityService.validate_maturity()`
   - `identify_expiring_soon()` → use `MaturityService.days_to_maturity()` and `bucket_maturity()`
   - `identify_matured_not_paid()` → use `MaturityService.calculate_days_past_maturity()`
3. Add new DQ panel showing maturity quality distribution

**Acceptance Criteria**:
- [ ] All maturity validations use `MaturityService`
- [ ] New "Maturity Quality" tab added to QA dashboard
- [ ] Shows counts by maturity_quality flag
- [ ] Links to maturity_dq_log for details

---

### 2.3 Refactor Loan Tape Display
**File**: `pages/loan_tape.py`
**Status**: Pending
**Impact**: Display and overdue checks

**Changes**:
1. Import `MaturityService` from `services.maturity`
2. Update overdue check to use `MaturityService.calculate_days_past_maturity()`
3. Add maturity quality indicator in loan details
4. Use `MaturityService.bucket_maturity()` for maturity grouping

**Acceptance Criteria**:
- [ ] Overdue checks use canonical API
- [ ] Maturity quality flags visible in UI
- [ ] No direct datetime comparisons for maturity

---

## Phase 3: Data Quality Tracking (PENDING ⏹️)

### 3.1 Add DQ Logging Helper
**File**: `services/maturity_dq.py` (new)
**Status**: Pending

**Deliverables**:
- [ ] `MaturityDQLogger` class with methods:
  - `log_issue()` - Log DQ issue to maturity_dq_log
  - `log_change()` - Log maturity date change
  - `get_unresolved_issues()` - Fetch unresolved issues
  - `resolve_issue()` - Mark issue as resolved
- [ ] Integration with Supabase

**Acceptance Criteria**:
- [ ] All DQ issues logged to database
- [ ] Includes old_value, new_value, issue description
- [ ] Auto-detects issue_type based on validation

---

### 3.2 Add Monitoring/Observability
**File**: `services/maturity_monitoring.py` (new)
**Status**: Pending

**Deliverables**:
- [ ] Metrics tracking:
  - `maturity_updates_total` counter
  - `maturity_invalid_total` counter
  - `maturity_missing_total` counter
  - `maturity_backdate_total` counter
- [ ] Structured logging for maturity operations
- [ ] Secret masking in logs

**Acceptance Criteria**:
- [ ] Counters emitted for each operation
- [ ] Logs include loan_id, quality, basis
- [ ] No secrets exposed in logs

---

## Phase 4: Testing (PENDING ⏹️)

### 4.1 Unit Tests
**File**: `tests/test_maturity_service.py` (new)
**Status**: Pending

**Test Coverage**:
- [ ] Test `resolve_maturity()` with 5 fixtures:
  1. Base case (original maturity)
  2. Amended maturity
  3. Renewal/extension
  4. Early payoff
  5. Invalid/missing maturity
- [ ] Test `select_amendment()` determinism
- [ ] Test `validate_maturity()` edge cases
- [ ] Test date parsing edge cases (None, invalid formats)
- [ ] Test bucket_maturity() boundaries

**Acceptance Criteria**:
- [ ] 95%+ code coverage on `services/maturity.py`
- [ ] All edge cases covered
- [ ] Property tests for determinism

---

### 4.2 Integration Tests
**File**: `tests/test_maturity_integration.py` (new)
**Status**: Pending

**Test Coverage**:
- [ ] End-to-end: Load loan → Prepare data → Validate maturity
- [ ] DQ logging: Invalid maturity → Logged to maturity_dq_log
- [ ] Amendment selection: Multiple amendments → Correct one selected

**Acceptance Criteria**:
- [ ] Tests use fixtures from `tests/fixtures/`
- [ ] Tests verify database state changes
- [ ] Tests verify DQ log entries

---

## Phase 5: Integration & UI (PENDING ⏹️)

### 5.1 Add Maturity DQ Panel to Streamlit
**File**: `pages/loan_qa.py` (update)
**Status**: Pending

**Changes**:
1. Add new tab "Maturity Data Quality"
2. Show summary metrics:
   - Total loans with maturity
   - By quality status (ok, missing, invalid, backdated)
   - By basis (original, amended, renewed, extended)
3. Show recent DQ issues from `maturity_dq_log`
4. Allow filtering and exporting issues

**Acceptance Criteria**:
- [ ] New tab shows maturity quality distribution
- [ ] Charts show trends over time
- [ ] Drill-down to individual issues
- [ ] Export to CSV

---

### 5.2 Update README
**File**: `README.md` (new/update)
**Status**: Pending

**Content**:
- [ ] "Maturity Date Contract" section:
  - HubSpot is sole source of truth
  - Amendment precedence rules
  - Quality flags and meanings
  - DQ tracking process
- [ ] Schema documentation
- [ ] API usage examples
- [ ] Known caveats and limitations

**Acceptance Criteria**:
- [ ] Clear documentation of maturity contract
- [ ] Examples for common use cases
- [ ] Links to relevant code sections

---

## Phase 6: Deployment (PENDING ⏹️)

### 6.1 Database Migration
**Actions**:
1. [ ] Run `SCHEMA_MIGRATION.sql` in Supabase (non-breaking)
2. [ ] Verify backfill results
3. [ ] Check maturity_quality distribution
4. [ ] Spot-check DQ log entries

**Rollback Plan**:
- Rollback SQL provided in `SCHEMA_MIGRATION.sql`
- No data loss (only adds columns)

---

### 6.2 Code Deployment
**Actions**:
1. [ ] Deploy canonical API (services/maturity.py)
2. [ ] Deploy refactored data preparation (utils/loan_tape_data.py)
3. [ ] Deploy refactored QA page (pages/loan_qa.py)
4. [ ] Deploy DQ tracking (services/maturity_dq.py)
5. [ ] Monitor for errors

**Feature Flags**:
- [ ] `USE_CANONICAL_MATURITY_API` (default: True)
- Allows fallback to old logic if issues arise

---

## Acceptance Criteria (Overall)

### Must-Have (Blocking)
- [x] Canonical `MaturityService` API created
- [x] Schema migration script created
- [ ] All existing code refactored to use canonical API
- [ ] DQ logging functional
- [ ] No regression in existing functionality
- [ ] Tests passing with >90% coverage

### Should-Have (Important)
- [ ] Maturity DQ panel in Streamlit
- [ ] Monitoring/metrics emitted
- [ ] Documentation complete
- [ ] Amendment handling documented

### Nice-to-Have (Optional)
- [ ] Backfill script for historical data
- [ ] Automated DQ alerts
- [ ] Maturity trend analysis

---

## Timeline

| Phase | Duration | Status |
|-------|----------|--------|
| 1. Foundation | 1 day | ✅ Complete |
| 2. Refactor Code | 1-2 days | ⏳ In Progress |
| 3. DQ Tracking | 1 day | ⏹️ Pending |
| 4. Testing | 1-2 days | ⏹️ Pending |
| 5. Integration | 1 day | ⏹️ Pending |
| 6. Deployment | 1 day | ⏹️ Pending |
| **Total** | **6-9 days** | |

---

## Risks & Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| HubSpot extraction not in this repo | Can't implement end-to-end | Document requirements for external ETL team |
| Breaking changes to API | Downstream failures | Maintain backward compatibility, feature flags |
| Performance degradation | Slow queries | Add indexes, cache maturity info |
| Existing data quality issues | Many DQ log entries | Expected - that's the point! Document baseline |

---

## Next Actions

1. ⏭️ **Refactor `utils/loan_tape_data.py`** to use `MaturityService`
2. ⏭️ **Refactor `pages/loan_qa.py`** to use `MaturityService`
3. ⏭️ **Create tests** for `MaturityService`
4. ⏭️ **Add DQ logging** to Supabase
5. ⏭️ **Run schema migration** in Supabase
6. ⏭️ **Create PR** with all changes

---

**Status**: Phase 1 Complete, Phase 2 In Progress
**Next Review**: After Phase 2 completion
