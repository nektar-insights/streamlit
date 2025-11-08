# Maturity Date Audit & Refactoring Plan
**Date**: 2025-11-08
**Goal**: Make `maturity_date` correct, traceable to HubSpot, and used consistently across Python modules and Streamlit views.

---

## 1. CONTEXTUALIZE: Current State Documentation

### Current Architecture Overview

The application is a Streamlit-based loan portfolio management system with the following data flow:

```
HubSpot (External ETL) → Supabase Tables → DataLoader → Streamlit Pages
```

#### 1.1 HubSpot Data Extraction

**Status**: ⚠️ **CRITICAL GAP IDENTIFIED**

- **No extraction code found in this repository**
- Data appears to be populated into Supabase via an **external ETL process** (not in this codebase)
- No incremental sync logic visible (no `updatedAt`/`lastModifiedDate` tracking)
- No API integration code with HubSpot found in:
  - `utils/` directory
  - `scripts/` directory
  - No `connectors/` or `hubspot/` modules exist

**Evidence**:
- `scripts/combine_hubspot_mca.py:18-22` - Only **reads** from Supabase `deals` table
- `utils/data_loader.py:113-140` - `load_deals()` fetches from Supabase, no HubSpot API calls
- All data loading functions use `supabase.table().select()` - no external API calls

**Ambiguity**: How does HubSpot data get into Supabase? Is there a separate ETL job?

#### 1.2 Maturity Date Persistence Layer

**Current Storage**:
- **Table**: `loan_summaries` (in Supabase)
- **Field**: `maturity_date` (datetime column)
- **Schema**: Simple datetime field with no metadata

**Missing Fields** (needed for traceability):
- ❌ `maturity_basis` (original vs amended)
- ❌ `maturity_version` (amendment tracking)
- ❌ `maturity_last_updated_at` (audit timestamp)
- ❌ `maturity_quality` (validation status)
- ❌ `maturity_source` (which HubSpot field?)
- ❌ `amendment_id` (reference to amendment record)

**Current Preprocessing**:
```python
# utils/loan_tape_data.py:54-56
for date_col in ["funding_date", "maturity_date", "payoff_date"]:
    if date_col in df.columns:
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
```
- Simple coercion to datetime
- No timezone handling (stored as naive datetime)
- No validation logging
- Errors silently converted to NaT

#### 1.3 Maturity Date Consumers

**Primary Consumers** (file:line references):

1. **`utils/loan_tape_data.py`**
   - `Line 54-56`: Normalizes maturity_date to datetime
   - `Line 89-98`: Calculates `remaining_maturity_months` from maturity_date
   - `Line 420-423`: Calculates `days_past_maturity` for risk scoring

2. **`pages/loan_qa.py`** (Extensive QA checks)
   - `Line 35-48`: Validates maturity_date > funding_date
   - `Line 50-72`: Identifies loans expiring in next 30 days
   - `Line 74-92`: Identifies loans past maturity (overdue)
   - Already has good DQ framework - can extend for HubSpot audit!

3. **`pages/loan_tape.py`**
   - `Line 930`: Displays maturity_date in loan details
   - `Line 969-972`: Checks if maturity_date < today for overdue flag

4. **`utils/preprocessing.py`**
   - `Line 64`: Includes maturity_date in default date column list

**Good News**: ✅ No shadow recomputation of maturity dates found
- Maturity is NOT derived from cash flows
- No alternative maturity calculation logic detected
- All consumers read the same `maturity_date` field

#### 1.4 Current Date/Time Handling

**Issues Identified**:
- **Timezone**: All dates stored as **naive datetime** (no timezone awareness)
- **Parsing**: Uses `pd.to_datetime(errors='coerce')` - silently fails
- **Validation**: No explicit validation of date ranges
- **Display**: Formatted as `YYYY-MM-DD` in Streamlit

**Recommendation**:
- Store as UTC date (DATE type, not DATETIME)
- Treat HubSpot date-only fields as end-of-day UTC for comparisons

---

## 2. AMBIGUITIES TO RESOLVE

### 2.1 Critical Questions (Must Answer Before Implementation)

| # | Question | Current State | Proposed Resolution |
|---|----------|---------------|---------------------|
| 1 | **What is the exact HubSpot field name for maturity?** | Unknown - no extraction code | ❓ **Need to identify**: Could be `closedate`, `maturity_date`, `deal_maturity`, or custom field |
| 2 | **Where is the HubSpot extraction code?** | Not in this repository | ❓ **Need to determine**: Separate ETL job? Zapier? Manual import? |
| 3 | **How are amendments/renewals handled?** | No amendment tracking found | **Proposed**: Add `deal_amendments` table with amendment history |
| 4 | **How are extensions handled?** | No extension logic found | **Proposed**: Extensions update maturity_date with audit trail |
| 5 | **What happens on early payoff?** | No special handling | **Proposed**: Track `actual_payoff_date` separately, keep original `maturity_date` |
| 6 | **Timezone handling for maturity dates?** | Naive datetime (no timezone) | **Proposed**: Store as DATE (UTC), no time component |
| 7 | **Policy for null/invalid maturity dates?** | Silently converted to NaT | **Proposed**: Log DQ issue, flag with `maturity_quality = 'invalid'` |
| 8 | **Can non-HubSpot sources override maturity?** | Unknown business rule | **Proposed**: No - HubSpot is sole source of truth |
| 9 | **Incremental sync strategy?** | No incremental sync found | **Proposed**: Use HubSpot `hs_lastmodifieddate` to pull only changed deals |
| 10 | **How to handle back-dated corrections?** | No versioning system | **Proposed**: Keep audit trail with `maturity_last_updated_at` |

### 2.2 Immediate Actions Required

**Before proceeding with implementation**:

1. ✅ **Identify HubSpot extraction mechanism**
   - Check for separate ETL repository
   - Review Supabase triggers/functions
   - Document data pipeline

2. ✅ **Identify exact HubSpot field mapping**
   - Query HubSpot API to see deal properties
   - Map HubSpot fields → Supabase columns
   - Document amendment structure in HubSpot

3. ✅ **Define amendment precedence rules**
   - How to select most recent amendment
   - What fields indicate an amendment vs. original deal
   - Document business logic

---

## 3. INTEGRATION MAP: Data Lineage

```
┌─────────────────────────────────────────────────────────────────────┐
│ HubSpot (Source of Truth)                                           │
│ ┌─────────────────┐      ┌──────────────────┐                      │
│ │ Deal Properties │      │ Deal Amendments  │                      │
│ │ - maturity_date │      │ - new_maturity   │                      │
│ │ - funding_date  │      │ - amendment_date │                      │
│ └─────────────────┘      └──────────────────┘                      │
└─────────────────────────────────────────────────────────────────────┘
                    │
                    ↓ [EXTERNAL ETL - Not in this repo]
┌─────────────────────────────────────────────────────────────────────┐
│ Supabase Storage                                                     │
│ ┌──────────────────┐  ┌──────────────────┐  ┌───────────────────┐  │
│ │ deals            │  │ loan_summaries   │  │ loan_schedules    │  │
│ │ - loan_id        │  │ - loan_id        │  │ - loan_id         │  │
│ │ - deal_name      │  │ - maturity_date  │  │ - payment_date    │  │
│ │ - partner_source │  │ - funding_date   │  │ - actual_payment  │  │
│ └──────────────────┘  │ - loan_status    │  └───────────────────┘  │
│                       └──────────────────┘                          │
└─────────────────────────────────────────────────────────────────────┘
                    │
                    ↓ [utils/data_loader.py]
┌─────────────────────────────────────────────────────────────────────┐
│ Data Loading Layer                                                   │
│ ┌──────────────────────────────────────────────────────────────────┐│
│ │ DataLoader.load_loan_summaries()                                 ││
│ │ DataLoader.load_deals()                                          ││
│ │ DataLoader.load_loan_schedules()                                 ││
│ └──────────────────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────────────┘
                    │
                    ↓ [utils/loan_tape_data.py]
┌─────────────────────────────────────────────────────────────────────┐
│ Data Preparation Layer                                               │
│ ┌──────────────────────────────────────────────────────────────────┐│
│ │ prepare_loan_data()                                              ││
│ │   - Normalizes maturity_date to datetime                         ││
│ │   - Calculates remaining_maturity_months                         ││
│ │   - Calculates days_past_maturity (for risk)                     ││
│ └──────────────────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────────────┘
                    │
                    ↓
┌─────────────────────────────────────────────────────────────────────┐
│ Streamlit Views (Consumers)                                          │
│ ┌─────────────────┐  ┌──────────────────┐  ┌───────────────────┐   │
│ │ loan_tape.py    │  │ loan_qa.py       │  │ capital_forecast  │   │
│ │ - Display       │  │ - QA Checks      │  │ - WAM/WA life     │   │
│ │ - Overdue flags │  │ - Expiring soon  │  │ - Maturity buckets│   │
│ │                 │  │ - Invalid dates  │  │                   │   │
│ └─────────────────┘  └──────────────────┘  └───────────────────┘   │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 4. FILE/LINE INDEX: All Maturity-Touching Locations

### Reading/Using maturity_date

| File | Lines | Operation | Notes |
|------|-------|-----------|-------|
| `utils/loan_tape_data.py` | 54-56 | Normalize to datetime | `pd.to_datetime(errors='coerce')` |
| `utils/loan_tape_data.py` | 89-98 | Calculate remaining_maturity_months | For active loans only |
| `utils/loan_tape_data.py` | 420-423 | Calculate days_past_maturity | For risk scoring |
| `pages/loan_qa.py` | 35-48 | Validate maturity > funding | QA check |
| `pages/loan_qa.py` | 50-72 | Identify expiring soon | QA check |
| `pages/loan_qa.py` | 74-92 | Identify past maturity | QA check |
| `pages/loan_qa.py` | 119, 228, 270 | Display formatting | Date columns |
| `pages/loan_tape.py` | 930, 950, 969-972 | Display and checks | Show maturity, check overdue |
| `utils/preprocessing.py` | 64 | Date column list | Default preprocessing |

### Loading maturity_date

| File | Lines | Operation | Notes |
|------|-------|-----------|-------|
| `utils/data_loader.py` | 193-209 | Load loan_summaries | Fetches from Supabase |
| `utils/data_loader.py` | 94-103 | Preprocess date columns | Converts to datetime |

### No recomputation found ✅

**Verified**: No code attempts to derive/recompute maturity_date from:
- Cash flow schedules
- Loan terms + funding_date
- Other date calculations

---

## 5. PROPOSED SOLUTION ARCHITECTURE

### 5.1 Canonical Maturity API (`services/maturity.py`)

```python
# services/maturity.py

from dataclasses import dataclass
from datetime import date
from typing import Optional
import pandas as pd

@dataclass
class MaturityInfo:
    """Canonical maturity information for a loan"""
    loan_id: str
    maturity_date: Optional[date]
    maturity_basis: str  # "original" | "amended" | "renewed" | "extended"
    maturity_version: int
    maturity_last_updated_at: pd.Timestamp
    maturity_quality: str  # "ok" | "missing" | "invalid" | "backdated"
    maturity_source: str  # HubSpot field name
    amendment_id: Optional[str]

def resolve_maturity(record: dict) -> Optional[date]:
    """Extract canonical maturity date from deal record"""

def select_amendment(amendments: list) -> Optional[dict]:
    """Select most recent valid amendment (deterministic)"""

def validate_maturity(mat_date: Optional[date], funding_date: Optional[date]) -> str:
    """Return quality flag: 'ok' | 'missing' | 'invalid' | 'backdated'"""

def days_to_maturity(as_of: date, mat_date: Optional[date]) -> Optional[int]:
    """Calculate days from as_of to maturity (negative if past)"""

def bucket_maturity(days: Optional[int]) -> str:
    """Bucket into: '0-30' | '31-60' | '61-90' | '>90' | 'past'"""
```

### 5.2 Storage Schema Extensions

**New columns for `loan_summaries` table**:
```sql
ALTER TABLE loan_summaries ADD COLUMN maturity_basis VARCHAR(20);
ALTER TABLE loan_summaries ADD COLUMN maturity_version INT DEFAULT 1;
ALTER TABLE loan_summaries ADD COLUMN maturity_last_updated_at TIMESTAMPTZ;
ALTER TABLE loan_summaries ADD COLUMN maturity_quality VARCHAR(20);
ALTER TABLE loan_summaries ADD COLUMN maturity_source VARCHAR(100);
ALTER TABLE loan_summaries ADD COLUMN amendment_id VARCHAR(50);
```

**New DQ tracking table**:
```sql
CREATE TABLE maturity_dq_log (
    id SERIAL PRIMARY KEY,
    loan_id VARCHAR(50) NOT NULL,
    issue_type VARCHAR(50) NOT NULL,
    old_value DATE,
    new_value DATE,
    detected_at TIMESTAMPTZ DEFAULT NOW(),
    resolved BOOLEAN DEFAULT FALSE
);
```

---

## 6. NEXT STEPS

1. ✅ **Identify HubSpot extraction process** (external to this repo?)
2. ✅ **Map exact HubSpot field names** for maturity and amendments
3. ⏭️ **Create `services/maturity.py`** with canonical API
4. ⏭️ **Extend Supabase schema** with audit fields
5. ⏭️ **Refactor all consumers** to use canonical API
6. ⏭️ **Add comprehensive tests** (unit, property, integration)
7. ⏭️ **Extend `pages/loan_qa.py`** with HubSpot traceability checks
8. ⏭️ **Document in README** the maturity date contract

---

## 7. RISKS & MITIGATIONS

| Risk | Impact | Mitigation |
|------|--------|------------|
| HubSpot extraction not in this repo | Can't implement end-to-end solution | Document requirements for external ETL team |
| Unknown HubSpot field mapping | Can't extract correct data | Query HubSpot API to identify fields |
| No amendment tracking in current system | Can't maintain audit trail | Design backward-compatible schema extension |
| Existing data has no audit metadata | Can't backfill quality flags | Run one-time validation script to assess data quality |
| Breaking changes to data model | Downstream consumers fail | Use feature flags and gradual rollout |

---

**Status**: ✅ Contextualization Complete
**Next Phase**: PLAN - Design canonical maturity API and implementation roadmap
