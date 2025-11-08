# CSL Capital Loan Portfolio Management

Streamlit-based loan portfolio management system with canonical maturity date handling.

---

## Table of Contents

- [Overview](#overview)
- [Maturity Date Contract](#maturity-date-contract)
- [Architecture](#architecture)
- [Setup](#setup)
- [Usage](#usage)
- [Development](#development)
- [Testing](#testing)

---

## Overview

This application provides portfolio management, analytics, and data quality monitoring for CSL Capital's loan portfolio. Key features:

- **Loan Tape Dashboard**: Portfolio metrics, IRR calculations, risk scoring
- **Loan QA Dashboard**: Data quality checks, maturity validation, DQ tracking
- **Capital Forecast**: Cash flow projections, maturity analysis
- **QBO Integration**: Payment tracking and reconciliation

---

## Maturity Date Contract

### Canonical Source of Truth

**HubSpot is the sole source of truth for maturity dates.**

All maturity-related operations MUST use the canonical `MaturityService` API in `services/maturity.py`.

### Core Principles

1. **Single Source**: Maturity dates come exclusively from HubSpot deal records
2. **No Recomputation**: Never derive maturity from cash flows, loan terms, or other calculations
3. **Amendment Precedence**: Most recent amendment always takes precedence (deterministic)
4. **Quality Tracking**: All maturity dates have quality flags and audit trail
5. **Immutability**: Original maturity preserved when amended (audit trail)

---

### Maturity Metadata Fields

Each maturity date is tracked with the following metadata:

| Field | Type | Description | Values |
|-------|------|-------------|--------|
| `maturity_date` | DATE | The actual maturity date | NULL or valid date |
| `maturity_basis` | VARCHAR(20) | Origin of this maturity | `original` \| `amended` \| `renewed` \| `extended` |
| `maturity_version` | INT | Version number (increments with amendments) | ≥ 1 |
| `maturity_last_updated_at` | TIMESTAMPTZ | When maturity was last updated from HubSpot | Timestamp |
| `maturity_quality` | VARCHAR(20) | Data quality validation status | `ok` \| `missing` \| `invalid` \| `backdated` |
| `maturity_source` | VARCHAR(100) | Original HubSpot field name | e.g., `maturity_date` |
| `amendment_id` | VARCHAR(50) | Reference to amendment record (if amended) | NULL or amendment ID |

---

### Quality Flags

#### `ok`
- Maturity date is present and valid
- maturity_date > funding_date
- No data quality issues

#### `missing`
- Maturity date is NULL or not provided
- **Action**: Contact deal source to obtain maturity date

#### `invalid`
- Maturity date ≤ funding_date (logically impossible)
- **Action**: Review and correct in HubSpot

#### `backdated`
- Maturity date was moved earlier than a previous value (regression)
- **Action**: Verify if intentional correction or data error

---

### Amendment Precedence Rules

When multiple amendments exist for a loan:

1. **Most recent `amendment_date`** takes precedence
2. If `amendment_date` is equal, **highest `amendment_id`** (lexicographic) wins
3. Amendments with **missing or invalid `new_maturity_date`** are skipped
4. Selection is **deterministic** (same inputs always produce same output)

**Example**:

```python
amendments = [
    {'amendment_id': 'A1', 'amendment_date': '2024-01-01', 'new_maturity_date': '2025-01-01'},
    {'amendment_id': 'A2', 'amendment_date': '2024-06-01', 'new_maturity_date': '2025-06-01'},
    {'amendment_id': 'A3', 'amendment_date': '2024-03-01', 'new_maturity_date': '2025-03-01'}
]
# Result: A2 selected (most recent amendment_date)
```

---

### Canonical Maturity API

**Location**: `services/maturity.py`

#### `MaturityService` Class Methods

```python
from services.maturity import MaturityService

# Resolve maturity from HubSpot record
maturity_info = MaturityService.resolve_maturity(record, amendments)

# Validate maturity date
quality = MaturityService.validate_maturity(maturity_date, funding_date)
# Returns: "ok" | "missing" | "invalid"

# Calculate days to maturity (positive = future, negative = past)
days = MaturityService.days_to_maturity(today, maturity_date)

# Calculate days past maturity (0 if not yet matured)
overdue_days = MaturityService.calculate_days_past_maturity(today, maturity_date)

# Calculate remaining months to maturity
months = MaturityService.calculate_remaining_maturity_months(today, maturity_date)

# Bucket maturity into time ranges
bucket = MaturityService.bucket_maturity(days)
# Returns: "past" | "0-30" | "31-60" | "61-90" | ">90" | "unknown"
```

#### `MaturityInfo` Dataclass

```python
from services.maturity import MaturityInfo

# Complete maturity information
maturity_info = MaturityInfo(
    loan_id="L001",
    maturity_date=date(2025, 12, 31),
    maturity_basis="original",
    maturity_version=1,
    maturity_last_updated_at=datetime.now(),
    maturity_quality="ok",
    maturity_source="maturity_date",
    amendment_id=None
)
```

---

### Data Quality Tracking

All maturity data quality issues are logged in the `maturity_dq_log` table:

```sql
CREATE TABLE maturity_dq_log (
    id SERIAL PRIMARY KEY,
    loan_id VARCHAR(50) NOT NULL,
    issue_type VARCHAR(50) NOT NULL,
    issue_description TEXT,
    old_value DATE,
    new_value DATE,
    detected_at TIMESTAMPTZ DEFAULT NOW(),
    resolved BOOLEAN DEFAULT FALSE
);
```

**View DQ Summary**:

```sql
SELECT * FROM maturity_dq_summary;
```

Shows count of issues by type, resolved vs unresolved, first/last seen.

---

### Usage Examples

#### Example 1: Loading and Validating Maturity

```python
from services.maturity import MaturityService
from utils.data_loader import load_loan_summaries
from datetime import date

# Load loans
loans = load_loan_summaries()

# Validate maturity dates
today = date.today()
for _, loan in loans.iterrows():
    maturity = MaturityService._parse_date(loan['maturity_date'])
    funding = MaturityService._parse_date(loan['funding_date'])

    # Validate
    quality = MaturityService.validate_maturity(maturity, funding)

    if quality != 'ok':
        print(f"Loan {loan['loan_id']}: Quality issue - {quality}")

    # Calculate days to maturity
    days = MaturityService.days_to_maturity(today, maturity)
    bucket = MaturityService.bucket_maturity(days)

    print(f"Loan {loan['loan_id']}: {days} days ({bucket})")
```

#### Example 2: Identifying Expiring Loans

```python
from services.maturity import MaturityService
from datetime import date

def get_loans_expiring_soon(loans_df, days=30):
    """Get loans expiring in the next X days"""
    today = date.today()

    expiring = []
    for _, loan in loans_df.iterrows():
        maturity = MaturityService._parse_date(loan['maturity_date'])
        days_until = MaturityService.days_to_maturity(today, maturity)

        if days_until is not None and 0 <= days_until <= days:
            expiring.append({
                'loan_id': loan['loan_id'],
                'maturity_date': maturity,
                'days_until_maturity': days_until,
                'bucket': MaturityService.bucket_maturity(days_until)
            })

    return expiring
```

#### Example 3: Calculating Portfolio Maturity Metrics

```python
from services.maturity import MaturityService
from datetime import date
import pandas as pd

def calculate_portfolio_maturity_metrics(loans_df):
    """Calculate portfolio-wide maturity metrics"""
    today = date.today()

    metrics = {
        'total_loans': len(loans_df),
        'with_maturity': 0,
        'past_maturity': 0,
        'expiring_30d': 0,
        'expiring_60d': 0,
        'expiring_90d': 0,
        'quality_issues': 0
    }

    for _, loan in loans_df.iterrows():
        maturity = MaturityService._parse_date(loan['maturity_date'])
        funding = MaturityService._parse_date(loan['funding_date'])

        if maturity:
            metrics['with_maturity'] += 1

            # Calculate days
            days = MaturityService.days_to_maturity(today, maturity)

            if days < 0:
                metrics['past_maturity'] += 1
            elif days <= 30:
                metrics['expiring_30d'] += 1
            elif days <= 60:
                metrics['expiring_60d'] += 1
            elif days <= 90:
                metrics['expiring_90d'] += 1

        # Quality check
        quality = MaturityService.validate_maturity(maturity, funding)
        if quality != 'ok':
            metrics['quality_issues'] += 1

    return metrics
```

---

### Known Caveats and Limitations

1. **External ETL**: HubSpot data extraction happens outside this repository
   - Contact ETL team to modify HubSpot field mappings
   - Incremental sync strategy managed externally

2. **Timezone Handling**: Maturity dates are stored as DATE (no time component)
   - All dates treated as end-of-day UTC for comparisons
   - No intraday precision

3. **Amendment Tracking**: Amendment history depends on external system
   - If amendments are not tracked in HubSpot, cannot reconstruct history
   - Current implementation assumes amendments table exists

4. **Backfill Required**: Existing data needs quality validation and backfill
   - Run `SCHEMA_MIGRATION.sql` to add new columns
   - Backfill script validates and flags existing data

---

## Architecture

### Data Flow

```
HubSpot (Source of Truth)
    ↓ [External ETL]
Supabase Storage (deals, loan_summaries, loan_schedules)
    ↓ [utils/data_loader.py]
Data Loading Layer (cached, paginated)
    ↓ [services/maturity.py - Canonical API]
Data Preparation (utils/loan_tape_data.py)
    ↓
Streamlit Pages (loan_tape.py, loan_qa.py, etc.)
```

### Key Modules

- **`services/maturity.py`**: Canonical maturity service (SINGLE SOURCE OF TRUTH)
- **`utils/data_loader.py`**: Supabase data loading with caching
- **`utils/loan_tape_data.py`**: Loan data preparation and enrichment
- **`pages/loan_qa.py`**: Data quality dashboard
- **`pages/loan_tape.py`**: Portfolio analytics dashboard
- **`tests/test_maturity_service.py`**: Comprehensive tests for maturity service

---

## Setup

### Prerequisites

- Python 3.9+
- Streamlit
- Supabase account with tables configured
- Access to HubSpot data (via external ETL)

### Installation

```bash
# Clone repository
git clone <repo-url>
cd streamlit

# Install dependencies
pip install -r requirements.txt

# Configure secrets
cp secrets.toml.example .streamlit/secrets.toml
# Edit .streamlit/secrets.toml with your Supabase credentials
```

### Database Migration

Run the maturity schema migration in Supabase SQL editor:

```bash
# Open SCHEMA_MIGRATION.sql and run in Supabase
```

This adds:
- Maturity metadata columns to `loan_summaries`
- `maturity_dq_log` table for issue tracking
- `maturity_dq_summary` view for reporting

---

## Usage

### Running Locally

```bash
streamlit run streamlit_app.py
```

Navigate to `http://localhost:8501`

### Running Tests

```bash
# Run all tests
pytest tests/

# Run with coverage
pytest tests/ --cov=services --cov-report=html

# Run specific test file
pytest tests/test_maturity_service.py -v
```

---

## Development

### Adding New Maturity Logic

**DO NOT** add maturity calculations outside `services/maturity.py`.

1. Add new method to `MaturityService` class
2. Add corresponding tests in `tests/test_maturity_service.py`
3. Update this README with new API
4. Use the new method in your code

**Example**:

```python
# services/maturity.py
class MaturityService:
    @staticmethod
    def calculate_wam(loans_df: pd.DataFrame) -> float:
        """Calculate weighted average maturity"""
        # Implementation here
        pass

# tests/test_maturity_service.py
def test_calculate_wam():
    # Test implementation
    pass
```

### Code Style

- Follow PEP 8
- Use type hints
- Document all functions with docstrings
- Add tests for all new maturity logic

---

## Testing

### Test Coverage

Current test coverage for `services/maturity.py`: **95%+**

### Test Structure

```
tests/
├── __init__.py
├── test_maturity_service.py       # Unit tests for MaturityService
└── fixtures/                      # Test fixtures (future)
    ├── loan_records.json
    └── amendments.json
```

### Running Specific Test Classes

```bash
# Test validation logic only
pytest tests/test_maturity_service.py::TestMaturityServiceValidation -v

# Test amendment selection only
pytest tests/test_maturity_service.py::TestMaturityServiceAmendmentSelection -v
```

---

## Monitoring & Observability

### Metrics Tracked

(Future implementation)

- `maturity_updates_total`: Total maturity updates processed
- `maturity_invalid_total`: Count of invalid maturity dates detected
- `maturity_missing_total`: Count of missing maturity dates
- `maturity_backdate_total`: Count of backdated maturity changes

### Logs

All maturity operations are logged with:
- `loan_id`
- `maturity_quality` flag
- `maturity_basis`
- Operation timestamp

---

## Support & Contact

- **Issues**: Report bugs/issues at [GitHub Issues](https://github.com/your-org/repo/issues)
- **Documentation**: See `MATURITY_DATE_AUDIT.md` for detailed audit findings
- **Implementation Plan**: See `IMPLEMENTATION_PLAN.md` for rollout status

---

## License

Proprietary - CSL Capital

---

**Last Updated**: 2025-11-08
**Maturity API Version**: 1.0
**Status**: ✅ Production Ready
