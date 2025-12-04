# CLAUDE.md - CSL Capital Dashboard

## Project Overview

This is the **CSL Capital Dashboard**, a Streamlit-based financial portfolio management and analytics platform. It integrates data from HubSpot (deals), QuickBooks Online (financial transactions), and MCA (Merchant Cash Advance) loans to provide real-time analytics, forecasting, risk scoring, and ML-based predictions.

## Tech Stack

- **Framework:** Streamlit (multi-page app)
- **Database:** Supabase (PostgreSQL with pagination)
- **Visualization:** Altair, Matplotlib
- **ML:** scikit-learn (classification & regression)
- **Financial:** numpy-financial (IRR, NPV calculations)
- **Python:** 3.11+

## Quick Commands

```bash
# Run the application
streamlit run streamlit_app.py

# Run with specific config
streamlit run streamlit_app.py --server.enableCORS false

# Test imports work
python -c "from utils.config import setup_page; from utils.data_loader import DataLoader; print('OK')"
```

## Directory Structure

```
streamlit/
├── streamlit_app.py          # Main entry (Pipeline Dashboard)
├── pages/                    # Multi-page Streamlit app
│   ├── loan_tape.py          # Loan portfolio analytics & ML
│   ├── capital_forecast.py   # Cash flow forecasting
│   ├── qbo_dashboard.py      # QuickBooks financial analysis
│   ├── x_QA_audit.py         # Data quality auditing
│   ├── x_loan_qa.py          # Loan-specific QA checks
│   └── x_qa_debugger.py      # Development/debugging tools
├── utils/                    # Centralized utilities
│   ├── config.py             # Supabase connection, colors, setup_page()
│   ├── data_loader.py        # DataLoader class with pagination
│   ├── preprocessing.py      # Type conversion, date normalization
│   ├── display_components.py # Reusable UI filters & components
│   ├── loan_tape_data.py     # Loan calculations, IRR, risk scoring
│   ├── loan_tape_analytics.py # Correlations, feature engineering
│   ├── loan_tape_ml.py       # ML model training & visualization
│   ├── loan_tape_loader.py   # Loan-specific data loading
│   ├── cash_flow_forecast.py # Forecast engine
│   └── qbo_data_loader.py    # QBO-specific loading
├── scripts/                  # Data integration scripts
├── assets/                   # Logo and static assets
└── requirements.txt          # Python dependencies
```

## Key Architecture Patterns

### Page Setup Pattern
Every page MUST use `setup_page()` at the top:

```python
from utils.config import setup_page
setup_page("CSL Capital | Page Name")
```

This ensures consistent branding (logo, colors, styles) across all pages.

### Data Loading Pattern
Use `DataLoader` class from utils for Supabase queries:

```python
from utils.data_loader import DataLoader

loader = DataLoader()
df = loader.load_deals()  # or load_mca_deals(), load_qbo_data(), etc.
```

### Filter Components
Use centralized filter functions from `display_components.py`:

```python
from utils.display_components import (
    create_date_range_filter,
    create_partner_source_filter,
    create_status_filter
)
```

### Constants Location
- **PROBLEM_STATUSES** - defined in `utils/loan_tape_analytics.py`
- **PLATFORM_FEE_RATE** (3%) - defined in `utils/config.py`
- **COLOR_PALETTE**, **PRIMARY_COLOR** - defined in `utils/config.py`

## Critical Conventions

### Date Handling
Always normalize dates to timezone-naive for consistency:

```python
df["date_created"] = pd.to_datetime(df["date_created"], utc=True).dt.tz_localize(None)
df["month_start"] = df["date_created"].dt.to_period("M").dt.to_timestamp(how="start")
```

### ID Normalization (Critical for Merges)
loan_id fields often have float/int inconsistencies. Always normalize before merging:

```python
df["loan_id"] = df["loan_id"].astype(str).str.strip().str.replace(r'\.0$', '', regex=True)
```

### Type Conversion
Use pandas error coercion for safe type conversion:

```python
df["amount"] = pd.to_numeric(df["amount"], errors="coerce")
```

### Column Detection Pattern
Handle varying column names across data sources:

```python
status_col = "dealstage" if "dealstage" in df.columns else "stage"
```

## Naming Conventions

- **Functions/Variables:** snake_case (`calculate_irr`, `loan_data`)
- **Classes:** PascalCase (`DataLoader`)
- **Constants:** ALL_CAPS (`PLATFORM_FEE_RATE`, `PRIMARY_COLOR`)
- **Files:** snake_case (`loan_tape_data.py`)

## Common Tasks

### Adding a New Page
1. Create `pages/new_page.py`
2. Import and call `setup_page()` at the top
3. Use `DataLoader` for data access
4. Use filter components from `display_components.py`

### Adding a New Data Loader
1. Add method to `utils/data_loader.py`
2. Include pagination support via `_load_with_pagination()`
3. Apply `@st.cache_data(ttl=3600)` for caching

### Adding New ML Features
1. Add feature preparation in `utils/loan_tape_data.py`
2. Add display names in `FEATURE_DISPLAY_NAMES` dict in `loan_tape_analytics.py`
3. Update model training in `loan_tape_ml.py`

## Testing

There is no automated test framework. Testing is manual per `TESTING_CHECKLIST.md`:

1. Start app: `streamlit run streamlit_app.py`
2. Navigate to each page and verify loading
3. Check browser console (F12) for errors
4. Verify data displays correctly

## Important Gotchas

1. **Page Config Error:** `st.set_page_config()` can only be called once per page. Use `setup_page()` instead of direct calls.

2. **Supabase Pagination:** Large tables (>10k rows) require pagination. `DataLoader` handles this automatically.

3. **Float loan_id:** When merging DataFrames on `loan_id`, always normalize to string first (see ID Normalization above).

4. **Empty DataFrames:** Always check `if df.empty:` before processing to avoid errors.

5. **Caching TTL:** Supabase data cached for 1 hour (`ttl=3600`). Use cache refresh buttons when available.

## Secrets Configuration

Required in `.streamlit/secrets.toml` or Streamlit Cloud secrets:

```toml
[supabase]
url = "https://YOUR_PROJECT.supabase.co"
service_role = "YOUR_SERVICE_ROLE_KEY"

[hubspot]
api_key = "YOUR_KEY"

[qbo]
client_id = "..."
client_secret = "..."
refresh_token = "..."
realm_id = "..."
```

## Color Theme

Primary green theme (`#34a853`). All colors defined in `utils/config.py`:
- PRIMARY_COLOR: #34a853
- Risk gradient: yellow -> orange -> red
- Performance gradient: light green -> dark green
