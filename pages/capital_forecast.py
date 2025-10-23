# pages/capital_forecast.py
"""
Capital Forecast Dashboard - Cash flow projections and forecasting

Provides forward-looking analysis of capital requirements and expected returns.
"""

from utils.imports import *
from utils.config import setup_page
from utils.data_loader import load_deals, load_qbo_data
from utils.preprocessing import preprocess_dataframe
from utils.cash_flow_forecast import create_cash_flow_forecast
from utils.display_components import create_date_range_filter, create_status_filter

# ----------------------------
# Page Configuration & Styles
# ----------------------------
setup_page("CSL Capital | Capital Forecast")

# ----------------------------
# Load and prepare data
# ----------------------------
# Load QBO data
qbo_df, gl_df = load_qbo_data()

# Load and preprocess deals
df = load_deals()

# Preprocess using centralized utility
df = preprocess_dataframe(
    df,
    numeric_cols=["amount", "total_funded_amount", "factor_rate", "loan_term", "commission"],
    date_cols=["date_created"]
)

# ----------------------------
# Sidebar Filters
# ----------------------------
st.sidebar.header("Filters")

# Date range filter
with st.sidebar:
    filtered_df, _ = create_date_range_filter(
        df,
        date_col="date_created",
        label="Select Date Range",
        checkbox_label="Filter by Date Created",
        default_enabled=False,
        key_prefix="capital_forecast_date"
    )

# Status filter (using dealstage if available, otherwise stage)
status_col = "dealstage" if "dealstage" in filtered_df.columns else ("stage" if "stage" in filtered_df.columns else None)
if status_col:
    with st.sidebar:
        filtered_df, selected_status = create_status_filter(
            filtered_df,
            status_col=status_col,
            label="Filter by Deal Stage",
            include_all_option=True,
            key_prefix="capital_forecast_status"
        )

st.sidebar.markdown("---")
st.sidebar.write(f"**Showing:** {len(filtered_df)} of {len(df)} deals")

# Filter for closed won deals
closed_won = filtered_df[filtered_df["is_closed_won"] == True] if "is_closed_won" in filtered_df.columns else filtered_df

# Create forecast
create_cash_flow_forecast(filtered_df, closed_won, qbo_df)
