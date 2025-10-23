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

# Filter for closed won deals
closed_won = df[df["is_closed_won"] == True]

# Create forecast
create_cash_flow_forecast(df, closed_won, qbo_df)
