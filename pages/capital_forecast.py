# pages/capital_forecast.py
from utils.imports import *
from utils.config import setup_page
from utils.data_loader import load_deals, load_qbo_data
from utils.cash_flow_forecast import create_cash_flow_forecast

# ----------------------------
# Page Configuration & Styles
# ----------------------------
setup_page("CSL Capital | Capital Forecast")

# ----------------------------
# Load data
# ----------------------------
# Load QBO data
qbo_df, gl_df = load_qbo_data()

# Load and preprocess deals
df = load_deals()
cols_to_convert = ["amount", "total_funded_amount", "factor_rate", "loan_term", "commission"]
df["date_created"] = pd.to_datetime(df["date_created"], errors="coerce")
df[cols_to_convert] = df[cols_to_convert].apply(pd.to_numeric, errors="coerce")
closed_won = df[df["is_closed_won"] == True]

# Create forecast - now with QBO data
create_cash_flow_forecast(df, closed_won, qbo_df)
