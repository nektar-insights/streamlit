# pages/capital_forecast.py
from utils.imports import *
from utils.config import (
    inject_global_styles,
    inject_logo,
    get_supabase_client,
)
from utils.qbo_data_loader import load_qbo_data
from utils.cash_flow_forecast import cash_flow_forecast

# Page config & branding
st.set_page_config(
    page_title="CSL Capital | Cash Forecast",
    layout="wide",
)
inject_global_styles()
inject_logo()

# Load data
supabase = get_supabase_client()

# Load deals data
@st.cache_data(ttl=3600)
def load_deals():
    res = supabase.table("deals").select("*").execute()
    return pd.DataFrame(res.data)

# Load QBO data using your existing loader
qbo_df, gl_df = load_qbo_data()

# Load and preprocess deals
deals_df = load_deals()
cols_to_convert = ["amount", "total_funded_amount", "factor_rate", "loan_term", "commission"]
deals_df["date_created"] = pd.to_datetime(deals_df["date_created"], errors="coerce")
deals_df[cols_to_convert] = deals_df[cols_to_convert].apply(pd.to_numeric, errors="coerce")
closed_won = deals_df[deals_df["is_closed_won"] == True]

# Create integrated forecast
create_integrated_cash_flow_forecast(deals_df, closed_won, qbo_df)
