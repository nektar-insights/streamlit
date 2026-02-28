# utils/config.py
import streamlit as st
from google.cloud import bigquery
from pathlib import Path
import base64
import json

# Import Altair theme
from utils.altair_theme import register_csl_theme

# ----------------------------
# BigQuery Table Map
# Supabase table name → fully-qualified BQ table reference
# ----------------------------
GCP_PROJECT = "csl-deal-pipeline"
_TABLE_MAP = {
    "deals":                       f"{GCP_PROJECT}.csl_hubspot.deals",
    "mca_deals":                   f"{GCP_PROJECT}.csl_hubspot.mca_deals",
    "loan_summaries":              f"{GCP_PROJECT}.csl_hubspot.loan_summaries",
    "loan_schedules":              f"{GCP_PROJECT}.csl_hubspot.loan_schedules",
    "naics_sector_risk_profile":   f"{GCP_PROJECT}.csl_hubspot.naics_sector_risk_profile",
    "cash_flow_forecast_history":  f"{GCP_PROJECT}.csl_hubspot.cash_flow_forecast_history",
    "qbo_invoice_payments":        f"{GCP_PROJECT}.csl_qbo.qbo_invoice_payments",
    "qbo_general_ledger":          f"{GCP_PROJECT}.csl_qbo.qbo_general_ledger",
    "qbo_loan_summary_view":       f"{GCP_PROJECT}.csl_qbo.loan_net_collections",
    "ml_model_results":            f"{GCP_PROJECT}.csl_hubspot.ml_model_results",
    "ml_loan_predictions":         f"{GCP_PROJECT}.csl_hubspot.ml_loan_predictions",
    "model_metrics_history":       f"{GCP_PROJECT}.csl_hubspot.model_metrics_history",
}

# ----------------------------
# BigQuery Connection
# Secrets format (.streamlit/secrets.toml):
#   [gcp]
#   project_id = "csl-deal-pipeline"
#   service_account_key = '{"type": "service_account", ...}'  # JSON string
# ----------------------------
@st.cache_resource
def get_bq_client():
    try:
        if "gcp" in st.secrets and "service_account_key" in st.secrets["gcp"]:
            sa_info = st.secrets["gcp"]["service_account_key"]
            if isinstance(sa_info, str):
                sa_info = json.loads(sa_info)
            from google.oauth2.service_account import Credentials
            credentials = Credentials.from_service_account_info(
                sa_info,
                scopes=["https://www.googleapis.com/auth/bigquery"]
            )
            project = st.secrets["gcp"].get("project_id", GCP_PROJECT)
            return bigquery.Client(project=project, credentials=credentials)
    except Exception:
        pass
    # Fall back to ADC (local dev: gcloud auth application-default login)
    return bigquery.Client(project=GCP_PROJECT)

# ----------------------------
# Color Palette Constants
# ----------------------------
PRIMARY_COLOR = "#34a853"  # Fresh green
TEXT_COLOR = "#222222"
BACKGROUND_COLOR = "#ffffff"
CARD_BACKGROUND = "#f5f5f5"

PERFORMANCE_GRADIENT = ["#e8f5e8", "#34a853", "#1e7e34"]
RISK_GRADIENT = ["#fef9e7", "#f39c12", "#dc3545"]
COLOR_PALETTE = [
    "#34a853", "#2d5a3d", "#4a90e2", "#6c757d",
    "#495057", "#28a745", "#17a2b8", "#6f42c1"
]

# ----------------------------
# Platform Constants
# ----------------------------
PLATFORM_FEE_RATE = 0.03

# ----------------------------
# Performance Grade Thresholds
# ----------------------------
# IRR Grade Thresholds (annualized IRR as decimal)
# Typical MCA deals have 30-50% gross return over 6-12 months
IRR_GRADE_THRESHOLDS = {
    "A": 0.60,  # >= 60% annualized IRR (excellent speed)
    "B": 0.40,  # >= 40% annualized IRR (good speed)
    "C": 0.20,  # >= 20% annualized IRR (acceptable)
    "D": 0.00,  # >= 0% annualized IRR (below target but positive)
    "F": float("-inf"),  # < 0% (loss)
}

# ROI Grade Thresholds (cumulative ROI as decimal)
# Expected full return is 30-50% on typical deals
ROI_GRADE_THRESHOLDS = {
    "A": 0.30,  # >= 30% ROI (fully achieved expected return)
    "B": 0.20,  # >= 20% ROI (strong return)
    "C": 0.10,  # >= 10% ROI (moderate return)
    "D": 0.00,  # >= 0% ROI (capital preserved)
    "F": float("-inf"),  # < 0% (loss)
}

# On-Time Payment Grade Thresholds (pct_on_time as decimal 0-1)
# Based on percentage of payments made on time
ONTIME_GRADE_THRESHOLDS = {
    "A": 0.95,  # >= 95% on-time (excellent borrower)
    "B": 0.80,  # >= 80% on-time (good borrower)
    "C": 0.60,  # >= 60% on-time (fair performance)
    "D": 0.40,  # >= 40% on-time (problematic)
    "F": 0.00,  # < 40% on-time (high-risk borrower)
}

# Grade Visual Indicators (emoji + color)
GRADE_INDICATORS = {
    "A": {"emoji": "🟢", "color": "#28a745", "label": "Excellent"},
    "B": {"emoji": "🟢", "color": "#34a853", "label": "Good"},
    "C": {"emoji": "🟡", "color": "#ffc107", "label": "Fair"},
    "D": {"emoji": "🟠", "color": "#fd7e14", "label": "Poor"},
    "F": {"emoji": "🔴", "color": "#dc3545", "label": "Failing"},
    "N/A": {"emoji": "⚪", "color": "#6c757d", "label": "Not Available"},
}

# ----------------------------
# Page Setup Functions
# ----------------------------
def setup_page(title: str = "CSL Capital | Dashboard", layout: str = "wide"):
    """
    Centralized page setup function that applies consistent configuration and branding

    Args:
        title: Page title to display in browser tab
        layout: Page layout ('wide' or 'centered')

    Usage:
        from utils.config import setup_page
        setup_page("CSL Capital | My Page")
    """
    st.set_page_config(page_title=title, layout=layout)
    inject_global_styles()
    inject_logo()
    # Register and enable the CSL Altair chart theme
    register_csl_theme()

# ----------------------------
# Branding Functions
# ----------------------------
def inject_logo():
    logo_path = Path("assets/CSL_Capital_Logo.png")
    if logo_path.exists():
        logo_base64 = base64.b64encode(logo_path.read_bytes()).decode()
        st.markdown(
            f"""
            <div style="display:flex; align-items:center; margin-bottom:1rem;">
                <img src="data:image/png;base64,{logo_base64}" height="60">
            </div>
            """,
            unsafe_allow_html=True,
        )

def inject_global_styles():
    """Load custom CSS theme from assets/styles.css"""
    css_path = Path("assets/styles.css")
    if css_path.exists():
        with open(css_path) as f:
            css_content = f.read()
            st.markdown(f"<style>{css_content}</style>", unsafe_allow_html=True)
    else:
        st.warning(f"CSS file not found at: {css_path.absolute()}")
        # Fallback to basic styles if CSS file not found
        st.markdown(
            f"""
            <style>
                html, body, [class*="css"]  {{
                    font-family: 'Segoe UI', sans-serif;
                    color: {TEXT_COLOR};
                    background-color: {BACKGROUND_COLOR};
                }}
                .stButton > button {{
                    background-color: {PRIMARY_COLOR};
                    color: white;
                    border-radius: 6px;
                    padding: 0.5rem 1rem;
                }}
                .stMetric label {{
                    color: {TEXT_COLOR};
                }}
            </style>
            """,
            unsafe_allow_html=True
        )
