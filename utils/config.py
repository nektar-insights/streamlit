# utils/config.py
import streamlit as st
from supabase import create_client
from pathlib import Path
import base64

# ----------------------------
# Supabase Connection
# ----------------------------
@st.cache_resource
def get_supabase_client():
    url = st.secrets["supabase"]["url"]
    key = st.secrets["supabase"]["service_role"]
    return create_client(url, key)

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
