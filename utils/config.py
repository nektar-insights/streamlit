# utils/config.py
import streamlit as st
from supabase import create_client

# ----------------------------
# Supabase Connection
# ----------------------------
@st.cache_resource
def get_supabase_client():
    """Create and cache Supabase client"""
    url = st.secrets["supabase"]["url"]
    key = st.secrets["supabase"]["service_role"]
    return create_client(url, key)

# ----------------------------
# Color Palette Constants
# ----------------------------
PRIMARY_COLOR = "#34a853"
PERFORMANCE_GRADIENT = ["#e8f5e8", "#34a853", "#1e7e34"]  # Light green → Mid → Dark green
RISK_GRADIENT = ["#fef9e7", "#f39c12", "#dc3545"]  # Light yellow → Orange → Red
COLOR_PALETTE = [
    "#34a853", "#2d5a3d", "#4a90e2", "#6c757d",
    "#495057", "#28a745", "#17a2b8", "#6f42c1"
]

# ----------------------------
# Common Configuration
# ----------------------------
PLATFORM_FEE_RATE = 0.03
