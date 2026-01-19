# utils/imports.py
# Standard libraries
import streamlit as st
import pandas as pd
import altair as alt
from datetime import datetime, timedelta
import io
import hashlib
import numpy_financial as npf

# Third-party libraries
from numpy import busday_count

# Optional PDF export (requires system cairo libraries)
try:
    from xhtml2pdf import pisa
except ImportError:
    pisa = None  # PDF export will be disabled

# Project imports
from utils.config import (
    get_supabase_client, 
    PRIMARY_COLOR, 
    PERFORMANCE_GRADIENT, 
    RISK_GRADIENT, 
    COLOR_PALETTE,
    PLATFORM_FEE_RATE
)
