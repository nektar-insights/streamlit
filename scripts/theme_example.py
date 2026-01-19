# ============================================
# CSL CAPITAL - EXAMPLE STREAMLIT APP
# Shows how to integrate the custom theme
# ============================================

import streamlit as st
import pandas as pd
import altair as alt
from altair_theme import register_csl_theme, create_bar_chart, create_line_chart, create_positive_negative_bar

# ============================================
# PAGE CONFIG (must be first Streamlit command)
# ============================================
st.set_page_config(
    page_title="CSL Capital Dashboard",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================
# LOAD CUSTOM CSS
# ============================================
with open("styles.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# ============================================
# REGISTER ALTAIR THEME
# ============================================
CSL_COLORS, CSL_CATEGORICAL = register_csl_theme()

# ============================================
# SIDEBAR
# ============================================
with st.sidebar:
    st.image("https://via.placeholder.com/200x60/00C853/ffffff?text=CSL+Capital", width=180)
    st.markdown("---")
    
    # Navigation would be automatic in multipage app
    # This is just for demo
    page = st.radio(
        "Navigation",
        ["Dashboard", "Deal Pipeline", "Loan Tape", "QA Dashboard"],
        label_visibility="collapsed"
    )
    
    st.markdown("---")
    
    # Filters
    st.subheader("Filters")
    partner = st.selectbox("Partner Source", ["All", "TVT Capital", "VitalCap", "Fresh Funding"])
    date_range = st.date_input("Date Range", value=[])
    
    st.markdown("---")
    st.caption("© 2025 CSL Capital")

# ============================================
# MAIN CONTENT
# ============================================

# Header
st.title("Portfolio Overview")
st.markdown("Real-time insights into deal flow and portfolio performance")

# ============================================
# KPI METRICS ROW
# ============================================
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        label="Total Deployed",
        value="$2.4M",
        delta="+$180K this month"
    )

with col2:
    st.metric(
        label="Active Deals",
        value="47",
        delta="+8"
    )

with col3:
    st.metric(
        label="Avg Net MOIC",
        value="1.28x",
        delta="+0.03x"
    )

with col4:
    st.metric(
        label="Default Rate",
        value="2.1%",
        delta="-0.4%",
        delta_color="inverse"
    )

st.markdown("---")

# ============================================
# CHARTS ROW
# ============================================
col_left, col_right = st.columns(2)

# Sample data
monthly_data = pd.DataFrame({
    'month': ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun'],
    'deals': [12, 18, 15, 22, 19, 25],
    'volume': [150000, 220000, 180000, 290000, 240000, 310000]
})

partner_data = pd.DataFrame({
    'partner': ['TVT Capital', 'VitalCap', 'Fresh Funding', 'Other'],
    'deals': [23, 15, 8, 5],
    'volume': [480000, 320000, 180000, 120000]
})

pnl_data = pd.DataFrame({
    'month': ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun'],
    'pnl': [12000, -5000, 18000, 8000, -3000, 22000]
})

with col_left:
    st.subheader("Deal Volume by Month")
    bar_chart = create_bar_chart(monthly_data, 'month', 'volume', title='')
    st.altair_chart(bar_chart, width="stretch")

with col_right:
    st.subheader("Deals by Partner")
    partner_chart = create_bar_chart(partner_data, 'partner', 'deals', title='', horizontal=False)
    st.altair_chart(partner_chart, width="stretch")

# ============================================
# SECOND CHARTS ROW
# ============================================
col_left2, col_right2 = st.columns(2)

with col_left2:
    st.subheader("Monthly P&L")
    pnl_chart = create_positive_negative_bar(pnl_data, 'month', 'pnl', title='')
    st.altair_chart(pnl_chart, width="stretch")

with col_right2:
    st.subheader("Cumulative Volume Trend")
    monthly_data['cumulative'] = monthly_data['volume'].cumsum()
    line_chart = create_line_chart(monthly_data, 'month', 'cumulative', title='')
    st.altair_chart(line_chart, width="stretch")

st.markdown("---")

# ============================================
# RECENT DEALS TABLE
# ============================================
st.subheader("Recent Deals")

deals_df = pd.DataFrame({
    'Deal Name': ['ABC Construction', 'XYZ Medical', 'Smith Trucking', 'Jones Electric', 'Metro Plumbing'],
    'Partner': ['TVT Capital', 'VitalCap', 'Fresh Funding', 'TVT Capital', 'VitalCap'],
    'Amount': ['$45,000', '$32,000', '$28,000', '$55,000', '$38,000'],
    'Factor Rate': ['1.38', '1.42', '1.35', '1.40', '1.36'],
    'Net MOIC': ['1.24x', '1.28x', '1.22x', '1.26x', '1.23x'],
    'Status': ['Active', 'Active', 'Pending', 'Active', 'Closed']
})

st.dataframe(
    deals_df,
    width="stretch",
    hide_index=True,
    column_config={
        "Status": st.column_config.TextColumn(
            "Status",
            help="Current deal status"
        )
    }
)

# ============================================
# TABS EXAMPLE
# ============================================
st.markdown("---")
st.subheader("Deal Analysis")

tab1, tab2, tab3 = st.tabs(["By Industry", "By Term Length", "By FICO Range"])

with tab1:
    industry_data = pd.DataFrame({
        'industry': ['Construction', 'Healthcare', 'Trucking', 'Restaurant', 'Manufacturing'],
        'count': [15, 12, 9, 8, 6]
    })
    chart = create_bar_chart(industry_data, 'industry', 'count')
    st.altair_chart(chart, width="stretch")

with tab2:
    term_data = pd.DataFrame({
        'term': ['3-6 months', '6-9 months', '9-12 months', '12+ months'],
        'count': [18, 22, 14, 8]
    })
    chart = create_bar_chart(term_data, 'term', 'count')
    st.altair_chart(chart, width="stretch")

with tab3:
    fico_data = pd.DataFrame({
        'fico_range': ['620-650', '650-700', '700-750', '750+'],
        'count': [8, 18, 20, 12]
    })
    chart = create_bar_chart(fico_data, 'fico_range', 'count')
    st.altair_chart(chart, width="stretch")

# ============================================
# EXPANDER EXAMPLE
# ============================================
with st.expander("Investment Rules Summary"):
    st.markdown("""
    **Hard Fail Conditions:**
    - Business under 5 years old
    - FICO below 620
    - Third position or worse
    - Debt leverage above 25%
    - Net MOIC below 1.15x
    
    **Fresh Funding Specific:**
    - Minimum 1.22x Net MOIC required
    
    **Excluded Industries (VICE):**
    - Cannabis, Alcohol, Tobacco, Gambling, Adult Entertainment
    """)
