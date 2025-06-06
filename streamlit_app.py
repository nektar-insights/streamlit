# pages/pipeline_dashboard.py

import streamlit as st
import pandas as pd
import altair as alt
from supabase import create_client
from datetime import datetime
import io
from xhtml2pdf import pisa

# ----------------------------
# Color Palette (matching MCA dashboard)
# ----------------------------
PERFORMANCE_GRADIENT = ["#e8f5e8", "#34a853", "#1e7e34"]
RISK_GRADIENT = ["#fef9e7", "#f39c12", "#dc3545"]
PRIMARY_COLOR = "#34a853"
COLOR_PALETTE = [
    "#34a853", "#2d5a3d", "#4a90e2", "#6c757d",
    "#495057", "#28a745", "#17a2b8", "#6f42c1"
]

# ----------------------------
# Supabase connection
# ----------------------------
url = st.secrets["supabase"]["url"]
key = st.secrets["supabase"]["service_role"]
supabase = create_client(url, key)

# ----------------------------
# Load and prepare data
# ----------------------------
@st.cache_data(ttl=3600)
def load_deals():
    res = supabase.table("deals").select("*").execute()
    return pd.DataFrame(res.data)

df = load_deals()
df["date_created"] = pd.to_datetime(df["date_created"], errors="coerce")
today = pd.to_datetime("today").normalize()
df["month"] = df["date_created"].dt.to_period("M").astype(str)
df["amount"] = pd.to_numeric(df["amount"], errors="coerce")
df["total_funded_amount"] = pd.to_numeric(df["total_funded_amount"], errors="coerce")
df["factor_rate"] = pd.to_numeric(df["factor_rate"], errors="coerce")
df["loan_term"] = pd.to_numeric(df["loan_term"], errors="coerce")
df["commission"] = pd.to_numeric(df["commission"], errors="coerce")
df["loan_id"] = df["loan_id"].astype("string")

# ----------------------------
# Rolling Window Deal Flow Calculations
# ----------------------------
periods = [
    ("0–30 Days", 0, 30),
    ("30–60 Days", 30, 60),
    ("60–90 Days", 60, 90),
    ("90–120 Days", 90, 120),
]
flow_data = []
for label, start, end in periods:
    window_start = today - pd.Timedelta(days=end)
    window_end = today - pd.Timedelta(days=start)
    window_df = df[(df["date_created"] >= window_start) & (df["date_created"] < window_end)]
    flow_data.append({
        "Period": label,
        "Deals": len(window_df),
        "Total Funded": window_df["total_funded_amount"].sum()
    })
flow_df = pd.DataFrame(flow_data)

# ----------------------------
# Filters
# ----------------------------
st.title("Pipeline Dashboard")

min_date = df["date_created"].min()
max_date = df["date_created"].max()

start_date, end_date = st.date_input(
    "Filter by Date Range", 
    [min_date, max_date], 
    min_value=min_date, 
    max_value=max_date
)

df = df[(df["date_created"] >= pd.to_datetime(start_date)) &
        (df["date_created"] <= pd.to_datetime(end_date))]

partner_options = sorted(df["partner_source"].dropna().unique())
all_label = "All Partners"
partner_options_with_all = [all_label] + partner_options

selected_partner = st.selectbox("Filter by Partner Source", options=partner_options_with_all)
if selected_partner != all_label:
    df = df[df["partner_source"] == selected_partner]

participation_options = ["All Deals", "Participated Only", "Not Participated"]
participation_filter = st.radio("Show Deals", participation_options)
if participation_filter == "Participated Only":
    df = df[df["is_closed_won"] == True]
elif participation_filter == "Not Participated":
    df = df[df["is_closed_won"] != True]

# ----------------------------
# Metric Calculations
# ----------------------------
closed_won = df[df["is_closed_won"] == True]
total_deals = len(df)
participation_ratio = len(closed_won) / total_deals if total_deals > 0 else 0
months = df["month"].nunique()
pacing = len(closed_won) / months if months > 0 else 0

avg_amount = closed_won["amount"].mean()
avg_factor = closed_won["factor_rate"].mean()
avg_term = closed_won["loan_term"].mean()
avg_participation_pct = (closed_won["amount"] / closed_won["total_funded_amount"]).mean()
avg_commission = closed_won["commission"].mean()

PLATFORM_FEE_RATE = 0.04
total_capital_deployed = closed_won["amount"].sum()
total_commissions_paid = (closed_won["amount"] * closed_won["commission"]).sum()
total_platform_fee = total_capital_deployed * PLATFORM_FEE_RATE
total_expected_return = (closed_won["amount"] * closed_won["factor_rate"]) - closed_won["commission"] - (closed_won["amount"] * PLATFORM_FEE_RATE)
total_expected_return_sum = total_expected_return.sum()
moic = total_expected_return_sum / total_capital_deployed if total_capital_deployed > 0 else 0
projected_irr = (moic ** (12 / avg_term) - 1) if avg_term > 0 else 0

# ----------------------------
# Display Summary Metrics
# ----------------------------
st.subheader("📊 Deal Overview")
col1, col2, col3 = st.columns(3)
col1.metric("Total Deals", total_deals)
col2.metric("Closed Won", len(closed_won))
col3.metric("Close Ratio", f"{participation_ratio:.2%}")

st.subheader("💰 Financial Performance")
col4, col5, col6 = st.columns(3)
col4.metric("Total Capital Deployed", f"${total_capital_deployed:,.0f}")
col5.metric("Total Expected Return", f"${total_expected_return_sum:,.0f}")
col6.metric("MOIC", f"{moic:.2f}")

col7, col8, col9 = st.columns(3)
col7.metric("Projected IRR", f"{projected_irr:.2%}")
col8.metric("Avg % of Deal", f"{avg_participation_pct:.2%}")
col9.metric("Commission Paid", f"${total_commissions_paid:,.0f}")

st.subheader("📋 Deal Characteristics")
col10, col11, col12 = st.columns(3)
col10.metric("Avg Participation ($)", f"${avg_amount:,.0f}")
col11.metric("Avg Factor", f"{avg_factor:.2f}")
col12.metric("Avg Term (mo)", f"{avg_term:.1f}")

# ----------------------------
# Insert Rolling Flow Table and Charts
# ----------------------------
st.subheader("📈 Rolling Deal Flow Trends")
flow_df_display = flow_df.copy()
flow_df_display["Total Funded"] = flow_df_display["Total Funded"].apply(lambda x: f"${x:,.0f}")
st.dataframe(flow_df_display, use_container_width=True)

flow_chart = alt.Chart(flow_df).mark_bar(size=40).encode(
    x=alt.X("Period:N", sort=["90–120 Days", "60–90 Days", "30–60 Days", "0–30 Days"]),
    y=alt.Y("Deals:Q", title="Deal Count"),
    tooltip=["Period", "Deals"]
).properties(height=300)

funded_chart = alt.Chart(flow_df).mark_bar(size=40, color="#4a90e2").encode(
    x=alt.X("Period:N", sort=["90–120 Days", "60–90 Days", "30–60 Days", "0–30 Days"]),
    y=alt.Y("Total Funded:Q", title="Total Funded ($)", axis=alt.Axis(format="$,.0f")),
    tooltip=["Period", alt.Tooltip("Total Funded", format="$,.0f")]
).properties(height=300)

st.altair_chart(flow_chart, use_container_width=True)
st.altair_chart(funded_chart, use_container_width=True)

# (continued with partner summary and exports...)
