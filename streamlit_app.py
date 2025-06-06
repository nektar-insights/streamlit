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
PERFORMANCE_GRADIENT = ["#e8f5e8", "#34a853", "#1e7e34"]  # Light green → Mid → Dark green
RISK_GRADIENT = ["#fef9e7", "#f39c12", "#dc3545"]  # Light yellow → Orange → Red
PRIMARY_COLOR = "#34a853"
COLOR_PALETTE = [
    "#34a853",  # Primary green
    "#2d5a3d",  # Dark green
    "#4a90e2",  # Professional blue
    "#6c757d",  # Neutral gray
    "#495057",  # Dark gray
    "#28a745",  # Success green
    "#17a2b8",  # Info blue
    "#6f42c1",  # Professional purple
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

today = pd.to_datetime("today").normalize()

def get_stats_by_window(days):
    recent = df[df["date_created"] >= today - pd.Timedelta(days=days)]
    return {
        "count": len(recent),
        "funded": recent["total_funded_amount"].sum()
    }

# Get stats
stats_30 = get_stats_by_window(30)
stats_60 = get_stats_by_window(60)
stats_90 = get_stats_by_window(90)

# Compute changes (30 vs 60) and (30 vs 90)
def compute_change(current, previous):
    delta = current - previous
    pct = (delta / previous * 100) if previous != 0 else 0
    return f"{delta:+,.0f} ({pct:+.1f}%)"

count_change_60 = compute_change(stats_30["count"], stats_60["count"])
count_change_90 = compute_change(stats_30["count"], stats_90["count"])

funded_change_60 = compute_change(stats_30["funded"], stats_60["funded"])
funded_change_90 = compute_change(stats_30["funded"], stats_90["funded"])

# ----------------------------
# Data type conversions and basic calculations
# ----------------------------
df["date_created"] = pd.to_datetime(df["date_created"], errors="coerce")
df["month"] = df["date_created"].dt.to_period("M").astype(str)
df["amount"] = pd.to_numeric(df["amount"], errors="coerce")
df["total_funded_amount"] = pd.to_numeric(df["total_funded_amount"], errors="coerce")
df["factor_rate"] = pd.to_numeric(df["factor_rate"], errors="coerce")
df["loan_term"] = pd.to_numeric(df["loan_term"], errors="coerce")
df["commission"] = pd.to_numeric(df["commission"], errors="coerce")
df["loan_id"] = df["loan_id"].astype("string")

# Platform fee rate
PLATFORM_FEE_RATE = 0.04

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
# Calculate all metrics
# ----------------------------

# Core deal metrics
closed_won = df[df["is_closed_won"] == True]
total_deals = len(df)
participation_ratio = len(closed_won) / total_deals if total_deals > 0 else 0
months = df["month"].nunique()
pacing = len(closed_won) / months if months > 0 else 0

# Deal characteristics
avg_amount = closed_won["amount"].mean()
avg_factor = closed_won["factor_rate"].mean()
avg_term = closed_won["loan_term"].mean()
avg_participation_pct = (closed_won["amount"] / closed_won["total_funded_amount"]).mean()
avg_commission = closed_won["commission"].mean()

# Financial calculations
total_capital_deployed = closed_won["amount"].sum()
total_commissions_paid = (closed_won["amount"] * closed_won["commission"]).sum()
total_platform_fee = total_capital_deployed * PLATFORM_FEE_RATE
total_expected_return = (closed_won["amount"] * closed_won["factor_rate"]) - closed_won["commission"] - (closed_won["amount"] * PLATFORM_FEE_RATE)
total_expected_return_sum = total_expected_return.sum()
moic = total_expected_return_sum / total_capital_deployed if total_capital_deployed > 0 else 0
projected_irr = (moic ** (12 / avg_term) - 1) if avg_term > 0 else 0

# Monthly aggregations for charts
monthly_funded = df.groupby("month")["total_funded_amount"].sum().round(0).reset_index()
monthly_partner = df.groupby(["month", "partner_source"]).size().reset_index(name="count")
monthly_amount = df.groupby("month")["amount"].sum().round(0).reset_index()

# Partner summary calculations
all_deals = df.groupby("partner_source").agg(
    total_deals=("id", "count"),
    total_amount=("total_funded_amount", "sum")
)
won_deals = closed_won.groupby("partner_source").agg(
    participated_deals=("id", "count"),
    participated_amount=("amount", "sum"),
    total_won_amount=("total_funded_amount", "sum")
)

partner_summary = all_deals.join(won_deals, how="left").fillna(0)
partner_summary["closed_won_pct"] = partner_summary["participated_deals"] / partner_summary["total_deals"]
partner_summary["avg_participation_pct"] = partner_summary["participated_amount"] / partner_summary["total_won_amount"]

# ----------------------------
# Display metrics sections
# ----------------------------

# Deal Overview
st.subheader("📊 Deal Overview")
col1, col2, col3 = st.columns(3)
col1.metric("Total Deals", total_deals)
col2.metric("Closed Won", len(closed_won))
col3.metric("Close Ratio", f"{participation_ratio:.2%}")

# Financial Performance
st.subheader("💰 Financial Performance")
col4, col5, col6 = st.columns(3)
col4.metric("Total Capital Deployed", f"${total_capital_deployed:,.0f}")
col5.metric("Total Expected Return", f"${total_expected_return_sum:,.0f}")
col6.metric("MOIC", f"{moic:.2f}")

col7, col8, col9 = st.columns(3)
col7.metric("Projected IRR", f"{projected_irr:.2%}")
col8.metric("Avg % of Deal", f"{avg_participation_pct:.2%}")
col9.metric("Commission Paid", f"${total_commissions_paid:,.0f}")

# Deal Characteristics
st.subheader("📋 Deal Characteristics")
col10, col11, col12 = st.columns(3)
col10.metric("Avg Participation ($)", f"${avg_amount:,.0f}")
col11.metric("Avg Factor", f"{avg_factor:.2f}")
col12.metric("Avg Term (mo)", f"{avg_term:.1f}")

st.subheader("📈 Deal Flow Trends")
st.markdown(f"**Last 30 days**: {stats_30['count']} deals, ${stats_30['funded']:,.0f} funded")

col1, col2 = st.columns(2)
col1.metric("vs. Last 60 Days (Count)", count_change_60)
col1.metric("vs. Last 90 Days (Count)", count_change_90)

col2.metric("vs. Last 60 Days (Funded)", funded_change_60)
col2.metric("vs. Last 90 Days (Funded)", funded_change_90)

# ----------------------------
# Charts and visualizations
# ----------------------------

# Total Funded Amount by Month
st.subheader("📈 Total Funded Amount by Month")
funded_chart = alt.Chart(monthly_funded).mark_bar(
    size=45, 
    color=PRIMARY_COLOR
).encode(
    x=alt.X("month:T", axis=alt.Axis(labelAngle=0, title="")),
    y=alt.Y("total_funded_amount:Q", 
            title="Total Funded ($)", 
            axis=alt.Axis(format="$.1s", titlePadding=25, labelPadding=20, tickCount=4, labelFontSize=10)),
    tooltip=[alt.Tooltip("total_funded_amount", title="Total Funded", format="$,.0f")]
)

# Add average line
avg_line = alt.Chart(monthly_funded).mark_rule(
    color="gray", 
    strokeWidth=2, 
    strokeDash=[4,2], 
    opacity=0.6
).encode(
    y=alt.Y("mean(total_funded_amount):Q")
)

funded_combined = (funded_chart + avg_line).properties(
    width=600, 
    height=450
).configure_axis(
    labelFontSize=10,
    titleFontSize=12
)

st.altair_chart(funded_combined, use_container_width=True)

# Deals per Month by Partner Source
st.subheader("🤝 Deals per Month by Partner Source")
partner_chart = alt.Chart(monthly_partner).mark_bar(size=45).encode(
    x=alt.X("month:T", title=None, axis=alt.Axis(labelAngle=0)),
    y=alt.Y("count:Q", 
            title="Deal Count", 
            axis=alt.Axis(titlePadding=25, labelPadding=20, tickCount=4, labelFontSize=10)),
    color=alt.Color(
        "partner_source:N",
        scale=alt.Scale(range=COLOR_PALETTE),
        title="Partner Source"
    ),
    tooltip=[
        alt.Tooltip("partner_source", title="Partner"),
        alt.Tooltip("count", title="Deal Count")
    ]
).properties(
    width=600, 
    height=450
).configure_axis(
    labelFontSize=10,
    titleFontSize=12
)

st.altair_chart(partner_chart, use_container_width=True)

# Participated Amount by Month
st.subheader("💵 Participated Amount by Month")
bar_color = PRIMARY_COLOR if df["partner_source"].nunique() <= 1 else COLOR_PALETTE[1]

amount_chart = alt.Chart(monthly_amount).mark_bar(
    size=45, 
    color=bar_color
).encode(
    x=alt.X("month:T", axis=alt.Axis(labelAngle=0, title="")),
    y=alt.Y("amount:Q", 
            title="Amount ($)", 
            axis=alt.Axis(format="$.1s", titlePadding=25, labelPadding=20, tickCount=4, labelFontSize=10)),
    tooltip=[alt.Tooltip("amount", title="Amount", format="$,.0f")]
)

# Add average line
amount_avg_line = alt.Chart(monthly_amount).mark_rule(
    color="gray", 
    strokeWidth=2, 
    strokeDash=[4,2], 
    opacity=0.6
).encode(
    y=alt.Y("mean(amount):Q")
)

amount_combined = (amount_chart + amount_avg_line).properties(
    width=600, 
    height=450
).configure_axis(
    labelFontSize=10,
    titleFontSize=12
)

st.altair_chart(amount_combined, use_container_width=True)

# ----------------------------
# Partner Summary Table
# ----------------------------
st.subheader("📊 Partner Summary Table")

# Format the summary for display
partner_summary["$ Opportunities"] = partner_summary["total_amount"].apply(lambda x: f"${x:,.0f}")
partner_summary["Participated $"] = partner_summary["participated_amount"].apply(lambda x: f"${x:,.0f}")
partner_summary["% Closed Won"] = partner_summary["closed_won_pct"].apply(lambda x: f"{x:.2%}")
partner_summary["Avg % of Deal"] = partner_summary["avg_participation_pct"].apply(lambda x: f"{x:.2%}")

summary_display = partner_summary.reset_index()[[
    "partner_source", "total_deals", "$ Opportunities",
    "Participated $", "% Closed Won", "Avg % of Deal"
]].rename(columns={
    "partner_source": "Partner", 
    "total_deals": "Total Deals"
})

st.dataframe(summary_display, use_container_width=True)

# ----------------------------
# Download functions
# ----------------------------
def create_pdf_from_html(html: str):
    result = io.BytesIO()
    pisa.CreatePDF(io.StringIO(html), dest=result)
    return result.getvalue()

# ----------------------------
# Download buttons
# ----------------------------
csv = summary_display.to_csv(index=False).encode("utf-8")
st.download_button(
    label="📥 Download Partner Summary as CSV",
    data=csv,
    file_name="partner_summary.csv",
    mime="text/csv"
)

html = summary_display.to_html(index=False)
pdf = create_pdf_from_html(html)

st.download_button(
    label="📄 Download Partner Summary as PDF",
    data=pdf,
    file_name="partner_summary.pdf",
    mime="application/pdf"
)
