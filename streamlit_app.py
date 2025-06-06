import streamlit as st
import pandas as pd
import altair as alt
from supabase import create_client
from datetime import datetime
import io
from xhtml2pdf import pisa

# ----------------------------
# Color Palette
# ----------------------------
PRIMARY_COLOR = "#34a853"
PERFORMANCE_GRADIENT = ["#e8f5e8", "#34a853", "#1e7e34"]
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
today = pd.to_datetime("today").normalize()

# ----------------------------
# Data preprocessing
# ----------------------------
cols_to_convert = ["amount", "total_funded_amount", "factor_rate", "loan_term", "commission"]
df["date_created"] = pd.to_datetime(df["date_created"], errors="coerce")
df["month"] = df["date_created"].dt.to_period("M").astype(str)
df[cols_to_convert] = df[cols_to_convert].apply(pd.to_numeric, errors="coerce")
df["loan_id"] = df["loan_id"].astype("string")
df["is_participated"] = df["is_closed_won"] == True

# ----------------------------
# Filters
# ----------------------------
st.title("Pipeline Dashboard")

min_date, max_date = df["date_created"].min(), df["date_created"].max()
start_date, end_date = st.date_input(
    "Filter by Date Range", 
    [min_date, max_date], 
    min_value=min_date, 
    max_value=max_date
)
df = df[(df["date_created"] >= pd.to_datetime(start_date)) & 
        (df["date_created"] <= pd.to_datetime(end_date))]

partner_options = sorted(df["partner_source"].dropna().unique())
selected_partner = st.selectbox(
    "Filter by Partner Source", 
    options=["All Partners"] + partner_options
)
if selected_partner != "All Partners":
    df = df[df["partner_source"] == selected_partner]

participation_filter = st.radio(
    "Show Deals", 
    ["All Deals", "Participated Only", "Not Participated"]
)
if participation_filter == "Participated Only":
    df = df[df["is_closed_won"] == True]
elif participation_filter == "Not Participated":
    df = df[df["is_closed_won"] != True]

# ----------------------------
# Calculate all metrics
# ----------------------------
closed_won = df[df["is_closed_won"] == True]
total_deals = len(df)
participation_ratio = len(closed_won) / total_deals if total_deals > 0 else 0
months = df["month"].nunique()

# Deal characteristics
avg_amount = closed_won["amount"].mean()
avg_factor = closed_won["factor_rate"].mean()
avg_term = closed_won["loan_term"].mean()
avg_participation_pct = (closed_won["amount"] / closed_won["total_funded_amount"]).mean()
avg_commission = closed_won["commission"].mean()

# Financial calculations
PLATFORM_FEE_RATE = 0.04
total_capital_deployed = closed_won["amount"].sum()
total_commissions_paid = (closed_won["amount"] * closed_won["commission"]).sum()
total_platform_fee = total_capital_deployed * PLATFORM_FEE_RATE
total_expected_return = ((closed_won["amount"] * closed_won["factor_rate"]) - 
                        closed_won["commission"] - 
                        (closed_won["amount"] * PLATFORM_FEE_RATE))
total_expected_return_sum = total_expected_return.sum()
moic = total_expected_return_sum / total_capital_deployed if total_capital_deployed > 0 else 0
projected_irr = (moic ** (12 / avg_term) - 1) if avg_term > 0 else 0

# Rolling deal flow calculations - Fixed to look back from today
periods = [
    ("0-30 Days", 0, 30), 
    ("31-60 Days", 31, 60), 
    ("61-90 Days", 61, 90), 
    ("91-120 Days", 91, 120)
]
flow_data = []
for label, start, end in periods:
    window = df[(df["date_created"] >= today - pd.Timedelta(days=end)) & 
                (df["date_created"] <= today - pd.Timedelta(days=start))]
    flow_data.append({
        "Period": label, 
        "Deals": len(window), 
        "Total Funded": window["total_funded_amount"].sum()
    })
flow_df = pd.DataFrame(flow_data)

# Monthly aggregations
monthly_funded = df.groupby("month")["total_funded_amount"].sum().reset_index()
monthly_deals = df.groupby("month").size().reset_index(name="deal_count")
participated_only = df[df["is_participated"] == True]
monthly_participation = participated_only.groupby("month").agg(
    deal_count=("id", "count"),
    total_amount=("amount", "sum")
).reset_index()
monthly_participation_ratio = df.groupby("month").agg(
    total_deals=("id", "count"),
    participated_deals=("is_participated", "sum")
).reset_index()
monthly_participation_ratio["participation_pct"] = (
    monthly_participation_ratio["participated_deals"] / 
    monthly_participation_ratio["total_deals"]
)

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
st.subheader("Deal Overview")
col1, col2, col3 = st.columns(3)
col1.metric("Total Deals", total_deals)
col2.metric("Closed Won", len(closed_won))
col3.metric("Close Ratio", f"{participation_ratio:.2%}")

st.subheader("Financial Performance")
col4, col5, col6 = st.columns(3)
col4.metric("Total Capital Deployed", f"${total_capital_deployed:,.0f}")
col5.metric("Total Expected Return", f"${total_expected_return_sum:,.0f}")
col6.metric("MOIC", f"{moic:.2f}")

col7, col8, col9 = st.columns(3)
col7.metric("Projected IRR", f"{projected_irr:.2%}")
col8.metric("Avg % of Deal", f"{avg_participation_pct:.2%}")
col9.metric("Commission Paid", f"${total_commissions_paid:,.0f}")

st.subheader("Deal Characteristics")
col10, col11, col12 = st.columns(3)
col10.metric("Avg Participation ($)", f"${avg_amount:,.0f}")
col11.metric("Avg Factor", f"{avg_factor:.2f}")
col12.metric("Avg Term (mo)", f"{avg_term:.1f}")

# ----------------------------
# Rolling Deal Flow
# ----------------------------
st.subheader("Rolling Deal Flow Trends")

# Add change calculations
flow_df["Deal Change"] = flow_df["Deals"].diff().fillna(0).astype(int)
flow_df["Deal Change %"] = flow_df["Deals"].pct_change().fillna(0).apply(lambda x: f"{x:.1%}")
flow_df["Funded Change"] = flow_df["Total Funded"].diff().fillna(0).astype(int)
flow_df["Funded Change %"] = flow_df["Total Funded"].pct_change().fillna(0).apply(lambda x: f"{x:.1%}")

# Display formatted table
flow_df_display = flow_df.copy()
flow_df_display["Total Funded Display"] = flow_df_display["Total Funded"].apply(lambda x: f"${x:,.0f}")
flow_df_display["Funded Change Display"] = flow_df_display["Funded Change"].apply(lambda x: f"${x:,.0f}")

st.dataframe(
    flow_df_display[["Period", "Deals", "Deal Change", "Deal Change %", 
                     "Total Funded Display", "Funded Change Display", "Funded Change %"]], 
    use_container_width=True,
    column_config={
        "Total Funded Display": "Total Funded",
        "Funded Change Display": "Funded Change"
    }
)

# Rolling flow charts
flow_chart = alt.Chart(flow_df).mark_bar(
    size=60, 
    color=PRIMARY_COLOR,
    cornerRadiusTopLeft=3,
    cornerRadiusTopRight=3
).encode(
    x=alt.X("Period:N", 
            sort=["91-120 Days", "61-90 Days", "31-60 Days", "0-30 Days"],
            axis=alt.Axis(labelAngle=0)),
    y=alt.Y("Deals:Q", title="Deal Count"),
    tooltip=[
        alt.Tooltip("Period", title="Period"),
        alt.Tooltip("Deals", title="Deal Count"),
        alt.Tooltip("Deal Change", title="Change vs Previous"),
        alt.Tooltip("Deal Change %", title="Percent Change")
    ]
).properties(
    height=350,
    title="Deal Count by Period"
)

funded_flow_chart = alt.Chart(flow_df).mark_bar(
    size=60, 
    color=COLOR_PALETTE[2],
    cornerRadiusTopLeft=3,
    cornerRadiusTopRight=3
).encode(
    x=alt.X("Period:N", 
            sort=["91-120 Days", "61-90 Days", "31-60 Days", "0-30 Days"],
            axis=alt.Axis(labelAngle=0)),
    y=alt.Y("Total Funded:Q", 
            title="Total Funded ($)", 
            axis=alt.Axis(format="$.1s", titlePadding=15)),
    tooltip=[
        alt.Tooltip("Period", title="Period"),
        alt.Tooltip("Total Funded", title="Total Funded", format="$,.0f"),
        alt.Tooltip("Funded Change", title="Change vs Previous", format="$,.0f"),
        alt.Tooltip("Funded Change %", title="Percent Change")
    ]
).properties(
    height=350,
    title="Total Funded by Period"
)

st.altair_chart(flow_chart, use_container_width=True)
st.altair_chart(funded_flow_chart, use_container_width=True)

# ----------------------------
# Monthly trend charts
# ----------------------------
st.subheader("Total Funded Amount by Month")
funded_chart = alt.Chart(monthly_funded).mark_bar(
    size=50, 
    color=PRIMARY_COLOR,
    cornerRadiusTopLeft=3,
    cornerRadiusTopRight=3
).encode(
    x=alt.X("month:T", axis=alt.Axis(labelAngle=0, title="Month")),
    y=alt.Y("total_funded_amount:Q", 
            title="Total Funded ($)", 
            axis=alt.Axis(format="$.1s", titlePadding=15)),
    tooltip=[
        alt.Tooltip("month", title="Month"),
        alt.Tooltip("total_funded_amount", title="Total Funded Amount", format="$,.0f")
    ]
)

funded_avg = alt.Chart(monthly_funded).mark_rule(
    color="gray", 
    strokeWidth=2, 
    strokeDash=[4, 2],
    opacity=0.7
).encode(
    y=alt.Y("mean(total_funded_amount):Q")
)

# Add regression line
funded_regression = alt.Chart(monthly_funded).mark_line(
    color="#1f77b4", 
    strokeWidth=3
).transform_regression(
    'month', 'total_funded_amount'
).encode(
    x='month:T',
    y='total_funded_amount:Q'
)

st.altair_chart(
    (funded_chart + funded_avg + funded_regression).properties(height=400), 
    use_container_width=True
)

st.subheader("Total Deal Count by Month")
deal_chart = alt.Chart(monthly_deals).mark_bar(
    size=50, 
    color=COLOR_PALETTE[2],
    cornerRadiusTopLeft=3,
    cornerRadiusTopRight=3
).encode(
    x=alt.X("month:T", title="Month", axis=alt.Axis(labelAngle=0)),
    y=alt.Y("deal_count:Q", title="Deal Count"),
    tooltip=[
        alt.Tooltip("month", title="Month"),
        alt.Tooltip("deal_count", title="Deal Count")
    ]
)

deal_avg = alt.Chart(monthly_deals).mark_rule(
    color="gray", 
    strokeWidth=2, 
    strokeDash=[4, 2],
    opacity=0.7
).encode(
    y=alt.Y("mean(deal_count):Q")
)

# Add regression line
deal_regression = alt.Chart(monthly_deals).mark_line(
    color="#e45756", 
    strokeWidth=3
).transform_regression(
    'month', 'deal_count'
).encode(
    x='month:T',
    y='deal_count:Q'
)

st.altair_chart(
    (deal_chart + deal_avg + deal_regression).properties(height=400), 
    use_container_width=True
)

st.subheader("Participation Trends by Month")
participation_chart = alt.Chart(monthly_participation).mark_bar(
    size=60, 
    color=PRIMARY_COLOR,
    cornerRadiusTopLeft=3,
    cornerRadiusTopRight=3
).encode(
    x=alt.X("month:T", title="Month", axis=alt.Axis(labelAngle=0)),
    y=alt.Y("deal_count:Q", title="Participated Deals"),
    tooltip=[
        alt.Tooltip("month", title="Month"),
        alt.Tooltip("deal_count", title="Participated Count")
    ]
)

participation_avg = alt.Chart(monthly_participation).mark_rule(
    color="gray", 
    strokeWidth=2, 
    strokeDash=[4, 2],
    opacity=0.7
).encode(
    y=alt.Y("mean(deal_count):Q")
)

# Add regression line
participation_regression = alt.Chart(monthly_participation).mark_line(
    color="#FF9900", 
    strokeWidth=3
).transform_regression(
    'month', 'deal_count'
).encode(
    x='month:T',
    y='deal_count:Q'
)

st.altair_chart(
    (participation_chart + participation_avg + participation_regression).properties(height=400), 
    use_container_width=True
)

st.subheader("Participation Amount by Month")
amount_chart = alt.Chart(monthly_participation).mark_bar(
    size=60, 
    color=PRIMARY_COLOR,
    cornerRadiusTopLeft=3,
    cornerRadiusTopRight=3
).encode(
    x=alt.X("month:T", title="Month", axis=alt.Axis(labelAngle=0)),
    y=alt.Y("total_amount:Q", 
            title="Participation Amount ($)", 
            axis=alt.Axis(format="$.1s", titlePadding=15)),
    tooltip=[
        alt.Tooltip("month", title="Month"),
        alt.Tooltip("total_amount", title="Participation Amount", format="$,.0f")
    ]
)

amount_avg = alt.Chart(monthly_participation).mark_rule(
    color="gray", 
    strokeWidth=2, 
    strokeDash=[4, 2],
    opacity=0.7
).encode(
    y=alt.Y("mean(total_amount):Q")
)

# Add regression line
amount_regression = alt.Chart(monthly_participation).mark_line(
    color="#17a2b8", 
    strokeWidth=3
).transform_regression(
    'month', 'total_amount'
).encode(
    x='month:T',
    y='total_amount:Q'
)

st.altair_chart(
    (amount_chart + amount_avg + amount_regression).properties(height=400), 
    use_container_width=True
)

st.subheader("Monthly Participation Rate")
rate_line = alt.Chart(monthly_participation_ratio).mark_line(
    color="#e45756", 
    strokeWidth=4,
    point=alt.OverlayMarkDef(color="#e45756", size=80, filled=True)
).encode(
    x=alt.X("month:T", title="Month", axis=alt.Axis(labelAngle=0)),
    y=alt.Y("participation_pct:Q", 
            title="Participation Rate", 
            axis=alt.Axis(format=".0%", titlePadding=15)),
    tooltip=[
        alt.Tooltip("month", title="Month"),
        alt.Tooltip("participation_pct", title="Participation Rate", format=".1%")
    ]
).properties(
    height=350
)

st.altair_chart(rate_line, use_container_width=True)

# ----------------------------
# Partner Summary Table
# ----------------------------
st.subheader("Partner Summary Table")

# Format summary for display
partner_summary["$ Opportunities"] = partner_summary["total_amount"].apply(lambda x: f"${x:,.0f}")
partner_summary["Participated $"] = partner_summary["participated_amount"].apply(lambda x: f"${x:,.0f}")
partner_summary["% Closed Won"] = partner_summary["closed_won_pct"].apply(lambda x: f"{x:.2%}")
partner_summary["Avg % of Deal"] = partner_summary["avg_participation_pct"].apply(lambda x: f"{x:.2%}")

summary_display = partner_summary.reset_index()[[
    "partner_source", "total_deals", "$ Opportunities",
    "Participated $", "% Closed Won", "Avg % of Deal"
]].rename(columns={"partner_source": "Partner", "total_deals": "Total Deals"})

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
    label="Download Partner Summary as CSV",
    data=csv,
    file_name="partner_summary.csv",
    mime="text/csv"
)

html = summary_display.to_html(index=False)
pdf = create_pdf_from_html(html)
st.download_button(
    label="Download Partner Summary as PDF",
    data=pdf,
    file_name="partner_summary.pdf",
    mime="application/pdf"
)
