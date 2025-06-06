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

# Preprocessing
cols_to_convert = ["amount", "total_funded_amount", "factor_rate", "loan_term", "commission"]
df["date_created"] = pd.to_datetime(df["date_created"], errors="coerce")
df["month"] = df["date_created"].dt.to_period("M").astype(str)
df[cols_to_convert] = df[cols_to_convert].apply(pd.to_numeric, errors="coerce")
df["loan_id"] = df["loan_id"].astype("string")

# ----------------------------
# Filters
# ----------------------------
st.title("Pipeline Dashboard")

min_date, max_date = df["date_created"].min(), df["date_created"].max()
start_date, end_date = st.date_input("Filter by Date Range", [min_date, max_date], min_value=min_date, max_value=max_date)
df = df[(df["date_created"] >= pd.to_datetime(start_date)) & (df["date_created"] <= pd.to_datetime(end_date))]

partner_options = sorted(df["partner_source"].dropna().unique())
selected_partner = st.selectbox("Filter by Partner Source", options=["All Partners"] + partner_options)
if selected_partner != "All Partners":
    df = df[df["partner_source"] == selected_partner]

participation_filter = st.radio("Show Deals", ["All Deals", "Participated Only", "Not Participated"])
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
# Rolling Deal Flow
# ----------------------------
periods = [("0–30 Days", 0, 30), ("30–60 Days", 30, 60), ("60–90 Days", 60, 90), ("90–120 Days", 90, 120)]
flow_data = []
for label, start, end in periods:
    window = df[(df["date_created"] >= today - pd.Timedelta(days=end)) & (df["date_created"] < today - pd.Timedelta(days=start))]
    flow_data.append({"Period": label, "Deals": len(window), "Total Funded": window["total_funded_amount"].sum()})
flow_df = pd.DataFrame(flow_data)

# ----------------------------
# Display Summary Metrics
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
# Insert Rolling Flow Table and Charts
# ----------------------------
st.subheader("Rolling Deal Flow Trends")

# Add % change columns
flow_df["Deals Δ"] = flow_df["Deals"].diff().fillna(0).astype(int)
flow_df["Deals %"] = flow_df["Deals"].pct_change().fillna(0).apply(lambda x: f"{x:.1%}")

flow_df["Total Funded Δ"] = flow_df["Total Funded"].diff().fillna(0).astype(int)
flow_df["Funded %"] = flow_df["Total Funded"].pct_change().fillna(0).apply(lambda x: f"{x:.1%}")

# Display formatted table
flow_df_display = flow_df.copy()
flow_df_display["Total Funded"] = flow_df_display["Total Funded"].apply(lambda x: f"${x:,.0f}")
flow_df_display["Total Funded Δ"] = flow_df_display["Total Funded Δ"].apply(lambda x: f"${x:,.0f}")
st.dataframe(flow_df_display[[
    "Period", "Deals", "Deals Δ", "Deals %", 
    "Total Funded", "Total Funded Δ", "Funded %"
]], use_container_width=True)

# Bar charts
flow_chart = alt.Chart(flow_df).mark_bar(size=40).encode(
    x=alt.X("Period:N", sort=["90–120 Days", "60–90 Days", "30–60 Days", "0–30 Days"]),
    y=alt.Y("Deals:Q", title="Deal Count"),
    tooltip=["Period", "Deals", "Deals Δ", "Deals %"]
).properties(height=300)

funded_chart = alt.Chart(flow_df).mark_bar(size=40, color="#4a90e2").encode(
    x=alt.X("Period:N", sort=["90–120 Days", "60–90 Days", "30–60 Days", "0–30 Days"]),
    y=alt.Y("Total Funded:Q", title="Total Funded ($)", axis=alt.Axis(format="$,.0f")),
    tooltip=["Period", "Total Funded", "Total Funded Δ", "Funded %"]
).properties(height=300)

st.altair_chart(flow_chart, use_container_width=True)
st.altair_chart(funded_chart, use_container_width=True)

# ----------------------------
# Monthly Aggregations (Post-Filter)
# ----------------------------
monthly_funded = df.groupby("month")["total_funded_amount"].sum().reset_index()
monthly_deals = df.groupby("month").size().reset_index(name="deal_count")
monthly_amount = df.groupby("month")["amount"].sum().reset_index()

df["is_participated"] = df["is_closed_won"] == True
participation_trend = df.groupby(["month", "is_participated"]).agg(
    deal_count=("id", "count"),
    total_amount=("amount", "sum")
).reset_index()

# ----------------------------
# Partner Summary Table
# ----------------------------

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
partner_summary["$ Opportunities"] = partner_summary["total_amount"].apply(lambda x: f"${x:,.0f}")
partner_summary["Participated $"] = partner_summary["participated_amount"].apply(lambda x: f"${x:,.0f}")
partner_summary["% Closed Won"] = partner_summary["closed_won_pct"].apply(lambda x: f"{x:.2%}")
partner_summary["Avg % of Deal"] = partner_summary["avg_participation_pct"].apply(lambda x: f"{x:.2%}")

summary_display = partner_summary.reset_index()[[
    "partner_source", "total_deals", "$ Opportunities",
    "Participated $", "% Closed Won", "Avg % of Deal"
]].rename(columns={"partner_source": "Partner", "total_deals": "Total Deals"})

st.subheader("Partner Summary Table")
st.dataframe(summary_display, use_container_width=True)

# Funded Amount by Month
monthly_funded = df.groupby("month")["total_funded_amount"].sum().round(0).reset_index()
st.subheader("Total Funded Amount by Month")
funded_chart = alt.Chart(monthly_funded).mark_bar(size=45, color=color_palette[0]).encode(
    x=alt.X("month:T", axis=alt.Axis(labelAngle=0, title="")),
    y=alt.Y("total_funded_amount:Q", title="Total Funded ($)", axis=alt.Axis(format="$,.0f", titlePadding=10)),
    tooltip=[alt.Tooltip("total_funded_amount", title="Total Funded", format="$,.0f")]
) + alt.Chart(monthly_funded).mark_rule(color="gray", strokeWidth=2, strokeDash=[4,2], opacity=0.6).encode(
    y=alt.Y("mean(total_funded_amount):Q", title="Average Funded", axis=alt.Axis(format="$,.0f"))
)
st.altair_chart(funded_chart.properties(width=850, height=300), use_container_width=True)

# Deals per Month by Partner Source
monthly_partner = df.groupby(["month", "partner_source"]).size().reset_index(name="count")
st.subheader("Deals per Month by Partner Source")
partner_chart = alt.Chart(monthly_partner).mark_bar(size=45).encode(
    x=alt.X("month:T", title=None, axis=alt.Axis(labelAngle=0)),
    y=alt.Y("count:Q", title="Deal Count"),
    color=alt.Color(
        "partner_source:N",
        scale=alt.Scale(domain=["Fresh Funding", "TVT", "VitalCap"], range=color_palette)
    ),
    tooltip=["partner_source", "count"]
).properties(width=850, height=400)
st.altair_chart(partner_chart, use_container_width=True)

# Participated Amount by Month
monthly_amount = df.groupby("month")["amount"].sum().round(0).reset_index()
bar_color = color_palette[0] if df["partner_source"].nunique() <= 1 else color_palette[1]
st.subheader("Participated Amount by Month")
amount_chart = alt.Chart(monthly_amount).mark_bar(size=45, color=bar_color).encode(
    x=alt.X("month:T", axis=alt.Axis(labelAngle=0, title="")),
    y=alt.Y("amount:Q", title="Amount ($)", axis=alt.Axis(format="$,.0f", titlePadding=10)),
    tooltip=[alt.Tooltip("amount", title="Amount", format="$,.0f")]
) + alt.Chart(monthly_amount).mark_rule(color="gray", strokeWidth=2, strokeDash=[4,2], opacity=0.6).encode(
    y=alt.Y("mean(amount):Q", title="Average Amount", axis=alt.Axis(format="$,.0f"))
)
st.altair_chart(amount_chart.properties(width=850, height=300), use_container_width=True)

# ----------------------------
# Downloads
# ----------------------------
def create_pdf_from_html(html: str):
    result = io.BytesIO()
    pisa.CreatePDF(io.StringIO(html), dest=result)
    return result.getvalue()

csv = summary_display.to_csv(index=False).encode("utf-8")
st.download_button(
    label="Download Partner Summary as CSV",  # Removed emoji to avoid Unicode error
    data=csv,
    file_name="partner_summary.csv",
    mime="text/csv"
)

html = summary_display.to_html(index=False)
pdf = create_pdf_from_html(html)
st.download_button(
    label="Download Partner Summary as PDF",  # Removed emoji to avoid Unicode error
    data=pdf,
    file_name="partner_summary.pdf",
    mime="application/pdf"
)
