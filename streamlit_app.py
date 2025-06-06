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

# Volume 
st.subheader("Total Funded Amount by Month")

funded_chart = alt.Chart(monthly_funded).mark_bar(size=45, color=PRIMARY_COLOR).encode(
    x=alt.X("month:T", axis=alt.Axis(labelAngle=0, title="Month")),
    y=alt.Y("total_funded_amount:Q", title="Total Funded ($)", axis=alt.Axis(format="$,.0f")),
    tooltip=[alt.Tooltip("total_funded_amount", title="Total Funded", format="$,.0f")]
)

funded_avg_line = alt.Chart(monthly_funded).mark_rule(
    color="gray", strokeWidth=2, strokeDash=[4, 2], opacity=0.6
).encode(
    y=alt.Y("mean(total_funded_amount):Q")
)

st.altair_chart((funded_chart + funded_avg_line).properties(height=300), use_container_width=True)

st.subheader("Total Deal Count by Month")

monthly_deals = df.groupby("month").size().reset_index(name="deal_count")

deal_count_chart = alt.Chart(monthly_deals).mark_bar(size=45, color=COLOR_PALETTE[2]).encode(
    x=alt.X("month:T", axis=alt.Axis(labelAngle=0, title="Month")),
    y=alt.Y("deal_count:Q", title="Deal Count"),
    tooltip=["month", "deal_count"]
)

st.altair_chart(deal_count_chart.properties(height=300), use_container_width=True)

st.subheader("Participation Trends by Month")

# Group by month and participation
df["is_participated"] = df["is_closed_won"] == True
participation_trend = df.groupby(["month", "is_participated"]).agg(
    deal_count=("id", "count"),
    total_amount=("amount", "sum")
).reset_index()

# Deal count chart
count_chart = alt.Chart(participation_trend).mark_bar().encode(
    x=alt.X("month:T", title="Month"),
    y=alt.Y("deal_count:Q", title="Deal Count"),
    color=alt.Color("is_participated:N", title="Participated", scale=alt.Scale(
        domain=[True, False],
        range=[PRIMARY_COLOR, COLOR_PALETTE[3]]
    )),
    tooltip=["month", "deal_count", "is_participated"]
).properties(title="Deal Count by Participation", height=300)

# Amount chart
amount_chart = alt.Chart(participation_trend).mark_bar().encode(
    x=alt.X("month:T", title="Month"),
    y=alt.Y("total_amount:Q", title="Amount ($)", axis=alt.Axis(format="$,.0f")),
    color=alt.Color("is_participated:N", title="Participated", scale=alt.Scale(
        domain=[True, False],
        range=[PRIMARY_COLOR, COLOR_PALETTE[3]]
    )),
    tooltip=["month", "total_amount", "is_participated"]
).properties(title="Participation Amount by Month", height=300)

st.altair_chart(count_chart, use_container_width=True)
st.altair_chart(amount_chart, use_container_width=True)

# ----------------------------
# 📈 Total Funded Amount by Month
# ----------------------------
monthly_funded = df.groupby("month")["total_funded_amount"].sum().reset_index()
funded_chart = alt.Chart(monthly_funded).mark_bar(size=50, color=PRIMARY_COLOR).encode(
    x=alt.X("month:T", axis=alt.Axis(labelAngle=0, title="Month")),
    y=alt.Y("total_funded_amount:Q", title="Total Funded ($)", axis=alt.Axis(format="$,.0f")),
    tooltip=[alt.Tooltip("total_funded_amount", title="Total Funded", format="$,.0f")]
)
funded_avg = alt.Chart(monthly_funded).mark_rule(color="gray", strokeDash=[4, 2]).encode(
    y=alt.Y("mean(total_funded_amount):Q")
)
funded_trend = alt.Chart(monthly_funded).mark_line(color="#1f77b4", strokeWidth=2).encode(
    x="month:T",
    y="total_funded_amount:Q"
)
st.subheader("Total Funded Amount by Month")
st.altair_chart((funded_chart + funded_avg + funded_trend).properties(height=320), use_container_width=True)

# ----------------------------
# 📊 Total Deal Count by Month
# ----------------------------
monthly_deals = df.groupby("month").size().reset_index(name="deal_count")
deal_chart = alt.Chart(monthly_deals).mark_bar(size=50, color=COLOR_PALETTE[2]).encode(
    x=alt.X("month:T", title="Month"),
    y=alt.Y("deal_count:Q", title="Deal Count"),
    tooltip=[alt.Tooltip("deal_count", title="Deal Count")]
)
deal_avg = alt.Chart(monthly_deals).mark_rule(color="gray", strokeDash=[4, 2]).encode(
    y=alt.Y("mean(deal_count):Q")
)
deal_trend = alt.Chart(monthly_deals).mark_line(color="#e45756", strokeWidth=2).encode(
    x="month:T",
    y="deal_count:Q"
)
st.subheader("Total Deal Count by Month")
st.altair_chart((deal_chart + deal_avg + deal_trend).properties(height=320), use_container_width=True)

# ----------------------------
# 📊 Participation Trends by Month (Count) – Only Participated = True
# ----------------------------
df["is_participated"] = df["is_closed_won"] == True
participated_only = df[df["is_participated"] == True]
monthly_participation = participated_only.groupby("month").agg(
    deal_count=("id", "count")
).reset_index()

participation_chart = alt.Chart(monthly_participation).mark_bar(size=60, color=PRIMARY_COLOR).encode(
    x=alt.X("month:T", title="Month"),
    y=alt.Y("deal_count:Q", title="Participated Deals"),
    tooltip=[alt.Tooltip("deal_count", title="Participated Count")]
)
participation_avg = alt.Chart(monthly_participation).mark_rule(color="gray", strokeDash=[4, 2]).encode(
    y=alt.Y("mean(deal_count):Q")
)
participation_trend = alt.Chart(monthly_participation).mark_line(color="#FF9900", strokeWidth=2).encode(
    x="month:T",
    y="deal_count:Q"
)
st.subheader("Participation Trends by Month")
st.altair_chart((participation_chart + participation_avg + participation_trend).properties(height=320), use_container_width=True)

# ----------------------------
# 💵 Participation Amount by Month (Unstacked)
# ----------------------------
participation_amount = participated_only.groupby("month").agg(
    total_amount=("amount", "sum")
).reset_index()

amount_chart = alt.Chart(participation_amount).mark_bar(size=60, color=PRIMARY_COLOR).encode(
    x=alt.X("month:T", title="Month"),
    y=alt.Y("total_amount:Q", title="Participation Amount ($)", axis=alt.Axis(format="$,.0f")),
    tooltip=[alt.Tooltip("total_amount", title="Amount", format="$,.0f")]
)
amount_avg = alt.Chart(participation_amount).mark_rule(color="gray", strokeDash=[4, 2]).encode(
    y=alt.Y("mean(total_amount):Q")
)
amount_trend = alt.Chart(participation_amount).mark_line(color="#17a2b8", strokeWidth=2).encode(
    x="month:T",
    y="total_amount:Q"
)
st.subheader("Participation Amount by Month")
st.altair_chart((amount_chart + amount_avg + amount_trend).properties(height=320), use_container_width=True)

# ----------------------------
# 📉 Participation Rate (% of Deals)
# ----------------------------
monthly_participation_ratio = df.groupby("month").agg(
    total_deals=("id", "count"),
    participated_deals=("is_participated", "sum")
).reset_index()
monthly_participation_ratio["participation_pct"] = monthly_participation_ratio["participated_deals"] / monthly_participation_ratio["total_deals"]

rate_line = alt.Chart(monthly_participation_ratio).mark_line(color="#e45756", strokeWidth=3).encode(
    x=alt.X("month:T", title="Month"),
    y=alt.Y("participation_pct:Q", title="Participation Rate", axis=alt.Axis(format=".0%")),
    tooltip=[alt.Tooltip("participation_pct", format=".1%")]
).properties(height=300)

st.subheader("Monthly Participation Rate (% of Deals)")
st.altair_chart(rate_line, use_container_width=True)


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
