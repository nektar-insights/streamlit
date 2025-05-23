import streamlit as st
import pandas as pd
import altair as alt
from supabase import create_client
from datetime import datetime

# Load Supabase
url = st.secrets["supabase"]["url"]
key = st.secrets["supabase"]["service_role"]
supabase = create_client(url, key)

# Load Data
@st.cache_data(ttl=3600)
def load_deals():
    res = supabase.table("deals").select("*").execute()
    return pd.DataFrame(res.data)

df = load_deals()

# Data Prep
df["date_created"] = pd.to_datetime(df["date_created"], errors="coerce")
df["month"] = df["date_created"].dt.to_period("M").astype(str)
df["amount"] = pd.to_numeric(df["amount"], errors="coerce")
df["total_funded_amount"] = pd.to_numeric(df["total_funded_amount"], errors="coerce")
df["factor_rate"] = pd.to_numeric(df["factor_rate"], errors="coerce")
df["loan_term"] = pd.to_numeric(df["loan_term"], errors="coerce")
df["commission"] = pd.to_numeric(df["commission"], errors="coerce")

# --- Filters ---
min_date = df["date_created"].min()
max_date = df["date_created"].max()

start_date, end_date = st.date_input(
    "Filter by Date Range", [min_date, max_date], min_value=min_date, max_value=max_date
)

df = df[(df["date_created"] >= pd.to_datetime(start_date)) &
        (df["date_created"] <= pd.to_datetime(end_date))]

partner_options = sorted(df["partner_source"].dropna().unique())
all_label = "All Partners"
partner_options_with_all = [all_label] + partner_options

selected_partner = st.selectbox("Filter by Partner Source", options=partner_options_with_all)
if selected_partner != all_label:
    df = df[df["partner_source"] == selected_partner]

# --- Participation Filter ---
participation_options = ["All Deals", "Participated Only", "Not Participated"]
participation_filter = st.radio("Show Deals", participation_options)
if participation_filter == "Participated Only":
    df = df[df["is_closed_won"] == True]
elif participation_filter == "Not Participated":
    df = df[df["is_closed_won"] != True]

# --- Metrics ---
closed_won = df[df["is_closed_won"] == True]
total_deals = len(df)
participation_ratio = len(closed_won) / total_deals if total_deals else 0
months = df["month"].nunique()
pacing = len(closed_won) / months if months else 0

avg_amount = closed_won["amount"].mean()
avg_factor = closed_won["factor_rate"].mean()
avg_term = closed_won["loan_term"].mean()
avg_participation_pct = (closed_won["amount"] / closed_won["total_funded_amount"]).mean()
avg_commission = closed_won["commission"].mean()
total_commissions_paid = (closed_won["amount"] * closed_won["commission"]).sum()

# --- Top Summary ---
st.title("HubSpot Deals Dashboard")
col1, col2, col3 = st.columns(3)
col1.metric("Total Deals", total_deals)
col2.metric("Closed Won", len(closed_won))
col3.metric("Close Ratio", f"{participation_ratio:.2%}")

col4, col5, col6 = st.columns(3)
col4.metric("Avg Participation ($)", f"${avg_amount:,.0f}")
col5.metric("Avg Factor", f"{avg_factor:.2f}")
col6.metric("Avg Term (mo)", f"{avg_term:.1f}")

col7, col8, col9 = st.columns(3)
col7.metric("Pacing (Deals/mo)", f"{pacing:.1f}")
col8.metric("Avg % of Deal", f"{avg_participation_pct:.2%}")
col9.metric("Commission Paid", f"${total_commissions_paid:,.0f}")

# --- Charts ---
color_palette = ['#34a853', '#394053', '#4E4A59', '#E6C79C', '#E5DCC5']

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

# Amount by Month
monthly_amount = df.groupby("month")["amount"].sum().round(0).reset_index()
bar_color = color_palette[0] if df["partner_source"].nunique() <= 1 else color_palette[1]
st.subheader("Amount by Month")
amount_chart = alt.Chart(monthly_amount).mark_bar(size=45, color=bar_color).encode(
    x=alt.X("month:T", axis=alt.Axis(labelAngle=0, title="")),
    y=alt.Y("amount:Q", title="Amount ($)", axis=alt.Axis(format="$,.0f", titlePadding=10)),
    tooltip=[alt.Tooltip("amount", title="Amount", format="$,.0f")]
) + alt.Chart(monthly_amount).mark_rule(color="gray", strokeWidth=2, strokeDash=[4,2], opacity=0.6).encode(
    y=alt.Y("mean(amount):Q", title="Average Amount", axis=alt.Axis(format="$,.0f"))
)
st.altair_chart(amount_chart.properties(width=850, height=300), use_container_width=True)

# --- Partner Summary ---
st.subheader("Partner Summary Table")

closed_won = df[df["is_closed_won"] == True]
all_deals = df.groupby("partner_source").agg(
    total_deals=("id", "count"),
    total_amount=("total_funded_amount", "sum")
)
won_deals = closed_won.groupby("partner_source").agg(
    participated_deals=("id", "count"),
    participated_amount=("amount", "sum"),
    total_won_amount=("total_funded_amount", "sum")
)

summary = all_deals.join(won_deals, how="left").fillna(0)
summary["closed_won_pct"] = summary["participated_deals"] / summary["total_deals"]
summary["avg_participation_pct"] = summary["participated_amount"] / summary["total_won_amount"]

summary["$ Opportunities"] = summary["total_amount"].apply(lambda x: f"${x:,.0f}")
summary["Participated $"] = summary["participated_amount"].apply(lambda x: f"${x:,.0f}")
summary["% Closed Won"] = summary["closed_won_pct"].apply(lambda x: f"{x:.2%}")
summary["Avg % of Deal"] = summary["avg_participation_pct"].apply(lambda x: f"{x:.2%}")

summary_display = summary.reset_index()[[
    "partner_source", "total_deals", "$ Opportunities",
    "Participated $", "% Closed Won", "Avg % of Deal"
]].rename(columns={"partner_source": "Partner", "total_deals": "Total Deals"})

st.dataframe(summary_display, use_container_width=True)
