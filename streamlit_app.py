import streamlit as st
import pandas as pd
import altair as alt
from supabase import create_client
from datetime import datetime

# Load secrets
url = st.secrets["supabase"]["url"]
key = st.secrets["supabase"]["service_role"]
supabase = create_client(url, key)

# Pull deals
@st.cache_data(ttl=3600)
def load_deals():
    res = supabase.table("deals").select("*").execute()
    return pd.DataFrame(res.data)

df = load_deals()

# Convert date_created
df["date_created"] = pd.to_datetime(df["date_created"], errors="coerce")
df["month"] = df["date_created"].dt.to_period("M").astype(str)

# --- Date filter ---
min_date = df["date_created"].min()
max_date = df["date_created"].max()

start_date, end_date = st.date_input(
    "Filter by Date Range", [min_date, max_date],
    min_value=min_date, max_value=max_date
)

df = df[(df["date_created"] >= pd.to_datetime(start_date)) &
        (df["date_created"] <= pd.to_datetime(end_date))]

# --- Calculations ---
df["amount"] = pd.to_numeric(df["amount"], errors="coerce")
df["total_funded_amount"] = pd.to_numeric(df["total_funded_amount"], errors="coerce")
df["factor_rate"] = pd.to_numeric(df["factor_rate"], errors="coerce")
df["loan_term"] = pd.to_numeric(df["loan_term"], errors="coerce")

total_deals = len(df)
closed_won = df[df["is_closed_won"] == "true"]
participation_ratio = len(closed_won) / total_deals if total_deals else 0
months = df["month"].nunique()
pacing = len(closed_won) / months if months else 0

avg_amount = closed_won["amount"].mean()
avg_factor = closed_won["factor_rate"].mean()
avg_term = closed_won["loan_term"].mean()

# --- UI Layout ---
st.title("HubSpot Deals Dashboard")

# Summary row
col1, col2, col3 = st.columns(3)
col1.metric("Total Deals", total_deals)
col2.metric("Closed Won", len(closed_won))
col3.metric("Close Ratio", f"{participation_ratio:.2%}")

col4, col5, col6 = st.columns(3)
col4.metric("Avg Participation ($)", f"${avg_amount:,.0f}")
col5.metric("Avg Factor", f"{avg_factor:.2f}")
col6.metric("Avg Term (mo)", f"{avg_term:.1f}")

st.metric("Pacing (Deals Closed Won per Month)", f"{pacing:.1f}")

# --- Charts ---
# Total Funded Amount by Month
monthly_funded = df.dropna(subset=["total_funded_amount"]).groupby("month")["total_funded_amount"].sum().round(0).reset_index()
st.subheader("Total Funded Amount by Month")
st.bar_chart(monthly_funded.set_index("month"))

# Count of Deals per Month by Partner Source
monthly_partner = df.groupby(["month", "partner_source"]).size().reset_index(name="count")
st.subheader("Deals per Month by Partner Source")
partner_chart = alt.Chart(monthly_partner).mark_bar().encode(
    x="month:T",
    y="count:Q",
    color="partner_source:N"
).properties(width=700, height=400)
st.altair_chart(partner_chart, use_container_width=True)

# Amount by Month
monthly_amount = df.groupby("month")["amount"].sum().round(0).reset_index()
st.subheader("Amount by Month")
st.bar_chart(monthly_amount.set_index("month"))
