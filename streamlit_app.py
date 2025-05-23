import streamlit as st
import pandas as pd
from supabase import create_client
from datetime import datetime

# Load secrets
url = st.secrets["supabase"]["url"]
key = st.secrets["supabase"]["service_role"]
supabase = create_client(url, key)

# Pull deals from Supabase
@st.cache_data(ttl=3600)
def load_deals():
    response = supabase.table("deals").select("*").execute()
    return pd.DataFrame(response.data)

df = load_deals()

# Convert date_created to datetime
df["date_created"] = pd.to_datetime(df["date_created"], errors="coerce")
df["month"] = df["date_created"].dt.to_period("M").astype(str)

# Create page layout
st.title("HubSpot Deals Dashboard")

# a) Total Count of Deals
total_deals = len(df)
st.metric("Total Deals", total_deals)

# b) Total Closed Won
closed_won = df[df["is_closed_won"] == True] 
st.metric("Deals Closed Won", len(closed_won))

# c) Participation Ratio
participation_ratio = len(closed_won) / total_deals if total_deals else 0
st.metric("Closed/Won Ratio", f"{participation_ratio:.2%}")

# d) Total Funded Amount by Month
df_funded = df.dropna(subset=["total_funded_amount"])
df_funded["total_funded_amount"] = pd.to_numeric(df_funded["total_funded_amount"], errors="coerce")
monthly_funded = df_funded.groupby("month")["total_funded_amount"].sum().reset_index()
st.subheader("Total Funded Amount by Month")
st.bar_chart(monthly_funded.set_index("month"))

# e) Count of Deals per Month by Partner Source
monthly_partner = df.groupby(["month", "partner_source"]).size().reset_index(name="count")
pivot_partner = monthly_partner.pivot(index="month", columns="partner_source", values="count").fillna(0)
st.subheader("Deals per Month by Partner Source")
st.line_chart(pivot_partner)

# f) Plot Dollars of Amount by Month
df["amount"] = pd.to_numeric(df["amount"], errors="coerce")
monthly_amount = df.groupby("month")["amount"].sum().reset_index()
st.subheader("Amount by Month")
st.bar_chart(monthly_amount.set_index("month"))
