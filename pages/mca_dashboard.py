# pages/mca_dashboard.py
import streamlit as st
import pandas as pd
from supabase import create_client

# Supabase connection
url = st.secrets["supabase"]["url"]
key = st.secrets["supabase"]["service_role"]
supabase = create_client(url, key)

@st.cache_data(ttl=3600)
def load_mca_deals():
    res = supabase.table("mca_deals").select("*").execute()
    return pd.DataFrame(res.data)

df = load_mca_deals()

# Data prep
df["funding_date"] = pd.to_datetime(df["funding_date"], errors="coerce")
df["purchase_price"] = pd.to_numeric(df["purchase_price"], errors="coerce")
df["receivables_amount"] = pd.to_numeric(df["receivables_amount"], errors="coerce")
df["current_balance"] = pd.to_numeric(df["current_balance"], errors="coerce")
df["past_due_amount"] = pd.to_numeric(df["past_due_amount"], errors="coerce")

# Filters
min_date = df["funding_date"].min()
max_date = df["funding_date"].max()
start_date, end_date = st.date_input("Filter by Funding Date", [min_date, max_date], min_value=min_date, max_value=max_date)
df = df[(df["funding_date"] >= pd.to_datetime(start_date)) & (df["funding_date"] <= pd.to_datetime(end_date))]

status_filter = st.multiselect("Status Category", df["status_category"].dropna().unique(), default=list(df["status_category"].dropna().unique()))
df = df[df["status_category"].isin(status_filter)]

# Metrics
total_deals = len(df)
total_funded = df["purchase_price"].sum()
total_receivables = df["receivables_amount"].sum()
total_past_due = df["past_due_amount"].sum()

st.title("MCA Deals Dashboard")
col1, col2, col3 = st.columns(3)
col1.metric("Total Deals", total_deals)
col2.metric("Total Funded", f"${total_funded:,.0f}")
col3.metric("Total Receivables", f"${total_receivables:,.0f}")

st.metric("Total Past Due", f"${total_past_due:,.0f}")

st.subheader("Preview")
st.dataframe(df[["deal_id", "dba", "funding_date", "status_category", "purchase_price", "receivables_amount", "past_due_amount"]], use_container_width=True)
