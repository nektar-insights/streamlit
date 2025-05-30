import streamlit as st
import pandas as pd
from supabase import create_client

# Supabase connection
url = st.secrets["supabase"]["url"]
key = st.secrets["supabase"]["service_role"]
supabase = create_client(url, key)

# Load Data
@st.cache_data(ttl=3600)
def load_qbo_data():
    res = supabase.table("qbo_transactions").select("*").execute()
    return pd.DataFrame(res.data)

df = load_qbo_data()
df["amount"] = pd.to_numeric(df["amount"], errors="coerce")

# Grouped by Name
by_name = df.groupby("name", dropna=False)["amount"].sum().reset_index()
by_name = by_name.sort_values(by="amount", ascending=False)
by_name["amount"] = by_name["amount"].map("${:,.2f}".format)

# Add total row
total_name = pd.DataFrame([{"name": "TOTAL", "amount": "${:,.2f}".format(df["amount"].sum())}])
by_name_display = pd.concat([by_name, total_name], ignore_index=True)

# Grouped by Transaction Type
by_type = df.groupby("transaction_type", dropna=False)["amount"].sum().reset_index()
by_type = by_type.sort_values(by="amount", ascending=False)
by_type["amount"] = by_type["amount"].map("${:,.2f}".format)

# Add total row
total_type = pd.DataFrame([{"transaction_type": "TOTAL", "amount": "${:,.2f}".format(df["amount"].sum())}])
by_type_display = pd.concat([by_type, total_type], ignore_index=True)

# Display
st.title("QBO Transaction Summary")

st.subheader("💰 Total Amount by Name")
st.dataframe(by_name_display, use_container_width=True)

st.subheader("📄 Total Amount by Transaction Type")
st.dataframe(by_type_display, use_container_width=True)
