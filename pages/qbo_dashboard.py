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

# Prep Data
df["amount"] = pd.to_numeric(df["amount"], errors="coerce")

# Grouped by Name
by_name = df.groupby("name")["amount"].sum().reset_index().sort_values(by="amount", ascending=False)

# Grouped by Transaction Type
by_type = df.groupby("transaction_type")["amount"].sum().reset_index().sort_values(by="amount", ascending=False)

# Display
st.title("QBO Transaction Summary")

st.subheader("💰 Total Amount by Name")
st.dataframe(by_name, use_container_width=True)

st.subheader("📄 Total Amount by Transaction Type")
st.dataframe(by_type, use_container_width=True)
