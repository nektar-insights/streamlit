import streamlit as st
import pandas as pd
from supabase import create_client

# Setup Supabase using secrets.toml
supabase = create_client(st.secrets["SUPABASE_URL"], st.secrets["SUPABASE_KEY"])

# Pull QBO transactions
res = supabase.table("qbo_transactions").select("*").execute()
df = pd.DataFrame(res.data)

# Convert amount to numeric
df["amount"] = pd.to_numeric(df["amount"], errors="coerce")

# Group by name
by_name = df.groupby("name")["amount"].sum().reset_index().sort_values(by="amount", ascending=False)

# Group by transaction type
by_type = df.groupby("transaction_type")["amount"].sum().reset_index().sort_values(by="amount", ascending=False)

# Display results in Streamlit
st.subheader("💰 Total Amount by Name")
st.dataframe(by_name)

st.subheader("📄 Total Amount by Transaction Type")
st.dataframe(by_type)
