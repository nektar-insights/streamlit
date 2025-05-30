import streamlit as st
import pandas as pd
from supabase import create_client
import os

# Setup Supabase
supabase = create_client(os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_KEY"))

# Pull QBO transactions
res = supabase.table("qbo_transactions").select("*").execute()
df = pd.DataFrame(res.data)

# Convert amount to numeric (if not already)
df["amount"] = pd.to_numeric(df["amount"], errors="coerce")

# Group by name
by_name = df.groupby("name")["amount"].sum().reset_index().sort_values(by="amount", ascending=False)

# Group by transaction type
by_type = df.groupby("txn_type")["amount"].sum().reset_index().sort_values(by="amount", ascending=False)

# Display
print("Total Amount by Name:\n", by_name)
print("\nTotal Amount by Transaction Type:\n", by_type)
