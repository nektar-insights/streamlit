import pandas as pd
import streamlit as st
from utils.imports import get_supabase_client

supabase = get_supabase_client()

def fetch_all_rows(table_name: str, chunk_size: int = 1000) -> pd.DataFrame:
    """Fetch all rows from a Supabase table using pagination to avoid limits"""
    rows = []
    start = 0
    while True:
        end = start + chunk_size - 1
        response = (
            supabase.table(table_name)
            .select("*")
            .range(start, end)
            .execute()
        )
        batch = response.data
        if not batch:
            break
        rows.extend(batch)
        start += chunk_size
    return pd.DataFrame(rows)

@st.cache_data(ttl=3600)
def load_qbo_data():
    df_txn = fetch_all_rows("qbo_invoice_payments")
    df_gl = fetch_all_rows("qbo_general_ledger")
    return df_txn, df_gl

@st.cache_data(ttl=3600)
def load_deals():
    return fetch_all_rows("deals")

@st.cache_data(ttl=3600)
def load_mca_deals():
    return fetch_all_rows("mca_deals")
