# utils/qbo_data_loader.py
import pandas as pd
import streamlit as st
from utils.config import get_bq_client, _TABLE_MAP


def _bq_fetch(table_name: str) -> pd.DataFrame:
    """Fetch all rows from a BigQuery table by logical name."""
    try:
        bq = get_bq_client()
        bq_table = _TABLE_MAP[table_name]
        return bq.query(f"SELECT * FROM `{bq_table}`").to_dataframe()
    except Exception as e:
        print(f"[BQ] Failed to load {table_name}: {e}")
        return pd.DataFrame()


@st.cache_data(ttl=3600)
def load_qbo_data():
    df_txn = _bq_fetch("qbo_invoice_payments")
    df_gl = _bq_fetch("qbo_general_ledger")
    return df_txn, df_gl


@st.cache_data(ttl=3600)
def load_deals():
    return _bq_fetch("deals")


@st.cache_data(ttl=3600)
def load_mca_deals():
    return _bq_fetch("mca_deals")
