# scripts/combine_hubspot_mca.py

import streamlit as st
import pandas as pd
from supabase import create_client
import os

# -------------------------
# Setup: Supabase Connection
# -------------------------
url = st.secrets["supabase"]["url"]
key = st.secrets["supabase"]["service_role"]
supabase = create_client(url, key)

def fetch_table(table_name):
    return pd.DataFrame(supabase.table(table_name).select("*").execute().data)

def combine_deals():
    deals = fetch_table("deals")
    mca = fetch_table("mca_deals")
    qbo = fetch_table("qbo_loan_summary_view")

    # Handle null loan_ids
    deals = deals.dropna(subset=["loan_id"])
    deals["loan_id"] = deals["loan_id"].astype(str)
    mca["deal_number"] = mca["deal_number"].astype(str)
    qbo["loan_id"] = qbo["loan_id"].astype(str)

    # Merge HubSpot + MCA
    combined = pd.merge(
        deals,
        mca,
        how="inner",
        left_on="loan_id",
        right_on="deal_number",
        suffixes=("_hubspot", "_mca")
    )

    # Merge QBO view onto combined deals
    combined = pd.merge(
        combined,
        qbo,
        how="left",
        left_on="loan_id",
        right_on="loan_id"  # Same field name, so no remapping needed
    )

    # Drop unnecessary columns
    drop_cols = [
        "id_hubspot", "pipeline", "is_closed_won", "id_mca", "extraction_run_id",
        "deal_id", "deal_type", "owner", "funding_type", "sales_rep",
        "nature_of_business", "sos_status", "google_score", "twitter_score"
    ]
    combined = combined.drop(columns=[col for col in drop_cols if col in combined.columns])

    return combined
