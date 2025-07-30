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
    # Load tables
    deals = fetch_table("deals")
    mca = fetch_table("mca_deals")
    qbo = fetch_table("qbo_loan_summary_view")

    # Clean and cast IDs
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

    # Merge QBO view
    combined = pd.merge(
        combined,
        qbo,
        how="left",
        on="loan_id"
    )

    # Exclude any deals marked as canceled in status_category
    combined = combined[combined["status_category"] != "Canceled"]

    # Compute tib as the rounded average of tib and years_in_business,
    # skipping nulls so a single non-null will carry through
    combined["tib"] = (
        combined[["tib", "years_in_business"]]
        .mean(axis=1)            # skips NaNs
        .round()                 
        .astype("Int64")         # nullable integer dtype
    )
    
    # Drop unnecessary columns
    drop_cols = [
        # HubSpot/MCA metadata
        "id_hubspot", "pipeline", "is_closed_won", "id_mca", "extraction_run_id",
        "deal_id", "deal_type", "owner", "funding_type", "sales_rep",
        "nature_of_business", "sos_status", "google_score", "twitter_score",
        # MCA-specific fields
        "deal_number", "purchase_price", "receivables_amount", "years_in_business",
        "payments_made", "total_payments_expected", "detail_url", "page_url",
        "extracted_at", "created_at", "amount_mca", "mca_app_date",
        "monthly_cc_processing", "monthly_bank_deposits", "avg_daily_bank_bal",
        "last_updated", "net_activity"
    ]
    combined = combined.drop(columns=[col for col in drop_cols if col in combined.columns])

    return combined
