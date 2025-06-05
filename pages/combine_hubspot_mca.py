# scripts/combine_hubspot_mca.py

import pandas as pd
from supabase import create_client
import os

# Setup Supabase
supabase = create_client(os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_KEY"))

def fetch_table(table_name):
    return pd.DataFrame(supabase.table(table_name).select("*").execute().data)

def combine_deals():
    deals = fetch_table("deals")
    mca = fetch_table("mca_deals")

    # Handle null loan_ids
    deals = deals.dropna(subset=["loan_id"])
    deals["loan_id"] = deals["loan_id"].astype(str)
    mca["deal_number"] = mca["deal_number"].astype(str)

    combined = pd.merge(
        deals,
        mca,
        how="inner",
        left_on="loan_id",
        right_on="deal_number",
        suffixes=("_hubspot", "_mca")
    )

    return combined
