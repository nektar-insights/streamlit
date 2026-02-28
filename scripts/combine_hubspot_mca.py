# scripts/combine_hubspot_mca.py

import pandas as pd
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


def combine_deals():
    # Load tables
    deals = _bq_fetch("deals")
    mca = _bq_fetch("mca_deals")
    qbo = _bq_fetch("qbo_loan_summary_view")

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
        "deal_id", "owner", "sales_rep",
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
