# utils/data_loader.py
"""
Centralized data loading utilities for the Streamlit application.
All data loading functions are consolidated here for reuse across pages.
"""

import pandas as pd
import streamlit as st
from typing import Tuple, Optional
from utils.imports import get_supabase_client


class DataLoader:
    """Centralized data loader with caching and error handling"""
    
    def __init__(self):
        self.supabase = get_supabase_client()
    
    def _fetch_all_rows(self, table_name: str, chunk_size: int = 1000) -> pd.DataFrame:
        """
        Fetch all rows from a Supabase table using pagination to avoid limits

        Args:
            table_name: Name of the table to fetch from
            chunk_size: Number of rows to fetch per batch

        Returns:
            pd.DataFrame: Complete dataset from the table
        """
        try:
            rows = []
            start = 0

            while True:
                end = start + chunk_size - 1
                response = (
                    self.supabase.table(table_name)
                    .select("*")
                    .range(start, end)
                    .execute()
                )
                batch = response.data
                if not batch:
                    break  # no more rows
                rows.extend(batch)
                start += chunk_size

            return pd.DataFrame(rows)

        except Exception:
            # Fallback to standard query
            try:
                response = self.supabase.table(table_name).select("*").execute()
                return pd.DataFrame(response.data)
            except Exception:
                return pd.DataFrame()
    
    def _preprocess_data(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        """
        Clean and preprocess dataframe with common transformations
        
        Args:
            dataframe: Raw dataframe to preprocess
            
        Returns:
            pd.DataFrame: Cleaned dataframe
        """
        if dataframe.empty:
            return dataframe
        
        df_clean = dataframe.copy()
        
        # Handle numeric columns
        numeric_cols = [
            'total_amount', 'balance', 'debit', 'credit', 'amount', 'purchase_price', 
            'receivables_amount', 'current_balance', 'past_due_amount', 'principal_amount', 
            'rtr_balance', 'amount_hubspot', 'total_funded_amount', 'factor_rate', 
            'loan_term', 'commission', 'tib', 'fico'
        ]
        
        for col in numeric_cols:
            if col in df_clean.columns:
                df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
        
        # Handle date columns
        date_cols = [
            'txn_date', 'due_date', 'date', 'date_created', 'funding_date', 
            'created_time', 'last_updated_time'
        ]
        
        for col in date_cols:
            if col in df_clean.columns:
                df_clean[col] = pd.to_datetime(df_clean[col], errors='coerce')
        
        # Handle string columns
        string_cols = ['loan_id', 'deal_number', 'customer_name', 'partner_source']
        for col in string_cols:
            if col in df_clean.columns:
                df_clean[col] = df_clean[col].astype(str).str.strip()
        
        return df_clean
    
    @st.cache_data(ttl=3600)
    def load_deals(_self) -> pd.DataFrame:
        """
        Load and preprocess deals data from HubSpot
        
        Returns:
            pd.DataFrame: Processed deals data
        """
        try:
            df = _self._fetch_all_rows("deals")
            df = _self._preprocess_data(df)
            
            # Additional deals-specific preprocessing
            if not df.empty:
                # Create derived columns
                if 'date_created' in df.columns:
                    df["month"] = df["date_created"].dt.to_period("M").astype(str)
                
                # Boolean flags
                if 'is_closed_won' in df.columns:
                    df["is_participated"] = df["is_closed_won"] == True

            return df

        except Exception:
            return pd.DataFrame()
    
    @st.cache_data(ttl=3600)
    def load_mca_deals(_self) -> pd.DataFrame:
        """
        Load and preprocess raw MCA deals data
        
        Returns:
            pd.DataFrame: Processed MCA deals data
        """
        try:
            df = _self._fetch_all_rows("mca_deals")
            df = _self._preprocess_data(df)
            return df

        except Exception:
            return pd.DataFrame()
    
    @st.cache_data(ttl=3600)
    def load_qbo_data(_self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Load and preprocess QBO transaction and general ledger data
        
        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: (invoice_payments, general_ledger)
        """
        try:
            # Load both QBO datasets
            df_txn = _self._fetch_all_rows("qbo_invoice_payments")
            df_gl = _self._fetch_all_rows("qbo_general_ledger")
            
            # Preprocess both
            df_txn = _self._preprocess_data(df_txn)
            df_gl = _self._preprocess_data(df_gl)
            
            # Additional QBO-specific preprocessing
            if not df_txn.empty and "txn_date" in df_txn.columns:
                df_txn["year_month"] = df_txn["txn_date"].dt.to_period("M")
                df_txn["week"] = df_txn["txn_date"].dt.isocalendar().week
                df_txn["day_of_week"] = df_txn["txn_date"].dt.day_name()
                df_txn["days_since_txn"] = (pd.Timestamp.now() - df_txn["txn_date"]).dt.days

            return df_txn, df_gl

        except Exception:
            return pd.DataFrame(), pd.DataFrame()
    
    @st.cache_data(ttl=3600)
    def load_loan_summaries(_self) -> pd.DataFrame:
        """
        Load and preprocess loan summaries data

        Returns:
            pd.DataFrame: Processed loan summaries data
        """
        try:
            df = _self._fetch_all_rows("loan_summaries")
            df = _self._preprocess_data(df)
            return df

        except Exception:
            return pd.DataFrame()

    @st.cache_data(ttl=3600)
    def load_loan_schedules(_self) -> pd.DataFrame:
        """
        Load and preprocess loan schedules data

        Returns:
            pd.DataFrame: Processed loan schedules data
        """
        try:
            df = _self._fetch_all_rows("loan_schedules")
            df = _self._preprocess_data(df)
            return df

        except Exception:
            return pd.DataFrame()

    @st.cache_data(ttl=3600)
    def load_naics_sector_risk(_self) -> pd.DataFrame:
        """
        Load and preprocess NAICS sector risk profile data

        Returns:
            pd.DataFrame: Processed NAICS sector risk data
        """
        try:
            df = _self._fetch_all_rows("naics_sector_risk_profile")
            df = _self._preprocess_data(df)
            return df

        except Exception:
            return pd.DataFrame()

    @st.cache_data(ttl=3600)
    def get_last_updated(_self) -> str:
        """
        Get the last updated timestamp from key tables

        Returns:
            str: Formatted timestamp of the most recent update
        """
        try:
            timestamps = []
            for table in ["loan_summaries", "deals", "loan_schedules", "mca_deals"]:
                try:
                    res = _self.supabase.table(table).select("updated_at").order("updated_at", desc=True).limit(1).execute()
                    if res.data and res.data[0].get("updated_at"):
                        timestamps.append(pd.to_datetime(res.data[0]["updated_at"]))
                except:
                    try:
                        res = _self.supabase.table(table).select("created_at").order("created_at", desc=True).limit(1).execute()
                        if res.data and res.data[0].get("created_at"):
                            timestamps.append(pd.to_datetime(res.data[0]["created_at"]))
                    except:
                        pass
            if timestamps:
                return max(timestamps).strftime("%B %d, %Y at %I:%M %p")
            return "Unable to determine"
        except Exception as e:
            return f"Error: {str(e)}"

    @st.cache_data(ttl=3600)
    def load_combined_mca_deals(_self) -> pd.DataFrame:
        """
        Load combined MCA deals using the combine_deals function
        
        Returns:
            pd.DataFrame: Combined and processed MCA deals data
        """
        try:
            from scripts.combine_hubspot_mca import combine_deals
            df = combine_deals()
            
            if not df.empty:
                df = _self._preprocess_data(df)
                
                # MCA-specific calculations
                if 'funding_date' in df.columns:
                    df["days_since_funding"] = (pd.Timestamp.now() - df["funding_date"]).dt.days
                
                # Status-based calculations
                if 'status_category' in df.columns:
                    # Set past_due_amount to 0 for Matured deals
                    if 'past_due_amount' in df.columns:
                        df.loc[df["status_category"] == "Matured", "past_due_amount"] = 0
                
                # Financial calculations
                for col in ["purchase_price", "receivables_amount", "current_balance", 
                           "past_due_amount", "principal_amount", "rtr_balance", 
                           "amount_hubspot", "total_funded_amount"]:
                    if col in df.columns:
                        df[col] = pd.to_numeric(df[col], errors="coerce")
                
                # Calculate derived metrics
                if 'amount_hubspot' in df.columns and 'total_funded_amount' in df.columns:
                    df["participation_ratio"] = df["amount_hubspot"] / df["total_funded_amount"].replace(0, pd.NA)
                
                if 'past_due_amount' in df.columns and 'current_balance' in df.columns:
                    df["past_due_pct"] = df.apply(
                        lambda row: row["past_due_amount"] / row["current_balance"]
                        if pd.notna(row["past_due_amount"]) and pd.notna(row["current_balance"]) and row["current_balance"] > 0
                        else 0,
                        axis=1
                    )
                
            return df

        except Exception:
            return pd.DataFrame()
    
    def clear_cache(self, dataset: Optional[str] = None):
        """
        Clear cached data for specific dataset or all datasets
        
        Args:
            dataset: Specific dataset to clear ('deals', 'mca', 'qbo', 'combined_mca') or None for all
        """
        if dataset == 'deals':
            if hasattr(st.session_state, '_cache'):
                for key in list(st.session_state._cache.keys()):
                    if 'load_deals' in key:
                        del st.session_state._cache[key]
        elif dataset == 'mca':
            if hasattr(st.session_state, '_cache'):
                for key in list(st.session_state._cache.keys()):
                    if 'load_mca_deals' in key:
                        del st.session_state._cache[key]
        elif dataset == 'qbo':
            if hasattr(st.session_state, '_cache'):
                for key in list(st.session_state._cache.keys()):
                    if 'load_qbo_data' in key:
                        del st.session_state._cache[key]
        elif dataset == 'combined_mca':
            if hasattr(st.session_state, '_cache'):
                for key in list(st.session_state._cache.keys()):
                    if 'load_combined_mca_deals' in key:
                        del st.session_state._cache[key]
        else:
            # Clear all caches
            st.cache_data.clear()
    
    def get_data_diagnostics(self) -> dict:
        """
        Get diagnostic information about the data join process
        
        Returns:
            dict: Diagnostic information for display in dashboard
        """
        try:
            # Load the required datasets
            deals_df = self.load_deals()
            qbo_df, _ = self.load_qbo_data()
            
            if deals_df.empty or qbo_df.empty:
                return {"error": "Missing required datasets for diagnostics"}
            
            # Convert amounts to numeric for calculations
            qbo_df["total_amount"] = pd.to_numeric(qbo_df["total_amount"], errors="coerce")
            deals_df["amount"] = pd.to_numeric(deals_df["amount"], errors="coerce")
            
            # Clean loan_ids
            qbo_df["loan_id"] = qbo_df["loan_id"].astype(str).str.strip()
            deals_df["loan_id"] = deals_df["loan_id"].astype(str).str.strip()
            
            # Transaction type breakdown
            transaction_types = {}
            if "transaction_type" in qbo_df.columns:
                txn_summary = qbo_df.groupby("transaction_type").agg({
                    "total_amount": "sum",
                    "transaction_id": "count" if "transaction_id" in qbo_df.columns else "size"
                })
                
                for txn_type in txn_summary.index:
                    transaction_types[txn_type] = {
                        "total_amount": float(txn_summary.loc[txn_type, "total_amount"]),
                        "count": int(txn_summary.loc[txn_type].iloc[1])  # second column
                    }
            
            # Basic counts and analysis
            diagnostics = {
                "raw_deals_count": len(deals_df),
                "raw_qbo_count": len(qbo_df),
                "total_qbo_amount": float(qbo_df["total_amount"].sum()),
                
                # Deal analysis
                "closed_won_deals": len(deals_df[deals_df["is_closed_won"] == True]) if "is_closed_won" in deals_df.columns else 0,
                "total_participation": float(deals_df[deals_df["is_closed_won"] == True]["amount"].sum()) if "is_closed_won" in deals_df.columns and "amount" in deals_df.columns else 0,
                
                # Transaction type breakdown
                "transaction_types": transaction_types,
                
                # Payment type filtering
                "payment_types_amount": float(qbo_df[qbo_df["transaction_type"].isin(["Payment", "Deposit", "Receipt"])]["total_amount"].sum()) if "transaction_type" in qbo_df.columns else 0,
                "payment_types_count": len(qbo_df[qbo_df["transaction_type"].isin(["Payment", "Deposit", "Receipt"])]) if "transaction_type" in qbo_df.columns else 0,
                
                # Loan ID analysis
                "qbo_with_loan_id": {
                    "count": len(qbo_df[qbo_df["loan_id"].notna() & (qbo_df["loan_id"] != "") & (qbo_df["loan_id"] != "nan")]),
                    "amount": float(qbo_df[qbo_df["loan_id"].notna() & (qbo_df["loan_id"] != "") & (qbo_df["loan_id"] != "nan")]["total_amount"].sum())
                },
                "qbo_without_loan_id": {
                    "count": len(qbo_df[qbo_df["loan_id"].isna() | (qbo_df["loan_id"] == "") | (qbo_df["loan_id"] == "nan")]),
                    "amount": float(qbo_df[qbo_df["loan_id"].isna() | (qbo_df["loan_id"] == "") | (qbo_df["loan_id"] == "nan")]["total_amount"].sum())
                },
                
                # Loan ID overlap
                "unique_deal_loan_ids": deals_df["loan_id"].nunique() if "loan_id" in deals_df.columns else 0,
                "unique_qbo_loan_ids": qbo_df["loan_id"].nunique() if "loan_id" in qbo_df.columns else 0,
                "overlapping_loan_ids": len(set(deals_df["loan_id"].unique()).intersection(set(qbo_df["loan_id"].unique()))) if "loan_id" in deals_df.columns and "loan_id" in qbo_df.columns else 0,
                
                # Top customers
                "top_customers": qbo_df.groupby("customer_name")["total_amount"].sum().nlargest(10).to_dict() if "customer_name" in qbo_df.columns else {}
            }
            
            return diagnostics
            
        except Exception as e:
            return {"error": str(e)}


# Global instance for easy importing
data_loader = DataLoader()

# Convenience functions for backward compatibility
def load_deals() -> pd.DataFrame:
    """Load deals data"""
    return data_loader.load_deals()

def load_mca_deals() -> pd.DataFrame:
    """Load MCA deals data"""
    return data_loader.load_mca_deals()

def load_qbo_data() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load QBO data"""
    return data_loader.load_qbo_data()

def load_combined_mca_deals() -> pd.DataFrame:
    """Load combined MCA deals data"""
    return data_loader.load_combined_mca_deals()

def load_loan_summaries() -> pd.DataFrame:
    """Load loan summaries data"""
    return data_loader.load_loan_summaries()

def load_loan_schedules() -> pd.DataFrame:
    """Load loan schedules data"""
    return data_loader.load_loan_schedules()

def load_naics_sector_risk() -> pd.DataFrame:
    """Load NAICS sector risk data"""
    return data_loader.load_naics_sector_risk()

def get_last_updated() -> str:
    """Get last updated timestamp"""
    return data_loader.get_last_updated()

def get_data_diagnostics() -> dict:
    """Get data diagnostics"""
    return data_loader.get_data_diagnostics()

def clear_data_cache(dataset: Optional[str] = None):
    """Clear data cache"""
    data_loader.clear_cache(dataset)


# Re-export get_supabase_client for convenience
from utils.imports import get_supabase_client
