# utils/loan_tape_loader.py
"""
Loan tape loading utilities that use the centralized data loader.
Updated to remove duplicate data loading code and use centralized functions.
"""

import pandas as pd
import numpy as np
from utils.data_loader import data_loader


def load_loan_tape_data():
    """
    Load and prepare loan tape data by joining deals and QBO payment data
    
    Returns:
        pd.DataFrame: Comprehensive loan tape with all required metrics
    """
    # Use centralized data loader
    deals_df = data_loader.load_deals()
    qbo_df, _ = data_loader.load_qbo_data()
    
    if deals_df.empty or qbo_df.empty:
        return pd.DataFrame()
    
    # Clean and prepare data
    deals_df = _prepare_deals_data(deals_df)
    qbo_df = _prepare_qbo_data(qbo_df)
    
    # Create loan tape by joining data
    loan_tape = _create_loan_tape(deals_df, qbo_df)
    
    return loan_tape


def load_unified_loan_customer_data():
    """
    Load unified loan and customer payment data in a single comprehensive table
    
    Returns:
        pd.DataFrame: Combined loan performance and customer payment analysis
    """
    # Use centralized data loader
    deals_df = data_loader.load_deals()
    qbo_df, _ = data_loader.load_qbo_data()
    
    if deals_df.empty or qbo_df.empty:
        print("Warning: Missing deals or QBO data for unified analysis")
        return pd.DataFrame()
    
    # Clean and prepare data
    deals_df = _prepare_deals_data(deals_df)
    qbo_df = _prepare_qbo_data(qbo_df)
    
    # Create unified loan-customer analysis
    unified_data = _create_unified_loan_customer_table(deals_df, qbo_df)
    
    return unified_data


def get_data_diagnostics():
    """
    Get diagnostic information about the data join process
    
    Returns:
        dict: Diagnostic information for display in dashboard
    """
    # Use centralized diagnostics function
    return data_loader.get_data_diagnostics()


def get_customer_payment_summary(qbo_df=None):
    """
    Get summary of payments by customer, including unattributed payments
    
    Args:
        qbo_df: QBO payment dataframe (optional - will load if not provided)
        
    Returns:
        pd.DataFrame: Customer payment summary
    """
    if qbo_df is None or qbo_df.empty:
        # Load QBO data using centralized loader
        qbo_df, _ = data_loader.load_qbo_data()
    
    if qbo_df.empty:
        return pd.DataFrame()
    
    # Prepare QBO data
    qbo_clean = _prepare_qbo_data(qbo_df)
    
    if qbo_clean.empty:
        return pd.DataFrame()
    
    # Customer level summary
    customer_summary = qbo_clean.groupby("customer_name").agg({
        "total_amount": "sum",
        "txn_date": "count",
        "loan_id": lambda x: x.nunique()  # Unique loans per customer
    }).reset_index()
    
    customer_summary.columns = ["Customer", "Total Payments", "Payment Count", "Unique Loans"]
    
    # Add unattributed payments (where loan_id is null or empty)
    unattributed = qbo_clean[qbo_clean["loan_id"].isin(["", "nan", "None", "NULL"]) | qbo_clean["loan_id"].isna()].groupby("customer_name").agg({
        "total_amount": "sum",
        "txn_date": "count"
    }).reset_index()
    unattributed.columns = ["Customer", "Unattributed Amount", "Unattributed Count"]
    
    # Merge with main summary
    customer_summary = customer_summary.merge(unattributed, on="Customer", how="left")
    customer_summary["Unattributed Amount"] = customer_summary["Unattributed Amount"].fillna(0)
    customer_summary["Unattributed Count"] = customer_summary["Unattributed Count"].fillna(0)
    
    # Sort by total payments descending
    customer_summary = customer_summary.sort_values("Total Payments", ascending=False)
    
    return customer_summary


def _prepare_deals_data(deals_df):
    """Prepare and clean deals data"""
    # Convert numeric columns (already handled by centralized loader, but ensure consistency)
    numeric_cols = ["amount", "factor_rate", "tib", "fico"]
    for col in numeric_cols:
        if col in deals_df.columns:
            deals_df[col] = pd.to_numeric(deals_df[col], errors="coerce")
    
    # Ensure loan_id is string and clean (already handled by centralized loader)
    if "loan_id" in deals_df.columns:
        deals_df["loan_id"] = deals_df["loan_id"].astype(str).str.strip()
    
    # Filter to only closed won deals (participated deals)
    if "is_closed_won" in deals_df.columns:
        deals_df = deals_df[deals_df["is_closed_won"] == True].copy()
    
    # Calculate total return (expected profit)
    if "amount" in deals_df.columns and "factor_rate" in deals_df.columns:
        deals_df["total_return"] = (deals_df["amount"] * deals_df["factor_rate"]) - deals_df["amount"]
    
    return deals_df


def _prepare_qbo_data(qbo_df):
    """Prepare and clean QBO payment data"""
    # Convert amount to numeric (already handled by centralized loader)
    if "total_amount" in qbo_df.columns:
        qbo_df["total_amount"] = pd.to_numeric(qbo_df["total_amount"], errors="coerce")
    
    # Ensure loan_id is string and clean (already handled by centralized loader)
    if "loan_id" in qbo_df.columns:
        qbo_df["loan_id"] = qbo_df["loan_id"].astype(str).str.strip()
    
    # Filter to only payment transactions (positive cash flow)
    if "transaction_type" in qbo_df.columns:
        payment_types = ["Payment", "Deposit", "Receipt"]
        qbo_df = qbo_df[qbo_df["transaction_type"].isin(payment_types)].copy()
    
    # Take absolute value to ensure positive amounts
    if "total_amount" in qbo_df.columns:
        qbo_df["total_amount"] = qbo_df["total_amount"].abs()
    
    return qbo_df


def _create_loan_tape(deals_df, qbo_df):
    """Create the loan tape by joining deals and payment data"""
    
    if "loan_id" not in deals_df.columns or "loan_id" not in qbo_df.columns:
        print("Warning: loan_id column missing from one or both datasets")
        return pd.DataFrame()
    
    # Aggregate QBO payments by loan_id
    loan_payments = qbo_df.groupby("loan_id").agg({
        "total_amount": "sum",
        "txn_date": "count"  # Count of payments
    }).reset_index()
    loan_payments.columns = ["loan_id", "rtr_amount", "payment_count"]
    
    # Join deals with payment data
    loan_tape = deals_df.merge(loan_payments, on="loan_id", how="left")
    
    # Fill missing payment data with zeros
    loan_tape["rtr_amount"] = loan_tape["rtr_amount"].fillna(0)
    loan_tape["payment_count"] = loan_tape["payment_count"].fillna(0)
    
    # Calculate RTR percentage
    if "amount" in loan_tape.columns:
        loan_tape["rtr_percentage"] = (loan_tape["rtr_amount"] / loan_tape["amount"]) * 100
        loan_tape["rtr_percentage"] = loan_tape["rtr_percentage"].fillna(0)
    
    # Select and rename columns for final loan tape based on actual table structure
    columns_to_select = []
    column_rename_map = {}
    
    # Define the columns we want and their mapping
    desired_columns = {
        "loan_id": "Loan ID",
        "deal_name": "Customer",  # deals table uses deal_name
        "factor_rate": "Factor Rate",
        "amount": "Total Participation",
        "total_return": "Total Return",
        "rtr_amount": "RTR Amount",
        "rtr_percentage": "RTR %",
        "payment_count": "Payment Count",
        "tib": "TIB",
        "fico": "FICO",
        "partner_source": "Partner Source",
        "date_created": "Date Created"
    }
    
    # Only include columns that exist in the dataframe
    for col, display_name in desired_columns.items():
        if col in loan_tape.columns:
            columns_to_select.append(col)
            column_rename_map[col] = display_name
    
    # Select available columns
    loan_tape_final = loan_tape[columns_to_select].copy()
    
    # Rename columns for display
    loan_tape_final = loan_tape_final.rename(columns=column_rename_map)
    
    # Sort by RTR percentage descending to show best performing loans first
    if "RTR %" in loan_tape_final.columns:
        loan_tape_final = loan_tape_final.sort_values("RTR %", ascending=False)
    
    return loan_tape_final


def _create_unified_loan_customer_table(deals_df, qbo_df):
    """
    Create a unified table combining loan performance with customer payment analysis
    """
    
    if "customer_name" not in qbo_df.columns:
        print("Warning: customer_name column missing from QBO data")
        return pd.DataFrame()
    
    # Step 1: Create customer-level aggregations from QBO data
    customer_summary = qbo_df.groupby("customer_name").agg({
        "total_amount": ["sum", "count", "mean"],
        "loan_id": lambda x: x.nunique(),  # Unique loans per customer
        "txn_date": ["min", "max"]
    }).reset_index()
    
    # Flatten column names
    customer_summary.columns = [
        "customer_name", "total_customer_payments", "total_payment_count", 
        "avg_payment_size", "unique_loans_with_payments", "first_payment", "last_payment"
    ]
    
    # Step 2: Calculate unattributed payments per customer
    unattributed = qbo_df[
        qbo_df["loan_id"].isin(["", "nan", "None", "NULL"]) | 
        qbo_df["loan_id"].isna()
    ].groupby("customer_name").agg({
        "total_amount": ["sum", "count"]
    }).reset_index()
    
    if not unattributed.empty:
        unattributed.columns = ["customer_name", "unattributed_amount", "unattributed_count"]
        customer_summary = customer_summary.merge(unattributed, on="customer_name", how="left")
    else:
        customer_summary["unattributed_amount"] = 0
        customer_summary["unattributed_count"] = 0
    
    customer_summary["unattributed_amount"] = customer_summary["unattributed_amount"].fillna(0)
    customer_summary["unattributed_count"] = customer_summary["unattributed_count"].fillna(0)
    
    # Step 3: Create loan-level data with payments
    if "loan_id" not in qbo_df.columns:
        print("Warning: loan_id column missing from QBO data")
        return pd.DataFrame()
    
    loan_payments = qbo_df.groupby("loan_id").agg({
        "total_amount": "sum",
        "txn_date": ["count", "min", "max"],
        "customer_name": "first"  # Get customer name for each loan
    }).reset_index()
    
    # Flatten columns
    loan_payments.columns = ["loan_id", "rtr_amount", "payment_count", "first_payment_date", "last_payment_date", "qbo_customer_name"]
    
    # Step 4: Join deals with loan payments
    if "loan_id" not in deals_df.columns:
        print("Warning: loan_id column missing from deals data")
        return pd.DataFrame()
    
    unified = deals_df.merge(loan_payments, on="loan_id", how="left")
    
    # Fill missing payment data
    unified["rtr_amount"] = unified["rtr_amount"].fillna(0)
    unified["payment_count"] = unified["payment_count"].fillna(0)
    
    # Calculate loan-level metrics
    if "amount" in unified.columns:
        unified["rtr_percentage"] = (unified["rtr_amount"] / unified["amount"]) * 100
        unified["rtr_percentage"] = unified["rtr_percentage"].fillna(0)
    
    # Step 5: Add customer-level summary data
    # First, try to match on deal_name to customer_name
    if "deal_name" in unified.columns:
        unified_with_customer = unified.merge(
            customer_summary, 
            left_on="deal_name", 
            right_on="customer_name", 
            how="left"
        )
    else:
        # Fallback if deal_name doesn't exist
        unified_with_customer = unified.copy()
        for col in customer_summary.columns:
            if col != "customer_name":
                unified_with_customer[col] = 0
    
    # If that doesn't work well, try matching on qbo_customer_name
    if "qbo_customer_name" in unified_with_customer.columns:
        missing_customer_data = unified_with_customer["customer_name"].isna() if "customer_name" in unified_with_customer.columns else pd.Series([True] * len(unified_with_customer))
        if missing_customer_data.any():
            # For rows missing customer data, try matching on qbo_customer_name
            customer_backup = unified_with_customer[missing_customer_data].merge(
                customer_summary,
                left_on="qbo_customer_name",
                right_on="customer_name",
                how="left",
                suffixes=("", "_backup")
            )
            
            # Fill in missing customer data
            for col in customer_summary.columns:
                if col != "customer_name" and col in customer_backup.columns:
                    unified_with_customer.loc[missing_customer_data, col] = customer_backup[col].values
    
    # Step 6: Calculate additional metrics
    unified_with_customer["customer_attributed_percentage"] = (
        (unified_with_customer["total_customer_payments"] - unified_with_customer["unattributed_amount"]) /
        unified_with_customer["total_customer_payments"] * 100
    ).fillna(0)
    
    if "last_payment" in unified_with_customer.columns:
        unified_with_customer["days_since_last_payment"] = (
            pd.Timestamp.now() - pd.to_datetime(unified_with_customer["last_payment"])
        ).dt.days
    
    # Step 7: Select and format final columns
    final_columns = {
        "loan_id": "Loan ID",
        "deal_name": "Deal Name",
        "customer_name": "QBO Customer",
        "factor_rate": "Factor Rate",
        "amount": "Participation Amount",
        "total_return": "Expected Return",
        "rtr_amount": "RTR Amount",
        "rtr_percentage": "RTR %",
        "payment_count": "Loan Payment Count",
        "first_payment_date": "First Payment",
        "last_payment_date": "Last Payment",
        "total_customer_payments": "Total Customer Payments",
        "total_payment_count": "Total Customer Payment Count",
        "unique_loans_with_payments": "Customer Active Loans",
        "unattributed_amount": "Unattributed Amount",
        "unattributed_count": "Unattributed Count",
        "customer_attributed_percentage": "Attribution %",
        "days_since_last_payment": "Days Since Last Payment",
        "tib": "TIB",
        "fico": "FICO",
        "partner_source": "Partner Source",
        "date_created": "Deal Date"
    }
    
    # Select available columns
    available_columns = [col for col in final_columns.keys() if col in unified_with_customer.columns]
    unified_final = unified_with_customer[available_columns].copy()
    
    # Rename columns
    unified_final = unified_final.rename(columns={
        col: final_columns[col] for col in available_columns
    })
    
    # Sort by RTR percentage descending
    if "RTR %" in unified_final.columns:
        unified_final = unified_final.sort_values("RTR %", ascending=False)
    
    return unified_final
