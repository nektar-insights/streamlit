# utils/loan_tape_loader.py
import pandas as pd
from utils.imports import get_supabase_client

def load_loan_tape_data():
    """
    Load and prepare loan tape data by joining deals and QBO payment data
    
    Returns:
        pd.DataFrame: Comprehensive loan tape with all required metrics
    """
    supabase = get_supabase_client()
    
    # Load deals data
    deals_res = supabase.table("deals").select("*").execute()
    deals_df = pd.DataFrame(deals_res.data)
    
    # Load QBO payment data
    qbo_res = supabase.table("qbo_invoice_payments").select("*").execute()
    qbo_df = pd.DataFrame(qbo_res.data)
    
    if deals_df.empty:
        print("Warning: No deals data found")
        return pd.DataFrame()
    
    if qbo_df.empty:
        print("Warning: No QBO payment data found")
        return pd.DataFrame()
    
    # Debug: Print column names
    print("Deals columns:", deals_df.columns.tolist())
    print("QBO columns:", qbo_df.columns.tolist())
    
    # Clean and prepare deals data
    deals_df = _prepare_deals_data(deals_df)
    
    # Clean and prepare QBO data  
    qbo_df = _prepare_qbo_data(qbo_df)
    
    # Create loan tape by joining data
    loan_tape = _create_loan_tape(deals_df, qbo_df)
    
    return loan_tape

def _prepare_deals_data(deals_df):
    """Prepare and clean deals data"""
    # Convert numeric columns
    numeric_cols = ["amount", "factor_rate", "tib", "fico"]
    for col in numeric_cols:
        if col in deals_df.columns:
            deals_df[col] = pd.to_numeric(deals_df[col], errors="coerce")
    
    # Ensure loan_id is string and clean
    deals_df["loan_id"] = deals_df["loan_id"].astype(str).str.strip()
    
    # Filter to only closed won deals (participated deals)
    deals_df = deals_df[deals_df["is_closed_won"] == True].copy()
    
    # Calculate total return (expected profit)
    deals_df["total_return"] = (deals_df["amount"] * deals_df["factor_rate"]) - deals_df["amount"]
    
    return deals_df

def _prepare_qbo_data(qbo_df):
    """Prepare and clean QBO payment data"""
    # Convert amount to numeric
    qbo_df["total_amount"] = pd.to_numeric(qbo_df["total_amount"], errors="coerce")
    
    # Ensure loan_id is string and clean
    qbo_df["loan_id"] = qbo_df["loan_id"].astype(str).str.strip()
    
    # Filter to only payment transactions (positive cash flow)
    payment_types = ["Payment", "Deposit", "Receipt"]
    qbo_df = qbo_df[qbo_df["transaction_type"].isin(payment_types)].copy()
    
    # Take absolute value to ensure positive amounts
    qbo_df["total_amount"] = qbo_df["total_amount"].abs()
    
    return qbo_df

def _create_loan_tape(deals_df, qbo_df):
    """Create the loan tape by joining deals and payment data"""
    
    # Debug: Print available columns
    print("Available deals columns:", deals_df.columns.tolist())
    print("Available QBO columns:", qbo_df.columns.tolist())
    
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
    loan_tape["rtr_percentage"] = (loan_tape["rtr_amount"] / loan_tape["amount"]) * 100
    loan_tape["rtr_percentage"] = loan_tape["rtr_percentage"].fillna(0)
    
    # Debug: Print loan_tape columns after join
    print("Loan tape columns after join:", loan_tape.columns.tolist())
    
    # Check what columns are actually available and build the selection dynamically
    available_cols = loan_tape.columns.tolist()
    
    # Define column mapping - map desired name to possible actual column names
    column_mapping = {
        "loan_id": ["loan_id"],
        "customer_name": ["customer_name", "dealname", "deal_name", "company", "account_name"],
        "factor_rate": ["factor_rate"],
        "amount": ["amount"],
        "total_return": ["total_return"],
        "rtr_amount": ["rtr_amount"],
        "rtr_percentage": ["rtr_percentage"],
        "payment_count": ["payment_count"],
        "tib": ["tib"],
        "fico": ["fico"],
        "partner_source": ["partner_source", "source", "lead_source"],
        "date_created": ["date_created", "createdate", "created_date"]
    }
    
    # Build final column selection
    final_columns = {}
    for desired_col, possible_cols in column_mapping.items():
        found_col = None
        for possible_col in possible_cols:
            if possible_col in available_cols:
                found_col = possible_col
                break
        if found_col:
            final_columns[found_col] = desired_col
        else:
            print(f"Warning: Could not find column for {desired_col}. Tried: {possible_cols}")
    
    # Select available columns
    loan_tape_final = loan_tape[list(final_columns.keys())].copy()
    
    # Rename columns for display
    loan_tape_final = loan_tape_final.rename(columns={
        old_name: new_name.replace("_", " ").title() for old_name, new_name in final_columns.items()
    })
    
    # Ensure we have the core required columns, if not, create empty ones
    required_display_cols = ["Loan Id", "Factor Rate", "Amount", "Total Return", "Rtr Amount", "Rtr Percentage", "Payment Count"]
    for col in required_display_cols:
        if col not in loan_tape_final.columns:
            loan_tape_final[col] = 0
    
    # Sort by RTR percentage descending to show best performing loans first
    if "Rtr Percentage" in loan_tape_final.columns:
        loan_tape_final = loan_tape_final.sort_values("Rtr Percentage", ascending=False)
    
    return loan_tape_final

def get_customer_payment_summary(qbo_df):
    """
    Get summary of payments by customer, including unattributed payments
    
    Args:
        qbo_df: QBO payment dataframe (pass the raw df from main script)
        
    Returns:
        pd.DataFrame: Customer payment summary
    """
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
