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
    
    if deals_df.empty or qbo_df.empty:
        return pd.DataFrame()
    
    # Clean and prepare data
    deals_df = _prepare_deals_data(deals_df)
    qbo_df = _prepare_qbo_data(qbo_df)
    
    # Create loan tape by joining data
    loan_tape = _create_loan_tape(deals_df, qbo_df)
    
    return loan_tape

def get_data_diagnostics():
    """
    Get diagnostic information about the data join process
    
    Returns:
        dict: Diagnostic information for display in dashboard
    """
    supabase = get_supabase_client()
    
    # Load raw data
    deals_res = supabase.table("deals").select("*").execute()
    deals_df = pd.DataFrame(deals_res.data)
    
    qbo_res = supabase.table("qbo_invoice_payments").select("*").execute()
    qbo_df = pd.DataFrame(qbo_res.data)
    
    if deals_df.empty or qbo_df.empty:
        return {}
    
    # Convert amounts to numeric for calculations
    qbo_df["total_amount"] = pd.to_numeric(qbo_df["total_amount"], errors="coerce")
    deals_df["amount"] = pd.to_numeric(deals_df["amount"], errors="coerce")
    
    # Clean loan_ids
    qbo_df["loan_id"] = qbo_df["loan_id"].astype(str).str.strip()
    deals_df["loan_id"] = deals_df["loan_id"].astype(str).str.strip()
    
    # Basic counts
    diagnostics = {
        "raw_deals_count": len(deals_df),
        "raw_qbo_count": len(qbo_df),
        "total_qbo_amount": qbo_df["total_amount"].sum(),
        
        # Deal analysis
        "closed_won_deals": len(deals_df[deals_df["is_closed_won"] == True]),
        "total_participation": deals_df[deals_df["is_closed_won"] == True]["amount"].sum(),
        
        # Transaction type breakdown
        "transaction_types": qbo_df.groupby("transaction_type").agg({
            "total_amount": "sum",
            "transaction_id": "count"
        }).to_dict(),
        
        # Payment type filtering
        "payment_types_amount": qbo_df[qbo_df["transaction_type"].isin(["Payment", "Deposit", "Receipt"])]["total_amount"].sum(),
        "payment_types_count": len(qbo_df[qbo_df["transaction_type"].isin(["Payment", "Deposit", "Receipt"])]),
        
        # Loan ID analysis
        "qbo_with_loan_id": {
            "count": len(qbo_df[qbo_df["loan_id"].notna() & (qbo_df["loan_id"] != "") & (qbo_df["loan_id"] != "nan")]),
            "amount": qbo_df[qbo_df["loan_id"].notna() & (qbo_df["loan_id"] != "") & (qbo_df["loan_id"] != "nan")]["total_amount"].sum()
        },
        "qbo_without_loan_id": {
            "count": len(qbo_df[qbo_df["loan_id"].isna() | (qbo_df["loan_id"] == "") | (qbo_df["loan_id"] == "nan")]),
            "amount": qbo_df[qbo_df["loan_id"].isna() | (qbo_df["loan_id"] == "") | (qbo_df["loan_id"] == "nan")]["total_amount"].sum()
        },
        
        # Loan ID overlap
        "unique_deal_loan_ids": deals_df["loan_id"].nunique(),
        "unique_qbo_loan_ids": qbo_df["loan_id"].nunique(),
        "overlapping_loan_ids": len(set(deals_df["loan_id"].unique()).intersection(set(qbo_df["loan_id"].unique()))),
        
        # Top customers
        "top_customers": qbo_df.groupby("customer_name")["total_amount"].sum().nlargest(10).to_dict()
    }
    
    return diagnostics

def load_unified_loan_customer_data():
    """
    Load unified loan and customer payment data in a single comprehensive table
    
    Returns:
        pd.DataFrame: Combined loan performance and customer payment analysis
    """
    supabase = get_supabase_client()
    
    # Load deals data
    deals_res = supabase.table("deals").select("*").execute()
    deals_df = pd.DataFrame(deals_res.data)
    
    # Load QBO payment data
    qbo_res = supabase.table("qbo_invoice_payments").select("*").execute()
    qbo_df = pd.DataFrame(qbo_res.data)
    
    if deals_df.empty or qbo_df.empty:
        print("Warning: Missing deals or QBO data for unified analysis")
        return pd.DataFrame()
    
    # Clean and prepare data
    deals_df = _prepare_deals_data(deals_df)
    qbo_df = _prepare_qbo_data(qbo_df)
    
    # Create unified loan-customer analysis
    unified_data = _create_unified_loan_customer_table(deals_df, qbo_df)
    
    return unified_data

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
    loan_payments = qbo_df.groupby("loan_id").agg({
        "total_amount": "sum",
        "txn_date": ["count", "min", "max"],
        "customer_name": "first"  # Get customer name for each loan
    }).reset_index()
    
    # Flatten columns
    loan_payments.columns = ["loan_id", "rtr_amount", "payment_count", "first_payment_date", "last_payment_date", "qbo_customer_name"]
    
    # Step 4: Join deals with loan payments
    unified = deals_df.merge(loan_payments, on="loan_id", how="left")
    
    # Fill missing payment data
    unified["rtr_amount"] = unified["rtr_amount"].fillna(0)
    unified["payment_count"] = unified["payment_count"].fillna(0)
    
    # Calculate loan-level metrics
    unified["rtr_percentage"] = (unified["rtr_amount"] / unified["amount"]) * 100
    unified["rtr_percentage"] = unified["rtr_percentage"].fillna(0)
    
    # Step 5: Add customer-level summary data
    # First, try to match on deal_name to customer_name
    unified_with_customer = unified.merge(
        customer_summary, 
        left_on="deal_name", 
        right_on="customer_name", 
        how="left"
    )
    
    # If that doesn't work well, try matching on qbo_customer_name
    missing_customer_data = unified_with_customer["customer_name"].isna()
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

def get_customer_payment_summary(qbo_df=None):
    """
    Get summary of payments by customer, including unattributed payments
    
    Args:
        qbo_df: QBO payment dataframe (optional - will load if not provided)
        
    Returns:
        pd.DataFrame: Customer payment summary
    """
    if qbo_df is None or qbo_df.empty:
        # Load all QBO data if not provided
        supabase = get_supabase_client()
        qbo_df = _load_all_data_with_fallback(supabase, "qbo_invoice_payments")
    
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
