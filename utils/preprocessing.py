# utils/preprocessing.py
"""
Consolidated data preprocessing utilities for Streamlit application.
Provides common data cleaning and transformation functions used across multiple pages.
"""

import pandas as pd
import numpy as np
from typing import List, Optional


def preprocess_numeric_columns(
    df: pd.DataFrame,
    columns: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Convert specified columns to numeric, handling errors gracefully.

    Args:
        df: DataFrame to process
        columns: List of column names to convert. If None, uses default set.

    Returns:
        pd.DataFrame: DataFrame with numeric columns converted
    """
    if columns is None:
        # Default numeric columns commonly used across pages
        columns = [
            'total_amount', 'balance', 'debit', 'credit', 'amount',
            'purchase_price', 'receivables_amount', 'current_balance',
            'past_due_amount', 'principal_amount', 'rtr_balance',
            'amount_hubspot', 'total_funded_amount', 'factor_rate',
            'loan_term', 'commission', 'tib', 'fico', 'roi',
            'commission_fee', 'csl_participation_amount', 'total_paid'
        ]

    df_clean = df.copy()

    for col in columns:
        if col in df_clean.columns:
            df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')

    return df_clean


def preprocess_date_columns(
    df: pd.DataFrame,
    columns: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Convert specified columns to datetime, handling errors gracefully.

    Args:
        df: DataFrame to process
        columns: List of column names to convert. If None, uses default set.

    Returns:
        pd.DataFrame: DataFrame with date columns converted
    """
    if columns is None:
        # Default date columns commonly used across pages
        columns = [
            'txn_date', 'due_date', 'date', 'date_created', 'funding_date',
            'created_time', 'last_updated_time', 'maturity_date', 'payoff_date'
        ]

    df_clean = df.copy()

    for col in columns:
        if col in df_clean.columns:
            df_clean[col] = pd.to_datetime(df_clean[col], errors='coerce')

    return df_clean


def preprocess_string_columns(
    df: pd.DataFrame,
    columns: Optional[List[str]] = None,
    strip: bool = True
) -> pd.DataFrame:
    """
    Convert specified columns to string and optionally strip whitespace.

    Args:
        df: DataFrame to process
        columns: List of column names to convert. If None, uses default set.
        strip: Whether to strip whitespace

    Returns:
        pd.DataFrame: DataFrame with string columns processed
    """
    if columns is None:
        # Default string columns commonly used across pages
        columns = [
            'loan_id', 'deal_number', 'customer_name', 'partner_source',
            'transaction_type', 'status_category', 'loan_status', 'dba'
        ]

    df_clean = df.copy()

    for col in columns:
        if col in df_clean.columns:
            df_clean[col] = df_clean[col].astype(str)
            if strip:
                df_clean[col] = df_clean[col].str.strip()

    return df_clean


def preprocess_dataframe(
    df: pd.DataFrame,
    numeric_cols: Optional[List[str]] = None,
    date_cols: Optional[List[str]] = None,
    string_cols: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Comprehensive dataframe preprocessing with numeric, date, and string conversions.

    This is the main preprocessing function that combines all preprocessing steps.
    It's the recommended function to use for most preprocessing needs.

    Args:
        df: DataFrame to process
        numeric_cols: Columns to convert to numeric (None = use defaults)
        date_cols: Columns to convert to datetime (None = use defaults)
        string_cols: Columns to convert to string (None = use defaults)

    Returns:
        pd.DataFrame: Fully preprocessed DataFrame
    """
    if df.empty:
        return df

    df_clean = df.copy()

    # Apply numeric preprocessing
    df_clean = preprocess_numeric_columns(df_clean, numeric_cols)

    # Apply date preprocessing
    df_clean = preprocess_date_columns(df_clean, date_cols)

    # Apply string preprocessing
    df_clean = preprocess_string_columns(df_clean, string_cols)

    return df_clean


def clean_null_values(
    df: pd.DataFrame,
    strategy: str = 'keep',
    subset: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Handle null values in DataFrame based on strategy.

    Args:
        df: DataFrame to clean
        strategy: How to handle nulls ('keep', 'drop_rows', 'drop_cols', 'fill_zero', 'fill_median')
        subset: Columns to consider for null handling (None = all columns)

    Returns:
        pd.DataFrame: Cleaned DataFrame
    """
    df_clean = df.copy()

    if strategy == 'keep':
        return df_clean

    columns_to_check = subset if subset else df_clean.columns.tolist()

    if strategy == 'drop_rows':
        df_clean = df_clean.dropna(subset=columns_to_check)
    elif strategy == 'drop_cols':
        df_clean = df_clean.drop(columns=[col for col in columns_to_check if df_clean[col].isna().all()])
    elif strategy == 'fill_zero':
        for col in columns_to_check:
            if col in df_clean.columns:
                df_clean[col] = df_clean[col].fillna(0)
    elif strategy == 'fill_median':
        for col in columns_to_check:
            if col in df_clean.columns and pd.api.types.is_numeric_dtype(df_clean[col]):
                df_clean[col] = df_clean[col].fillna(df_clean[col].median())

    return df_clean


def add_derived_date_features(
    df: pd.DataFrame,
    date_col: str,
    prefix: str = ''
) -> pd.DataFrame:
    """
    Add derived features from a date column.

    Args:
        df: DataFrame to process
        date_col: Name of date column to derive features from
        prefix: Prefix for new column names

    Returns:
        pd.DataFrame: DataFrame with added date features
    """
    if date_col not in df.columns:
        return df

    df_clean = df.copy()

    # Ensure column is datetime
    df_clean[date_col] = pd.to_datetime(df_clean[date_col], errors='coerce')

    # Add derived features
    prefix_str = f"{prefix}_" if prefix else ""

    try:
        df_clean[f"{prefix_str}year"] = df_clean[date_col].dt.year
        df_clean[f"{prefix_str}month"] = df_clean[date_col].dt.month
        df_clean[f"{prefix_str}quarter"] = df_clean[date_col].dt.quarter
        df_clean[f"{prefix_str}year_month"] = df_clean[date_col].dt.to_period("M").astype(str)
        df_clean[f"{prefix_str}day_of_week"] = df_clean[date_col].dt.day_name()
        df_clean[f"{prefix_str}days_ago"] = (pd.Timestamp.now() - df_clean[date_col]).dt.days
    except Exception as e:
        # If any derivation fails, continue without raising error
        pass

    return df_clean


def standardize_currency_columns(
    df: pd.DataFrame,
    currency_cols: Optional[List[str]] = None,
    negative_to_zero: bool = False
) -> pd.DataFrame:
    """
    Standardize currency columns (convert to numeric, handle formatting).

    Args:
        df: DataFrame to process
        currency_cols: Columns containing currency values
        negative_to_zero: Whether to convert negative values to zero

    Returns:
        pd.DataFrame: DataFrame with standardized currency columns
    """
    if currency_cols is None:
        # Common currency columns
        currency_cols = [
            'amount', 'total_amount', 'balance', 'purchase_price',
            'total_funded_amount', 'current_balance', 'past_due_amount',
            'commission_fees', 'platform_fees', 'total_invested', 'total_paid'
        ]

    df_clean = df.copy()

    for col in currency_cols:
        if col in df_clean.columns:
            # Remove currency symbols and commas if present
            if df_clean[col].dtype == 'object':
                df_clean[col] = df_clean[col].astype(str).str.replace('$', '').str.replace(',', '')

            # Convert to numeric
            df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')

            # Handle negative values if requested
            if negative_to_zero:
                df_clean[col] = df_clean[col].clip(lower=0)

    return df_clean


def deduplicate_dataframe(
    df: pd.DataFrame,
    subset: Optional[List[str]] = None,
    keep: str = 'first'
) -> pd.DataFrame:
    """
    Remove duplicate rows from DataFrame.

    Args:
        df: DataFrame to deduplicate
        subset: Columns to use for identifying duplicates (None = all columns)
        keep: Which duplicate to keep ('first', 'last', False)

    Returns:
        pd.DataFrame: Deduplicated DataFrame
    """
    return df.drop_duplicates(subset=subset, keep=keep)


def filter_by_date_range(
    df: pd.DataFrame,
    date_col: str,
    start_date: Optional[pd.Timestamp] = None,
    end_date: Optional[pd.Timestamp] = None
) -> pd.DataFrame:
    """
    Filter DataFrame to a specific date range.

    Args:
        df: DataFrame to filter
        date_col: Name of date column to filter on
        start_date: Start of date range (None = no lower bound)
        end_date: End of date range (None = no upper bound)

    Returns:
        pd.DataFrame: Filtered DataFrame
    """
    if date_col not in df.columns:
        return df

    df_filtered = df.copy()
    df_filtered[date_col] = pd.to_datetime(df_filtered[date_col], errors='coerce')

    if start_date is not None:
        df_filtered = df_filtered[df_filtered[date_col] >= start_date]

    if end_date is not None:
        df_filtered = df_filtered[df_filtered[date_col] <= end_date]

    return df_filtered
