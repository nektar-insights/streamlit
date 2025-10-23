# utils/display_components.py
"""
Reusable UI components for Streamlit dashboards.
Provides consistent metric displays, charts, and data tables across pages.
"""

import streamlit as st
import pandas as pd
import altair as alt
from typing import List, Dict, Optional, Any, Tuple


def display_metric_row(
    metrics: List[Dict[str, Any]],
    columns: Optional[int] = None
):
    """
    Display a row of metrics using Streamlit columns.

    Args:
        metrics: List of dicts with keys: 'label', 'value', 'delta' (optional), 'help' (optional)
        columns: Number of columns (None = auto-calculate based on metrics count)

    Example:
        display_metric_row([
            {'label': 'Total Loans', 'value': 150, 'delta': '+10'},
            {'label': 'Total Amount', 'value': '$1.5M', 'help': 'Sum of all loans'}
        ])
    """
    if not metrics:
        return

    n_cols = columns or len(metrics)
    cols = st.columns(n_cols)

    for i, metric in enumerate(metrics):
        with cols[i % n_cols]:
            st.metric(
                label=metric.get('label', ''),
                value=metric.get('value', 0),
                delta=metric.get('delta'),
                help=metric.get('help')
            )


def display_kpi_cards(
    kpis: Dict[str, Any],
    columns: int = 4,
    format_currency: bool = True
):
    """
    Display KPI cards in a grid layout.

    Args:
        kpis: Dictionary mapping KPI names to values
        columns: Number of columns in grid
        format_currency: Whether to format values as currency

    Example:
        display_kpi_cards({
            'Total Invested': 1500000,
            'Total Returned': 1200000,
            'Net Position': 300000,
            'ROI': 0.25
        }, columns=4)
    """
    cols = st.columns(columns)

    for i, (label, value) in enumerate(kpis.items()):
        with cols[i % columns]:
            if format_currency and isinstance(value, (int, float)):
                if 'rate' in label.lower() or 'roi' in label.lower() or '%' in label.lower():
                    formatted_value = f"{value:.2%}" if isinstance(value, float) else value
                else:
                    formatted_value = f"${value:,.0f}"
            else:
                formatted_value = value

            st.metric(label=label, value=formatted_value)


def display_summary_table(
    df: pd.DataFrame,
    title: str = "",
    columns: Optional[List[str]] = None,
    format_dict: Optional[Dict[str, str]] = None,
    hide_index: bool = True
):
    """
    Display a formatted summary table.

    Args:
        df: DataFrame to display
        title: Optional title for the table
        columns: Columns to display (None = all)
        format_dict: Dictionary mapping column names to format strings
        hide_index: Whether to hide the index

    Example:
        display_summary_table(
            df,
            title="Loan Summary",
            columns=['loan_id', 'amount', 'status'],
            format_dict={'amount': '${:,.2f}'}
        )
    """
    if title:
        st.subheader(title)

    display_df = df[columns] if columns else df.copy()

    if format_dict:
        styled_df = display_df.style.format(format_dict)
        st.dataframe(styled_df, use_container_width=True, hide_index=hide_index)
    else:
        st.dataframe(display_df, use_container_width=True, hide_index=hide_index)


def create_comparison_chart(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    color_col: Optional[str] = None,
    title: str = "",
    chart_type: str = "bar"
) -> alt.Chart:
    """
    Create a comparison chart (bar, line, or area).

    Args:
        df: DataFrame with data
        x_col: Column for x-axis
        y_col: Column for y-axis
        color_col: Optional column for color encoding
        title: Chart title
        chart_type: Type of chart ('bar', 'line', 'area')

    Returns:
        alt.Chart: Altair chart object
    """
    base = alt.Chart(df).encode(
        x=alt.X(f"{x_col}:N", title=x_col.replace('_', ' ').title()),
        y=alt.Y(f"{y_col}:Q", title=y_col.replace('_', ' ').title()),
    )

    if color_col:
        base = base.encode(color=alt.Color(f"{color_col}:N"))

    if chart_type == "bar":
        chart = base.mark_bar()
    elif chart_type == "line":
        chart = base.mark_line(point=True)
    elif chart_type == "area":
        chart = base.mark_area()
    else:
        chart = base.mark_bar()

    chart = chart.properties(title=title, width=600, height=400)

    return chart


def display_data_quality_metrics(
    df: pd.DataFrame,
    show_null_analysis: bool = True,
    show_duplicates: bool = True
):
    """
    Display data quality metrics for a DataFrame.

    Args:
        df: DataFrame to analyze
        show_null_analysis: Whether to show null value analysis
        show_duplicates: Whether to show duplicate analysis
    """
    st.subheader("Data Quality Metrics")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Total Rows", f"{len(df):,}")

    with col2:
        st.metric("Total Columns", len(df.columns))

    with col3:
        memory_usage = df.memory_usage(deep=True).sum() / 1024**2
        st.metric("Memory Usage", f"{memory_usage:.2f} MB")

    if show_null_analysis:
        st.write("**Null Values by Column:**")
        null_counts = df.isnull().sum()
        null_pcts = (null_counts / len(df) * 100).round(2)

        null_df = pd.DataFrame({
            'Column': null_counts.index,
            'Null Count': null_counts.values,
            'Null %': null_pcts.values
        }).sort_values('Null Count', ascending=False)

        null_df = null_df[null_df['Null Count'] > 0]

        if not null_df.empty:
            st.dataframe(null_df, use_container_width=True, hide_index=True)
        else:
            st.success("No null values found!")

    if show_duplicates:
        duplicate_count = df.duplicated().sum()
        st.metric("Duplicate Rows", f"{duplicate_count:,}")

        if duplicate_count > 0:
            st.warning(f"Found {duplicate_count} duplicate rows")


def create_download_button(
    df: pd.DataFrame,
    filename: str = "data.csv",
    label: str = "Download Data",
    file_format: str = "csv"
):
    """
    Create a download button for a DataFrame.

    Args:
        df: DataFrame to download
        filename: Name of downloaded file
        label: Button label
        file_format: Format ('csv' or 'excel')
    """
    if file_format == "csv":
        data = df.to_csv(index=False).encode('utf-8')
        mime = "text/csv"
    elif file_format == "excel":
        import io
        buffer = io.BytesIO()
        with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
            df.to_excel(writer, index=False)
        data = buffer.getvalue()
        mime = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    else:
        raise ValueError(f"Unsupported format: {file_format}")

    st.download_button(
        label=label,
        data=data,
        file_name=filename,
        mime=mime
    )


def display_filter_section(
    df: pd.DataFrame,
    filter_columns: List[str],
    key_prefix: str = ""
) -> pd.DataFrame:
    """
    Display a filter section for a DataFrame and return filtered result.

    Args:
        df: DataFrame to filter
        filter_columns: Columns to create filters for
        key_prefix: Prefix for widget keys (for multiple filter sections)

    Returns:
        pd.DataFrame: Filtered DataFrame
    """
    st.subheader("Filters")

    filtered_df = df.copy()

    for col in filter_columns:
        if col not in df.columns:
            continue

        unique_values = df[col].dropna().unique()

        if len(unique_values) <= 20:
            # Use multiselect for categorical with few values
            selected = st.multiselect(
                f"Filter by {col.replace('_', ' ').title()}",
                options=sorted(unique_values.tolist()),
                key=f"{key_prefix}_{col}_filter"
            )
            if selected:
                filtered_df = filtered_df[filtered_df[col].isin(selected)]

        elif pd.api.types.is_numeric_dtype(df[col]):
            # Use slider for numeric
            min_val, max_val = float(df[col].min()), float(df[col].max())
            selected_range = st.slider(
                f"Filter by {col.replace('_', ' ').title()}",
                min_value=min_val,
                max_value=max_val,
                value=(min_val, max_val),
                key=f"{key_prefix}_{col}_slider"
            )
            filtered_df = filtered_df[
                (filtered_df[col] >= selected_range[0]) &
                (filtered_df[col] <= selected_range[1])
            ]

    return filtered_df


def display_progress_indicator(
    current: int,
    total: int,
    label: str = "Progress",
    show_percentage: bool = True
):
    """
    Display a progress indicator.

    Args:
        current: Current value
        total: Total value
        label: Label for the progress bar
        show_percentage: Whether to show percentage text
    """
    progress = current / total if total > 0 else 0

    if show_percentage:
        st.write(f"**{label}:** {current:,} / {total:,} ({progress:.1%})")

    st.progress(progress)


def create_time_series_chart(
    df: pd.DataFrame,
    date_col: str,
    value_col: str,
    title: str = "",
    color: str = "#1f77b4"
) -> alt.Chart:
    """
    Create a time series line chart.

    Args:
        df: DataFrame with time series data
        date_col: Column with dates
        value_col: Column with values
        title: Chart title
        color: Line color

    Returns:
        alt.Chart: Altair chart object
    """
    chart = alt.Chart(df).mark_line(point=True, color=color).encode(
        x=alt.X(f"{date_col}:T", title="Date"),
        y=alt.Y(f"{value_col}:Q", title=value_col.replace('_', ' ').title()),
        tooltip=[
            alt.Tooltip(f"{date_col}:T", title="Date", format="%Y-%m-%d"),
            alt.Tooltip(f"{value_col}:Q", title=value_col.replace('_', ' ').title())
        ]
    ).properties(
        title=title,
        width=700,
        height=400
    )

    return chart


def create_date_range_filter(
    df: pd.DataFrame,
    date_col: str,
    label: str = "Filter by Date Range",
    checkbox_label: str = "Enable Date Filter",
    default_enabled: bool = False,
    key_prefix: str = "date_filter"
) -> Tuple[pd.DataFrame, bool]:
    """
    Standardized date range filter with checkbox toggle.

    Args:
        df: DataFrame to filter
        date_col: Name of the date column
        label: Label for the date input widget
        checkbox_label: Label for the checkbox toggle
        default_enabled: Whether filter is enabled by default
        key_prefix: Prefix for widget keys (for multiple filters on same page)

    Returns:
        Tuple of (filtered_df, is_filter_active)

    Example:
        filtered_df, is_active = create_date_range_filter(
            df,
            "funding_date",
            label="Select Funding Date Range",
            checkbox_label="Filter by Funding Date"
        )
    """
    if date_col not in df.columns or df[date_col].isna().all():
        st.info(f"No date data available in column '{date_col}'")
        return df.copy(), False

    # Extract min and max dates
    min_date = df[date_col].min().date()
    max_date = df[date_col].max().date()

    # Checkbox to enable/disable filter
    use_filter = st.checkbox(checkbox_label, value=default_enabled, key=f"{key_prefix}_checkbox")

    if use_filter:
        # Date range picker
        date_range = st.date_input(
            label,
            value=[min_date, max_date],
            min_value=min_date,
            max_value=max_date,
            key=f"{key_prefix}_input"
        )

        # Apply filter if valid range
        if isinstance(date_range, (list, tuple)) and len(date_range) == 2:
            filtered_df = df[
                (df[date_col].dt.date >= date_range[0]) &
                (df[date_col].dt.date <= date_range[1])
            ].copy()
            return filtered_df, True
        else:
            return df.copy(), False
    else:
        return df.copy(), False


def create_partner_source_filter(
    df: pd.DataFrame,
    partner_col: str = "partner_source",
    label: str = "Filter by Partner Source",
    default_all: bool = True,
    key_prefix: str = "partner_filter"
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Standardized partner source multiselect filter.

    Args:
        df: DataFrame to filter
        partner_col: Name of the partner source column
        label: Label for the multiselect widget
        default_all: Whether to select all partners by default
        key_prefix: Prefix for widget keys (for multiple filters on same page)

    Returns:
        Tuple of (filtered_df, selected_partners)

    Example:
        filtered_df, selected = create_partner_source_filter(
            df,
            partner_col="partner_source",
            label="Select Partner Sources"
        )
    """
    if partner_col not in df.columns:
        st.warning(f"Column '{partner_col}' not found in data")
        return df.copy(), []

    # Get unique partner sources
    partners = sorted(df[partner_col].dropna().unique().tolist())

    if not partners:
        st.info(f"No partner source data available")
        return df.copy(), []

    # Multiselect widget
    selected_partners = st.multiselect(
        label,
        options=partners,
        default=partners if default_all else [],
        key=f"{key_prefix}_multiselect"
    )

    # Apply filter if selections made
    if selected_partners:
        filtered_df = df[df[partner_col].isin(selected_partners)].copy()
        return filtered_df, selected_partners
    else:
        # If nothing selected, return empty or all (based on default_all)
        if default_all:
            return df.copy(), partners
        else:
            return df[df[partner_col].isin([])].copy(), []


def create_status_filter(
    df: pd.DataFrame,
    status_col: str,
    label: str = "Filter by Status",
    include_all_option: bool = True,
    key_prefix: str = "status_filter"
) -> Tuple[pd.DataFrame, str]:
    """
    Standardized status selectbox filter.

    Args:
        df: DataFrame to filter
        status_col: Name of the status column
        label: Label for the selectbox widget
        include_all_option: Whether to include "All" option
        key_prefix: Prefix for widget keys (for multiple filters on same page)

    Returns:
        Tuple of (filtered_df, selected_status)

    Example:
        filtered_df, status = create_status_filter(
            df,
            status_col="loan_status",
            label="Filter by Loan Status"
        )
    """
    if status_col not in df.columns:
        st.warning(f"Column '{status_col}' not found in data")
        return df.copy(), "All"

    # Get unique statuses
    statuses = sorted(df[status_col].dropna().unique().tolist())

    if not statuses:
        st.info(f"No status data available")
        return df.copy(), "All"

    # Add "All" option if requested
    if include_all_option:
        options = ["All"] + statuses
    else:
        options = statuses

    # Selectbox widget
    selected_status = st.selectbox(
        label,
        options=options,
        index=0,
        key=f"{key_prefix}_selectbox"
    )

    # Apply filter if not "All"
    if selected_status == "All":
        return df.copy(), selected_status
    else:
        filtered_df = df[df[status_col] == selected_status].copy()
        return filtered_df, selected_status
