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
