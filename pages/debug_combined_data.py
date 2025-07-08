# debug_combined_data.py
import streamlit as st
import pandas as pd
from scripts.combine_hubspot_mca import combine_deals

st.set_page_config(page_title="Combined Deals Sample", layout="wide")
st.title("🔍 Sample Combined Deals Dataset Structure")

# Load data
with st.spinner("Loading combined deals..."):
    df = combine_deals()

st.success(f"✅ Data loaded successfully!")

# Show basic info
st.subheader("Dataset Overview")
col1, col2, col3 = st.columns(3)
col1.metric("Total Rows", len(df))
col2.metric("Total Columns", len(df.columns))
col3.metric("Memory Usage", f"{df.memory_usage(deep=True).sum() / 1024**2:.1f} MB")

# Show column names
st.subheader("Column Names")
st.write("**All columns in the combined dataset:**")
columns_df = pd.DataFrame({
    "Column Name": df.columns.tolist(),
    "Data Type": [str(dtype) for dtype in df.dtypes],
    "Non-Null Count": [df[col].count() for col in df.columns],
    "Null Count": [df[col].isnull().sum() for col in df.columns]
})
st.dataframe(columns_df, use_container_width=True)

# Show sample data
st.subheader("Sample Data (First 5 Rows)")
st.dataframe(df.head(), use_container_width=True)

# Look for key fields we need
st.subheader("🔍 Key Fields Analysis")

# Check for CSL participation field
csl_candidates = [col for col in df.columns if any(term in col.lower() for term in ['amount', 'participation', 'hubspot', 'csl'])]
st.write("**Potential CSL participation fields:**")
if csl_candidates:
    for col in csl_candidates:
        st.write(f"- `{col}` (Type: {df[col].dtype}, Non-null: {df[col].count()})")
else:
    st.write("❌ No obvious CSL participation fields found")

# Check for payment/RTR fields
payment_candidates = [col for col in df.columns if any(term in col.lower() for term in ['paid', 'payment', 'rtr', 'total'])]
st.write("**Potential payment/RTR fields:**")
if payment_candidates:
    for col in payment_candidates:
        st.write(f"- `{col}` (Type: {df[col].dtype}, Non-null: {df[col].count()})")
else:
    st.write("❌ No obvious payment/RTR fields found")

# Check for balance fields
balance_candidates = [col for col in df.columns if any(term in col.lower() for term in ['balance', 'outstanding', 'remaining'])]
st.write("**Potential balance fields:**")
if balance_candidates:
    for col in balance_candidates:
        st.write(f"- `{col}` (Type: {df[col].dtype}, Non-null: {df[col].count()})")
else:
    st.write("❌ No obvious balance fields found")

# Check for other important fields
other_important = [col for col in df.columns if any(term in col.lower() for term in ['principal', 'factor', 'term', 'commission', 'status'])]
st.write("**Other important fields:**")
if other_important:
    for col in other_important:
        st.write(f"- `{col}` (Type: {df[col].dtype}, Non-null: {df[col].count()})")
else:
    st.write("❌ No other important fields found")

# Show data types summary
st.subheader("Data Types Summary")
dtype_summary = df.dtypes.value_counts()
st.bar_chart(dtype_summary)

# Download options
st.subheader("📥 Export Options")
col1, col2 = st.columns(2)

with col1:
    # Download column list
    columns_csv = columns_df.to_csv(index=False)
    st.download_button(
        label="Download Column Info as CSV",
        data=columns_csv,
        file_name="combined_dataset_columns.csv",
        mime="text/csv"
    )

with col2:
    # Download sample data
    sample_csv = df.head(10).to_csv(index=False)
    st.download_button(
        label="Download Sample Data as CSV",
        data=sample_csv,
        file_name="combined_dataset_sample.csv",
        mime="text/csv"
    )
