# Debug DataFrame Issue
import streamlit as st
import pandas as pd

st.title("🔍 Debug DataFrame Issue")

# Check if the combine_deals function is accessible
try:
    from scripts.combine_hubspot_mca import combine_deals
    st.success("✅ combine_deals function imported successfully")
    
    # Try to load the data
    try:
        df = combine_deals()
        st.success(f"✅ Data loaded successfully - Shape: {df.shape}")
        
        # Show basic info
        st.subheader("Data Overview")
        st.write(f"Rows: {len(df)}")
        st.write(f"Columns: {len(df.columns)}")
        
        # Check key columns
        key_columns = ['amount_hubspot', 'total_funded_amount', 'current_balance', 'past_due_amount', 'status_category']
        st.subheader("Key Column Check")
        for col in key_columns:
            if col in df.columns:
                st.write(f"✅ {col}: {df[col].dtype}, {df[col].count()} non-null values")
            else:
                st.write(f"❌ {col}: MISSING")
        
        # Show first few rows
        st.subheader("Sample Data")
        st.dataframe(df.head())
        
        # Test filtering
        st.subheader("Filter Test")
        try:
            filtered_df = df[df["status_category"] != "Canceled"]
            st.write(f"After filtering out Canceled: {len(filtered_df)} rows remaining")
            
            # Test status categories
            status_counts = df["status_category"].value_counts()
            st.write("Status categories:")
            st.write(status_counts)
            
        except Exception as filter_error:
            st.error(f"Error filtering data: {filter_error}")
        
    except Exception as load_error:
        st.error(f"❌ Error loading data: {load_error}")
        
except ImportError as import_error:
    st.error(f"❌ Error importing combine_deals: {import_error}")

# Check if the main dashboard file structure
st.subheader("File Structure Check")
try:
    with open("/mount/src/streamlit/pages/mca_dashboard.py", "r") as f:
        first_50_lines = f.readlines()[:50]
        st.text("First 50 lines of mca_dashboard.py:")
        st.code("".join(first_50_lines), language="python")
except Exception as file_error:
    st.error(f"Error reading dashboard file: {file_error}")
