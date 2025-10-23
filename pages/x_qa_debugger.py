# pages/debugger.py
import streamlit as st
import pandas as pd
from utils.config import setup_page

# 1) Page config & branding
setup_page("🔍 Debug DataFrame Issue")

# 3) Page header
with st.container():
    st.title("🔍 Debug DataFrame Issue")
    st.write("---")

# 4) Import & data-load check
with st.container():
    st.header("1️⃣ combine_deals import + load")
    try:
        from scripts.combine_hubspot_mca import combine_deals
        st.success("✅ combine_deals imported")
        df = combine_deals()
        st.success(f"✅ Loaded — {df.shape[0]:,} rows × {df.shape[1]:,} cols")
    except Exception as e:
        st.error(f"❌ {e}")

# 5) Data overview
with st.container():
    st.header("2️⃣ Data Overview")
    if 'df' in locals():
        st.metric("Rows", len(df))
        st.metric("Columns", len(df.columns))
        st.write("---")
        st.subheader("Key Columns")
        for col in [
            'amount_hubspot', 'total_funded_amount',
            'current_balance', 'past_due_amount', 'status_category'
        ]:
            if col in df:
                st.write(f"• ✅ **{col}**: {df[col].dtype}, {df[col].count():,} non-null")
            else:
                st.write(f"• ❌ **{col}**: MISSING")
        st.subheader("Sample Data")
        st.dataframe(df.head(), use_container_width=True)

# 6) Filter test
with st.container():
    st.header("3️⃣ Filter Test")
    if 'df' in locals() and 'status_category' in df:
        filtered = df[df.status_category != "Canceled"]
        st.write(f"Rows after filter: {len(filtered):,}")
        st.subheader("Status Counts")
        st.bar_chart(df.status_category.value_counts())
    else:
        st.warning("No status_category to test against.")

# 7) File-structure check
with st.container():
    st.header("4️⃣ File Structure")
    try:
        path = "/mnt/src/streamlit/pages/mca_dashboard.py"   # ← corrected
        lines = open(path).read().splitlines()[:50]
        st.code("\n".join(lines), language="python")
    except Exception as e:
        st.error(f"❌ Unable to read `{path}`:\n{e}")
