import streamlit as st
import pandas as pd
import altair as alt
from numpy import busday_count
from supabase import create_client

# -------------------------
# Setup: Supabase Connection
# -------------------------
url = st.secrets["supabase"]["url"]
key = st.secrets["supabase"]["service_role"]
supabase = create_client(url, key)

# -------------------------
# Load Data
# -------------------------
@st.cache_data(ttl=3600)
def load_data():
    txn_data = supabase.table("qbo_transactions").select("*").execute()
    gl_data = supabase.table("qbo_general_ledger").select("*").execute()
    return pd.DataFrame(txn_data.data), pd.DataFrame(gl_data.data)

df, gl_df = load_data()

# -------------------------
# Preprocess Data
# -------------------------
for d in [df, gl_df]:
    d["amount"] = pd.to_numeric(d["amount"], errors="coerce")

df["date"] = pd.to_datetime(df["date"], errors="coerce")
gl_df["txn_date"] = pd.to_datetime(gl_df["txn_date"], errors="coerce")

# -------------------------
# Working Days
# -------------------------
start_date = gl_df["txn_date"].min().date()
today = pd.Timestamp.today().date()
working_days = busday_count(start_date, today)

# -------------------------
# Inflows (Payments)
# -------------------------
payments_df = df[df["transaction_type"] == "Payment"].copy()
payments_df["amount"] = payments_df["amount"].abs()
total_inflows = payments_df["amount"].sum()

avg_inflow_day = total_inflows / working_days if working_days else 0
avg_inflow_week = avg_inflow_day * 5
avg_inflow_month = avg_inflow_day * 21

# -------------------------
# Outflows (Expenses, Checks, Bills)
# -------------------------
expenses_df = df[df["transaction_type"].isin(["Expense", "Check", "Bill"])].copy()
expenses_df["amount"] = expenses_df["amount"].abs()
total_outflows = expenses_df["amount"].sum()

avg_outflow_day = total_outflows / working_days if working_days else 0
avg_outflow_week = avg_outflow_day * 5
avg_outflow_month = avg_outflow_day * 21

# -------------------------
# Display Summary
# -------------------------
st.title("💸 Cash Flow Forecast Dashboard")

st.subheader("🔢 Summary")
st.markdown(f"""
- **Working Days:** {working_days}  
- **Total Inflows:** ${total_inflows:,.2f}  
- **Total Outflows:** ${total_outflows:,.2f}  
- **Avg Inflow/Day:** ${avg_inflow_day:,.2f}  
- **Avg Outflow/Day:** ${avg_outflow_day:,.2f}  
- **Avg Inflow/Week:** ${avg_inflow_week:,.2f}  
- **Avg Outflow/Week:** ${avg_outflow_week:,.2f}  
- **Avg Inflow/Month:** ${avg_inflow_month:,.2f}  
- **Avg Outflow/Month:** ${avg_outflow_month:,.2f}
""")

# -------------------------
# Net Burn Forecast
# -------------------------
st.subheader("📅 Projected Net Cash Burn")
daily_net = avg_inflow_day - avg_outflow_day

forecast_df = pd.DataFrame({
    "Forecast Horizon (Days)": [30, 60, 90],
    "Projected Net Change ($)": [daily_net * 30, daily_net * 60, daily_net * 90]
})

forecast_df["Projected Net Change ($)"] = forecast_df["Projected Net Change ($)"].map("${:,.2f}".format)

st.dataframe(forecast_df, use_container_width=True)

# -------------------------
# Bar Chart: Inflow vs Outflow Averages
# -------------------------
bar_data = pd.DataFrame({
    "Type": ["Inflow", "Outflow"],
    "Daily Avg": [avg_inflow_day, avg_outflow_day],
    "Weekly Avg": [avg_inflow_week, avg_outflow_week],
    "Monthly Avg": [avg_inflow_month, avg_outflow_month]
})

bar_chart = alt.Chart(bar_data.melt(id_vars=["Type"], var_name="Interval", value_name="Amount"))\
    .mark_bar().encode(
        x=alt.X("Interval:N", title="Interval"),
        y=alt.Y("Amount:Q", title="Avg Amount ($)", axis=alt.Axis(format="$,.0f")),
        color="Type:N",
        column="Type:N",
        tooltip=["Type", "Interval", alt.Tooltip("Amount", format="$,.2f")]
    ).properties(
        width=200,
        height=300
    )

st.altair_chart(bar_chart, use_container_width=True)
