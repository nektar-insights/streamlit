import streamlit as st
import pandas as pd
import altair as alt
from numpy import busday_count
from supabase import create_client
import io
from xhtml2pdf import pisa

# -------------------------
# Setup: Supabase Connection
# -------------------------
url = st.secrets["supabase"]["url"]
key = st.secrets["supabase"]["service_role"]
supabase = create_client(url, key)

# -------------------------
# Load and Prepare Data
# -------------------------
@st.cache_data(ttl=3600)
def load_qbo_data():
    tx_res = supabase.table("qbo_transactions").select("*").execute()
    gl_res = supabase.table("qbo_general_ledger").select("*").execute()
    return pd.DataFrame(tx_res.data), pd.DataFrame(gl_res.data)

df, gl_df = load_qbo_data()

# Preprocess Transactions
df["amount"] = pd.to_numeric(df["amount"], errors="coerce")
df["date"] = pd.to_datetime(df["date"], errors="coerce")

gl_df["amount"] = pd.to_numeric(gl_df["amount"], errors="coerce")
gl_df["txn_date"] = pd.to_datetime(gl_df["txn_date"], errors="coerce")

st.title("QBO Dashboard")

# -------------------------
# Loan Performance by Deal
# -------------------------
filtered_df = df[df["transaction_type"].isin(["Invoice", "Payment"])].copy()
filtered_df = filtered_df[~filtered_df["name"].isin(["CSL", "VEEM"])]
filtered_df["amount"] = filtered_df["amount"].abs()

pivot = filtered_df.pivot_table(
    index="name",
    columns="transaction_type",
    values="amount",
    aggfunc="sum",
    fill_value=0
).reset_index()

pivot["balance"] = pivot.get("Invoice", 0) - pivot.get("Payment", 0)
pivot["balance_ratio"] = pivot["balance"] / pivot["Invoice"]
pivot["indicator"] = pivot["balance_ratio"].apply(
    lambda x: "🔴" if x >= 0.25 else ("🟡" if x >= 0.10 else "🟢")
)

pivot_display = pivot.copy()
pivot_display["Invoice"] = pivot_display["Invoice"].map("${:,.2f}".format)
pivot_display["Payment"] = pivot_display["Payment"].map("${:,.2f}".format)
pivot_display["balance"] = pivot_display["balance"].map("${:,.2f}".format)
pivot_display["Deal Name"] = pivot_display["name"]
pivot_display["Balance (with Risk)"] = pivot_display["indicator"] + " " + pivot_display["balance"]

st.subheader("\U0001F4BC Loan Performance by Deal")
st.dataframe(
    pivot_display[["Deal Name", "Invoice", "Payment", "Balance (with Risk)"]].sort_values("Balance (with Risk)", ascending=False),
    use_container_width=True
)

# -------------------------
# Bar Chart: Top Balances
# -------------------------
top_balances = pivot.sort_values("balance", ascending=False).head(15)

bar_chart = alt.Chart(top_balances).mark_bar(color="#e45756").encode(
    x=alt.X("balance:Q", title="Outstanding Balance ($)", axis=alt.Axis(format="$,.0f")),
    y=alt.Y("name:N", sort="-x", title="Deal Name"),
    tooltip=["name", alt.Tooltip("balance:Q", format="$,.2f")]
).properties(
    width=800,
    height=400,
    title="\U0001F50E Top 15 Deals by Outstanding Balance"
)

st.altair_chart(bar_chart, use_container_width=True)

# -------------------------
# Ratio Chart: Balance as % of Invoice
# -------------------------
problem_loans = pivot[pivot["Invoice"] > 0].copy()
problem_loans["percentage"] = (problem_loans["balance"] / problem_loans["Invoice"]) * 100
problem_loans = problem_loans.sort_values("percentage", ascending=False).head(15)

ratio_chart = (
    alt.Chart(problem_loans)
    .transform_calculate(
        risk_color="""
        datum.percentage >= 25 ? '#e45756' :
        datum.percentage >= 10 ? '#ffcc00' :
        '#34a853'
        """
    )
    .mark_bar()
    .encode(
        x=alt.X("percentage:Q", title="Balance as % of Invoice", axis=alt.Axis(format=".1f")),
        y=alt.Y("name:N", title="Deal Name", sort="-x"),
        tooltip=[
            alt.Tooltip("name", title="Deal Name"),
            alt.Tooltip("percentage:Q", title="Balance %", format=".2f")
        ],
        color=alt.Color("risk_color:N", scale=None, legend=None)
    )
    .properties(
        width=800,
        height=400,
        title="\U0001F6A8 Problem Loan Ratios (Top 15 by Balance %)"
    )
)

st.altair_chart(ratio_chart, use_container_width=True)

# -------------------------
# Net Cash Flow Forecast
# -------------------------
st.subheader("📅 Net Cash Flow Forecast (Projected Change in Cash)")

# Use general ledger for date boundaries (more reliable)
gl_dates = gl_df["txn_date"].dropna()
gl_dates = pd.to_datetime(gl_dates, errors="coerce")
start = gl_dates.min().date()
end = gl_dates.max().date()
working_days = busday_count(start, end)

# Sum inflows & outflows from qbo transactions
inflows_df = df[df["transaction_type"].isin(["Payment", "SalesReceipt", "BankDeposit"])].copy()
outflows_df = df[df["transaction_type"].isin(["Expense", "Check", "Bill", "VendorCredit"])].copy()

inflows_df["amount"] = inflows_df["amount"].abs()
outflows_df["amount"] = outflows_df["amount"].abs()

total_inflows = inflows_df["amount"].sum()
total_outflows = outflows_df["amount"].sum()

avg_inflow_day = total_inflows / working_days if working_days else 0
avg_outflow_day = total_outflows / working_days if working_days else 0

avg_inflow_week = avg_inflow_day * 5
avg_outflow_week = avg_outflow_day * 5
avg_inflow_month = avg_inflow_day * 21
avg_outflow_month = avg_outflow_day * 21

net_cash_gain_per_day = avg_inflow_day - avg_outflow_day

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
- **Net Cash Change/Day:** ${net_cash_gain_per_day:,.2f}
""")

# Forecast over next 30/60/90 days
forecast_days = [30, 60, 90]
forecast_df = pd.DataFrame({
    "Forecast Horizon (Days)": forecast_days,
    "Projected Net Change ($)": [
        f"${day * net_cash_gain_per_day:,.2f}" for day in forecast_days
    ]
})

st.dataframe(forecast_df, use_container_width=True)
