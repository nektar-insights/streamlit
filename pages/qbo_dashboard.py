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

# -----------------------------------------
# 🔁 Cash Flow Summary & Forecast (Hybrid)
# -----------------------------------------

st.subheader("🧮 Cash Flow Analysis Based on General Ledger + Transactions")

# 1. Working Days from General Ledger
gl_start = gl_df["txn_date"].min().date()
gl_end = gl_df["txn_date"].max().date()
working_days = busday_count(gl_start, gl_end)

# 2. Total Inflows (Payments)
inflows = df[df["transaction_type"] == "Payment"].copy()
inflows_amt = inflows["amount"].abs().sum()

# 3. Total Outflows (Expenses, Checks, Bills)
outflows = df[df["transaction_type"].isin(["Expense", "Check", "Bill"])].copy()
outflows_amt = outflows["amount"].abs().sum()

# 4. Averages
avg_inflow_day = inflows_amt / working_days if working_days else 0
avg_outflow_day = outflows_amt / working_days if working_days else 0

avg_inflow_week = avg_inflow_day * 5
avg_outflow_week = avg_outflow_day * 5

avg_inflow_month = avg_inflow_day * 21
avg_outflow_month = avg_outflow_day * 21

net_burn_day = avg_outflow_day - avg_inflow_day

# 5. Display
st.markdown(f"""
- **Working Days:** {working_days}  
- **Total Inflows:** ${inflows_amt:,.2f}  
- **Total Outflows:** ${outflows_amt:,.2f}  
- **Avg Inflow/Day:** ${avg_inflow_day:,.2f}  
- **Avg Outflow/Day:** ${avg_outflow_day:,.2f}  
- **Avg Inflow/Week:** ${avg_inflow_week:,.2f}  
- **Avg Outflow/Week:** ${avg_outflow_week:,.2f}  
- **Avg Inflow/Month:** ${avg_inflow_month:,.2f}  
- **Avg Outflow/Month:** ${avg_outflow_month:,.2f}
""")

# 6. Forecast Net Burn
st.subheader("📉 Net Burn Forecast")
forecast_days = [30, 60, 90]
forecast_df = pd.DataFrame({
    "Days": forecast_days,
    "Net Burn ($)": [f"${day * net_burn_day:,.2f}" for day in forecast_days]
})
st.dataframe(forecast_df, use_container_width=True)

# 7. Net Burn Visualization
st.subheader("📊 Net Cash Flow Breakdown")

burn_chart = pd.DataFrame({
    "Category": ["Inflows", "Outflows", "Net Burn"],
    "Amount": [inflows_amt, outflows_amt, outflows_amt - inflows_amt]
})

bar = alt.Chart(burn_chart).mark_bar().encode(
    x="Category:N",
    y=alt.Y("Amount:Q", title="Total ($)", axis=alt.Axis(format="$,.0f")),
    color=alt.Color("Category:N", scale=alt.Scale(
        domain=["Inflows", "Outflows", "Net Burn"],
        range=["#34a853", "#ea4335", "#fbbc04"]
    )),
    tooltip=["Category", alt.Tooltip("Amount", format="$,.2f")]
).properties(width=700, height=300)

st.altair_chart(bar, use_container_width=True)
