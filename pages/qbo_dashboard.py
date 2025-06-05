import streamlit as st
import pandas as pd
import altair as alt
from numpy import busday_count
from supabase import create_client
import io
from xhtml2pdf import pisa
import hashlib

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
    df_txn = pd.DataFrame(supabase.table("qbo_transactions").select("*").execute().data)
    df_gl = pd.DataFrame(supabase.table("qbo_general_ledger").select("*").execute().data)
    return df_txn, df_gl

df, gl_df = load_qbo_data()

# Preprocess Transactions
for d in [df, gl_df]:
    if "amount" in d.columns:
        d["amount"] = pd.to_numeric(d["amount"], errors="coerce")
    if "date" in d.columns:
        d["date"] = pd.to_datetime(d["date"], errors="coerce")

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

st.subheader("Loan Performance by Deal")
st.dataframe(
    pivot_display[["Deal Name", "Invoice", "Payment", "Balance (with Risk)"]]
    .sort_values("Balance (with Risk)", ascending=False),
    use_container_width=True
)

# -------------------------
# Top Outstanding Balances
# -------------------------
top_balances = pivot.sort_values("balance", ascending=False).head(15)

bar_chart = alt.Chart(top_balances).mark_bar().encode(
    x=alt.X("balance:Q", title="Outstanding Balance ($)", axis=alt.Axis(format="$,.0f")),
    y=alt.Y("name:N", sort="-x", title="Deal Name"),
    tooltip=["name", alt.Tooltip("balance:Q", format="$,.2f")]
).properties(
    width=800,
    height=400,
    title="Top 15 Deals by Outstanding Balance"
)

st.altair_chart(bar_chart, use_container_width=True)

# -------------------------
# Problem Loan Ratios
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
        tooltip=["name", alt.Tooltip("percentage:Q", format=".2f")],
        color=alt.Color("risk_color:N", scale=None, legend=None)
    )
    .properties(
        width=800,
        height=400,
        title="Problem Loan Ratios (Top 15 by Balance %)"
    )
)

st.altair_chart(ratio_chart, use_container_width=True)

# -------------------------
# Monthly Payment Trends
# -------------------------
payments_df = df[df["transaction_type"] == "Payment"].copy()
payments_df["amount"] = payments_df["amount"].abs()
payments_df["month"] = payments_df["date"].dt.strftime("%b %Y")

monthly_payments = payments_df.groupby("month")["amount"].sum().reset_index()

st.subheader("Monthly Payments Received")
payment_trend = alt.Chart(monthly_payments).mark_line(point=True).encode(
    x=alt.X("month:N", title="Month"),
    y=alt.Y("amount:Q", title="Total Payments ($)", axis=alt.Axis(format="$,.0f")),
    tooltip=["month", alt.Tooltip("amount", format="$,.2f")]
).properties(width=800, height=300)

st.altair_chart(payment_trend, use_container_width=True)

# -------------------------
# Cash Flow Stats
# -------------------------
from numpy import busday_count

first_day = gl_df["date"].min().date()
today = pd.Timestamp.today().date()
working_days = busday_count(first_day, today)

inflows = df[df["transaction_type"].isin(["Payment"])]
outflows = df[df["transaction_type"].isin(["Expense", "Check", "Bill"])]

inflows_total = inflows["amount"].sum()
outflows_total = outflows["amount"].sum()

avg_inflow_day = inflows_total / working_days if working_days else 0
avg_outflow_day = outflows_total / working_days if working_days else 0

avg_inflow_week = avg_inflow_day * 5
avg_outflow_week = avg_outflow_day * 5
avg_inflow_month = avg_inflow_day * 21
avg_outflow_month = avg_outflow_day * 21

st.subheader("Cash Flow Averages")
st.markdown(f"""
- Working Days: {working_days}
- Total Inflows: ${inflows_total:,.2f}
- Total Outflows: ${outflows_total:,.2f}
- Avg Inflow/Day: ${avg_inflow_day:,.2f}
- Avg Outflow/Day: ${avg_outflow_day:,.2f}
- Avg Inflow/Week: ${avg_inflow_week:,.2f}
- Avg Outflow/Week: ${avg_outflow_week:,.2f}
- Avg Inflow/Month: ${avg_inflow_month:,.2f}
- Avg Outflow/Month: ${avg_outflow_month:,.2f}
""")

# -------------------------
# Net Burn Forecast
# -------------------------
st.subheader("Cash Flow Forecast")
net_daily = avg_inflow_day - avg_outflow_day
forecast_df = pd.DataFrame({
    "Days": [30, 60, 90],
    "Projected Net Burn": [net_daily * d for d in [30, 60, 90]]
})
forecast_df["Projected Net Burn"] = forecast_df["Projected Net Burn"].map("${:,.2f}".format)
st.dataframe(forecast_df, use_container_width=True)

# -------------------------
# Deduplication Strategy Suggestion (not executed)
# -------------------------
# UUID hash by date, amount, description, txn_type
if "dedupe" not in gl_df.columns:
    gl_df["dedupe"] = gl_df.apply(
        lambda row: hashlib.md5(f"{row['date']}-{row['amount']}-{row['description']}-{row['txn_type']}".encode()).hexdigest(),
        axis=1
    )
