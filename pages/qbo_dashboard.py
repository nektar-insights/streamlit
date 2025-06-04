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
# Load and Prepare Data
# -------------------------
@st.cache_data(ttl=3600)
def load_qbo_data():
    res = supabase.table("qbo_transactions").select("*").execute()
    return pd.DataFrame(res.data)

df = load_qbo_data()
df["amount"] = pd.to_numeric(df["amount"], errors="coerce")
df["date"] = pd.to_datetime(df["date"], errors="coerce")

st.title("QBO Transaction Dashboard")

# -------------------------
# Loan Performance by Deal
# -------------------------
# Only consider Invoices and Payments, exclude internal accounts
filtered_df = df[df["transaction_type"].isin(["Invoice", "Payment"])].copy()
filtered_df = filtered_df[~filtered_df["name"].isin(["CSL", "VEEM"])]
filtered_df["amount"] = filtered_df["amount"].abs()

# Pivot: group by name and transaction type
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

# Format for display
pivot_display = pivot.copy()
pivot_display["Invoice"] = pivot_display["Invoice"].map("${:,.2f}".format)
pivot_display["Payment"] = pivot_display["Payment"].map("${:,.2f}".format)
pivot_display["balance"] = pivot_display["balance"].map("${:,.2f}".format)
pivot_display["Deal Name"] = pivot_display["name"]
pivot_display["Balance (with Risk)"] = pivot_display["indicator"] + " " + pivot_display["balance"]

# Display table with risk indicators
st.subheader("💼 Loan Performance by Deal")
st.dataframe(
    pivot_display[["Deal Name", "Invoice", "Payment", "Balance (with Risk)"]]
    .sort_values("Balance (with Risk)", ascending=False),
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
    title="🔎 Top 15 Deals by Outstanding Balance"
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
        title="🚨 Problem Loan Ratios (Top 15 by Balance %)"
    )
)

st.altair_chart(ratio_chart, use_container_width=True)

# -------------------------
# Monthly Payments Trend
# -------------------------
payments_df = df[df["transaction_type"] == "Payment"].copy()
payments_df["amount"] = payments_df["amount"].abs()
payments_df["month"] = payments_df["date"].dt.strftime("%b %Y")

monthly_payments = payments_df.groupby("month")["amount"].sum().reset_index()

st.subheader("📈 Monthly Payments Received")
payment_trend = alt.Chart(monthly_payments).mark_line(point=True).encode(
    x=alt.X("month:N", title="Month"),
    y=alt.Y("amount:Q", title="Total Payments ($)", axis=alt.Axis(format="$,.0f")),
    tooltip=["month", alt.Tooltip("amount", format="$,.2f")]
).properties(width=800, height=300)

st.altair_chart(payment_trend, use_container_width=True)

# -------------------------
# Average Payment Inflow
# -------------------------
first_payment_date = payments_df["date"].min().date()
today = pd.Timestamp.today().date()
working_days = busday_count(first_payment_date, today)

total_payments = payments_df["amount"].sum()
avg_per_day = total_payments / working_days if working_days else 0
avg_per_week = avg_per_day * 5
avg_per_month = avg_per_day * 21

st.subheader("💵 Average Payment Inflow")
st.markdown(f"""
- **Total Payments:** ${total_payments:,.2f}  
- **Working Days:** {working_days}  
- **Avg/Day:** ${avg_per_day:,.2f}  
- **Avg/Week:** ${avg_per_week:,.2f}  
- **Avg/Month:** ${avg_per_month:,.2f}
""")

# -------------------------
# Average Expense Outflow
# -------------------------
expenses_df = df[df["transaction_type"].isin(["Expense", "Check", "Bill"])].copy()
expenses_df["amount"] = expenses_df["amount"].abs()
total_expenses = expenses_df["amount"].sum()

avg_exp_day = total_expenses / working_days if working_days else 0
avg_exp_week = avg_exp_day * 5
avg_exp_month = avg_exp_day * 21

st.subheader("📉 Average Expense Outflow")
st.markdown(f"""
- **Total Expenses:** ${total_expenses:,.2f}  
- **Avg/Day:** ${avg_exp_day:,.2f}  
- **Avg/Week:** ${avg_exp_week:,.2f}  
- **Avg/Month:** ${avg_exp_month:,.2f}
""")

# -------------------------
# Monthly Payments vs Expenses Trend
# -------------------------
expenses_df["month"] = expenses_df["date"].dt.strftime("%b %Y")

monthly_expenses = expenses_df.groupby("month")["amount"].sum().reset_index()
monthly_expenses["type"] = "Expense"
monthly_payments["type"] = "Payment"

combined = pd.concat([monthly_payments.rename(columns={"amount": "amount"}), monthly_expenses])
combined["month"] = pd.Categorical(combined["month"], categories=monthly_payments["month"], ordered=True)

trend_chart = alt.Chart(combined).mark_line(point=True).encode(
    x="month:N",
    y=alt.Y("amount:Q", title="Amount ($)", axis=alt.Axis(format="$,.0f")),
    color="type:N",
    tooltip=["type", "month", alt.Tooltip("amount", format="$,.2f")]
).properties(width=850, height=300, title="📊 Monthly Payments vs Expenses")

st.altair_chart(trend_chart, use_container_width=True)

import io
from xhtml2pdf import pisa

# -------------------------
# Cash Flow Forecast Based on Trends
# -------------------------
st.subheader("📅 Cash Flow Forecast (Projected Net Inflow)")

daily_net_inflow = avg_per_day - avg_exp_day

forecast_df = pd.DataFrame({
    "Forecast Horizon (Days)": [30, 60, 90],
    "Business Days": [30, 60, 90],
    "Projected Net Inflow ($)": [
        f"${daily_net_inflow * 30:,.2f}",
        f"${daily_net_inflow * 60:,.2f}",
        f"${daily_net_inflow * 90:,.2f}"
    ]
})

st.dataframe(forecast_df, use_container_width=True)

# -------------------------
# CSV + PDF Download for Loan Performance
# -------------------------
st.subheader("📥 Download Loan Performance Summary")

# Prepare cleaned download version
download_df = pivot[["name", "Invoice", "Payment", "balance", "balance_ratio"]].copy()
download_df.columns = ["Deal Name", "Invoice ($)", "Payment ($)", "Balance ($)", "Balance %"]
download_df["Invoice ($)"] = download_df["Invoice ($)"].map("${:,.2f}".format)
download_df["Payment ($)"] = download_df["Payment ($)"].map("${:,.2f}".format)
download_df["Balance ($)"] = download_df["Balance ($)"].map("${:,.2f}".format)
download_df["Balance %"] = download_df["Balance %"].apply(lambda x: f"{x:.2%}")

# Download as CSV
csv_data = download_df.to_csv(index=False).encode("utf-8")
st.download_button(
    label="📄 Download as CSV",
    data=csv_data,
    file_name="loan_performance_summary.csv",
    mime="text/csv"
)

# Download as PDF
def create_pdf_from_html(html: str):
    result = io.BytesIO()
    pisa.CreatePDF(io.StringIO(html), dest=result)
    return result.getvalue()

html = download_df.to_html(index=False)
pdf_data = create_pdf_from_html(html)

st.download_button(
    label="📄 Download as PDF",
    data=pdf_data,
    file_name="loan_performance_summary.pdf",
    mime="application/pdf"
)
