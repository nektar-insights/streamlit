import streamlit as st
import pandas as pd
import altair as alt
from numpy import busday_count
from supabase import create_client

# Supabase connection
url = st.secrets["supabase"]["url"]
key = st.secrets["supabase"]["service_role"]
supabase = create_client(url, key)

# Load Data
@st.cache_data(ttl=3600)
def load_qbo_data():
    res = supabase.table("qbo_transactions").select("*").execute()
    return pd.DataFrame(res.data)

df = load_qbo_data()
df["amount"] = pd.to_numeric(df["amount"], errors="coerce")
df["date"] = pd.to_datetime(df["date"], errors="coerce")

st.title("QBO Transaction Summary")

# --- Loan Performance ---
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

pivot_display = pivot.copy()
for col in ["Invoice", "Payment", "balance"]:
    if col in pivot_display.columns:
        pivot_display[col] = pivot_display[col].map("${:,.2f}".format)

st.subheader("📊 Loan Performance by Customer")
st.dataframe(pivot_display.sort_values("balance", ascending=False), use_container_width=True)

# --- Charts ---
top_balances = pivot.sort_values("balance", ascending=False).head(15)

bar_chart = alt.Chart(top_balances).mark_bar(color="#e45756").encode(
    x=alt.X("balance:Q", title="Outstanding Balance ($)", axis=alt.Axis(format="$,.0f")),
    y=alt.Y("name:N", sort="-x", title="Customer"),
    tooltip=["name", alt.Tooltip("balance:Q", format="$,.2f")]
).properties(width=800, height=400)

st.altair_chart(bar_chart, use_container_width=True)

melted = pivot.melt(id_vars="name", value_vars=["Invoice", "Payment"], var_name="Type", value_name="Amount")
melted = melted[melted["name"].isin(top_balances["name"])]

bar_compare = alt.Chart(melted).mark_bar().encode(
    x=alt.X("Amount:Q", axis=alt.Axis(format="$,.0f")),
    y=alt.Y("name:N", sort=alt.EncodingSortField(field="Amount", op="max", order="descending")),
    color=alt.Color("Type:N", scale=alt.Scale(scheme="category10")),
    tooltip=["name", "Type", alt.Tooltip("Amount:Q", format="$,.2f")]
).properties(width=800, height=400)

st.altair_chart(bar_compare, use_container_width=True)

# --- Monthly Payments Trend ---
payments_df = df[df["transaction_type"] == "Payment"].copy()
payments_df["amount"] = payments_df["amount"].abs()
payments_df["month"] = payments_df["date"].dt.to_period("M").astype(str)

monthly_payments = payments_df.groupby("month")["amount"].sum().reset_index()

st.subheader("📈 Monthly Payments Received")
payment_trend = alt.Chart(monthly_payments).mark_line(point=True).encode(
    x=alt.X("month:T", title="Month"),
    y=alt.Y("amount:Q", title="Total Payments ($)", axis=alt.Axis(format="$,.0f")),
    tooltip=["month", alt.Tooltip("amount", format="$,.2f")]
).properties(width=800, height=300)

st.altair_chart(payment_trend, use_container_width=True)

# --- Payment Averages ---
first_payment_date = payments_df["date"].min().date()
today = pd.Timestamp.today().date()
working_days = busday_count(first_payment_date, today)

total_payments = payments_df["amount"].sum()
avg_per_day = total_payments / working_days if working_days else 0
avg_per_week = avg_per_day * 5
avg_per_month = avg_per_day * 21

st.subheader("📊 Average Payment Inflow")
st.markdown(f"""
- **Total Payments:** ${total_payments:,.2f}  
- **Working Days:** {working_days}  
- **Avg/Day:** ${avg_per_day:,.2f}  
- **Avg/Week:** ${avg_per_week:,.2f}  
- **Avg/Month:** ${avg_per_month:,.2f}
""")

# --- Expenses ---
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

# --- Monthly Trend: Payments vs Expenses ---
expenses_df["month"] = expenses_df["date"].dt.to_period("M").astype(str)

monthly_expenses = expenses_df.groupby("month")["amount"].sum().reset_index()
monthly_expenses["type"] = "Expense"
monthly_payments["type"] = "Payment"

combined = pd.concat([monthly_payments.rename(columns={"amount": "amount"}), monthly_expenses])
combined["month"] = pd.to_datetime(combined["month"])

trend_chart = alt.Chart(combined).mark_line(point=True).encode(
    x="month:T",
    y=alt.Y("amount:Q", title="Amount ($)", axis=alt.Axis(format="$,.0f")),
    color="type:N",
    tooltip=["type", "month", alt.Tooltip("amount", format="$,.2f")]
).properties(width=850, height=300)

st.subheader("📊 Monthly Payments vs Expenses")
st.altair_chart(trend_chart, use_container_width=True)
