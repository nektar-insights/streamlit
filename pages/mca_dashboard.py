# pages/mca_dashboard.py
import streamlit as st
import pandas as pd
from supabase import create_client

# Supabase connection
url = st.secrets["supabase"]["url"]
key = st.secrets["supabase"]["service_role"]
supabase = create_client(url, key)

@st.cache_data(ttl=3600)
def load_mca_deals():
    res = supabase.table("mca_deals").select("*").execute()
    return pd.DataFrame(res.data)

df = load_mca_deals()

# Data prep
df["funding_date"] = pd.to_datetime(df["funding_date"], errors="coerce")
df["purchase_price"] = pd.to_numeric(df["purchase_price"], errors="coerce")
df["receivables_amount"] = pd.to_numeric(df["receivables_amount"], errors="coerce")
df["current_balance"] = pd.to_numeric(df["current_balance"], errors="coerce")
df["past_due_amount"] = pd.to_numeric(df["past_due_amount"], errors="coerce")
df["funding_date"] = pd.to_datetime(df["funding_date"], errors="coerce").dt.date
df["past_due_amount"] = pd.to_numeric(df["past_due_amount"], errors="coerce")
df["current_balance"] = pd.to_numeric(df["current_balance"], errors="coerce")
df["purchase_price"] = pd.to_numeric(df["purchase_price"], errors="coerce")
df["principal_amount"] = pd.to_numeric(df["principal_amount"], errors="coerce")
df["rtr_balance"] = pd.to_numeric(df["rtr_balance"], errors="coerce")

df["past_due_pct"] = df.apply(
    lambda row: row["past_due_amount"] / row["current_balance"]
    if row["current_balance"] and row["past_due_amount"] else 0,
    axis=1
)

loan_tape = df[[
    "deal_number", "dba", "funding_date", "status_category", "past_due_amount", "past_due_pct",
    "performance_ratio", "rtr_balance", "performance_details"
]].copy()

loan_tape.rename(columns={
    "deal_number": "Loan ID",
    "dba": "Deal",
    "funding_date": "Funding Date",
    "status_category": "Status",
    "past_due_amount": "Past Due ($)",
    "past_due_pct": "Past Due Amount",
    "performance_ratio": "Performance Ratio",
    "rtr_balance": "Remaining to Recover ($)",
    "performance_details": "Performance Notes"
}, inplace=True)

loan_tape["Past Due Amount"] = loan_tape["Past Due Amount"].apply(lambda x: f"{x:.1%}")
loan_tape["Past Due ($)"] = loan_tape["Past Due ($)"].apply(lambda x: f"${x:,.0f}")
loan_tape["Remaining to Recover ($)"] = loan_tape["Remaining to Recover ($)"].apply(lambda x: f"${x:,.0f}")
st.subheader("📋 Loan Tape")
st.dataframe(loan_tape, use_container_width=True)

# Filters
min_date = df["funding_date"].min()
max_date = df["funding_date"].max()
start_date, end_date = st.date_input("Filter by Funding Date", [min_date, max_date], min_value=min_date, max_value=max_date)
df = df[(df["funding_date"] >= pd.to_datetime(start_date)) & (df["funding_date"] <= pd.to_datetime(end_date))]

status_filter = st.multiselect("Status Category", df["status_category"].dropna().unique(), default=list(df["status_category"].dropna().unique()))
df = df[df["status_category"].isin(status_filter)]

# Metrics
total_deals = len(df)
total_funded = df["purchase_price"].sum()
total_receivables = df["receivables_amount"].sum()
total_past_due = df["past_due_amount"].sum()

st.title("MCA Deals Dashboard")
col1, col2, col3 = st.columns(3)
col1.metric("Total Deals", total_deals)
col2.metric("Total Funded", f"${total_funded:,.0f}")
col3.metric("Total Receivables", f"${total_receivables:,.0f}")

st.metric("Total Past Due", f"${total_past_due:,.0f}")

st.subheader("Preview")
st.dataframe(df[["deal_id", "dba", "funding_date", "status_category", "purchase_price", "receivables_amount", "past_due_amount"]], use_container_width=True)

status_chart = (
    df["status_category"]
    .value_counts(normalize=True)
    .reset_index()
    .rename(columns={"index": "Status", "status_category": "Share"})
)

bar = alt.Chart(status_chart).mark_bar().encode(
    x=alt.X("Status:N", title="Status Category"),
    y=alt.Y("Share:Q", title="Percent of Deals", axis=alt.Axis(format=".0%")),
    tooltip=["Status", alt.Tooltip("Share:Q", format=".2%")]
).properties(
    width=700,
    height=350,
    title="📊 Distribution of Deal Status"
)

st.altair_chart(bar, use_container_width=True)

not_current = df[df["status_category"] != "Current"].copy()
not_current["at_risk_pct"] = not_current["past_due_amount"] / not_current["current_balance"]

risk_chart = alt.Chart(not_current).mark_bar().encode(
    x=alt.X("dba:N", title="Deal", sort="-y"),
    y=alt.Y("at_risk_pct:Q", title="% of Balance at Risk", axis=alt.Axis(format=".0%")),
    tooltip=[
        alt.Tooltip("dba:N", title="Deal"),
        alt.Tooltip("past_due_amount:Q", title="Past Due ($)", format="$,.0f"),
        alt.Tooltip("current_balance:Q", title="Current Balance ($)", format="$,.0f"),
        alt.Tooltip("at_risk_pct:Q", title="% at Risk", format=".2%")
    ]
).properties(width=850, height=400, title="🚨 % of Balance at Risk (Non-Current Deals)")

st.altair_chart(risk_chart, use_container_width=True)

df["risk_score"] = (
    df["past_due_amount"] / df["current_balance"].clip(lower=1) * 0.5 +  # Past Due % = high weight
    df["rtr_balance"] / df["principal_amount"].clip(lower=1) * 0.3 +      # Unrecovered capital
    (df["funding_date"] < pd.Timestamp.today() - pd.Timedelta(days=180)).astype(int) * 0.2  # Aged deal
)
