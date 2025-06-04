# pages/mca_dashboard.py

import streamlit as st
import pandas as pd
import altair as alt
from supabase import create_client

# ----------------------------
# Supabase connection
# ----------------------------
url = st.secrets["supabase"]["url"]
key = st.secrets["supabase"]["service_role"]
supabase = create_client(url, key)

# ----------------------------
# Load and prepare data
# ----------------------------
@st.cache_data(ttl=3600)
def load_mca_deals():
    res = supabase.table("mca_deals").select("*").execute()
    return pd.DataFrame(res.data)

df = load_mca_deals()

# Filter out Canceled deals
df = df[df["status_category"] != "Canceled"]

# Convert data types
df["funding_date"] = pd.to_datetime(df["funding_date"], errors="coerce").dt.date
df["purchase_price"] = pd.to_numeric(df["purchase_price"], errors="coerce")
df["receivables_amount"] = pd.to_numeric(df["receivables_amount"], errors="coerce")
df["current_balance"] = pd.to_numeric(df["current_balance"], errors="coerce")
df["past_due_amount"] = pd.to_numeric(df["past_due_amount"], errors="coerce")
df["principal_amount"] = pd.to_numeric(df["principal_amount"], errors="coerce")
df["rtr_balance"] = pd.to_numeric(df["rtr_balance"], errors="coerce")

# Set past_due_amount to 0 for Matured deals
df.loc[df["status_category"] == "Matured", "past_due_amount"] = 0

# Add derived field for percent past due
df["past_due_pct"] = df.apply(
    lambda row: row["past_due_amount"] / row["current_balance"]
    if row["current_balance"] and row["past_due_amount"] else 0,
    axis=1
)

# ----------------------------
# Filters
# ----------------------------
min_date = df["funding_date"].min()
max_date = df["funding_date"].max()

start_date, end_date = st.date_input("Filter by Funding Date", [min_date, max_date], min_value=min_date, max_value=max_date)
df = df[(df["funding_date"] >= start_date) & (df["funding_date"] <= end_date)]

# Filter out Canceled deals
status_category_filter = st.multiselect("status_category Category", df["status_category"].dropna().unique(), default=list(df["status_category"].dropna().unique()))
df = df[df["status_category"].isin(status_category_filter)]

# ----------------------------
# Metrics Summary
# ----------------------------
st.title("MCA Deals Dashboard")

total_deals = len(df)
total_funded = df["purchase_price"].sum()
total_receivables = df["receivables_amount"].sum()
total_past_due = df["past_due_amount"].sum()

col1, col2, col3 = st.columns(3)
col1.metric("Total Deals", total_deals)
col2.metric("Total Funded", f"${total_funded:,.0f}")
col3.metric("Total Receivables", f"${total_receivables:,.0f}")

st.metric("Total Past Due", f"${total_past_due:,.0f}")

# ----------------------------
# Loan Tape Display
# ----------------------------
loan_tape = df[[
    "deal_number", "dba", "funding_date", "status_category",
    "past_due_amount", "past_due_pct", "performance_ratio",
    "rtr_balance", "performance_details"
]].copy()

loan_tape.rename(columns={
    "deal_number": "Loan ID",
    "dba": "Deal",
    "funding_date": "Funding Date",
    "status_category": "Status Category",
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

# ----------------------------
# Distribution of Deal status_category (Bar Chart)
# ----------------------------

# Step 1: calculate normalized value counts safely
status_category_counts = df["status_category"].fillna("Unknown").value_counts(normalize=True)

# Step 2: convert to DataFrame
status_category_chart = pd.DataFrame({
    "status_category": status_category_counts.index.astype(str),
    "Share": status_category_counts.values
})

# Step 3: ensure clean types
status_category_chart["Share"] = pd.to_numeric(status_category_chart["Share"], errors="coerce")

# Step 4: build chart
bar = alt.Chart(status_category_chart).mark_bar().encode(
    x=alt.X("status_category:N", title="Status Category", sort=alt.EncodingSortField(field="Share", order="ascending")),
    y=alt.Y("Share:Q", title="Percent of Deals", axis=alt.Axis(format=".0%")),
    tooltip=[
        alt.Tooltip("status_category", title="Status"),
        alt.Tooltip("Share:Q", title="Share", format=".2%")
    ]
).properties(
    width=700,
    height=350,
    title="📊 Distribution of Deal Status"
)

st.altair_chart(bar, use_container_width=True)

# ----------------------------
# Risk Chart: % of Balance at Risk
# ----------------------------
not_current = df[df["status_category"] != "Current"].copy()
not_current["at_risk_pct"] = not_current["past_due_amount"] / not_current["current_balance"]
not_current = not_current[not_current["at_risk_pct"] > 0]

risk_chart = alt.Chart(not_current).mark_bar().encode(
    x=alt.X("dba:N", title="Deal", sort="-y"),
    y=alt.Y("at_risk_pct:Q", title="% of Balance at Risk", axis=alt.Axis(format=".0%")),
    tooltip=[
        alt.Tooltip("dba:N", title="Deal"),
        alt.Tooltip("past_due_amount:Q", title="Past Due ($)", format="$,.0f"),
        alt.Tooltip("current_balance:Q", title="Current Balance ($)", format="$,.0f"),
        alt.Tooltip("at_risk_pct:Q", title="% at Risk", format=".2%")
    ]
).properties(
    width=850,
    height=400,
    title="🚨 % of Balance at Risk (Non-Current Deals)"
)

st.altair_chart(risk_chart, use_container_width=True)

# ----------------------------
# Risk Scoring
# ----------------------------
st.subheader("🔥 Top 10 Highest Risk Deals* (Excludes New and Performing Loans)")

# Calculate days since funding
df["days_since_funding"] = (pd.Timestamp.today() - pd.to_datetime(df["funding_date"])).dt.days

# Define risk pool: exclude new deals, $0 past due, and current status
risk_df = df[
    (df["days_since_funding"] > 30) &
    (df["past_due_amount"] > df["current_balance"] * 0.01) &  # must be >1% past due
    (df["status_category"] != "Current")
].copy()

# Calculate percent past due
risk_df["past_due_pct"] = risk_df["past_due_amount"] / risk_df["current_balance"].clip(lower=1)

# Normalize age for relative weight (to avoid unfairly penalizing very old deals)
max_days = risk_df["days_since_funding"].max()
risk_df["age_weight"] = risk_df["days_since_funding"] / max_days

# Final weighted risk score
risk_df["risk_score"] = risk_df["past_due_pct"] * 0.7 + risk_df["age_weight"] * 0.3

# Top 10 by risk score
top_risk = risk_df.sort_values("risk_score", ascending=False).head(10).copy()

# Format output table
# Remove string formatting from earlier step
top_risk_display = top_risk[[
    "deal_number", "dba", "status_category", "funding_date", "risk_score",
    "past_due_amount", "current_balance"
]].rename(columns={
    "deal_number": "Loan ID",
    "dba": "Deal",
    "status_category": "Status",
    "funding_date": "Funded",
    "risk_score": "Risk Score",
    "past_due_amount": "Past Due ($)",
    "current_balance": "Current Balance ($)"
})

# Highlight Risk Score with styling, and format values
styled_df = top_risk_display.style.background_gradient(
    subset=["Risk Score"], cmap="Reds", axis=None  # axis=None avoids vertical bar
).format({
    "Past Due ($)": "${:,.0f}",
    "Current Balance ($)": "${:,.0f}",
    "Risk Score": "{:.2f}"
})

st.dataframe(styled_df, use_container_width=True)

bar_chart = alt.Chart(top_risk).mark_bar().encode(
    x=alt.X("dba:N", title="Deal", sort="-y"),
    y=alt.Y("risk_score:Q", title="Risk Score"),
    color=alt.Color("risk_score:Q", scale=alt.Scale(scheme="orangered")),
    tooltip=["deal_number", "status_category", "funding_date", "past_due_amount", "risk_score"]
).properties(
    width=700,
    height=400,
    title="🔥 Top 10 Risk Scores"
)

st.altair_chart(bar_chart, use_container_width=True)

from streamlit.components.v1 import html

# Format again after color styling
styled_df = top_risk_display.style.background_gradient(
    subset=["Risk Score"], cmap="Reds"
).format({
    "Past Due ($)": "${:,.0f}",
    "Current Balance ($)": "${:,.0f}",
    "Remaining to Recover ($)": "${:,.0f}",
    "Risk Score": "{:.2f}"
})

st.dataframe(styled_df, use_container_width=True)

# ----------------------------
# Scatter Plot
# ----------------------------
scatter = alt.Chart(risk_df).mark_circle().encode(
    x=alt.X("past_due_pct:Q", title="% Past Due", axis=alt.Axis(format=".0%")),
    y=alt.Y("days_since_funding:Q", title="Days Since Funding"),
    size=alt.Size("risk_score:Q", title="Risk Score"),
    color=alt.Color("risk_score:Q", scale=alt.Scale(scheme="orangered"), title="Risk Score"),
    tooltip=["dba", "status_category", "funding_date", "risk_score", "past_due_pct", "days_since_funding"]
).properties(
    width=700,
    height=400,
    title="📉 Risk Score by Past Due % and Deal Age"
)

st.altair_chart(scatter, use_container_width=True)

st.markdown("""
* Risk Score is calculated using:
- **70% weight** on the percentage of the loan that is past due,
- **30% weight** on how long the loan has been outstanding.
New deals (< 30 days old), those with low delinquency (<1%), or with status 'Current' are excluded.
""")

#-------
csv = loan_tape.to_csv(index=False).encode("utf-8")
st.download_button(
    label="📄 Download Loan Tape as CSV",
    data=csv,
    file_name="loan_tape.csv",
    mime="text/csv"
)

csv_risk = top_risk_display.to_csv(index=False).encode("utf-8")
st.download_button(
    label="📄 Download Top 10 Risk Deals as CSV",
    data=csv_risk,
    file_name="top_risk_deals.csv",
    mime="text/csv"
)
