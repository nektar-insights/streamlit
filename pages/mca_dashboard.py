# pages/mca_dashboard.py

import streamlit as st
import pandas as pd
import altair as alt
from supabase import create_client
from scripts.combine_hubspot_mca import combine_deals

PERFORMANCE_GRADIENT = ["#e8f5e8", "#34a853", "#1e7e34"]  # Light green → Mid → Dark green
RISK_GRADIENT = ["#fef9e7", "#f39c12", "#dc3545"]  # Light yellow → Orange → Red

# ----------------------------
# Supabase connection
# ----------------------------
url = st.secrets["supabase"]["url"]
key = st.secrets["supabase"]["service_role"]
supabase = create_client(url, key)

# ----------------------------
# Load and prepare single dataframe
# ----------------------------
@st.cache_data(ttl=3600)
def load_mca_deals():
    res = supabase.table("mca_deals").select("*").execute()
    return pd.DataFrame(res.data)

# Use combined dataframe as the single source of truth
df = combine_deals()

# Filter out Canceled deals completely
df = df[df["status_category"] != "Canceled"]

# ----------------------------
# Data type conversions and basic calculations
# ----------------------------
df["funding_date"] = pd.to_datetime(df["funding_date"], errors="coerce").dt.date
for col in ["purchase_price", "receivables_amount", "current_balance", "past_due_amount", 
            "principal_amount", "rtr_balance", "amount_hubspot", "total_funded_amount"]:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# Set past_due_amount to 0 for Matured deals
df.loc[df["status_category"] == "Matured", "past_due_amount"] = 0

# Calculate derived fields
df.rename(columns={"amount_hubspot": "csl_participation"}, inplace=True)
df["past_due_pct"] = df.apply(
    lambda row: row["past_due_amount"] / row["current_balance"]
    if pd.notna(row["past_due_amount"]) and pd.notna(row["current_balance"]) and row["current_balance"] > 0
    else 0,
    axis=1
)

# CSL-specific calculations
df["participation_ratio"] = df["csl_participation"] / df["total_funded_amount"].replace(0, pd.NA)
df["csl_past_due"] = df["participation_ratio"] * df["past_due_amount"]
df["principal_remaining_est"] = df.apply(
    lambda row: row["principal_amount"] if pd.isna(row["payments_made"])
    else max(row["principal_amount"] - row["payments_made"], 0),
    axis=1
)
df["csl_principal_at_risk"] = df["participation_ratio"] * df["principal_remaining_est"]

# Set CSL past due to 0 for Current deals
df.loc[df["status_category"] == "Current", "csl_past_due"] = 0

# Commission calculations
df["commission_rate"] = pd.to_numeric(df["commission"], errors="coerce")

# Risk scoring calculations
df["days_since_funding"] = (pd.Timestamp.today() - pd.to_datetime(df["funding_date"])).dt.days

# ----------------------------
# Filters
# ----------------------------
st.title("MCA Deals Dashboard")

min_date = df["funding_date"].min()
max_date = df["funding_date"].max()
start_date, end_date = st.date_input(
    "Filter by Funding Date",
    [min_date, max_date],
    min_value=min_date,
    max_value=max_date
)
df = df[(df["funding_date"] >= start_date) & (df["funding_date"] <= end_date)]

status_options = ["All"] + list(df["status_category"].dropna().unique())
status_category_filter = st.radio("Status Category", status_options, index=0)
if status_category_filter != "All":
    df = df[df["status_category"] == status_category_filter]

# ----------------------------
# Calculate all metrics
# ----------------------------

# Portfolio metrics
total_deals = len(df)
total_matured = (df["status_category"] == "Matured").sum()
total_current = (df["status_category"] == "Current").sum()
total_non_current = (df["status_category"] == "Not Current").sum()
outstanding_total = total_current + total_non_current
pct_current = total_current / outstanding_total if outstanding_total > 0 else 0
pct_non_current = total_non_current / outstanding_total if outstanding_total > 0 else 0

# CSL investment metrics
csl_capital_deployed = df["csl_participation"].sum()
total_csl_past_due = df["csl_past_due"].sum()
at_risk = df[df["status_category"] == "Not Current"]
total_csl_at_risk = (at_risk["participation_ratio"] * at_risk["principal_remaining_est"]).sum()

# Commission metrics
average_commission_pct = df["commission_rate"].mean()
total_commission_paid = (df["csl_participation"] * df["commission_rate"]).sum()
average_commission_on_loan = total_commission_paid / df["csl_participation"].sum() if df["csl_participation"].sum() > 0 else 0

# Risk analysis dataframes
not_current_df = df[(df["status_category"] != "Current") & (df["status_category"] != "Matured")].copy()
not_current_df["at_risk_pct"] = not_current_df["past_due_amount"] / not_current_df["current_balance"]
not_current_df = not_current_df[not_current_df["at_risk_pct"] > 0]

# Risk scoring for top 10
risk_df = df[
    (df["days_since_funding"] > 30) &
    (df["past_due_amount"] > df["current_balance"] * 0.01) &
    (df["status_category"] != "Current") &
    (df["status_category"] != "Matured")
].copy()

if len(risk_df) > 0:
    risk_df["past_due_pct_calc"] = risk_df["past_due_amount"] / risk_df["current_balance"].clip(lower=1)
    max_days = risk_df["days_since_funding"].max()
    if max_days > 0:
        risk_df["age_weight"] = risk_df["days_since_funding"] / max_days
    else:
        risk_df["age_weight"] = 0
    risk_df["risk_score"] = risk_df["past_due_pct_calc"] * 0.7 + risk_df["age_weight"] * 0.3
    top_risk = risk_df.sort_values("risk_score", ascending=False).head(10).copy()

# ----------------------------
# Display metrics sections
# ----------------------------

# Portfolio Summary
st.subheader("📊 CSL Portfolio Summary")
col1, col2, col3 = st.columns(3)
col1.metric("Total Deals", total_deals)
col2.metric("Matured Deals", total_matured)
col3.metric("Current Deals", total_current)

col4, col5, col6 = st.columns(3)
col4.metric("Not Current Deals", total_non_current)
col5.metric("Pct. Outstanding Deals Current", f"{pct_current:.1%}")
col6.metric("Pct. Outstanding Deals Not Current", f"{pct_non_current:.1%}")

# CSL Investment Overview
st.subheader("💰 CSL Investment Overview")
col7, col8, col9 = st.columns(3)
col7.metric("Capital Deployed", f"${csl_capital_deployed:,.0f}")
col8.metric("Past Due Exposure", f"${total_csl_past_due:,.0f}")
col9.metric("Outstanding CSL Principal", f"${total_csl_at_risk:,.0f}")

# CSL Commission Summary
st.subheader("💼 CSL Commission Summary")
col10, col11, col12 = st.columns(3)
col10.metric("Avg. Commission Rate", f"{average_commission_pct:.2%}")
col11.metric("Avg. Applied to Participation", f"{average_commission_on_loan:.2%}")
col12.metric("Total Commission Paid", f"${total_commission_paid:,.0f}")

# ----------------------------
# Tables and visualizations
# ----------------------------

# Loan Tape Display
st.subheader("📋 Loan Tape")
loan_tape = df[[
    "deal_number", "dba", "funding_date", "status_category",
    "csl_past_due", "past_due_pct", "performance_ratio",
    "rtr_balance", "performance_details"
]].copy()

loan_tape.rename(columns={
    "deal_number": "Loan ID",
    "dba": "Deal",
    "funding_date": "Funding Date",
    "status_category": "Status Category",
    "csl_past_due": "CSL Past Due ($)",
    "past_due_pct": "Past Due %",
    "performance_ratio": "Performance Ratio",
    "rtr_balance": "Remaining to Recover ($)",
    "performance_details": "Performance Notes"
}, inplace=True)

loan_tape["Past Due %"] = pd.to_numeric(loan_tape["Past Due %"], errors="coerce").fillna(0)*100
loan_tape["CSL Past Due ($)"] = pd.to_numeric(loan_tape["CSL Past Due ($)"], errors="coerce").fillna(0)
loan_tape["Remaining to Recover ($)"] = pd.to_numeric(loan_tape["Remaining to Recover ($)"], errors="coerce").fillna(0)
loan_tape["Performance Ratio"] = pd.to_numeric(loan_tape["Performance Ratio"], errors="coerce").fillna(0)

st.dataframe(
    loan_tape,
    use_container_width=True,
    column_config={
        "Past Due %": st.column_config.NumberColumn("Past Due %", format="%.2f"),
        "CSL Past Due ($)": st.column_config.NumberColumn("CSL Past Due ($)", format="$%.0f"),
        "Remaining to Recover ($)": st.column_config.NumberColumn("Remaining to Recover ($)", format="$%.0f"),
        "Performance Ratio": st.column_config.NumberColumn("Performance Ratio", format="%.2f"),
    }
)

# Distribution of Deal Status (Bar Chart)
status_category_counts = df["status_category"].fillna("Unknown").value_counts(normalize=True)

# Calculate unpaid CSL Principal by status category
status_csl_principal = df.groupby(df["status_category"].fillna("Unknown"))["csl_principal_at_risk"].sum()

status_category_chart = pd.DataFrame({
    "status_category": status_category_counts.index.astype(str),
    "Share": status_category_counts.values,
    "unpaid_csl_principal": status_csl_principal.reindex(status_category_counts.index).fillna(0).values
})

bar = alt.Chart(status_category_chart).mark_bar().encode(
    x=alt.X(
        "status_category:N",
        title="Status Category",
        sort=alt.EncodingSortField(field="Share", order="descending"),
        axis=alt.Axis(labelAngle=-90)
    ),
    y=alt.Y("Share:Q", title="Percent of Deals", axis=alt.Axis(format=".0%")),
    color=alt.Color(
        "Share:Q",
        scale=alt.Scale(range=PERFORMANCE_GRADIENT),
        legend=None
    ),
    tooltip=[
        alt.Tooltip("status_category", title="Status"),
        alt.Tooltip("Share:Q", title="Share", format=".2%"),
        alt.Tooltip("unpaid_csl_principal:Q", title="Unpaid CSL Principal (Est.)", format="$,.0f")
    ]
).properties(
    width=700,
    height=350,
    title="📊 Distribution of Deal Status"
)

st.altair_chart(bar, use_container_width=True)

# Risk Chart: Percentage of Balance at Risk
if len(not_current_df) > 0:
    risk_chart = alt.Chart(not_current_df).mark_bar().encode(
        x=alt.X(
            "deal_number:N",
            title="Loan ID",
            sort="-y",
            axis=alt.Axis(labelAngle=-90)
        ),
        y=alt.Y("at_risk_pct:Q", title="Pct. of Balance at Risk", axis=alt.Axis(format=".0%")),
        color=alt.Color(
            "at_risk_pct:Q",
            scale=alt.Scale(range=RISK_GRADIENT),
            legend=alt.Legend(title="Risk Level")
        ),
        tooltip=[
            alt.Tooltip("deal_number:N", title="Loan ID"),
            alt.Tooltip("dba:N", title="Deal Name"),
            alt.Tooltip("past_due_amount:Q", title="Past Due ($)", format="$,.0f"),
            alt.Tooltip("current_balance:Q", title="Current Balance ($)", format="$,.0f"),
            alt.Tooltip("at_risk_pct:Q", title="% at Risk", format=".2%")
        ]
    ).properties(
        width=850,
        height=400,
        title="🚨 Percentage of Balance at Risk (Non-Current, Non-Matured Deals)"
    )

    st.altair_chart(risk_chart, use_container_width=True)
else:
    st.info("No non-current, non-matured deals with past due amounts to display.")

# Top 10 Highest Risk Deals
st.subheader("🔥 Top 10 Highest Risk Deals")

if len(risk_df) > 0:
    # Risk score bar chart with Loan ID on x-axis
    bar_chart = alt.Chart(top_risk).mark_bar().encode(
        x=alt.X(
            "deal_number:N",
            title="Loan ID",
            sort="-y",
            axis=alt.Axis(labelAngle=-90)
        ),
        y=alt.Y("risk_score:Q", title="Risk Score"),
        color=alt.Color(
            "risk_score:Q",
            scale=alt.Scale(range=RISK_GRADIENT),
            legend=alt.Legend(title="Risk Score")
        ),
        tooltip=[
            alt.Tooltip("deal_number:N", title="Loan ID"),
            alt.Tooltip("dba:N", title="Deal Name"),
            alt.Tooltip("status_category:N", title="Status Category"),
            alt.Tooltip("funding_date:T", title="Funding Date"),
            alt.Tooltip("past_due_amount:Q", title="Past Due Amount", format="$,.0f"),
            alt.Tooltip("current_balance:Q", title="Current Balance", format="$,.0f"),
            alt.Tooltip("risk_score:Q", title="Risk Score", format=".3f")
        ]
    ).properties(
        width=700,
        height=400,
        title="(Excludes New and Performing Loans)"
    )

    st.altair_chart(bar_chart, use_container_width=True)

    # Top 10 Risk table with CSL Past Due
    top_risk_display = top_risk[[
        "deal_number", "dba", "status_category", "funding_date", "risk_score",
        "csl_past_due", "current_balance"
    ]].copy()

    top_risk_display.rename(columns={
        "deal_number": "Loan ID",
        "dba": "Deal",
        "status_category": "Status",
        "funding_date": "Funded",
        "risk_score": "Risk Score",
        "csl_past_due": "CSL Past Due ($)",
        "current_balance": "Current Balance ($)"
    }, inplace=True)

    # Clean up numeric data
    top_risk_display["Risk Score"] = pd.to_numeric(top_risk_display["Risk Score"], errors="coerce").fillna(0)
    top_risk_display["CSL Past Due ($)"] = pd.to_numeric(top_risk_display["CSL Past Due ($)"], errors="coerce").fillna(0)
    top_risk_display["Current Balance ($)"] = pd.to_numeric(top_risk_display["Current Balance ($)"], errors="coerce").fillna(0)

    st.dataframe(
        top_risk_display,
        use_container_width=True,
        column_config={
            "CSL Past Due ($)": st.column_config.NumberColumn(
                "CSL Past Due ($)",
                format="$%.0f",
                help="CSL portion of past due amount"
            ),
            "Current Balance ($)": st.column_config.NumberColumn(
                "Current Balance ($)",
                format="$%.0f", 
                help="Current outstanding balance"
            ),
            "Risk Score": st.column_config.NumberColumn(
                "Risk Score",
                format="%.3f",
                help="Calculated risk score"
            ),
        }
    )

    # Scatter Plot
    scatter = alt.Chart(risk_df).mark_circle(size=80).encode(
        x=alt.X("past_due_pct_calc:Q", title="% Past Due", axis=alt.Axis(format=".0%")),
        y=alt.Y("days_since_funding:Q", title="Days Since Funding"),
        size=alt.Size("risk_score:Q", title="Risk Score", scale=alt.Scale(range=[50, 400])),
        color=alt.Color(
            "risk_score:Q",
            scale=alt.Scale(range=RISK_GRADIENT),
            title="Risk Score"
        ),
        tooltip=[
            alt.Tooltip("deal_number:N", title="Loan ID"),
            alt.Tooltip("dba:N", title="Deal Name"),
            alt.Tooltip("status_category:N", title="Status"),
            alt.Tooltip("funding_date:T", title="Funded"),
            alt.Tooltip("risk_score:Q", title="Risk Score", format=".2f"),
            alt.Tooltip("past_due_pct_calc:Q", title="% Past Due", format=".2%"),
            alt.Tooltip("days_since_funding:Q", title="Days Since Funding")
        ]
    ).properties(
        width=700,
        height=400,
        title="📉 Risk Score by Past Due % and Deal Age"
    )

    threshold_x = alt.Chart(pd.DataFrame({"x": [0.10]})).mark_rule(
        strokeDash=[4, 4], color="gray"
    ).encode(x="x:Q")

    threshold_y = alt.Chart(pd.DataFrame({"y": [90]})).mark_rule(
        strokeDash=[4, 4], color="gray"
    ).encode(y="y:Q")

    st.altair_chart(
        scatter + threshold_x + threshold_y,
        use_container_width=True
    )
else:
    st.info("No deals meet the risk criteria for analysis.")

st.markdown("""
#### 🧠 Understanding the Risk Score

The **Risk Score** ranges between **0.00 and 1.00** and blends two key factors:
- **70% weight**: The **percentage of the balance that is past due**
- **30% weight**: The **age of the deal** (older deals score higher)

| Risk Score | Interpretation            |
|------------|---------------------------|
| 0.00–0.19  | Very Low Risk             |
| 0.20–0.49  | Low to Moderate Risk      |
| 0.50–0.74  | Elevated Risk             |
| 0.75–1.00  | High Risk of Loss         |

🔎 *Example*: A score of **0.80** implies the deal is either **very delinquent**, **very old**, or both.
New deals (< 30 days old), those with low delinquency (<1%), or with status 'Current' are excluded.
""")

# Download buttons
csv = loan_tape.to_csv(index=False).encode("utf-8")
st.download_button(
    label="📄 Download Loan Tape as CSV",
    data=csv,
    file_name="loan_tape.csv",
    mime="text/csv"
)

if len(risk_df) > 0:
    csv_risk = top_risk_display.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="📄 Download Top 10 Risk Deals as CSV",
        data=csv_risk,
        file_name="top_risk_deals.csv",
        mime="text/csv"
    )
