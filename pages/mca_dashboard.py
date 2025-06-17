# pages/mca_dashboard.py
from utils.imports import *
from scripts.combine_hubspot_mca import combine_deals
from scripts.get_naics_sector_risk import get_naics_sector_risk

# ----------------------------
# Define risk gradient color scheme (dark red to light red with 5 shades)
# ----------------------------
RISK_GRADIENT = ["#FFB6C1", "#F08080", "#DC143C", "#8B0000", "#B22222"]

# ----------------------------
# Supabase connection
# ----------------------------
supabase = get_supabase_client()

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

# Set CSL principal at risk to 0 for Matured deals (they're closed, no remaining principal)
df.loc[df["status_category"] == "Matured", "csl_principal_at_risk"] = 0

# Set CSL past due to 0 for Current deals
df.loc[df["status_category"] == "Current", "csl_past_due"] = 0

# Commission calculations
df["commission_rate"] = pd.to_numeric(df["commission"], errors="coerce")

# Risk scoring calculations
df["days_since_funding"] = (pd.Timestamp.today() - pd.to_datetime(df["funding_date"])).dt.days

# ----------------------------
# Industry/NAICS Processing
# ----------------------------
# Load NAICS sector risk data
naics_risk_df = get_naics_sector_risk()

# Extract 2-digit sector code from industry (full NAICS code)
if 'industry' in df.columns:
    df['sector_code'] = df['industry'].astype(str).str[:2].str.zfill(2)
    
    # Consolidate manufacturing sectors (31, 32, 33 -> Manufacturing)
    df['sector_code_consolidated'] = df['sector_code'].copy()
    df.loc[df['sector_code'].isin(['31', '32', '33']), 'sector_code_consolidated'] = 'Manufacturing'
    
    # Join with NAICS sector risk data
    if not naics_risk_df.empty:
        # Create consolidated risk data for manufacturing
        manufacturing_risk = naics_risk_df[naics_risk_df['sector_code'].isin(['31', '32', '33'])].iloc[0:1].copy()
        if not manufacturing_risk.empty:
            manufacturing_risk['sector_code'] = 'Manufacturing'
            manufacturing_risk['sector_name'] = 'Manufacturing (31-33)'
            # Use average risk score for manufacturing
            avg_risk_score = naics_risk_df[naics_risk_df['sector_code'].isin(['31', '32', '33'])]['risk_score'].mean()
            manufacturing_risk['risk_score'] = avg_risk_score
            manufacturing_risk['risk_profile'] = 'Medium'  # Default or calculate based on avg
            
            # Add manufacturing row to naics_risk_df
            naics_risk_consolidated = pd.concat([naics_risk_df, manufacturing_risk], ignore_index=True)
        else:
            naics_risk_consolidated = naics_risk_df
        
        # Join on consolidated sector codes
        df = df.merge(naics_risk_consolidated, left_on='sector_code_consolidated', right_on='sector_code', how='left', suffixes=('', '_risk'))
else:
    st.warning("Industry column not found in data")

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
st.subheader("CSL Portfolio Summary")
col1, col2, col3 = st.columns(3)
col1.metric("Total Deals", total_deals)
col2.metric("Matured Deals", total_matured)
col3.metric("Current Deals", total_current)

col4, col5, col6 = st.columns(3)
col4.metric("Not Current Deals", total_non_current)
col5.metric("Pct. Outstanding Deals Current", f"{pct_current:.1%}")
col6.metric("Pct. Outstanding Deals Not Current", f"{pct_non_current:.1%}")

# CSL Investment Overview
st.subheader("CSL Investment Overview")
col7, col8, col9 = st.columns(3)
col7.metric("Capital Deployed", f"${csl_capital_deployed:,.0f}")
col8.metric("Past Due Exposure", f"${total_csl_past_due:,.0f}")
col9.metric("Outstanding CSL Principal", f"${total_csl_at_risk:,.0f}")

# CSL Commission Summary
st.subheader("CSL Commission Summary")
col10, col11, col12 = st.columns(3)
col10.metric("Avg. Commission Rate", f"{average_commission_pct:.2%}")
col11.metric("Avg. Applied to Participation", f"{average_commission_on_loan:.2%}")
col12.metric("Total Commission Paid", f"${total_commission_paid:,.0f}")

# ----------------------------
# Tables and visualizations
# ----------------------------

# Loan Tape Display
st.subheader("Loan Tape")

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
        scale=alt.Scale(range=["#d73027", "#fc8d59", "#fee08b", "#d9ef8b", "#91bfdb", "#4575b4"]),
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
    title="Distribution of Deal Status"
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
        title="Percentage of Balance at Risk (Non-Current, Non-Matured Deals)"
    )

    st.altair_chart(risk_chart, use_container_width=True)
else:
    st.info("No non-current, non-matured deals with past due amounts to display.")

# Top 10 Highest Risk Deals
st.subheader("Top 10 Highest Risk Deals")

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

    # Top 10 Risk table
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
        title="Risk Score by Past Due % and Deal Age"
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
#### Understanding the Risk Score

The **Risk Score** ranges between **0.00 and 1.00** and blends two key factors:
- **70% weight**: The **percentage of the balance that is past due**
- **30% weight**: The **age of the deal** (older deals score higher)

| Risk Score | Interpretation            |
|------------|---------------------------|
| 0.00–0.19  | Very Low Risk             |
| 0.20–0.49  | Low to Moderate Risk      |
| 0.50–0.74  | Elevated Risk             |
| 0.75–1.00  | High Risk of Loss         |

*Example*: A score of **0.80** implies the deal is either **very delinquent**, **very old**, or both.
New deals (< 30 days old), those with low delinquency (<1%), or with status 'Current' are excluded.
""")

# ----------------------------
# NEW INDUSTRY & PORTFOLIO INSIGHTS
# ----------------------------

st.markdown("---")
st.header("Portfolio Composition & Risk Insights")

# Industry Analysis - Modified to group by risk_score
if 'sector_name' in df.columns and not df['sector_name'].isna().all():
    st.subheader("Deal Distribution by Industry Sector")
    st.caption("*Risk scores are based on industry sector risk profiles from NAICS data. Risk Score 5 represents the highest industry risk (darkest red).*")
    
    # Group by risk_score from naics_sector_risk_profile table
    industry_summary = df.groupby(['risk_score']).agg({
        'deal_number': 'count',
        'csl_participation': 'sum',
        'csl_principal_at_risk': 'sum',
        'risk_profile': 'first',
        'sector_name': lambda x: ', '.join(x.unique()[:3])  # Show up to 3 sector names per risk score
    }).reset_index()
    
    industry_summary.columns = ['Risk Score', 'Deal Count', 'CSL Capital Deployed', 'CSL Capital at Risk', 'Risk Profile', 'Sectors']
    industry_summary = industry_summary.sort_values('Deal Count', ascending=False)
    
    # Industry deals chart grouped by risk score
    industry_chart = alt.Chart(industry_summary).mark_bar().encode(
        x=alt.X('Risk Score:O', title='Risk Score', axis=alt.Axis(labelAngle=0)),
        y=alt.Y('Deal Count:Q', title='Number of Deals'),
        color=alt.Color('Risk Score:O', 
                       scale=alt.Scale(range=RISK_GRADIENT),
                       title='Industry Risk Score'),
        tooltip=[
            alt.Tooltip('Risk Score:O', title='Risk Score'),
            alt.Tooltip('Deal Count:Q', title='Number of Deals'),
            alt.Tooltip('CSL Capital Deployed:Q', title='Capital Deployed', format='$,.0f'),
            alt.Tooltip('CSL Capital at Risk:Q', title='Capital at Risk', format='$,.0f'),
            alt.Tooltip('Risk Profile:N', title='Risk Profile'),
            alt.Tooltip('Sectors:N', title='Primary Sectors')
        ]
    ).properties(
        width=800,
        height=400,
        title='Number of Deals by Risk Score'
    )
    
    st.altair_chart(industry_chart, use_container_width=True)
    
    # Capital exposure by risk score
    st.subheader("CSL Capital Exposure by Industry")
    st.caption("*Risk scores are based on industry sector risk profiles from NAICS data. Risk Score 5 represents the highest industry risk (darkest red).*")
    
    capital_chart = alt.Chart(industry_summary).mark_bar().encode(
        x=alt.X('Risk Score:O', title='Risk Score', axis=alt.Axis(labelAngle=0)),
        y=alt.Y('CSL Capital at Risk:Q', title='CSL Capital at Risk ($)', axis=alt.Axis(format='$,.0f')),
        color=alt.Color('Risk Score:O',
                       scale=alt.Scale(range=RISK_GRADIENT),
                       title='Industry Risk Score'),
        tooltip=[
            alt.Tooltip('Risk Score:O', title='Risk Score'),
            alt.Tooltip('CSL Capital Deployed:Q', title='Total Capital Deployed', format='$,.0f'),
            alt.Tooltip('CSL Capital at Risk:Q', title='Capital at Risk', format='$,.0f'),
            alt.Tooltip('Deal Count:Q', title='Number of Deals'),
            alt.Tooltip('Risk Profile:N', title='Risk Profile'),
            alt.Tooltip('Sectors:N', title='Primary Sectors')
        ]
    ).properties(
        width=800,
        height=400,
        title='CSL Capital at Risk by Risk Score'
    )
    
    st.altair_chart(capital_chart, use_container_width=True)
    
    # Industry summary table
    st.dataframe(
        industry_summary,
        use_container_width=True,
        column_config={
            "CSL Capital Deployed": st.column_config.NumberColumn("CSL Capital Deployed", format="$%.0f"),
            "CSL Capital at Risk": st.column_config.NumberColumn("CSL Capital at Risk", format="$%.0f"),
        }
    )

# FICO Score Analysis
if 'fico' in df.columns:
    st.subheader("Portfolio Distribution by FICO Score")
    
    # Create FICO bands
    df['fico_band'] = pd.cut(df['fico'], 
                            bins=[0, 580, 620, 660, 700, 740, 850], 
                            labels=['<580', '580-619', '620-659', '660-699', '700-739', '740+'],
                            include_lowest=True)
    
    # FICO analysis with percentage of total
    fico_summary = df.groupby('fico_band').agg({
        'deal_number': 'count',
        'csl_participation': 'sum',
        'csl_principal_at_risk': 'sum'
    }).reset_index()
    
    # Calculate percentage of total deals
    fico_summary['pct_of_total'] = (fico_summary['deal_number'] / fico_summary['deal_number'].sum()) * 100
    
    fico_summary.columns = ['FICO Band', 'Deal Count', 'CSL Capital Deployed', 'CSL Capital at Risk', 'Pct of Total']
    
    # FICO deals chart
    fico_deals_chart = alt.Chart(fico_summary).mark_bar().encode(
        x=alt.X('FICO Band:N', title='FICO Score Band'),
        y=alt.Y('Deal Count:Q', title='Number of Deals'),
        color=alt.Color('FICO Band:N', 
                       scale=alt.Scale(range=['#d62728', '#ff7f0e', '#ffbb78', '#2ca02c', '#98df8a', '#1f77b4']),
                       legend=None),
        tooltip=[
            alt.Tooltip('FICO Band:N', title='FICO Band'),
            alt.Tooltip('Deal Count:Q', title='Number of Deals'),
            alt.Tooltip('Pct of Total:Q', title='% of Total Deals', format='.1f'),
            alt.Tooltip('CSL Capital Deployed:Q', title='Capital Deployed', format='$,.0f'),
            alt.Tooltip('CSL Capital at Risk:Q', title='Capital at Risk', format='$,.0f')
        ]
    ).properties(
        width=600,
        height=400,
        title='Deal Count by FICO Score Band'
    )
    
    # FICO capital exposure chart
    fico_capital_chart = alt.Chart(fico_summary).mark_bar().encode(
        x=alt.X('FICO Band:N', title='FICO Score Band'),
        y=alt.Y('CSL Capital at Risk:Q', title='CSL Capital at Risk ($)', axis=alt.Axis(format='$,.0f')),
        color=alt.Color('FICO Band:N',
                       scale=alt.Scale(range=['#d62728', '#ff7f0e', '#ffbb78', '#2ca02c', '#98df8a', '#1f77b4']),
                       legend=None),
        tooltip=[
            alt.Tooltip('FICO Band:N', title='FICO Band'),
            alt.Tooltip('CSL Capital Deployed:Q', title='Total Capital Deployed', format='$,.0f'),
            alt.Tooltip('CSL Capital at Risk:Q', title='Capital at Risk', format='$,.0f'),
            alt.Tooltip('Deal Count:Q', title='Number of Deals'),
            alt.Tooltip('Pct of Total:Q', title='% of Total Deals', format='.1f')
        ]
    ).properties(
        width=600,
        height=400,
        title='CSL Capital at Risk by FICO Score Band'
    )
    
    col1, col2 = st.columns(2)
    with col1:
        st.altair_chart(fico_deals_chart, use_container_width=True)
    with col2:
        st.altair_chart(fico_capital_chart, use_container_width=True)
    
    # FICO summary table
    st.dataframe(
        fico_summary,
        use_container_width=True,
        column_config={
            "CSL Capital Deployed": st.column_config.NumberColumn("CSL Capital Deployed", format="$%.0f"),
            "CSL Capital at Risk": st.column_config.NumberColumn("CSL Capital at Risk", format="$%.0f"),
            "Pct of Total": st.column_config.NumberColumn("% of Total", format="%.1f%%"),
        }
    )

# Term Analysis - Modified to use years as requested
if 'tib' in df.columns:
    st.subheader("Capital Exposure by Time in Business")
    
    # Create TIB bands using years (5, 10, 15, 25, 35, 45)
    df['tib_years'] = df['tib'] 
    df['tib_band'] = pd.cut(df['tib_years'], 
                           bins=[0, 5, 10, 15, 25, 35, 45, 1000], 
                           labels=['≤5 years', '5-10 years', '10-15 years', '15-25 years', '25-35 years', '35-45 years', '>45 years'],
                           include_lowest=True)
    
    # TIB analysis
    tib_summary = df.groupby('tib_band').agg({
        'deal_number': 'count',
        'csl_participation': 'sum',
        'csl_principal_at_risk': 'sum'
    }).reset_index()
    
    tib_summary.columns = ['TIB Band', 'Deal Count', 'CSL Capital Deployed', 'CSL Capital at Risk']
    
    # TIB capital exposure chart
    tib_chart = alt.Chart(tib_summary).mark_bar().encode(
        x=alt.X('TIB Band:N', title='Time in Business'),
        y=alt.Y('CSL Capital at Risk:Q', title='CSL Capital at Risk ($)', axis=alt.Axis(format='$,.0f')),
        color=alt.Color('TIB Band:N',
                       scale=alt.Scale(range=RISK_GRADIENT),
                       legend=None),
        tooltip=[
            alt.Tooltip('TIB Band:N', title='TIB Band'),
            alt.Tooltip('Deal Count:Q', title='Number of Deals'),
            alt.Tooltip('CSL Capital Deployed:Q', title='Total Capital Deployed', format='$,.0f'),
            alt.Tooltip('CSL Capital at Risk:Q', title='Capital at Risk', format='$,.0f')
        ]
    ).properties(
        width=700,
        height=400,
        title='CSL Capital at Risk by Time in Business'
    )
    
    st.altair_chart(tib_chart, use_container_width=True)
    
    # TIB summary table
    st.dataframe(
        tib_summary,
        use_container_width=True,
        column_config={
            "CSL Capital Deployed": st.column_config.NumberColumn("CSL Capital Deployed", format="$%.0f"),
            "CSL Capital at Risk": st.column_config.NumberColumn("CSL Capital at Risk", format="$%.0f"),
        }
    )

# Download buttons
csv = loan_tape.to_csv(index=False).encode("utf-8")
st.download_button(
    label="Download Loan Tape as CSV",
    data=csv,
    file_name="loan_tape.csv",
    mime="text/csv"
)

if len(risk_df) > 0:
    csv_risk = top_risk_display.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Download Top 10 Risk Deals as CSV",
        data=csv_risk,
        file_name="top_risk_deals.csv",
        mime="text/csv"
    )
