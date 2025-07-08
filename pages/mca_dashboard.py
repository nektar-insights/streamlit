# pages/mca_dashboard.py
"""
Updated MCA dashboard using centralized data loader
"""

from utils.imports import *
from utils.data_loader import load_combined_mca_deals, clear_data_cache
from scripts.get_naics_sector_risk import get_naics_sector_risk
import numpy as np

# ----------------------------
# Define risk gradient color scheme
# ----------------------------
RISK_GRADIENT = ["#fff600","#ffc302", "#ff8f00", "#ff5b00","#ff0505"]

# ----------------------------
# Load and prepare data using centralized loader
# ----------------------------
df = load_combined_mca_deals()

# Filter out Canceled deals completely
if not df.empty and "status_category" in df.columns:
    df = df[df["status_category"] != "Canceled"]

# ----------------------------
# Data type conversions and basic calculations
# ----------------------------
if not df.empty:
    # Convert date columns (already handled by centralized loader, but ensure proper format)
    if "funding_date" in df.columns:
        df["funding_date"] = pd.to_datetime(df["funding_date"], errors="coerce").dt.date

    # Convert all financial columns to numeric (already handled by centralized loader)
    financial_cols = ["purchase_price", "receivables_amount", "current_balance", "past_due_amount", 
                     "principal_amount", "rtr_balance", "amount_hubspot", "total_funded_amount"]
    
    for col in financial_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # CALCULATION 1: Set past_due_amount to 0 for Matured deals (already handled by centralized loader)
    if "status_category" in df.columns and "past_due_amount" in df.columns:
        df.loc[df["status_category"] == "Matured", "past_due_amount"] = 0

    # CALCULATION 2: Rename CSL participation column for clarity
    if "amount_hubspot" in df.columns:
        df.rename(columns={"amount_hubspot": "csl_participation"}, inplace=True)

    # CALCULATION 3: Calculate past due percentage (may already be handled by centralized loader)
    if "past_due_amount" in df.columns and "current_balance" in df.columns:
        df["past_due_pct"] = df.apply(
            lambda row: row["past_due_amount"] / row["current_balance"]
            if pd.notna(row["past_due_amount"]) and pd.notna(row["current_balance"]) and row["current_balance"] > 0
            else 0,
            axis=1
        )

    # CALCULATION 4: Calculate CSL participation ratio
    if "csl_participation" in df.columns and "total_funded_amount" in df.columns:
        df["participation_ratio"] = df["csl_participation"] / df["total_funded_amount"].replace(0, pd.NA)

    # CALCULATION 5: Calculate CSL's portion of past due amount
    if "participation_ratio" in df.columns and "past_due_amount" in df.columns:
        df["csl_past_due"] = df["participation_ratio"] * df["past_due_amount"]

    # CALCULATION 6: Estimate remaining principal balance
    if "principal_amount" in df.columns:
        if "payments_made" in df.columns:
            df["principal_remaining_est"] = df.apply(
                lambda row: row["principal_amount"] if pd.isna(row["payments_made"])
                else max(row["principal_amount"] - row["payments_made"], 0),
                axis=1
            )
        else:
            df["principal_remaining_est"] = df["principal_amount"]

    # CALCULATION 7: Calculate CSL's principal at risk
    if "participation_ratio" in df.columns and "principal_remaining_est" in df.columns:
        df["csl_principal_at_risk"] = df["participation_ratio"] * df["principal_remaining_est"]

    # CALCULATION 8: Set CSL principal at risk to 0 for Matured deals
    if "status_category" in df.columns and "csl_principal_at_risk" in df.columns:
        df.loc[df["status_category"] == "Matured", "csl_principal_at_risk"] = 0

    # CALCULATION 9: Set CSL past due to 0 for Current deals
    if "status_category" in df.columns and "csl_past_due" in df.columns:
        df.loc[df["status_category"] == "Current", "csl_past_due"] = 0

    # CALCULATION 10: Commission rate conversion
    if "commission" in df.columns:
        df["commission_rate"] = pd.to_numeric(df["commission"], errors="coerce")

    # CALCULATION 11: Calculate days since funding (may already be handled by centralized loader)
    if "funding_date" in df.columns and "days_since_funding" not in df.columns:
        df["days_since_funding"] = (pd.Timestamp.today() - pd.to_datetime(df["funding_date"])).dt.days

    # CALCULATION 23: Calculate RTR percentage
    if "principal_amount" in df.columns and "rtr_balance" in df.columns:
        df["rtr_pct"] = df.apply(
            lambda row: (row["principal_amount"] - row["rtr_balance"]) / row["principal_amount"]
            if pd.notna(row["principal_amount"]) and pd.notna(row["rtr_balance"]) and row["principal_amount"] > 0
            else 0,
            axis=1
        )

# ----------------------------
# Industry/NAICS Processing
# ----------------------------
# Load NAICS sector risk data
naics_risk_df = get_naics_sector_risk()

# Extract 2-digit sector code from industry (full NAICS code)
if not df.empty and 'industry' in df.columns:
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
elif not df.empty:
    st.warning("Industry column not found in data")

# ----------------------------
# Page Header and Filters
# ----------------------------
st.title("MCA Deals Dashboard")

if df.empty:
    st.error("No MCA deal data available. Please check your data connection.")
    st.stop()

# Date filter
if "funding_date" in df.columns:
    min_date = df["funding_date"].min()
    max_date = df["funding_date"].max()
    
    if pd.notna(min_date) and pd.notna(max_date):
        start_date, end_date = st.date_input(
            "Filter by Funding Date",
            [min_date, max_date],
            min_value=min_date,
            max_value=max_date
        )
        df = df[(df["funding_date"] >= start_date) & (df["funding_date"] <= end_date)]

# Status filter
if "status_category" in df.columns:
    status_options = ["All"] + list(df["status_category"].dropna().unique())
    status_category_filter = st.radio("Status Category", status_options, index=0)
    if status_category_filter != "All":
        df = df[df["status_category"] == status_category_filter]

# ----------------------------
# Calculate all metrics with detailed comments
# ----------------------------

# PORTFOLIO METRICS
total_deals = len(df)

if "status_category" in df.columns:
    total_matured = (df["status_category"] == "Matured").sum()
    total_current = (df["status_category"] == "Current").sum()
    total_non_current = (df["status_category"] == "Not Current").sum()
    
    # Calculate outstanding deals (Current + Not Current, excludes Matured)
    outstanding_total = total_current + total_non_current
    
    # Calculate percentages of outstanding deals
    pct_current = total_current / outstanding_total if outstanding_total > 0 else 0
    pct_non_current = total_non_current / outstanding_total if outstanding_total > 0 else 0
else:
    total_matured = total_current = total_non_current = 0
    pct_current = pct_non_current = 0

# CSL INVESTMENT METRICS
if "csl_participation" in df.columns:
    csl_capital_deployed = df["csl_participation"].sum()
    
    if "csl_past_due" in df.columns:
        total_csl_past_due = df["csl_past_due"].sum()
    else:
        total_csl_past_due = 0
    
    # Outstanding CSL Principal (Capital at Risk)
    if "status_category" in df.columns and "participation_ratio" in df.columns and "principal_remaining_est" in df.columns:
        at_risk = df[df["status_category"] == "Not Current"]
        total_csl_at_risk = (at_risk["participation_ratio"] * at_risk["principal_remaining_est"]).sum()
    else:
        total_csl_at_risk = 0
else:
    csl_capital_deployed = total_csl_past_due = total_csl_at_risk = 0

# COMMISSION METRICS
if "commission_rate" in df.columns and "csl_participation" in df.columns:
    average_commission_pct = df["commission_rate"].mean()
    total_commission_paid = (df["csl_participation"] * df["commission_rate"]).sum()
    average_commission_on_loan = total_commission_paid / df["csl_participation"].sum() if df["csl_participation"].sum() > 0 else 0
else:
    average_commission_pct = total_commission_paid = average_commission_on_loan = 0

# RISK ANALYSIS DATAFRAMES
if "status_category" in df.columns and "past_due_amount" in df.columns and "current_balance" in df.columns:
    not_current_df = df[(df["status_category"] != "Current") & (df["status_category"] != "Matured")].copy()
    
    # Calculate at-risk percentage for visualization
    if not not_current_df.empty:
        not_current_df["at_risk_pct"] = not_current_df["past_due_amount"] / not_current_df["current_balance"]
        not_current_df = not_current_df[not_current_df["at_risk_pct"] > 0]
    
    # Risk scoring for top 10 highest risk deals
    if "days_since_funding" in df.columns:
        risk_df = df[
            (df["days_since_funding"] > 30) &
            (df["past_due_amount"] > df["current_balance"] * 0.01) &
            (df["status_category"] != "Current") &
            (df["status_category"] != "Matured")
        ].copy()
        
        if len(risk_df) > 0:
            # Calculate risk score components
            risk_df["past_due_pct_calc"] = risk_df["past_due_amount"] / risk_df["current_balance"].clip(lower=1)
            
            max_days = risk_df["days_since_funding"].max()
            if max_days > 0:
                risk_df["age_weight"] = risk_df["days_since_funding"] / max_days
            else:
                risk_df["age_weight"] = 0
            
            # Final risk score
            risk_df["risk_score"] = risk_df["past_due_pct_calc"] * 0.7 + risk_df["age_weight"] * 0.3
            
            # Get top 10 highest risk deals
            top_risk = risk_df.sort_values("risk_score", ascending=False).head(10).copy()
        else:
            top_risk = pd.DataFrame()
    else:
        risk_df = pd.DataFrame()
        top_risk = pd.DataFrame()
else:
    not_current_df = pd.DataFrame()
    risk_df = pd.DataFrame()
    top_risk = pd.DataFrame()

# ----------------------------
# Display metrics sections
# ----------------------------

# Portfolio Summary
st.subheader("CSL Portfolio Summary")
col1, col2, col3 = st.columns(3)
col1.metric("Total Deals", total_deals, help="Total number of deals in the portfolio (excludes Canceled deals)")
col2.metric("Matured Deals", total_matured, help="Number of deals that have been fully paid off and closed")
col3.metric("Current Deals", total_current, help="Number of deals that are performing and up-to-date on payments")

col4, col5, col6 = st.columns(3)
col4.metric("Not Current Deals", total_non_current, help="Number of deals that are delinquent or past due on payments")
col5.metric("Pct. Outstanding Deals Current", f"{pct_current:.1%}", help="Percentage of active deals (Current + Not Current) that are performing well")
col6.metric("Pct. Outstanding Deals Not Current", f"{pct_non_current:.1%}", help="Percentage of active deals (Current + Not Current) that are delinquent")

# CSL Investment Overview
st.subheader("CSL Investment Overview")
col7, col8, col9 = st.columns(3)
col7.metric("Capital Deployed", f"${csl_capital_deployed:,.0f}", help="Total amount of capital that CSL has invested across all deals")
col8.metric("Past Due Exposure", f"${total_csl_past_due:,.0f}", help="CSL's proportional share of all past due amounts based on participation ratio")
col9.metric("Outstanding CSL Principal", f"${total_csl_at_risk:,.0f}", help="CSL's share of remaining principal on 'Not Current' deals")

# CSL Commission Summary
st.subheader("CSL Commission Summary")
col10, col11, col12 = st.columns(3)
col10.metric("Avg. Commission Rate", f"{average_commission_pct:.2%}", help="Average commission rate across all deals")
col11.metric("Avg. Applied to Participation", f"{average_commission_on_loan:.2%}", help="Average commission rate weighted by CSL participation amounts")
col12.metric("Total Commission Paid", f"${total_commission_paid:,.0f}", help="Total commission payments made by CSL across all deals")

# ----------------------------
# Deal Type Composition Visual
# ----------------------------

# Deal Type Composition
if 'deal_type' in df.columns:
    st.subheader("Deal Type Composition")
    
    # Calculate deal type percentages
    deal_type_counts = df["deal_type"].fillna("Unknown").value_counts(normalize=True)
    deal_type_summary = pd.DataFrame({
        "deal_type": deal_type_counts.index.astype(str),
        "percentage": deal_type_counts.values,
        "count": df["deal_type"].fillna("Unknown").value_counts().values
    })
    
    # Deal type bar chart
    deal_type_chart = alt.Chart(deal_type_summary).mark_bar().encode(
        x=alt.X("deal_type:N", title="Deal Type", axis=alt.Axis(labelAngle=-45)),
        y=alt.Y("percentage:Q", title="% of Total Deals", axis=alt.Axis(format=".0%")),
        color=alt.Color("deal_type:N", scale=alt.Scale(range=RISK_GRADIENT), legend=None),
        tooltip=[
            alt.Tooltip("deal_type:N", title="Deal Type"),
            alt.Tooltip("count:Q", title="Number of Deals"),
            alt.Tooltip("percentage:Q", title="% of Total", format=".1%")
        ]
    ).properties(
        width=700,
        height=350,
        title="Distribution of Deal Types"
    )
    
    st.altair_chart(deal_type_chart, use_container_width=True)

# ----------------------------
# Loan Tape Filter and Display
# ----------------------------

# Status Category Filter for Loan Tape
st.subheader("Loan Tape")
loan_tape_status_options = ["All"] + list(df["status_category"].dropna().unique()) if "status_category" in df.columns else ["All"]
loan_tape_status_filter = st.radio("Filter Loan Tape by Status Category", loan_tape_status_options, index=0, key="loan_tape_filter")

# Apply filter to dataframe for loan tape
loan_tape_df = df.copy()
if loan_tape_status_filter != "All" and "status_category" in loan_tape_df.columns:
    loan_tape_df = loan_tape_df[loan_tape_df["status_category"] == loan_tape_status_filter]

# Create loan tape display
display_columns = [
    "deal_number", "dba", "funding_date", "status_category",
    "csl_past_due", "past_due_pct", "rtr_balance"
]

# Only include columns that exist
available_display_columns = [col for col in display_columns if col in loan_tape_df.columns]
loan_tape = loan_tape_df[available_display_columns].copy() if available_display_columns else pd.DataFrame()

if not loan_tape.empty:
    # Rename columns
    column_renames = {
        "deal_number": "Loan ID",
        "dba": "Deal",
        "funding_date": "Funding Date",
        "status_category": "Status Category",
        "csl_past_due": "CSL Past Due ($)",
        "past_due_pct": "Past Due %",
        "rtr_balance": "Remaining to Recover ($)"
    }
    
    # Only rename columns that exist
    existing_renames = {k: v for k, v in column_renames.items() if k in loan_tape.columns}
    loan_tape.rename(columns=existing_renames, inplace=True)
    
    # Format numeric columns
    if "Past Due %" in loan_tape.columns:
        loan_tape["Past Due %"] = pd.to_numeric(loan_tape["Past Due %"], errors="coerce").fillna(0) * 100
    if "CSL Past Due ($)" in loan_tape.columns:
        loan_tape["CSL Past Due ($)"] = pd.to_numeric(loan_tape["CSL Past Due ($)"], errors="coerce").fillna(0)
    if "Remaining to Recover ($)" in loan_tape.columns:
        loan_tape["Remaining to Recover ($)"] = pd.to_numeric(loan_tape["Remaining to Recover ($)"], errors="coerce").fillna(0)
    
    st.dataframe(
        loan_tape,
        use_container_width=True,
        column_config={
            "Past Due %": st.column_config.NumberColumn("Past Due %", format="%.2f%%"),
            "CSL Past Due ($)": st.column_config.NumberColumn("CSL Past Due ($)", format="$%.0f"),
            "Remaining to Recover ($)": st.column_config.NumberColumn("Remaining to Recover ($)", format="$%.0f"),
        }
    )

# ----------------------------
# Top 5 Biggest Loans Outstanding
# ----------------------------

st.subheader("Top 5 Biggest CSL Investments Outstanding")

if "status_category" in df.columns and "csl_participation" in df.columns:
    # Filter to non-matured deals and sort by CSL participation amount
    biggest_csl_loans = df[df["status_category"] != "Matured"].copy()
    biggest_csl_loans = biggest_csl_loans.sort_values("csl_participation", ascending=False).head(5)
    
    display_cols = [
        "deal_number", "dba", "status_category", "csl_participation", "csl_principal_at_risk", 
        "csl_past_due", "principal_amount", "principal_remaining_est", "current_balance", 
        "rtr_balance", "rtr_pct", "participation_ratio"
    ]
    
    # Only include columns that exist
    available_cols = [col for col in display_cols if col in biggest_csl_loans.columns]
    biggest_csl_loans_display = biggest_csl_loans[available_cols].copy() if available_cols else pd.DataFrame()
    
    if not biggest_csl_loans_display.empty:
        # Rename columns
        column_renames = {
            "deal_number": "Loan ID",
            "dba": "Deal Name",
            "status_category": "Status",
            "csl_participation": "CSL Participation ($)",
            "csl_principal_at_risk": "CSL Principal at Risk ($)",
            "csl_past_due": "CSL Past Due ($)",
            "principal_amount": "Original Principal ($)",
            "principal_remaining_est": "Principal Outstanding ($)",
            "current_balance": "Total Loan Outstanding ($)",
            "rtr_balance": "RTR ($)",
            "rtr_pct": "RTR %",
            "participation_ratio": "CSL Participation %"
        }
        
        existing_renames = {k: v for k, v in column_renames.items() if k in biggest_csl_loans_display.columns}
        biggest_csl_loans_display.rename(columns=existing_renames, inplace=True)
        
        # Clean up numeric data
        numeric_cols = ["CSL Participation ($)", "CSL Principal at Risk ($)", "CSL Past Due ($)", 
                       "Original Principal ($)", "Principal Outstanding ($)", "Total Loan Outstanding ($)", "RTR ($)"]
        
        for col in numeric_cols:
            if col in biggest_csl_loans_display.columns:
                biggest_csl_loans_display[col] = pd.to_numeric(biggest_csl_loans_display[col], errors="coerce").fillna(0)
        
        percentage_cols = ["RTR %", "CSL Participation %"]
        for col in percentage_cols:
            if col in biggest_csl_loans_display.columns:
                biggest_csl_loans_display[col] = pd.to_numeric(biggest_csl_loans_display[col], errors="coerce").fillna(0)
        
        st.dataframe(
            biggest_csl_loans_display,
            use_container_width=True,
            column_config={
                "CSL Participation ($)": st.column_config.NumberColumn("CSL Participation ($)", format="$%.0f"),
                "CSL Principal at Risk ($)": st.column_config.NumberColumn("CSL Principal at Risk ($)", format="$%.0f"),
                "CSL Past Due ($)": st.column_config.NumberColumn("CSL Past Due ($)", format="$%.0f"),
                "Original Principal ($)": st.column_config.NumberColumn("Original Principal ($)", format="$%.0f"),
                "Principal Outstanding ($)": st.column_config.NumberColumn("Principal Outstanding ($)", format="$%.0f"),
                "Total Loan Outstanding ($)": st.column_config.NumberColumn("Total Loan Outstanding ($)", format="$%.0f"),
                "RTR ($)": st.column_config.NumberColumn("RTR ($)", format="$%.0f"),
                "RTR %": st.column_config.NumberColumn("RTR %", format="%.1%"),
                "CSL Participation %": st.column_config.NumberColumn("CSL Participation %", format="%.1%"),
            }
        )

# ----------------------------
# Charts and visualizations
# ----------------------------

# Distribution of Deal Status (Bar Chart)
if "status_category" in df.columns:
    status_category_counts = df["status_category"].fillna("Unknown").value_counts(normalize=True)
    
    # Calculate unpaid CSL Principal by status category
    if "csl_principal_at_risk" in df.columns:
        status_csl_principal = df.groupby(df["status_category"].fillna("Unknown"))["csl_principal_at_risk"].sum()
    else:
        status_csl_principal = pd.Series(0, index=status_category_counts.index)
    
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
if not not_current_df.empty and "deal_number" in not_current_df.columns and "at_risk_pct" in not_current_df.columns:
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
            alt.Tooltip("dba:N", title="Deal Name") if "dba" in not_current_df.columns else alt.value("N/A"),
            alt.Tooltip("past_due_amount:Q", title="Past Due ($)", format="$,.0f") if "past_due_amount" in not_current_df.columns else alt.value("N/A"),
            alt.Tooltip("current_balance:Q", title="Current Balance ($)", format="$,.0f") if "current_balance" in not_current_df.columns else alt.value("N/A"),
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

if not top_risk.empty and len(top_risk) > 0:
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
            alt.Tooltip("dba:N", title="Deal Name") if "dba" in top_risk.columns else alt.value("N/A"),
            alt.Tooltip("status_category:N", title="Status Category") if "status_category" in top_risk.columns else alt.value("N/A"),
            alt.Tooltip("funding_date:T", title="Funding Date") if "funding_date" in top_risk.columns else alt.value("N/A"),
            alt.Tooltip("past_due_amount:Q", title="Past Due Amount", format="$,.0f") if "past_due_amount" in top_risk.columns else alt.value("N/A"),
            alt.Tooltip("current_balance:Q", title="Current Balance", format="$,.0f") if "current_balance" in top_risk.columns else alt.value("N/A"),
            alt.Tooltip("risk_score:Q", title="Risk Score", format=".3f")
        ]
    ).properties(
        width=700,
        height=400,
        title="(Excludes New and Performing Loans)"
    )
    
    st.altair_chart(bar_chart, use_container_width=True)
    
    # Top 10 Risk table
    display_columns = ["deal_number", "dba", "status_category", "funding_date", "risk_score", "csl_past_due", "current_balance"]
    available_columns = [col for col in display_columns if col in top_risk.columns]
    
    if available_columns:
        top_risk_display = top_risk[available_columns].copy()
        
        # Rename columns
        rename_map = {
            "deal_number": "Loan ID",
            "dba": "Deal",
            "status_category": "Status",
            "funding_date": "Funded",
            "risk_score": "Risk Score",
            "csl_past_due": "CSL Past Due ($)",
            "current_balance": "Current Balance ($)"
        }
        
        existing_renames = {k: v for k, v in rename_map.items() if k in top_risk_display.columns}
        top_risk_display = top_risk_display.rename(columns=existing_renames)
        
        # Clean up numeric data
        numeric_columns = ["Risk Score", "CSL Past Due ($)", "Current Balance ($)"]
        for col in numeric_columns:
            if col in top_risk_display.columns:
                top_risk_display[col] = pd.to_numeric(top_risk_display[col], errors="coerce").fillna(0)
        
        st.dataframe(
            top_risk_display,
            use_container_width=True,
            column_config={
                "CSL Past Due ($)": st.column_config.NumberColumn("CSL Past Due ($)", format="$%.0f"),
                "Current Balance ($)": st.column_config.NumberColumn("Current Balance ($)", format="$%.0f"), 
                "Risk Score": st.column_config.NumberColumn("Risk Score", format="%.3f"),
            }
        )
else:
    st.info("No deals meet the risk criteria for analysis.")

# ----------------------------
# Cache Management
# ----------------------------
st.header("🔧 Data Management")

st.subheader("Cache Management")
st.info("💡 Use these buttons to refresh cached data. The centralized data loader automatically handles all data preprocessing.")

col1, col2 = st.columns(2)

with col1:
    if st.button("🔄 Refresh MCA Data", help="Clears cache for combined MCA data"):
        clear_data_cache('combined_mca')
        st.success("MCA data cache cleared!")

with col2:
    if st.button("🔄 Refresh All Data", help="Clears all cached data"):
        clear_data_cache()
        st.success("All data caches cleared!")

# ----------------------------
# Download functionality
# ----------------------------
if not df.empty:
    st.subheader("Export Data")
    
    # Download loan tape
    if not loan_tape.empty:
        csv = loan_tape.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="📥 Download Loan Tape as CSV",
            data=csv,
            file_name=f"mca_loan_tape_{pd.Timestamp.now().strftime('%Y%m%d')}.csv",
            mime="text/csv"
        )
    
    # Download risk deals
    if not top_risk.empty:
        risk_csv = top_risk_display.to_csv(index=False).encode("utf-8") if 'top_risk_display' in locals() else top_risk.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="📥 Download Top 10 Risk Deals as CSV",
            data=risk_csv,
            file_name=f"top_risk_deals_{pd.Timestamp.now().strftime('%Y%m%d')}.csv",
            mime="text/csv"
        )

# ----------------------------
# Footer
# ----------------------------
st.divider()
st.caption(f"Dashboard last updated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')} | Using centralized data loader")
