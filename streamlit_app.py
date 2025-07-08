# streamlit_app.py
"""
Updated main pipeline dashboard using centralized data loader
"""

from utils.imports import *
from utils.data_loader import load_deals, clear_data_cache
import io

# ----------------------------
# Load and prepare data using centralized loader
# ----------------------------
df = load_deals()
today = pd.to_datetime("today").normalize()

# ----------------------------
# Data preprocessing
# ----------------------------
# Note: Most preprocessing is now handled by the centralized data loader
# Only add page-specific transformations here

if not df.empty:
    # Page-specific calculations
    df["is_participated"] = df["is_closed_won"] == True if "is_closed_won" in df.columns else False

# ----------------------------
# Filters
# ----------------------------
st.title("HubSpot Pipeline Dashboard")

if df.empty:
    st.error("No deal data available. Please check your data connection.")
    st.stop()

# Date filter
if "date_created" in df.columns:
    min_date, max_date = df["date_created"].min(), df["date_created"].max()
    start_date, end_date = st.date_input(
        "Filter by Date Range", 
        [min_date, max_date], 
        min_value=min_date, 
        max_value=max_date
    )
    df = df[(df["date_created"] >= pd.to_datetime(start_date)) & 
            (df["date_created"] <= pd.to_datetime(end_date))]

# Partner filter
if "partner_source" in df.columns:
    partner_options = sorted(df["partner_source"].dropna().unique())
    selected_partner = st.selectbox(
        "Filter by Partner Source", 
        options=["All Partners"] + partner_options
    )
    if selected_partner != "All Partners":
        df = df[df["partner_source"] == selected_partner]

# Participation filter
participation_filter = st.radio(
    "Show Deals", 
    ["All Deals", "Participated Only", "Not Participated"]
)
if participation_filter == "Participated Only":
    df = df[df["is_closed_won"] == True] if "is_closed_won" in df.columns else df
elif participation_filter == "Not Participated":
    df = df[df["is_closed_won"] != True] if "is_closed_won" in df.columns else df

# ----------------------------
# Calculate all metrics
# ----------------------------
closed_won = df[df["is_closed_won"] == True] if "is_closed_won" in df.columns else pd.DataFrame()
total_deals = len(df)
participation_ratio = len(closed_won) / total_deals if total_deals > 0 else 0

# Calculate date range and deal flow metrics
if "date_created" in df.columns and not df.empty:
    date_range_days = (df["date_created"].max() - df["date_created"].min()).days + 1
    date_range_weeks = date_range_days / 7
    date_range_months = date_range_days / 30.44  # Average days per month
    
    # Deal flow averages (across ALL deals, not just participated)
    avg_deals_per_day = total_deals / date_range_days if date_range_days > 0 else 0
    avg_deals_per_week = total_deals / date_range_weeks if date_range_weeks > 0 else 0
    avg_deals_per_month = total_deals / date_range_months if date_range_months > 0 else 0
else:
    avg_deals_per_day = avg_deals_per_week = avg_deals_per_month = 0

# Average deal characteristics
numeric_columns = ["total_funded_amount", "factor_rate", "commission", "loan_term"]
available_numeric_columns = [col for col in numeric_columns if col in df.columns]

avg_total_funded = df["total_funded_amount"].mean() if "total_funded_amount" in df.columns else 0
avg_factor_all = df["factor_rate"].mean() if "factor_rate" in df.columns else 0
avg_commission_all = df["commission"].mean() if "commission" in df.columns else 0
avg_term_all = df["loan_term"].mean() if "loan_term" in df.columns else 0

# Participated deal characteristics
if not closed_won.empty:
    avg_amount = closed_won["amount"].mean() if "amount" in closed_won.columns else 0
    avg_factor = closed_won["factor_rate"].mean() if "factor_rate" in closed_won.columns else 0
    avg_term = closed_won["loan_term"].mean() if "loan_term" in closed_won.columns else 0
    avg_commission = closed_won["commission"].mean() if "commission" in closed_won.columns else 0
    
    if "amount" in closed_won.columns and "total_funded_amount" in closed_won.columns:
        avg_participation_pct = (closed_won["amount"] / closed_won["total_funded_amount"]).mean()
    else:
        avg_participation_pct = 0
    
    # TIB and FICO data availability
    has_tib_data = "tib" in closed_won.columns and closed_won["tib"].count() > 0
    has_fico_data = "fico" in closed_won.columns and closed_won["fico"].count() > 0
    
    avg_tib = closed_won["tib"].mean() if has_tib_data else None
    avg_fico = closed_won["fico"].mean() if has_fico_data else None
else:
    avg_amount = avg_factor = avg_term = avg_commission = avg_participation_pct = 0
    has_tib_data = has_fico_data = False
    avg_tib = avg_fico = None

# Financial calculations
if not closed_won.empty and "amount" in closed_won.columns:
    total_capital_deployed = closed_won["amount"].sum()
    
    if "commission" in closed_won.columns:
        total_commissions_paid = (closed_won["amount"] * closed_won["commission"]).sum()
    else:
        total_commissions_paid = 0
    
    # Platform fee calculation (define PLATFORM_FEE_RATE if not already defined)
    PLATFORM_FEE_RATE = 0.02  # 2% platform fee
    total_platform_fee = total_capital_deployed * PLATFORM_FEE_RATE
    
    if "factor_rate" in closed_won.columns:
        total_expected_return = ((closed_won["amount"] * closed_won["factor_rate"]) - 
                               closed_won["amount"] - 
                               (closed_won["amount"] * closed_won.get("commission", 0)) - 
                               (closed_won["amount"] * PLATFORM_FEE_RATE))
        total_expected_return_sum = total_expected_return.sum()
        moic = total_expected_return_sum / total_capital_deployed if total_capital_deployed > 0 else 0
        projected_irr = (moic ** (12 / avg_term) - 1) if avg_term > 0 else 0
    else:
        total_expected_return_sum = moic = projected_irr = 0
else:
    total_capital_deployed = total_commissions_paid = total_platform_fee = 0
    total_expected_return_sum = moic = projected_irr = 0

# ----------------------------
# Display metrics sections
# ----------------------------
st.subheader("Deal Overview")
col1, col2, col3 = st.columns(3)
col1.metric("Total Deals", total_deals)
col2.metric("Closed Won", len(closed_won))
col3.metric("Close Ratio", f"{participation_ratio:.2%}")

# Deal Flow Metrics
st.write("**Deal Flow Averages**")
col4, col5, col6 = st.columns(3)
col4.metric("Avg Deals/Day", f"{avg_deals_per_day:.1f}")
col5.metric("Avg Deals/Week", f"{avg_deals_per_week:.1f}")
col6.metric("Avg Deals/Month", f"{avg_deals_per_month:.1f}")

# Average Deal Characteristics 
st.write("**Average Deal Characteristics (All Deals)**")
col7, col8 = st.columns(2)
col7.metric("Avg Total Funded", f"${avg_total_funded:,.0f}")
col8.metric("Avg Factor", f"{avg_factor_all:.2f}")

col9, col10 = st.columns(2)
col9.metric("Avg Commission", f"{avg_commission_all:.2%}")
col10.metric("Avg Term (mo)", f"{avg_term_all:.1f}")

# Financial Performance
st.subheader("Financial Performance")
col11, col12, col13 = st.columns(3)
col11.metric("Total Capital Deployed", f"${total_capital_deployed:,.0f}")
col12.metric("Total Expected Return", f"${total_expected_return_sum:,.0f}")
col13.metric("MOIC", f"{moic:.2f}")

col14, col15, col16 = st.columns(3)
col14.metric("Projected IRR", f"{projected_irr:.2%}")
col15.metric("Avg % of Deal", f"{avg_participation_pct:.2%}")
col16.metric("Commission Paid", f"${total_commissions_paid:,.0f}")

# Deal Characteristics (Participated Only)
st.subheader("Deal Characteristics (Participated Only)")
col17, col18, col19 = st.columns(3)
col17.metric("Avg Participation ($)", f"${avg_amount:,.0f}")
col18.metric("Avg Factor", f"{avg_factor:.2f}")
col19.metric("Avg Term (mo)", f"{avg_term:.1f}")

# TIB and FICO metrics
if has_tib_data and has_fico_data:
    col20, col21 = st.columns(2)
    col20.metric("Avg TIB", f"{avg_tib:,.0f}")
    col21.metric("Avg FICO", f"{avg_fico:.0f}")
elif has_tib_data:
    col20, _ = st.columns(2)
    col20.metric("Avg TIB", f"{avg_tib:,.0f}")
elif has_fico_data:
    col21, _ = st.columns(2)
    col21.metric("Avg FICO", f"{avg_fico:.0f}")
else:
    st.write("*TIB and FICO data not yet available*")

# ----------------------------
# Visualizations
# ----------------------------
if not closed_won.empty:
    st.subheader("Deal Distribution Analysis")
    
    # Color palette for charts
    PRIMARY_COLOR = "#1f77b4"
    COLOR_PALETTE = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]
    
    # Box plots for key metrics
    col1, col2 = st.columns(2)
    
    with col1:
        if "amount" in closed_won.columns:
            st.write("**Participation Amount Distribution**")
            participation_box = alt.Chart(closed_won).mark_boxplot(
                size=60,
                color=PRIMARY_COLOR,
                outliers={"color": COLOR_PALETTE[1], "size": 40}
            ).encode(
                y=alt.Y("amount:Q", 
                        title="Participation Amount ($)",
                        axis=alt.Axis(format="$.2s")),
                tooltip=[alt.Tooltip("amount:Q", title="Amount", format="$,.0f")]
            ).properties(height=300)
            
            st.altair_chart(participation_box, use_container_width=True)
    
    with col2:
        if "factor_rate" in closed_won.columns:
            st.write("**Factor Rate Distribution**")
            factor_box = alt.Chart(closed_won).mark_boxplot(
                size=60,
                color=COLOR_PALETTE[2],
                outliers={"color": COLOR_PALETTE[3], "size": 40}
            ).encode(
                y=alt.Y("factor_rate:Q", 
                        title="Factor Rate",
                        axis=alt.Axis(format=".2f")),
                tooltip=[alt.Tooltip("factor_rate:Q", title="Factor Rate", format=".3f")]
            ).properties(height=300)
            
            st.altair_chart(factor_box, use_container_width=True)

# ----------------------------
# Monthly Analysis
# ----------------------------
if "month" in df.columns and not df.empty:
    st.subheader("Monthly Trends")
    
    # Monthly aggregations
    monthly_funded = df.groupby("month")["total_funded_amount"].sum().reset_index() if "total_funded_amount" in df.columns else pd.DataFrame()
    monthly_deals = df.groupby("month").size().reset_index(name="deal_count")
    
    if not monthly_funded.empty:
        monthly_funded["month_date"] = pd.to_datetime(monthly_funded["month"])
        
        # Monthly funded chart
        funded_chart = alt.Chart(monthly_funded).mark_bar(
            size=40,
            color=PRIMARY_COLOR,
            cornerRadiusTopLeft=3,
            cornerRadiusTopRight=3
        ).encode(
            x=alt.X("month_date:T", title="Month", axis=alt.Axis(labelAngle=-45, format="%b %Y")),
            y=alt.Y("total_funded_amount:Q", title="Total Funded ($)", axis=alt.Axis(format="$.1s")),
            tooltip=[
                alt.Tooltip("month_date:T", title="Month", format="%B %Y"),
                alt.Tooltip("total_funded_amount:Q", title="Total Funded Amount", format="$,.0f")
            ]
        ).properties(height=400, title="Total Funded Amount by Month")
        
        st.altair_chart(funded_chart, use_container_width=True)

# ----------------------------
# Cache Management
# ----------------------------
st.subheader("Data Management")

col1, col2 = st.columns(2)

with col1:
    if st.button("🔄 Refresh Deal Data", help="Clear cached deal data and reload from database"):
        clear_data_cache('deals')
        st.success("Deal data cache cleared! Please refresh the page to see updated data.")

with col2:
    if st.button("🔄 Refresh All Data", help="Clear all cached data"):
        clear_data_cache()
        st.success("All data caches cleared! Please refresh the page to see updated data.")

# ----------------------------
# Export functionality
# ----------------------------
if not df.empty:
    st.subheader("Export Data")
    
    # Create summary for export
    export_data = df[["id", "date_created", "partner_source", "amount", "total_funded_amount", 
                     "factor_rate", "loan_term", "is_closed_won"]].copy() if all(col in df.columns for col in ["id", "date_created"]) else df
    
    csv_data = export_data.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="📥 Download Deal Data (CSV)",
        data=csv_data,
        file_name=f"deal_data_{pd.Timestamp.now().strftime('%Y%m%d')}.csv",
        mime="text/csv"
    )

# ----------------------------
# Footer
# ----------------------------
st.divider()
st.caption(f"Dashboard last updated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')} | Using centralized data loader")
