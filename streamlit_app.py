# streamlit_app.py
"""
Updated main pipeline dashboard using centralized data loader
COMPLETE VERSION - All original charts and calculations restored
"""

from utils.imports import *
from utils.data_loader import load_deals, clear_data_cache
import io
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
import pisa

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
    # Page-specific calculations that weren't in centralized loader
    cols_to_convert = ["amount", "total_funded_amount", "factor_rate", "loan_term", "commission"]
    
    # Ensure all required columns are properly converted (centralized loader should handle this, but double-check)
    for col in cols_to_convert:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    
    # Ensure date columns are properly formatted
    if "date_created" in df.columns:
        df["date_created"] = pd.to_datetime(df["date_created"], errors="coerce")
        df["month"] = df["date_created"].dt.to_period("M").astype(str)
    
    # Ensure loan_id is string type
    if "loan_id" in df.columns:
        df["loan_id"] = df["loan_id"].astype("string")
    
    # Boolean flags
    df["is_participated"] = df["is_closed_won"] == True if "is_closed_won" in df.columns else False

# ----------------------------
# Constants and Configuration
# ----------------------------
PLATFORM_FEE_RATE = 0.02  # 2% platform fee

# Color palette for charts
PRIMARY_COLOR = "#1f77b4"
COLOR_PALETTE = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]

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
months = df["month"].nunique() if "month" in df.columns else 1

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

# Average deal characteristics (ALL DEALS)
avg_total_funded = df["total_funded_amount"].mean() if "total_funded_amount" in df.columns else 0
avg_factor_all = df["factor_rate"].mean() if "factor_rate" in df.columns else 0
avg_commission_all = df["commission"].mean() if "commission" in df.columns else 0
avg_term_all = df["loan_term"].mean() if "loan_term" in df.columns else 0

# Deal characteristics (PARTICIPATED ONLY)
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

# Rolling deal flow calculations - Fixed to look back from today
periods = [
    ("91-120 Days", 91, 120),
    ("61-90 Days", 61, 90), 
    ("31-60 Days", 31, 60), 
    ("0-30 Days", 0, 30)
]
flow_data = []
for label, start, end in periods:
    if "date_created" in df.columns:
        window = df[(df["date_created"] >= today - pd.Timedelta(days=end)) & 
                    (df["date_created"] <= today - pd.Timedelta(days=start))]
        flow_data.append({
            "Period": label, 
            "Deals": len(window), 
            "Total Funded": window["total_funded_amount"].sum() if "total_funded_amount" in window.columns else 0
        })
    else:
        flow_data.append({"Period": label, "Deals": 0, "Total Funded": 0})

flow_df = pd.DataFrame(flow_data)

# Monthly aggregations
if "month" in df.columns:
    monthly_funded = df.groupby("month")["total_funded_amount"].sum().reset_index() if "total_funded_amount" in df.columns else pd.DataFrame()
    monthly_funded["month_date"] = pd.to_datetime(monthly_funded["month"]) if not monthly_funded.empty else pd.Series()

    monthly_deals = df.groupby("month").size().reset_index(name="deal_count")
    monthly_deals["month_date"] = pd.to_datetime(monthly_deals["month"])

    participated_only = df[df["is_participated"] == True] if "is_participated" in df.columns else pd.DataFrame()
    if not participated_only.empty:
        monthly_participation = participated_only.groupby("month").agg(
            deal_count=("id", "count") if "id" in participated_only.columns else ("index", "count"),
            total_amount=("amount", "sum") if "amount" in participated_only.columns else ("total_funded_amount", "sum")
        ).reset_index()
        monthly_participation["month_date"] = pd.to_datetime(monthly_participation["month"])
    else:
        monthly_participation = pd.DataFrame()

    monthly_participation_ratio = df.groupby("month").agg(
        total_deals=("id", "count") if "id" in df.columns else ("index", "count"),
        participated_deals=("is_participated", "sum") if "is_participated" in df.columns else ("is_closed_won", "sum")
    ).reset_index()
    monthly_participation_ratio["participation_pct"] = (
        monthly_participation_ratio["participated_deals"] / 
        monthly_participation_ratio["total_deals"]
    )
    monthly_participation_ratio["month_date"] = pd.to_datetime(monthly_participation_ratio["month"])

    # Dollar-based participation rate
    monthly_participation_ratio_dollar = df.groupby("month").agg(
        total_funded_amount=("total_funded_amount", "sum") if "total_funded_amount" in df.columns else ("amount", "sum"),
        participated_amount=("amount", "sum") if "amount" in df.columns else ("total_funded_amount", "sum")
    ).reset_index()
    monthly_participation_ratio_dollar["participation_pct_dollar"] = (
        monthly_participation_ratio_dollar["participated_amount"] / 
        monthly_participation_ratio_dollar["total_funded_amount"]
    ).fillna(0)
    monthly_participation_ratio_dollar["month_date"] = pd.to_datetime(monthly_participation_ratio_dollar["month"])
else:
    monthly_funded = monthly_deals = monthly_participation = pd.DataFrame()
    monthly_participation_ratio = monthly_participation_ratio_dollar = pd.DataFrame()

# Partner summary calculations
if "partner_source" in df.columns:
    all_deals = df.groupby("partner_source").agg(
        total_deals=("id", "count") if "id" in df.columns else ("index", "count"),
        total_amount=("total_funded_amount", "sum") if "total_funded_amount" in df.columns else ("amount", "sum")
    )
    
    if not closed_won.empty:
        won_deals = closed_won.groupby("partner_source").agg(
            participated_deals=("id", "count") if "id" in closed_won.columns else ("index", "count"),
            participated_amount=("amount", "sum") if "amount" in closed_won.columns else ("total_funded_amount", "sum"),
            total_won_amount=("total_funded_amount", "sum") if "total_funded_amount" in closed_won.columns else ("amount", "sum")
        )
    else:
        won_deals = pd.DataFrame()
    
    if not won_deals.empty:
        partner_summary = all_deals.join(won_deals, how="left").fillna(0)
        partner_summary["closed_won_pct"] = partner_summary["participated_deals"] / partner_summary["total_deals"]
        partner_summary["avg_participation_pct"] = partner_summary["participated_amount"] / partner_summary["total_won_amount"]
    else:
        partner_summary = all_deals.copy()
        partner_summary["closed_won_pct"] = 0
        partner_summary["avg_participation_pct"] = 0
else:
    partner_summary = pd.DataFrame()

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

# Average Deal Characteristics (All Deals)
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
# Box Plot Visualizations in Grid Layout
# ----------------------------
st.subheader("Deal Distribution Analysis")

if not closed_won.empty:
    # First row - Participation Amount and Factor Rate
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
                tooltip=[
                    alt.Tooltip("amount:Q", title="Amount", format="$,.0f")
                ]
            ).properties(
                height=300
            )
            
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
                tooltip=[
                    alt.Tooltip("factor_rate:Q", title="Factor Rate", format=".3f")
                ]
            ).properties(
                height=300
            )
            
            st.altair_chart(factor_box, use_container_width=True)

    # Second row - Loan Term and TIB/FICO
    col3, col4 = st.columns(2)

    with col3:
        if "loan_term" in closed_won.columns:
            st.write("**Loan Term Distribution**")
            term_box = alt.Chart(closed_won).mark_boxplot(
                size=60,
                color=COLOR_PALETTE[4],
                outliers={"color": COLOR_PALETTE[0], "size": 40}
            ).encode(
                y=alt.Y("loan_term:Q", 
                        title="Loan Term (months)",
                        axis=alt.Axis(format=".0f")),
                tooltip=[
                    alt.Tooltip("loan_term:Q", title="Term (months)", format=".1f")
                ]
            ).properties(
                height=300
            )
            
            st.altair_chart(term_box, use_container_width=True)

    with col4:
        # Show TIB if available, otherwise FICO, otherwise placeholder
        if has_tib_data:
            st.write("**TIB Distribution**")
            tib_box = alt.Chart(closed_won.dropna(subset=['tib'])).mark_boxplot(
                size=60,
                color=COLOR_PALETTE[1],
                outliers={"color": COLOR_PALETTE[2], "size": 40}
            ).encode(
                y=alt.Y("tib:Q", 
                        title="TIB",
                        axis=alt.Axis(format=",.0f")),
                tooltip=[
                    alt.Tooltip("tib:Q", title="TIB", format=",.0f")
                ]
            ).properties(
                height=300
            )
            
            st.altair_chart(tib_box, use_container_width=True)
        elif has_fico_data:
            st.write("**FICO Score Distribution**")
            fico_box = alt.Chart(closed_won.dropna(subset=['fico'])).mark_boxplot(
                size=60,
                color=COLOR_PALETTE[3],
                outliers={"color": COLOR_PALETTE[4], "size": 40}
            ).encode(
                y=alt.Y("fico:Q", 
                        title="FICO Score",
                        axis=alt.Axis(format=".0f")),
                tooltip=[
                    alt.Tooltip("fico:Q", title="FICO Score", format=".0f")
                ]
            ).properties(
                height=300
            )
            
            st.altair_chart(fico_box, use_container_width=True)
        else:
            st.write("**Additional Data**")
            st.info("TIB and FICO data not yet available for visualization")

    # Third row - Only if both TIB and FICO have data
    if has_tib_data and has_fico_data:
        col5, col6 = st.columns(2)
        
        with col5:
            st.write("**FICO Score Distribution**")
            fico_box = alt.Chart(closed_won.dropna(subset=['fico'])).mark_boxplot(
                size=60,
                color="#2E8B57",      # Experian green for good credit
                outliers={"color": "#DC143C", "size": 40}  # Red for outliers (poor credit indicators)
            ).encode(
                y=alt.Y("fico:Q", 
                        title="FICO Score",
                        axis=alt.Axis(format=".0f"),
                        scale=alt.Scale(domain=[300, 850])),  
                tooltip=[
                    alt.Tooltip("fico:Q", title="FICO Score", format=".0f")
                ]
            ).properties(
                height=300
            )
            
            st.altair_chart(fico_box, use_container_width=True)

# ----------------------------
# Rolling Deal Flow
# ----------------------------
st.subheader("Rolling Deal Flow Trends")

# Add change calculations
flow_df["Deal Change"] = flow_df["Deals"].diff().fillna(0).astype(int)
flow_df["Deal Change %"] = flow_df["Deals"].pct_change().fillna(0).apply(lambda x: f"{x:.1%}")
flow_df["Funded Change"] = flow_df["Total Funded"].diff().fillna(0).astype(int)
flow_df["Funded Change %"] = flow_df["Total Funded"].pct_change().fillna(0).apply(lambda x: f"{x:.1%}")

# Display formatted table
flow_df_display = flow_df.copy()
flow_df_display["Total Funded Display"] = flow_df_display["Total Funded"].apply(lambda x: f"${x:,.0f}")
flow_df_display["Funded Change Display"] = flow_df_display["Funded Change"].apply(lambda x: f"${x:,.0f}")

st.dataframe(
    flow_df_display[["Period", "Deals", "Deal Change", "Deal Change %", 
                     "Total Funded Display", "Funded Change Display", "Funded Change %"]], 
    use_container_width=True,
    column_config={
        "Total Funded Display": "Total Funded",
        "Funded Change Display": "Funded Change"
    }
)

# Rolling flow charts
flow_chart = alt.Chart(flow_df).mark_bar(
    size=60, 
    color=PRIMARY_COLOR,
    cornerRadiusTopLeft=3,
    cornerRadiusTopRight=3
).encode(
    x=alt.X("Period:N", 
            sort=["91-120 Days", "61-90 Days", "31-60 Days", "0-30 Days"],
            axis=alt.Axis(labelAngle=0)),
    y=alt.Y("Deals:Q", title="Deal Count"),
    tooltip=[
        alt.Tooltip("Period", title="Period"),
        alt.Tooltip("Deals", title="Deal Count"),
        alt.Tooltip("Deal Change", title="Change vs Previous"),
        alt.Tooltip("Deal Change %", title="Percent Change")
    ]
).properties(
    height=350,
    title="Deal Count by Period"
)

funded_flow_chart = alt.Chart(flow_df).mark_bar(
    size=60, 
    color=COLOR_PALETTE[2],
    cornerRadiusTopLeft=3,
    cornerRadiusTopRight=3
).encode(
    x=alt.X("Period:N", 
            sort=["91-120 Days", "61-90 Days", "31-60 Days", "0-30 Days"],
            axis=alt.Axis(labelAngle=0)),
    y=alt.Y("Total Funded:Q", 
            title="Total Funded ($)", 
            axis=alt.Axis(format="$.1s", titlePadding=15)),
    tooltip=[
        alt.Tooltip("Period", title="Period"),
        alt.Tooltip("Total Funded", title="Total Funded", format="$,.0f"),
        alt.Tooltip("Funded Change", title="Change vs Previous", format="$,.0f"),
        alt.Tooltip("Funded Change %", title="Percent Change")
    ]
).properties(
    height=350,
    title="Total Funded by Period"
)

st.altair_chart(flow_chart, use_container_width=True)
st.altair_chart(funded_flow_chart, use_container_width=True)

# ----------------------------
# MONTHLY PARTICIPATION RATE CHART (COUNT-BASED)
# ----------------------------
if not monthly_participation_ratio.empty:
    st.subheader("Monthly Participation Rate by Deal Count")
    rate_line = alt.Chart(monthly_participation_ratio).mark_line(
        color="#e45756", 
        strokeWidth=4,
        point=alt.OverlayMarkDef(color="#e45756", size=80, filled=True)
    ).encode(
        x=alt.X("month_date:T", 
                title="Month", 
                axis=alt.Axis(labelAngle=-45, format="%b %Y", labelPadding=10)),
        y=alt.Y("participation_pct:Q", 
                title="Participation Rate (by Count)", 
                axis=alt.Axis(format=".0%", titlePadding=20, labelPadding=5)),
        tooltip=[
            alt.Tooltip("month_date:T", title="Month", format="%B %Y"),
            alt.Tooltip("participation_pct:Q", title="Participation Rate", format=".1%")
        ]
    ).properties(
        height=350,
        width=800,
        padding={"left": 80, "top": 20, "right": 20, "bottom": 60}
    )

    st.altair_chart(rate_line, use_container_width=True)

# ----------------------------
# MONTHLY PARTICIPATION RATE CHART (DOLLAR-BASED)
# ----------------------------
if not monthly_participation_ratio_dollar.empty:
    st.subheader("Monthly Participation Rate by Dollar Amount")
    rate_line_dollar = alt.Chart(monthly_participation_ratio_dollar).mark_line(
        color="#17a2b8", 
        strokeWidth=4,
        point=alt.OverlayMarkDef(color="#17a2b8", size=80, filled=True)
    ).encode(
        x=alt.X("month_date:T", 
                title="Month", 
                axis=alt.Axis(labelAngle=-45, format="%b %Y", labelPadding=10)),
        y=alt.Y("participation_pct_dollar:Q", 
                title="Participation Rate (by $)", 
                axis=alt.Axis(format=".0%", titlePadding=20, labelPadding=5)),
        tooltip=[
            alt.Tooltip("month_date:T", title="Month", format="%B %Y"),
            alt.Tooltip("participation_pct_dollar:Q", title="Participation Rate ($)", format=".1%"),
            alt.Tooltip("total_funded_amount:Q", title="Total Opportunities", format="$,.0f"),
            alt.Tooltip("participated_amount:Q", title="Amount Participated", format="$,.0f")
        ]
    ).properties(
        height=350,
        width=800,
        padding={"left": 80, "top": 20, "right": 20, "bottom": 60}
    )

    st.altair_chart(rate_line_dollar, use_container_width=True)

# ----------------------------
# Monthly trend charts 
# ----------------------------
st.subheader("Total Funded Amount by Month")

if not monthly_funded.empty:
    funded_chart = alt.Chart(monthly_funded).mark_bar(
        size=40,
        color=PRIMARY_COLOR,
        cornerRadiusTopLeft=3,
        cornerRadiusTopRight=3
    ).encode(
        x=alt.X("month_date:T", 
                title="",
                axis=alt.Axis(labelAngle=-45, format="%b %Y"),
                scale=alt.Scale(padding=0.2)),
        y=alt.Y("total_funded_amount:Q", 
                title="Total Funded ($)",
                axis=alt.Axis(
                    format="$.1s",
                    tickCount=4,        # Reduced to just 4 ticks
                    labelPadding=12,    # More space between labels and axis
                    titlePadding=25,    # More space between title and labels
                    grid=True,
                    labelFontSize=11    # Smaller font size for labels
                ),
                scale=alt.Scale(
                    nice=True,
                    padding=0.15        # More padding at top/bottom
                )),
        tooltip=[
            alt.Tooltip("month_date:T", title="Month", format="%B %Y"),
            alt.Tooltip("total_funded_amount:Q", title="Total Funded Amount", format="$,.0f")
        ]
    ).properties(
        height=400,
        width=800,
        padding={"left": 85, "top": 30, "right": 20, "bottom": 60}  # Even more left padding
    )

    st.altair_chart(funded_chart, use_container_width=True)

# Total Deal Count by Month
st.subheader("Total Deal Count by Month")
if not monthly_deals.empty:
    deal_chart = alt.Chart(monthly_deals).mark_bar(
        size=40,  # Reduced bar size
        color=COLOR_PALETTE[2],
        cornerRadiusTopLeft=3,
        cornerRadiusTopRight=3
    ).encode(
        x=alt.X("month_date:T", 
                title="Month", 
                axis=alt.Axis(labelAngle=-45, format="%b %Y", labelPadding=10)),
        y=alt.Y("deal_count:Q", 
                title="Deal Count",
                axis=alt.Axis(titlePadding=20, labelPadding=5)),
        tooltip=[
            alt.Tooltip("month_date:T", title="Month", format="%B %Y"),
            alt.Tooltip("deal_count:Q", title="Deal Count")
        ]
    )

    deal_avg = alt.Chart(monthly_deals).mark_rule(
        color="gray", 
        strokeWidth=2, 
        strokeDash=[4, 2],
        opacity=0.7
    ).encode(
        y=alt.Y("mean(deal_count):Q")
    )

    # Add regression line
    deal_regression = alt.Chart(monthly_deals).mark_line(
        color="#e45756", 
        strokeWidth=3
    ).transform_regression(
        'month_date', 'deal_count'
    ).encode(
        x='month_date:T',
        y='deal_count:Q'
    )

    # Combine charts and set properties
    deal_combined = alt.layer(deal_chart, deal_avg, deal_regression).properties(
        height=400,
        width=800,
        padding={"left": 60, "top": 20, "right": 20, "bottom": 60}
    )

    st.altair_chart(deal_combined, use_container_width=True)

st.subheader("Participation Trends by Month")
if not monthly_participation.empty:
    participation_chart = alt.Chart(monthly_participation).mark_bar(
        size=40,  # Reduced bar size
        color=PRIMARY_COLOR,
        cornerRadiusTopLeft=3,
        cornerRadiusTopRight=3
    ).encode(
        x=alt.X("month_date:T", 
                title="Month", 
                axis=alt.Axis(labelAngle=-45, format="%b %Y", labelPadding=10)),
        y=alt.Y("deal_count:Q", 
                title="Participated Deals",
                axis=alt.Axis(titlePadding=20, labelPadding=5)),
        tooltip=[
            alt.Tooltip("month_date:T", title="Month", format="%B %Y"),
            alt.Tooltip("deal_count:Q", title="Participated Count")
        ]
    )

    participation_avg = alt.Chart(monthly_participation).mark_rule(
        color="gray", 
        strokeWidth=2, 
        strokeDash=[4, 2],
        opacity=0.7
    ).encode(
        y=alt.Y("mean(deal_count):Q")
    )

    # Add regression line
    participation_regression = alt.Chart(monthly_participation).mark_line(
        color="#FF9900", 
        strokeWidth=3
    ).transform_regression(
        'month_date', 'deal_count'
    ).encode(
        x='month_date:T',
        y='deal_count:Q'
    )

    # Combine charts and set properties
    participation_combined = alt.layer(participation_chart, participation_avg, participation_regression).properties(
        height=400,
        width=800,
        padding={"left": 60, "top": 20, "right": 20, "bottom": 60}
    )

    st.altair_chart(participation_combined, use_container_width=True)

# ----------------------------
# PARTNER SUMMARY TABLES
# ----------------------------
st.subheader("Partner Performance Summary")

if not partner_summary.empty:
    # Calculate additional metrics for partner summary
    partner_summary_enhanced = partner_summary.copy()

    # Deal-based metrics
    partner_summary_enhanced["participated_deal_count"] = partner_summary_enhanced["participated_deals"].astype(int)
    partner_summary_enhanced["deal_participation_rate"] = (
        partner_summary_enhanced["participated_deals"] / partner_summary_enhanced["total_deals"]
    ).fillna(0)

    # Dollar-based metrics  
    partner_summary_enhanced["dollar_participation_rate"] = (
        partner_summary_enhanced["participated_amount"] / partner_summary_enhanced["total_amount"]
    ).fillna(0)

    # ----------------------------
    # DEAL COUNT SUMMARY TABLE
    # ----------------------------
    st.write("**Deal Count Performance**")

    deal_summary = partner_summary_enhanced.reset_index()[[
        "partner_source", "total_deals", "participated_deal_count", "deal_participation_rate"
    ]].copy()

    # Format the deal summary
    deal_summary["Deal Participation Rate"] = deal_summary["deal_participation_rate"].apply(lambda x: f"{x:.2%}")
    deal_summary = deal_summary.rename(columns={
        "partner_source": "Partner", 
        "total_deals": "Total Deals",
        "participated_deal_count": "CSL Deals"
    })[["Partner", "Total Deals", "CSL Deals", "Deal Participation Rate"]]

    st.dataframe(deal_summary, use_container_width=True)

    # ----------------------------
    # DOLLAR AMOUNT SUMMARY TABLE
    # ----------------------------
    st.write("**Dollar Amount Performance**")

    dollar_summary = partner_summary_enhanced.reset_index()[[
        "partner_source", "total_amount", "participated_amount", "dollar_participation_rate"
    ]].copy()

    # Format the dollar summary
    dollar_summary["$ Opportunities"] = dollar_summary["total_amount"].apply(lambda x: f"${x:,.0f}")
    dollar_summary["$ Participated"] = dollar_summary["participated_amount"].apply(lambda x: f"${x:,.0f}")
    dollar_summary["$ Participation Rate"] = dollar_summary["dollar_participation_rate"].apply(lambda x: f"{x:.2%}")

    dollar_summary = dollar_summary.rename(columns={
        "partner_source": "Partner"
    })[["Partner", "$ Opportunities", "$ Participated", "$ Participation Rate"]]

    st.dataframe(dollar_summary, use_container_width=True)

# ----------------------------
# PDF Creation Helper Function
# ----------------------------
def create_pdf_from_html(html: str):
    """Create PDF from HTML string using reportlab/pisa"""
    try:
        result = io.BytesIO()
        pisa_status = pisa.CreatePDF(io.StringIO(html), dest=result)
        if pisa_status.err:
            return None
        return result.getvalue()
    except Exception as e:
        st.error(f"Error creating PDF: {e}")
        return None

# ----------------------------
# DOWNLOAD FUNCTIONS FOR BOTH TABLES
# ----------------------------
col_download1, col_download2 = st.columns(2)

with col_download1:
    if not partner_summary.empty:
        # Deal Count Summary Downloads
        deal_csv = deal_summary.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="📥 Download Deal Count Summary (CSV)",
            data=deal_csv,
            file_name="partner_deal_count_summary.csv",
            mime="text/csv"
        )
        
        # PDF download
        deal_html = deal_summary.to_html(index=False)
        deal_pdf = create_pdf_from_html(deal_html)
        if deal_pdf:
            st.download_button(
                label="📥 Download Deal Count Summary (PDF)",
                data=deal_pdf,
                file_name="partner_deal_count_summary.pdf",
                mime="application/pdf"
            )

with col_download2:
    if not partner_summary.empty:
        # Dollar Summary Downloads
        dollar_csv = dollar_summary.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="📥 Download Dollar Summary (CSV)",
            data=dollar_csv,
            file_name="partner_dollar_summary.csv",
            mime="text/csv"
        )
        
        # PDF download
        dollar_html = dollar_summary.to_html(index=False)
        dollar_pdf = create_pdf_from_html(dollar_html)
        if dollar_pdf:
            st.download_button(
                label="📥 Download Dollar Summary (PDF)",
                data=dollar_pdf,
                file_name="partner_dollar_summary.pdf",
                mime="application/pdf"
            )

# ----------------------------
# Cache Management and Additional Features
# ----------------------------
st.subheader("📊 Additional Analysis")

# Add data quality indicators
if not df.empty:
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        missing_amounts = df["amount"].isna().sum() if "amount" in df.columns else 0
        st.metric("Deals Missing Amount", missing_amounts)
    
    with col2:
        missing_factors = df["factor_rate"].isna().sum() if "factor_rate" in df.columns else 0
        st.metric("Deals Missing Factor Rate", missing_factors)
    
    with col3:
        missing_terms = df["loan_term"].isna().sum() if "loan_term" in df.columns else 0
        st.metric("Deals Missing Term", missing_terms)
    
    with col4:
        missing_partners = df["partner_source"].isna().sum() if "partner_source" in df.columns else 0
        st.metric("Deals Missing Partner", missing_partners)

# ----------------------------
# Data Management
# ----------------------------
st.subheader("🔧 Data Management")

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
st.subheader("📥 Export Data")

if not df.empty:
    # Create comprehensive export data
    export_columns = [
        "id", "date_created", "partner_source", "amount", "total_funded_amount", 
        "factor_rate", "loan_term", "is_closed_won", "commission", "loan_id"
    ]
    
    # Only include columns that exist
    available_export_columns = [col for col in export_columns if col in df.columns]
    export_data = df[available_export_columns].copy() if available_export_columns else df
    
    # Format for better readability
    if "date_created" in export_data.columns:
        export_data["date_created"] = export_data["date_created"].dt.strftime("%Y-%m-%d")
    
    # Download buttons
    col1, col2, col3 = st.columns(3)
    
    with col1:
        csv_data = export_data.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="📥 Download Deal Data (CSV)",
            data=csv_data,
            file_name=f"deal_data_{pd.Timestamp.now().strftime('%Y%m%d')}.csv",
            mime="text/csv"
        )
    
    with col2:
        # Export only participated deals
        if not closed_won.empty:
            participated_export = closed_won[available_export_columns].copy() if available_export_columns else closed_won
            if "date_created" in participated_export.columns:
                participated_export["date_created"] = participated_export["date_created"].dt.strftime("%Y-%m-%d")
            
            participated_csv = participated_export.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="📥 Download Participated Deals (CSV)",
                data=participated_csv,
                file_name=f"participated_deals_{pd.Timestamp.now().strftime('%Y%m%d')}.csv",
                mime="text/csv"
            )
    
    with col3:
        # Export summary metrics
        summary_data = {
            "Metric": [
                "Total Deals", "Participated Deals", "Participation Rate", 
                "Total Capital Deployed", "Expected Return", "MOIC", 
                "Projected IRR", "Avg Deal Size", "Avg Factor Rate", 
                "Avg Loan Term", "Avg Commission Rate"
            ],
            "Value": [
                total_deals, len(closed_won), f"{participation_ratio:.2%}",
                f"${total_capital_deployed:,.0f}", f"${total_expected_return_sum:,.0f}", 
                f"{moic:.2f}", f"{projected_irr:.2%}", f"${avg_amount:,.0f}", 
                f"{avg_factor:.3f}", f"{avg_term:.1f}", f"{avg_commission:.2%}"
            ]
        }
        
        summary_df = pd.DataFrame(summary_data)
        summary_csv = summary_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="📥 Download Summary Metrics (CSV)",
            data=summary_csv,
            file_name=f"pipeline_summary_{pd.Timestamp.now().strftime('%Y%m%d')}.csv",
            mime="text/csv"
        )

# ----------------------------
# Footer with Data Info
# ----------------------------
st.divider()

# Data source and quality information
col1, col2, col3 = st.columns(3)

with col1:
    st.caption("**Data Source**")
    st.caption("• HubSpot Deals API")
    st.caption("• Centralized Data Loader")
    st.caption("• Auto-pagination enabled")

with col2:
    st.caption("**Data Quality**")
    total_missing = sum([
        df["amount"].isna().sum() if "amount" in df.columns else 0,
        df["factor_rate"].isna().sum() if "factor_rate" in df.columns else 0,
        df["loan_term"].isna().sum() if "loan_term" in df.columns else 0
    ])
    st.caption(f"• {total_missing} missing critical values")
    st.caption(f"• {len(df)} total records loaded")

with col3:
    st.caption("**Last Updated**")
    st.caption(pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"))
    st.caption("• Using centralized data loader")
    st.caption("• 1-hour cache TTL")

# Performance indicator
if total_deals > 0:
    st.success(f"✅ Dashboard loaded successfully with {total_deals:,} deals")
else:
    st.warning("⚠️ No deal data loaded - check data connection")
