# pages/loan_tape.py
"""
Loan Tape Dashboard - CSL Capital
---------------------------------
This dashboard provides comprehensive analytics for loan portfolio management:
- Portfolio summary metrics and performance indicators
- Loan status distribution and tracking
- Return on investment (ROI) analysis
- Risk assessment and scoring
"""

from utils.imports import *
from utils.config import (
    inject_global_styles,
    inject_logo,
    get_supabase_client,
    PRIMARY_COLOR,
    COLOR_PALETTE,
    PLATFORM_FEE_RATE,
)
from datetime import datetime

# ------------------------------
# Page Configuration
# ------------------------------
st.set_page_config(
    page_title="CSL Capital | Loan Tape",
    layout="wide",
)
inject_global_styles()
inject_logo()

# Constants
PLATFORM_FEE = 0.03  # 3% platform fee
RISK_GRADIENT = [
    "#fff600",  # Light yellow
    "#ffc302",  # Yellow
    "#ff8f00",  # Orange
    "#ff5b00",  # Dark orange
    "#ff0505",  # Red
]

# ------------------------------
# Data Loading Functions
# ------------------------------
supabase = get_supabase_client()

@st.cache_data(ttl=3600)
def load_loan_summaries():
    """Load loan summary data from Supabase."""
    res = supabase.table("loan_summaries").select("*").execute()
    return pd.DataFrame(res.data)

@st.cache_data(ttl=3600)
def load_deals():
    """Load deal data from Supabase."""
    res = supabase.table("deals").select("*").execute()
    return pd.DataFrame(res.data)

@st.cache_data(ttl=3600)
def load_naics_sector_risk():
    """Load NAICS sector risk profiles."""
    res = supabase.table("naics_sector_risk_profile").select("*").execute()
    return pd.DataFrame(res.data)

# ------------------------------
# Data Processing Functions
# ------------------------------
def prepare_loan_data(loans_df, deals_df):
    """
    Merge and prepare loan and deal data.
    
    Args:
        loans_df: DataFrame with loan summaries
        deals_df: DataFrame with deal information
        
    Returns:
        Processed DataFrame with combined data
    """
    # Merge dataframes if data exists
    if not loans_df.empty and not deals_df.empty:
        df = loans_df.merge(
            deals_df[["loan_id", "deal_name", "partner_source", "industry", "commission"]], 
            on="loan_id", 
            how="left"
        )
    else:
        df = loans_df.copy()
    
    # Convert dates
    for date_col in ["funding_date", "maturity_date", "payoff_date"]:
        if date_col in df.columns:
            df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    
    # Handle numeric fields
    df['commission'] = pd.to_numeric(df['commission'], errors='coerce').fillna(0)
    
    # Calculate derived fields
    df['total_invested'] = (
        df['csl_participation_amount'] + 
        (df['csl_participation_amount'] * PLATFORM_FEE) +  # Platform fee
        (df['csl_participation_amount'] * df['commission_fee'])  # Commission fee
    )
    
    df['commission_fees'] = df['csl_participation_amount'] * df['commission_fee']
    df['platform_fees'] = df['csl_participation_amount'] * PLATFORM_FEE
    df['net_balance'] = df['total_invested'] - df['total_paid']
    
    # Calculate ROI
    df['current_roi'] = df.apply(
        lambda x: (x['total_paid'] / x['total_invested']) - 1 if x['total_invested'] > 0 else 0, 
        axis=1
    )
    
    # Add status flags and time calculations
    df['is_unpaid'] = df['loan_status'] != "Paid Off"
    df["days_since_funding"] = (pd.Timestamp.today() - df["funding_date"]).dt.days
    
    # Calculate remaining maturity in months (only for active loans)
    df["remaining_maturity_months"] = 0.0
    active_loans_mask = (df['loan_status'] != "Paid Off") & (df['maturity_date'] > pd.Timestamp.today())
    if 'maturity_date' in df.columns:
        df.loc[active_loans_mask, "remaining_maturity_months"] = (
            (df.loc[active_loans_mask, 'maturity_date'] - pd.Timestamp.today()).dt.days / 30
        )
    
    # Add cohort information
    df['cohort'] = df['funding_date'].dt.to_period('Q').astype(str)
    df['funding_month'] = df['funding_date'].dt.to_period('M')
    
    # Extract NAICS sector if industry data exists
    if 'industry' in df.columns:
        df['sector_code'] = df['industry'].astype(str).str[:2]
    
    return df

def calculate_irr(df):
    """
    Calculate IRR metrics for loans.
    
    IRR (Internal Rate of Return) is calculated as the discount rate that makes the NPV
    of all cash flows equal to zero. For loans, we consider:
    - Initial outflow: Total invested amount (negative)
    - Final inflow: Total paid amount (positive)
    
    Args:
        df: DataFrame with loan data
        
    Returns:
        DataFrame with added IRR calculations
    """
    result_df = df.copy()
    
    # Calculate Realized IRR for paid-off loans using the actual payoff date
    def calc_realized_irr(row):
        if pd.isna(row['funding_date']) or pd.isna(row['payoff_date']) or row['total_invested'] <= 0:
            return None
            
        # Calculate days between funding and payoff
        funding_date = pd.to_datetime(row['funding_date'])
        payoff_date = pd.to_datetime(row['payoff_date'])
        
        if payoff_date <= funding_date:
            # Can't calculate IRR if dates are invalid
            return None
            
        # Calculate days and convert to years
        days_to_payoff = (payoff_date - funding_date).days
        years_to_payoff = days_to_payoff / 365.0
        
        # If years is too small, IRR calculation might be unstable
        if years_to_payoff < 0.01:  # Less than ~3-4 days
            # For very short periods, use simple return formula
            simple_return = (row['total_paid'] / row['total_invested']) - 1
            annualized = (1 + simple_return) ** (1 / years_to_payoff) - 1
            return annualized
            
        # For normal periods, use financial IRR calculation
        # This creates a cash flow series: [initial investment (negative), final payment (positive)]
        try:
            irr = npf.irr([-row['total_invested'], row['total_paid']])
            
            # Check if IRR is reasonable (sometimes npf.irr can return unrealistic values)
            if irr < -1 or irr > 10:  # -100% to 1000%
                # Fallback to simple annualized return for extreme values
                simple_return = (row['total_paid'] / row['total_invested']) - 1
                annualized = (1 + simple_return) ** (1 / years_to_payoff) - 1
                return annualized
                
            return irr
        except:
            # If IRR calculation fails, fall back to simple annualized return
            simple_return = (row['total_paid'] / row['total_invested']) - 1
            annualized = (1 + simple_return) ** (1 / years_to_payoff) - 1
            return annualized
    
    # Calculate Expected IRR based on maturity date and projected payment
    def calc_expected_irr(row):
        if pd.isna(row['funding_date']) or pd.isna(row['maturity_date']) or row['total_invested'] <= 0:
            return None
            
        # For paid off loans, use realized IRR
        if row['loan_status'] == 'Paid Off' and not pd.isna(row['realized_irr']):
            return row['realized_irr']
            
        # Calculate days between funding and maturity
        funding_date = pd.to_datetime(row['funding_date'])
        maturity_date = pd.to_datetime(row['maturity_date'])
        
        if maturity_date <= funding_date:
            # Can't calculate IRR if dates are invalid
            return None
            
        # Expected payment at maturity (total RTR)
        expected_payment = row['our_rtr'] if 'our_rtr' in row and pd.notnull(row['our_rtr']) else row['total_invested'] * (1 + row['roi'])
        
        # Calculate expected IRR based on expected payment at maturity
        try:
            days_to_maturity = (maturity_date - funding_date).days
            years_to_maturity = days_to_maturity / 365.0
            
            # For very short periods, use simple return
            if years_to_maturity < 0.01:
                simple_return = (expected_payment / row['total_invested']) - 1
                annualized = (1 + simple_return) ** (1 / years_to_maturity) - 1
                return annualized
                
            # Normal IRR calculation
            irr = npf.irr([-row['total_invested'], expected_payment])
            
            # Check for reasonable values
            if irr < -1 or irr > 10:
                simple_return = (expected_payment / row['total_invested']) - 1
                annualized = (1 + simple_return) ** (1 / years_to_maturity) - 1
                return annualized
                
            return irr
        except:
            # Fallback calculation
            simple_return = (expected_payment / row['total_invested']) - 1
            annualized = (1 + simple_return) ** (1 / years_to_maturity) - 1
            return annualized
    
    # Apply IRR calculations
    result_df['realized_irr'] = result_df.apply(calc_realized_irr, axis=1)
    
    # First calculate realized IRR, then use it for expected IRR calculation where applicable
    result_df['expected_irr'] = result_df.apply(calc_expected_irr, axis=1)
    
    # Add annualized percentage fields for display purposes
    result_df['realized_irr_pct'] = result_df['realized_irr'].apply(
        lambda x: f"{x:.2%}" if pd.notnull(x) else "N/A"
    )
    result_df['expected_irr_pct'] = result_df['expected_irr'].apply(
        lambda x: f"{x:.2%}" if pd.notnull(x) else "N/A"
    )
    
    return result_df

def calculate_risk_scores(df):
    """
    Calculate risk scores for active loans.
    
    Args:
        df: DataFrame with loan data
        
    Returns:
        DataFrame with risk score calculations
    """
    # Only include active loans (exclude paid off loans)
    risk_df = df[df['loan_status'] != 'Paid Off'].copy()
    
    # Skip calculation if empty
    if risk_df.empty:
        return risk_df
    
    # Calculate performance-based risk score
    # Score = 70% payment performance shortfall + 30% deal age
    max_age = risk_df['days_since_funding'].max()
    risk_df['performance_gap'] = 1 - risk_df['payment_performance'].clip(upper=1.0)
    risk_df['age_weight'] = risk_df['days_since_funding'] / max_age if max_age > 0 else 0
    risk_df['risk_score'] = (risk_df['performance_gap'] * 0.7) + (risk_df['age_weight'] * 0.3)
    
    # Create risk bands
    risk_bins = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
    risk_labels = ["0–0.2", "0.2–0.4", "0.4–0.6", "0.6–0.8", "0.8–1.0"]
    risk_df["risk_band"] = pd.cut(risk_df["risk_score"], bins=risk_bins, labels=risk_labels)
    
    return risk_df

# ------------------------------
# Visualization Functions
# ------------------------------
def format_dataframe_for_display(df, columns=None, rename_map=None):
    """
    Format DataFrame for display by renaming columns and formatting values.
    
    Args:
        df: Source DataFrame
        columns: List of columns to include (defaults to all)
        rename_map: Dictionary mapping old column names to new display names
        
    Returns:
        Formatted DataFrame ready for display
    """
    # Select columns if specified
    if columns:
        # Filter to only include columns that exist
        display_columns = [col for col in columns if col in df.columns]
        display_df = df[display_columns].copy()
    else:
        display_df = df.copy()
    
    # Rename columns if specified
    if rename_map:
        # Apply renaming for columns that exist
        display_df.rename(
            columns={k: v for k, v in rename_map.items() if k in display_df.columns}, 
            inplace=True
        )
    
    # Format numeric columns
    for col in display_df.select_dtypes(include=['float64', 'float32']).columns:
        if any(term in col for term in ["ROI", "Rate", "Percentage", "Performance"]):
            display_df[col] = display_df[col].map(lambda x: f"{x:.2%}" if pd.notnull(x) else "")
        elif "Maturity" in col:
            display_df[col] = display_df[col].map(lambda x: f"{x:.1f}" if pd.notnull(x) else "")
        elif any(term in col for term in ["Capital", "Invested", "Paid", "Balance", "Fees"]):
            display_df[col] = display_df[col].map(lambda x: f"${x:,.2f}" if pd.notnull(x) else "")
    
    # Format date columns
    for col in display_df.columns:
        if any(term in col for term in ["Date", "Funding", "Maturity"]) and pd.api.types.is_datetime64_dtype(df[col.replace(" ", "_").lower()]):
            display_df[col] = pd.to_datetime(df[col.replace(" ", "_").lower()]).dt.strftime('%Y-%m-%d')
    
    return display_df

def plot_status_distribution(df):
    """Create and display a pie chart of loan status distribution (excluding Paid Off)."""
    active_df = df[df["loan_status"] != "Paid Off"].copy()
    
    if active_df.empty:
        st.info("No active loans to display in status distribution chart.")
        return
        
    # Calculate status percentages for active loans
    status_counts = active_df["loan_status"].value_counts(normalize=True)
    status_summary = pd.DataFrame({
        "status": status_counts.index.astype(str),
        "percentage": status_counts.values,
        "count": active_df["loan_status"].value_counts().values,
        "balance": active_df.groupby("loan_status")["net_balance"].sum().reindex(status_counts.index).values
    })
    
    # Add note about excluding Paid Off loans
    st.caption("Note: 'Paid Off' loans are excluded from this chart")
    
    # Create pie chart
    pie_chart = alt.Chart(status_summary).mark_arc().encode(
        theta=alt.Theta(field="percentage", type="quantitative"),
        color=alt.Color(
            field="status", 
            type="nominal", 
            scale=alt.Scale(range=RISK_GRADIENT),
            legend=alt.Legend(title="Loan Status", orient="right")
        ),
        tooltip=[
            alt.Tooltip("status:N", title="Loan Status"),
            alt.Tooltip("count:Q", title="Number of Loans"),
            alt.Tooltip("percentage:Q", title="% of Active Loans", format=".1%"),
            alt.Tooltip("balance:Q", title="Net Balance", format="$,.0f")
        ]
    ).properties(
        width=600,
        height=400,
        title={
            "text": "Distribution of Active Loan Status",
            "subtitle": "By count of loans in each status category",
            "fontSize": 16
        }
    )
    
    st.altair_chart(pie_chart, use_container_width=True)

def plot_roi_distribution(df):
    """Create and display a bar chart showing ROI distribution by loan."""
    # Create ROI visualization for non-zero investment loans
    roi_df = df[df['total_invested'] > 0].copy()
    roi_df = roi_df.sort_values('current_roi', ascending=False)
    
    if roi_df.empty:
        st.info("No loans with investment data to display ROI distribution.")
        return
        
    roi_chart = alt.Chart(roi_df).mark_bar().encode(
        x=alt.X(
            "loan_id:N",
            title="Loan ID",
            sort="-y",
            axis=alt.Axis(labelAngle=-90, labelLimit=150)
        ),
        y=alt.Y(
            "current_roi:Q", 
            title="Return on Investment (ROI)",
            axis=alt.Axis(format=".0%", grid=True)
        ),
        color=alt.Color(
            "current_roi:Q",
            scale=alt.Scale(domain=[-0.5, 0, 0.5], range=["#ff0505", "#ffc302", "#2ca02c"]),
            legend=alt.Legend(title="ROI", format=".0%")
        ),
        tooltip=[
            alt.Tooltip("loan_id:N", title="Loan ID"),
            alt.Tooltip("deal_name:N", title="Deal Name"),
            alt.Tooltip("loan_status:N", title="Status"),
            alt.Tooltip("current_roi:Q", title="Current ROI", format=".2%"),
            alt.Tooltip("total_invested:Q", title="Total Invested", format="$,.2f"),
            alt.Tooltip("total_paid:Q", title="Total Paid", format="$,.2f")
        ]
    ).properties(
        width=800,
        height=400,
        title={
            "text": "Return on Investment by Loan",
            "subtitle": "Sorted from highest to lowest ROI",
            "fontSize": 16
        }
    )

    st.altair_chart(roi_chart, use_container_width=True)
    
    # Add explanatory text
    st.caption(
        "**ROI Color Legend:** Red indicates negative returns, yellow indicates break-even, " +
        "and green indicates positive returns. Hover over bars for detailed information."
    )

def plot_capital_flow(df):
    """
    Create and display a line chart showing capital deployment vs returns over time.
    
    This function analyzes the funding dates and payment dates to accurately show
    when capital was deployed versus when returns were received.
    """
    # Get loan schedules for actual payment dates
    @st.cache_data(ttl=3600)
    def load_loan_schedules():
        """Load loan schedule data from Supabase."""
        res = supabase.table("loan_schedules").select("*").execute()
        return pd.DataFrame(res.data)
    
    loan_schedules = load_loan_schedules()
    
    # Convert funding_date to datetime if not already
    df['funding_date'] = pd.to_datetime(df['funding_date'], errors='coerce')
    
    # Capital Deployed - by funding date
    deploy_df = df[['funding_date', 'csl_participation_amount']].dropna().sort_values('funding_date')
    deploy_df = deploy_df.groupby('funding_date').sum().cumsum().reset_index()
    deploy_df.rename(columns={'csl_participation_amount': 'capital_deployed'}, inplace=True)
    
    # Capital Returned - by actual payment date (from loan_schedules)
    if not loan_schedules.empty and 'payment_date' in loan_schedules.columns:
        # Convert payment dates to datetime
        loan_schedules['payment_date'] = pd.to_datetime(loan_schedules['payment_date'], errors='coerce')
        
        # Filter for rows with actual payments
        payment_data = loan_schedules[
            loan_schedules['actual_payment'].notna() & 
            (loan_schedules['actual_payment'] > 0) &
            loan_schedules['payment_date'].notna()
        ]
        
        # Sum payments by date
        return_df = payment_data.groupby('payment_date')['actual_payment'].sum().reset_index()
        return_df = return_df.sort_values('payment_date')
        
        # Calculate cumulative sum
        return_df['capital_returned'] = return_df['actual_payment'].cumsum()
    else:
        # Fallback if no payment schedule data available
        st.warning("Payment schedule data not available. Using estimated return dates.")
        return_df = pd.DataFrame(columns=['payment_date', 'capital_returned'])
    
    # Skip if data is insufficient
    if deploy_df.empty or return_df.empty:
        st.info("Insufficient data to display capital flow chart.")
        return
    
    # Create separate charts with shared x-axis scale
    deploy_chart = alt.Chart(deploy_df).mark_line(color="red").encode(
        x=alt.X('funding_date:T', title="Date", axis=alt.Axis(format="%b %Y")),
        y=alt.Y('capital_deployed:Q', title="Capital Deployed ($)"),
        tooltip=[
            alt.Tooltip('funding_date:T', title="Date", format="%Y-%m-%d"),
            alt.Tooltip('capital_deployed:Q', title="Capital Deployed", format="$,.0f")
        ]
    ).properties(
        width=800,
        height=400,
    )
    
    # Only create return chart if we have return data
    if not return_df.empty:
        return_chart = alt.Chart(return_df).mark_line(color="green").encode(
            x=alt.X('payment_date:T', title="Date"),
            y=alt.Y('capital_returned:Q', title="Capital Returned ($)"),
            tooltip=[
                alt.Tooltip('payment_date:T', title="Date", format="%Y-%m-%d"),
                alt.Tooltip('capital_returned:Q', title="Capital Returned", format="$,.0f")
            ]
        )
        
        # Combine charts
        capital_chart = alt.layer(deploy_chart, return_chart).resolve_scale(
            x='shared', y='independent'
        ).properties(
            title={
                "text": "Capital Deployed vs. Capital Returned Over Time",
                "subtitle": "Red = Capital Deployed, Green = Capital Returned",
                "fontSize": 16
            }
        )
    else:
        capital_chart = deploy_chart.properties(
            title={
                "text": "Capital Deployed Over Time",
                "fontSize": 16
            }
        )
    
    # Milestone Annotations
    milestones = [500_000, 1_000_000, 2_000_000, 3_000_000]
    milestone_df = pd.DataFrame()
    for value in milestones:
        row = deploy_df[deploy_df['capital_deployed'] >= value].head(1)
        if not row.empty:
            row = row.copy()
            row['milestone'] = f"${value:,.0f}"
            milestone_df = pd.concat([milestone_df, row])
    
    # Add milestone points if we have any
    if not milestone_df.empty:
        milestone_points = alt.Chart(milestone_df).mark_point(filled=True, size=80, color="red").encode(
            x='funding_date:T',
            y='capital_deployed:Q',
            tooltip=[
                alt.Tooltip('milestone:N', title="Milestone"), 
                alt.Tooltip('funding_date:T', title="Date Reached", format="%Y-%m-%d")
            ]
        )
        capital_chart = alt.layer(capital_chart, milestone_points)
    
    st.altair_chart(capital_chart, use_container_width=True)
    
    # Add explanatory note
    st.caption(
        "**Note:** Capital deployment is tracked by funding date. Capital returns are based on actual payment dates " +
        "from loan schedule records. The chart shows the cumulative amounts over time."
    )

def plot_irr_by_partner(df):
    """Create and display a bar chart showing average IRR by partner."""
    # Filter to paid off loans
    paid_df = df[df['loan_status'] == "Paid Off"].copy()
    
    if paid_df.empty or 'realized_irr' not in paid_df.columns:
        st.info("No paid-off loans with IRR data to display.")
        return
        
    irr_by_partner = paid_df.groupby('partner_source').agg(
        avg_irr=('realized_irr', 'mean'),
        deal_count=('loan_id', 'count'),
        total_invested=('total_invested', 'sum'),
        total_returned=('total_paid', 'sum')
    ).dropna().reset_index()
    
    if irr_by_partner.empty:
        st.info("No partner data available for IRR analysis.")
        return
        
    irr_chart = alt.Chart(irr_by_partner).mark_bar().encode(
        x=alt.X(
            'avg_irr:Q', 
            title="Average IRR", 
            axis=alt.Axis(format=".0%", grid=True)
        ),
        y=alt.Y(
            'partner_source:N', 
            title="Partner", 
            sort='-x',
            axis=alt.Axis(labelLimit=150)
        ),
        color=alt.Color(
            'avg_irr:Q',
            scale=alt.Scale(domain=[-0.1, 0, 0.5], range=["#d62728", "#ffc302", "#2ca02c"]),
            legend=alt.Legend(title="IRR", format=".0%")
        ),
        tooltip=[
            alt.Tooltip('partner_source:N', title="Partner"),
            alt.Tooltip('avg_irr:Q', title="Average IRR", format=".2%"),
            alt.Tooltip('deal_count:Q', title="Number of Deals"),
            alt.Tooltip('total_invested:Q', title="Total Invested", format="$,.2f"),
            alt.Tooltip('total_returned:Q', title="Total Returned", format="$,.2f")
        ]
    ).properties(
        width=700,
        height=400,
        title={
            "text": "Average IRR by Partner",
            "subtitle": "For paid-off loans only",
            "fontSize": 16
        }
    )
    
    # Add count text
    text = alt.Chart(irr_by_partner).mark_text(
        align='left',
        baseline='middle',
        dx=5,
        fontSize=12
    ).encode(
        x='avg_irr:Q',
        y='partner_source:N',
        text=alt.Text('deal_count:Q', format='d', title="# Deals")
    )
    
    st.altair_chart(irr_chart + text, use_container_width=True)
    
    # Add explanatory text
    st.caption(
        "**Note:** This chart shows the average Internal Rate of Return (IRR) for each partner, " +
        "calculated only from fully paid-off loans. The number at the end of each bar indicates " +
        "the count of deals with that partner."
    )

def plot_repayment_heatmap(df):
    """Create and display a heatmap of cohort repayment performance over time."""
    # Filter relevant loans
    cohort_df = df[
        (df['funding_date'].notna()) &
        (df['our_rtr'] > 0) &
        (df['total_paid'] >= 0)
    ].copy()
    
    if cohort_df.empty:
        st.info("Insufficient data to display repayment heatmap.")
        return
        
    # Calculate "months since funding"
    cohort_df['current_month'] = pd.Timestamp.today().to_period('M')
    cohort_df['months_since_funding'] = (
        (cohort_df['current_month'] - cohort_df['funding_month']).apply(lambda x: x.n)
    ).clip(lower=0)
    
    # Create base repayment % for each loan
    cohort_df['repayment_pct'] = (cohort_df['total_paid'] / cohort_df['our_rtr']).clip(upper=1.0)
    
    # Group by cohort and months since funding
    repayment_grid = cohort_df.groupby(['cohort', 'months_since_funding']).agg(
        avg_repayment_pct=('repayment_pct', 'mean'),
        loans=('loan_id', 'count')
    ).reset_index()
    
    # Build heatmap
    heatmap = alt.Chart(repayment_grid).mark_rect().encode(
        x=alt.X(
            'months_since_funding:O', 
            title="Months Since Funding",
            axis=alt.Axis(labelAngle=0, grid=True)
        ),
        y=alt.Y(
            'cohort:N', 
            title="Funding Quarter", 
            sort='-y',
            axis=alt.Axis(tickMinStep=1)
        ),
        color=alt.Color(
            'avg_repayment_pct:Q', 
            title="% of RTR Paid", 
            scale=alt.Scale(scheme="greens", domain=[0, 1]),
            legend=alt.Legend(format=".0%")
        ),
        tooltip=[
            alt.Tooltip('cohort:N', title="Funding Quarter"),
            alt.Tooltip('months_since_funding:O', title="Months Since Funding"),
            alt.Tooltip('avg_repayment_pct:Q', title="Avg % Repaid", format=".0%"),
            alt.Tooltip('loans:Q', title="# of Loans")
        ]
    ).properties(
        width=700,
        height=400,
        title={
            "text": "Cohort Repayment Progress Over Time",
            "subtitle": "Shows average percentage of return-to-repay (RTR) received by funding cohort",
            "fontSize": 16
        }
    )
    
    st.altair_chart(heatmap, use_container_width=True)
    
    # Add explanatory note
    st.caption(
        "**How to read this chart:** Each row represents a cohort of loans funded in the same quarter. " +
        "Each column shows how many months have passed since funding. The color intensity indicates " +
        "the average percentage of the total expected return (RTR) that has been paid back. " +
        "Darker green indicates higher repayment percentages."
    )

def plot_capital_waterfall(df):
    """
    Create and display a waterfall chart showing capital flow.
    
    This function calculates the flow of capital from initial deployment through
    fees and returns to show the net gain or loss.
    """
    # Summarize values - all values are calculated from filtered dataframe
    total_capital_deployed = df['csl_participation_amount'].sum()
    platform_fees = df['platform_fees'].sum()
    commission_fees = df['commission_fees'].sum()
    bad_debt_allowance = df['bad_debt_allowance'].sum() if 'bad_debt_allowance' in df.columns else 0
    
    # Net investment (after fees and allowances)
    net_investment = total_capital_deployed # Don't subtract fees as they're included in total_invested
    
    # Total amount returned
    capital_returned = df['total_paid'].sum()
    
    # Net gain or loss (returns minus deployed capital)
    net_position = capital_returned - total_capital_deployed
    
    # Build waterfall data
    waterfall_data = pd.DataFrame([
        {"label": "Capital Deployed", "value": total_capital_deployed, "type": "start"},
        {"label": "Platform Fees", "value": platform_fees, "type": "info"},
        {"label": "Commission Fees", "value": commission_fees, "type": "info"},
        {"label": "Bad Debt Allowance", "value": bad_debt_allowance, "type": "info"},
        {"label": "Capital Returned", "value": capital_returned, "type": "end"},
        {"label": "Net Position", "value": net_position, "type": "net"}
    ])
    
    # Add percentage of capital deployed for fees
    if total_capital_deployed > 0:
        waterfall_data["pct_of_capital"] = waterfall_data["value"] / total_capital_deployed
    else:
        waterfall_data["pct_of_capital"] = 0
    
    # Compute cumulative position for visualization
    waterfall_data["running_total"] = 0
    waterfall_data.loc[0, "running_total"] = waterfall_data.loc[0, "value"]  # Start with capital deployed
    waterfall_data.loc[4, "running_total"] = waterfall_data.loc[4, "value"]  # Capital returned
    waterfall_data.loc[5, "running_total"] = waterfall_data.loc[0, "value"] + waterfall_data.loc[5, "value"]  # Net position
    
    # Format waterfall data for visualization
    cumulative = []
    for _, row in waterfall_data.iterrows():
        if row['type'] == "start":
            # Capital deployed (starting point)
            cumulative.append({
                "label": row['label'],
                "start": 0,
                "end": row['value'],
                "color": "#1f77b4",  # Blue
                "value": row['value'],
                "pct": row['pct_of_capital']
            })
        elif row['type'] == "info":
            # Fees (informational only, doesn't affect waterfall)
            cumulative.append({
                "label": row['label'],
                "start": 0,
                "end": row['value'],
                "color": "#ff7f0e",  # Orange
                "value": row['value'],
                "pct": row['pct_of_capital']
            })
        elif row['type'] == "end":
            # Capital returned 
            cumulative.append({
                "label": row['label'],
                "start": 0,
                "end": row['value'],
                "color": "#2ca02c",  # Green
                "value": row['value'],
                "pct": row['pct_of_capital'] if total_capital_deployed > 0 else 0
            })
        elif row['type'] == "net":
            # Net position (gain or loss)
            cumulative.append({
                "label": row['label'],
                "start": total_capital_deployed,
                "end": row['running_total'],
                "color": "#2ca02c" if row['value'] >= 0 else "#d62728",  # Green if gain, red if loss
                "value": row['value'],
                "pct": row['pct_of_capital']
            })
    
    wf_df = pd.DataFrame(cumulative)
    
    # Create bars
    bars = alt.Chart(wf_df).mark_bar().encode(
        x=alt.X('label:N', title="", axis=alt.Axis(labelAngle=0)),
        y=alt.Y('start:Q', title="Amount ($)", axis=alt.Axis(format="$,.0f", grid=True)),
        y2='end:Q',
        color=alt.Color('color:N', scale=None, legend=None),
        tooltip=[
            alt.Tooltip('label:N', title="Category"),
            alt.Tooltip('value:Q', title="Amount", format="$,.2f"),
            alt.Tooltip('pct:Q', title="% of Capital Deployed", format=".1%")
        ]
    ).properties(
        width=700,
        height=400,
        title={
            "text": "Capital Flow Analysis",
            "subtitle": "Breakdown of capital deployment, fees, and returns",
            "fontSize": 16
        }
    )
    
    # Add labels
    labels = alt.Chart(wf_df).mark_text(
        dy=-10,
        size=12,
        fontWeight="bold"
    ).encode(
        x='label:N',
        y='end:Q',
        text=alt.Text('value:Q', format="$,.0f")
    )
    
    # Add percentage labels for fees and returns
    pct_labels = alt.Chart(wf_df.query("type != 'net'")).mark_text(
        dy=10,
        size=10,
        color="white",
        fontWeight="bold"
    ).encode(
        x='label:N',
        y='end:Q',
        text=alt.Text('pct:Q', format=".1%")
    )
    
    st.altair_chart(bars + labels, use_container_width=True)
    
    # Create a summary table to explain the waterfall
    st.subheader("Capital Flow Summary")
    summary_df = pd.DataFrame({
        "Category": waterfall_data["label"],
        "Amount": waterfall_data["value"].map(lambda x: f"${x:,.2f}"),
        "% of Capital": waterfall_data["pct_of_capital"].map(lambda x: f"{x:.2%}")
    })
    
    st.dataframe(summary_df, use_container_width=True, hide_index=True)
    
    # Add explanatory note
    st.caption(
        "**How to read this chart:** The waterfall shows the flow of capital from initial deployment " +
        "to final return. Blue represents initial capital deployed, orange shows fees (which are included " +
        "in the total investment amount), green shows capital returned, and the final bar shows net gain/loss " +
        "(green for gain, red for loss)."
    )

def plot_risk_scatter(risk_df):
    """Create and display a scatter plot of risk factors."""
    if risk_df.empty:
        st.info("No active loans to display in risk assessment chart.")
        return
        
    scatter = alt.Chart(risk_df).mark_circle(size=75).encode(
        x=alt.X(
            "performance_gap:Q", 
            title="Performance Gap (1 - Paid %)",
            axis=alt.Axis(format=".0%", grid=True)
        ),
        y=alt.Y(
            "days_since_funding:Q", 
            title="Days Since Funding",
            axis=alt.Axis(grid=True)
        ),
        size=alt.Size(
            "risk_score:Q", 
            title="Risk Score",
            scale=alt.Scale(domain=[0, 1], range=[20, 300])
        ),
        color=alt.Color(
            "risk_score:Q", 
            scale=alt.Scale(scheme="orangered", domain=[0, 1]), 
            title="Risk Score",
            legend=alt.Legend(format=".1f")
        ),
        tooltip=[
            alt.Tooltip("loan_id:N", title="Loan ID"),
            alt.Tooltip("deal_name:N", title="Deal Name"),
            alt.Tooltip("loan_status:N", title="Status"),
            alt.Tooltip("performance_gap:Q", format=".2f", title="Gap"),
            alt.Tooltip("days_since_funding:Q", title="Age (Days)"),
            alt.Tooltip("risk_score:Q", format=".2f", title="Risk Score"),
            alt.Tooltip("net_balance:Q", format="$,.0f", title="Net Balance")
        ]
    ).properties(
        width=750,
        height=400,
        title={
            "text": "Loan Risk Assessment: Performance Gap vs. Loan Age",
            "subtitle": "Active loans only (Paid Off loans excluded)",
            "fontSize": 16
        }
    )
    
    # Add reference lines at key thresholds
    threshold_x = alt.Chart(pd.DataFrame({"x": [0.10]})).mark_rule(
        strokeDash=[4, 4], 
        color="gray", 
        strokeWidth=1
    ).encode(
        x="x:Q",
        tooltip=alt.Tooltip("x:Q", title="Performance Gap Threshold", format=".0%")
    )
    
    threshold_y = alt.Chart(pd.DataFrame({"y": [90]})).mark_rule(
        strokeDash=[4, 4], 
        color="gray",
        strokeWidth=1
    ).encode(
        y="y:Q",
        tooltip=alt.Tooltip("y:Q", title="Age Threshold (Days)")
    )
    
    # Add annotations
    annotations = pd.DataFrame([
        {"x": 0.05, "y": 45, "text": "Low Risk"},
        {"x": 0.15, "y": 45, "text": "Moderate Risk"},
        {"x": 0.05, "y": 135, "text": "Moderate Risk"},
        {"x": 0.15, "y": 135, "text": "High Risk"}
    ])
    
    text = alt.Chart(annotations).mark_text(
        align="center",
        baseline="middle",
        fontSize=12,
        fontWeight="bold",
        opacity=0.7
    ).encode(
        x="x:Q",
        y="y:Q",
        text="text:N"
    )
    
    st.altair_chart(scatter + threshold_x + threshold_y + text, use_container_width=True)
    
    # Add explanatory text
    st.caption(
        "**Risk Score Calculation:** Risk Score = 70% × Performance Gap + 30% × Age Weight. " +
        "Performance Gap measures how far behind schedule payments are (1 - Payment Performance). " +
        "Age Weight is normalized based on the oldest loan in the portfolio. " +
        "Higher values in either dimension increase risk."
    )

def plot_sector_risk(df):
    """Create and display a bar chart showing industry sector risk."""
    if 'sector_code' not in df.columns or df['sector_code'].isna().all():
        st.warning("Industry sector data not available for risk analysis.")
        return
        
    # Get sector risk data
    sector_risk_df = load_naics_sector_risk()
    
    # Join with loan data
    df_with_risk = df.merge(
        sector_risk_df,
        on='sector_code',
        how='left'
    )
    
    # Summary by sector
    sector_summary = df_with_risk.groupby(['sector_name', 'risk_score']).agg(
        loan_count=('loan_id', 'count'),
        total_deployed=('csl_participation_amount', 'sum'),
        avg_payment_performance=('payment_performance', 'mean')
    ).reset_index()
    
    sector_summary = sector_summary.sort_values('loan_count', ascending=False)
    
    if sector_summary.empty:
        st.info("No sector risk data available to display.")
        return
        
    # Display data table
    st.dataframe(
        sector_summary.rename(columns={
            'sector_name': 'Sector',
            'risk_score': 'Risk Score',
            'loan_count': 'Loan Count',
            'total_deployed': 'Capital Deployed',
            'avg_payment_performance': 'Avg Payment Performance'
        }),
        use_container_width=True,
        column_config={
            "Capital Deployed": st.column_config.NumberColumn(format="$%.0f"),
            "Avg Payment Performance": st.column_config.NumberColumn(format="%.2f")
        }
    )
    
    # Visualization
    chart = alt.Chart(sector_summary).mark_bar().encode(
        x=alt.X('sector_name:N', sort='-y', title="Industry Sector"),
        y=alt.Y('loan_count:Q', title="Number of Loans"),
        color=alt.Color('risk_score:Q', title="Risk Score", scale=alt.Scale(scheme="orangered")),
        tooltip=[
            alt.Tooltip('sector_name:N', title="Sector"),
            alt.Tooltip('loan_count:Q', title="Loans"),
            alt.Tooltip('total_deployed:Q', title="Capital", format="$,.0f"),
            alt.Tooltip('avg_payment_performance:Q', title="Avg Payment Performance", format=".2f")
        ]
    ).properties(
        width=800,
        height=400,
        title="Loan Count by Industry Sector and Risk"
    )
    
    st.altair_chart(chart, use_container_width=True)

# ------------------------------
# Dashboard Sections
# ------------------------------
def display_filters(df):
    """Display and process dashboard filters."""
    # Date filter
    if 'funding_date' in df.columns and not df['funding_date'].isna().all():
        min_date = df["funding_date"].min().date()
        max_date = df["funding_date"].max().date()
        start_date, end_date = st.date_input(
            "Filter by Funding Date",
            [min_date, max_date],
            min_value=min_date,
            max_value=max_date
        )
        filtered_df = df[(df["funding_date"].dt.date >= start_date) & (df["funding_date"].dt.date <= end_date)]
    else:
        filtered_df = df.copy()
    
    # Status filter as radio buttons with "All" option
    all_statuses = sorted(df["loan_status"].unique().tolist())
    status_options = ["All"] + all_statuses
    
    selected_status = st.radio(
        "Filter by Status:", 
        status_options,
        horizontal=True
    )
    
    # Apply status filter
    if selected_status != "All":
        filtered_df = filtered_df[filtered_df["loan_status"] == selected_status]
    
    return filtered_df

def display_portfolio_metrics(df):
    """Display portfolio overview metrics."""
    st.subheader("Portfolio Overview")
    
    # Calculate portfolio metrics
    total_positions = len(df)
    total_paid_off = (df["loan_status"] == "Paid Off").sum()
    total_active = total_positions - total_paid_off
    
    # Financial metrics
    total_capital_deployed = df['csl_participation_amount'].sum()
    total_invested = df['total_invested'].sum()
    total_capital_returned = df['total_paid'].sum()
    net_balance = df['net_balance'].sum()
    
    # Fee metrics
    total_commission_fees = df['commission_fees'].sum()
    total_platform_fees = df['platform_fees'].sum()
    total_bad_debt_allowance = df['bad_debt_allowance'].sum() if 'bad_debt_allowance' in df.columns else 0
    
    # Average metrics
    avg_total_paid = df['total_paid'].mean()
    avg_payment_performance = df['payment_performance'].mean() if 'payment_performance' in df.columns else 0
    
    # Calculate average remaining maturity for active loans
    active_loans_mask = (df['loan_status'] != "Paid Off") & (df['maturity_date'] > pd.Timestamp.today())
    avg_remaining_maturity = df.loc[active_loans_mask, 'remaining_maturity_months'].mean() if not df.loc[active_loans_mask].empty else 0
    
    # First row: Position counts and capital metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Positions", total_positions)
    with col2:
        st.metric("Total Capital Deployed", f"${total_capital_deployed:,.2f}")
    with col3:
        st.metric("Total Capital Returned", f"${total_capital_returned:,.2f}")
    
    # Second row: Fee metrics
    col4, col5, col6 = st.columns(3)
    with col4:
        st.metric("Total Commission Fees", f"${total_commission_fees:,.2f}")
    with col5:
        st.metric("Total Platform Fees", f"${total_platform_fees:,.2f}")
    with col6:
        st.metric("Total Bad Debt Allowance", f"${total_bad_debt_allowance:,.2f}")
    
    # Third row: Average metrics
    col7, col8, col9 = st.columns(3)
    with col7:
        st.metric("Average Total Paid", f"${avg_total_paid:,.2f}")
    with col8:
        # Use tooltip on the metric for better explanation
        st.metric(
            "Average Payment Performance", 
            f"{avg_payment_performance:.2%}", 
            help="Payment Performance measures the ratio of actual payments to expected payments. 100% means payments are on schedule."
        )
    with col9:
        st.metric("Average Remaining Maturity", f"{avg_remaining_maturity:.1f} months")

def display_top_positions(df):
    """Display top outstanding positions."""
    st.subheader("Top 5 Largest Outstanding Positions")
    
    # Filter for unpaid positions
    top_positions = (
        df[df['is_unpaid']]
        .sort_values('net_balance', ascending=False)
        .head(5)
    )
    
    if not top_positions.empty:
        # Calculate total net balance of top 5 positions
        top_5_total_balance = top_positions['net_balance'].sum()
        # Calculate percentage of total net balance
        top_5_pct_of_total = (top_5_total_balance / df['net_balance'].sum() * 100) if df['net_balance'].sum() > 0 else 0
        
        st.caption(f"Total Value: ${top_5_total_balance:,.2f} ({top_5_pct_of_total:.1f}% of total net balance)")
        
        # Format for display
        display_columns = ['loan_id', 'deal_name', 'loan_status', 'total_invested', 'total_paid', 'net_balance', 'remaining_maturity_months']
        column_rename = {
            "loan_id": "Loan ID",
            "deal_name": "Deal Name",
            "loan_status": "Loan Status",
            "total_invested": "Total Invested",
            "total_paid": "Total Paid",
            "net_balance": "Net Balance",
            "remaining_maturity_months": "Months to Maturity"
        }
        
        top_positions_display = format_dataframe_for_display(
            top_positions, 
            columns=display_columns,
            rename_map=column_rename
        )
        
        st.dataframe(
            top_positions_display,
            use_container_width=True,
            hide_index=True
        )
    else:
        st.info("No outstanding positions found with the current filters.")

def display_loan_tape(df):
    """Display the main loan tape table."""
    st.subheader("Loan Tape")
    
    # Select columns for display
    display_columns = ["loan_id"]
    
    # Add deal columns if available
    for col in ["deal_name", "partner_source", "industry"]:
        if col in df.columns:
            display_columns.append(col)
    
    # Add financial columns
    display_columns.extend([
        "loan_status",
        "funding_date",
        "maturity_date",
        "csl_participation_amount",
        "total_invested",
        "total_paid",
        "net_balance",
        "current_roi",
        "participation_percentage",
        "on_time_rate",
        "payment_performance",
        "remaining_maturity_months"
    ])
    
    # Column display names
    column_rename = {
        "loan_id": "Loan ID",
        "deal_name": "Deal Name",
        "partner_source": "Partner Source",
        "industry": "Industry",
        "loan_status": "Loan Status",
        "funding_date": "Funding Date",
        "maturity_date": "Maturity Date",
        "csl_participation_amount": "Capital Deployed",
        "total_invested": "Total Invested",
        "total_paid": "Total Paid",
        "net_balance": "Net Balance",
        "current_roi": "Current ROI",
        "participation_percentage": "Participation Percentage",
        "on_time_rate": "On Time Rate",
        "payment_performance": "Payment Performance",
        "remaining_maturity_months": "Remaining Maturity Months"
    }
    
    # Format for display
    loan_tape = format_dataframe_for_display(
        df,
        columns=display_columns,
        rename_map=column_rename
    )
    
    st.dataframe(
        loan_tape,
        use_container_width=True,
        hide_index=True
    )
    
    # Export functionality
    csv = loan_tape.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Download Loan Tape as CSV",
        data=csv,
        file_name="loan_tape.csv",
        mime="text/csv"
    )

def display_irr_analysis(df):
    """Display IRR analysis for paid-off loans."""
    st.subheader("IRR Analysis for Paid-Off Loans")
    
    # Filter to paid-off loans
    paid_df = df[df['loan_status'] == "Paid Off"].copy()
    
    if paid_df.empty:
        st.info("No paid-off loans to analyze for IRR.")
        return
    
    # Calculate weighted average IRR (weighted by investment amount)
    weighted_realized_irr = (paid_df['realized_irr'] * paid_df['total_invested']).sum() / paid_df['total_invested'].sum()
    weighted_expected_irr = (paid_df['expected_irr'] * paid_df['total_invested']).sum() / paid_df['total_invested'].sum()
    
    # Calculate simple averages for comparison
    avg_realized_irr = paid_df['realized_irr'].mean()
    avg_expected_irr = paid_df['expected_irr'].mean()
    
    # Display metrics with detailed tooltips
    col1, col2 = st.columns(2)
    with col1:
        st.metric(
            "Weighted Average Realized IRR", 
            f"{weighted_realized_irr:.2%}" if pd.notnull(weighted_realized_irr) else "N/A",
            help="Weighted by investment amount to reflect the actual portfolio return"
        )
        st.metric(
            "Simple Average Realized IRR", 
            f"{avg_realized_irr:.2%}" if pd.notnull(avg_realized_irr) else "N/A",
            help="Simple arithmetic mean of all IRRs (gives equal weight to each loan)"
        )
    with col2:
        st.metric(
            "Weighted Average Expected IRR", 
            f"{weighted_expected_irr:.2%}" if pd.notnull(weighted_expected_irr) else "N/A",
            help="Weighted by investment amount based on expected returns at maturity"
        )
        st.metric(
            "Simple Average Expected IRR", 
            f"{avg_expected_irr:.2%}" if pd.notnull(avg_expected_irr) else "N/A",
            help="Simple arithmetic mean of all expected IRRs"
        )
    
    # Display explanation of IRR calculation
    with st.expander("How IRR is Calculated"):
        st.markdown("""
        **Internal Rate of Return (IRR)** is the annualized rate of return that makes the net present value of all cash flows equal to zero.

        For our loan IRR calculations:
        - **Realized IRR** uses actual cash flows: initial investment (negative) and total amount paid back (positive)
        - **Expected IRR** uses initial investment and projected return at maturity
        - Time period is calculated using actual funding date and payoff/maturity date
        - Weighted average weights each loan's IRR by its investment amount
        
        The calculation uses the numpy financial `irr` function with fallback to annualized simple return for extreme values.
        """)
    
    # Display IRR table
    st.subheader("IRR by Loan")
    
    irr_columns = [
        'loan_id', 'deal_name', 'partner_source', 'funding_date', 'payoff_date',
        'total_invested', 'total_paid', 'realized_irr', 'expected_irr'
    ]
    
    column_rename = {
        'loan_id': 'Loan ID',
        'deal_name': 'Deal Name',
        'partner_source': 'Partner Source',
        'funding_date': 'Funding Date',
        'payoff_date': 'Payoff Date',
        'total_invested': 'Total Invested',
        'total_paid': 'Total Paid',
        'realized_irr': 'Realized IRR',
        'expected_irr': 'Expected IRR'
    }
    
    irr_display = format_dataframe_for_display(
        paid_df, 
        columns=irr_columns,
        rename_map=column_rename
    )
    
    st.dataframe(
        irr_display.sort_values(by='Realized IRR', ascending=False),
        use_container_width=True,
        column_config={
            "Realized IRR": st.column_config.NumberColumn(format="%.2%"),
            "Expected IRR": st.column_config.NumberColumn(format="%.2%"),
        }
    )

def display_cohort_analysis(df):
    """Display cohort and vintage analysis."""
    st.subheader("Cohort & Vintage Analysis")
    
    # Create cohort summary
    cohort_summary = df.groupby('cohort').agg(
        loans=('loan_id', 'count'),
        capital_deployed=('csl_participation_amount', 'sum'),
        capital_returned=('total_paid', 'sum'),
        avg_roi=('current_roi', 'mean')
    ).reset_index()
    
    # Format for display
    cohort_display = cohort_summary.copy()
    cohort_display['capital_deployed'] = cohort_display['capital_deployed'].map(lambda x: f"${x:,.2f}")
    cohort_display['capital_returned'] = cohort_display['capital_returned'].map(lambda x: f"${x:,.2f}")
    cohort_display['avg_roi'] = cohort_display['avg_roi'].map(lambda x: f"{x:.2%}")
    
    # Rename columns
    cohort_display.columns = ['Cohort', 'Loans', 'Capital Deployed', 'Capital Returned', 'Avg ROI']
    
    st.dataframe(cohort_display.sort_values('Cohort'), use_container_width=True)

def display_risk_analytics(df):
    """Display risk analytics section."""
    st.header("📈 Portfolio Risk Analytics")
    
    # Calculate risk scores for active loans
    risk_df = calculate_risk_scores(df)
    
    if risk_df.empty:
        st.info("No active loans to display risk analytics.")
        return
    
    # Display top risk loans
    st.subheader("Top 10 Underperforming Loans by Risk Score")
    
    top_risk = risk_df.sort_values("risk_score", ascending=False).head(10)
    
    risk_columns = [
        'loan_id', 'deal_name', 'loan_status', 'funding_date', 'payment_performance',
        'days_since_funding', 'performance_gap', 'risk_score'
    ]
    
    column_rename = {
        'loan_id': 'Loan ID',
        'deal_name': 'Deal Name',
        'loan_status': 'Status',
        'funding_date': 'Funded',
        'payment_performance': 'Payment Performance',
        'days_since_funding': 'Days Since Funding',
        'performance_gap': 'Performance Gap',
        'risk_score': 'Risk Score'
    }
    
    top_risk_display = format_dataframe_for_display(
        top_risk,
        columns=risk_columns,
        rename_map=column_rename
    )
    
    st.dataframe(
        top_risk_display.sort_values("Risk Score", ascending=False),
        use_container_width=True,
        column_config={
            "Payment Performance": st.column_config.NumberColumn(format="%.2f"),
            "Performance Gap": st.column_config.NumberColumn(format="%.2f"),
            "Risk Score": st.column_config.NumberColumn(format="%.3f")
        }
    )
    
    # Loan Status Distribution
    st.subheader("Loan Status Distribution")
    
    status_counts = df["loan_status"].fillna("Unknown").value_counts(normalize=True)
    status_csl_balance = df.groupby("loan_status")["net_balance"].sum()
    
    status_chart_df = pd.DataFrame({
        "Status": status_counts.index,
        "Share": status_counts.values,
        "Net Balance ($)": status_csl_balance.reindex(status_counts.index).fillna(0).values,
        "Count": df["loan_status"].value_counts().values
    })
    
    bar = alt.Chart(status_chart_df).mark_bar().encode(
        x=alt.X("Status:N", title="Loan Status", sort="-y"),
        y=alt.Y("Share:Q", title="% of Loans", axis=alt.Axis(format=".0%")),
        color=alt.Color("Share:Q", scale=alt.Scale(scheme="redyellowgreen"), legend=None),
        tooltip=[
            alt.Tooltip("Status:N"),
            alt.Tooltip("Count:Q", title="Number of Loans"),
            alt.Tooltip("Share:Q", format=".1%", title="Share of Loans"),
            alt.Tooltip("Net Balance ($):Q", format="$,.0f", title="Net Balance")
        ]
    ).properties(
        width=700,
        height=350,
        title={
            "text": "Loan Count by Status",
            "fontSize": 16
        }
    )
    
    # Add count labels
    text = alt.Chart(status_chart_df).mark_text(
        align='center',
        baseline='bottom',
        dy=-5,
        fontSize=12
    ).encode(
        x="Status:N",
        y="Share:Q",
        text=alt.Text("Count:Q", format="d")
    )
    
    st.altair_chart(bar + text, use_container_width=True)
    
    # Risk Score Distribution
    st.subheader("Risk Score Distribution")
    
    band_summary = risk_df.groupby("risk_band").agg(
        loan_count=("loan_id", "count"),
        net_balance=("net_balance", "sum")
    ).reset_index()
    
    if band_summary.empty:
        st.info("No risk band data available.")
        return
        
    risk_bar = alt.Chart(band_summary).mark_bar().encode(
        x=alt.X("risk_band:N", title="Risk Score Band"),
        y=alt.Y("loan_count:Q", title="Loan Count"),
        color=alt.Color(
            "risk_band:N", 
            scale=alt.Scale(
                domain=["0–0.2", "0.2–0.4", "0.4–0.6", "0.6–0.8", "0.8–1.0"],
                range=["#2ca02c", "#98df8a", "#ffbb78", "#ff7f0e", "#d62728"]
            ),
            legend=alt.Legend(title="Risk Band")
        ),
        tooltip=[
            alt.Tooltip("risk_band:N", title="Risk Band"),
            alt.Tooltip("loan_count:Q", title="Loan Count"),
            alt.Tooltip("net_balance:Q", title="Net Balance ($)", format="$,.0f")
        ]
    ).properties(
        width=650,
        height=350,
        title={
            "text": "Loan Count by Risk Score Band",
            "subtitle": "Active loans only (Paid Off loans excluded)",
            "fontSize": 16
        }
    )
    
    st.altair_chart(risk_bar, use_container_width=True)
    
    # Risk scatter plot
    plot_risk_scatter(risk_df)
    
    # Sector risk
    st.subheader("Industry Risk Composition")
    plot_sector_risk(df)

# ------------------------------
# Main Application
# ------------------------------
def main():
    """Main application entry point."""
    st.title("Loan Tape Dashboard")
    
    # Load data
    loans_df = load_loan_summaries()
    deals_df = load_deals()
    
    # Process data
    df = prepare_loan_data(loans_df, deals_df)
    
    # Add IRR calculations
    df = calculate_irr(df)
    
    # Display filters
    filtered_df = display_filters(df)
    
    # Display portfolio summary
    display_portfolio_metrics(filtered_df)
    
    # Display top positions
    display_top_positions(filtered_df)
    
    # Display loan tape
    display_loan_tape(filtered_df)
    
    # Create tabs for additional sections
    tabs = st.tabs(["📊 Visualizations", "📈 Analytics"])
    
    with tabs[0]:
        # Status distribution chart
        if 'loan_status' in filtered_df.columns and not filtered_df['loan_status'].isna().all():
            st.subheader("Distribution of Loan Status")
            plot_status_distribution(filtered_df)
        
        # ROI Distribution Chart
        st.subheader("ROI Distribution by Loan")
        plot_roi_distribution(filtered_df)
        
        # Capital Flow Chart
        st.subheader("Capital Flow: Deployment vs. Returns")
        plot_capital_flow(filtered_df)
        
        # IRR Analysis
        display_irr_analysis(filtered_df)
        
        # IRR by Partner
        st.subheader("Average IRR by Partner")
        plot_irr_by_partner(filtered_df)
        
        # Cohort Analysis
        display_cohort_analysis(filtered_df)
        
        # Cohort Repayment Heatmap
        st.subheader("Cohort Repayment Heatmap (Cumulative % of RTR Paid)")
        plot_repayment_heatmap(filtered_df)
        
        # Capital Waterfall
        st.subheader("Capital Flow Waterfall")
        plot_capital_waterfall(filtered_df)
    
    with tabs[1]:
        # Risk analytics section
        display_risk_analytics(filtered_df)

# Run the application
if __name__ == "__main__":
    main()
