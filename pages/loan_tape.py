# pages/loan_tape.py
"""
Loan Tape Dashboard
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
# Improved color palette for better visual distinction
LOAN_STATUS_COLORS = {
    "Active": "#2ca02c",        # Green
    "Late": "#ffbb78",          # Light orange
    "Default": "#ff7f0e",       # Orange
    "Bankrupt": "#d62728",      # Red
    "Severe": "#990000",        # Dark red
    "Minor Delinquency": "#88c999",  # Light green
    "Moderate Delinquency": "#ffcc88", # Light orange
    "Past Delinquency": "#aaaaaa",   # Gray
    "Severe Delinquency": "#cc4444", # Dark red
    "Active - Frequently Late": "#66aa66" # Medium green
}

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

@st.cache_data(ttl=3600)
def load_loan_schedules():
    """Load loan schedule data from Supabase."""
    res = supabase.table("loan_schedules").select("*").execute()
    return pd.DataFrame(res.data)

@st.cache_data(ttl=3600)
def get_last_updated():
    """Get the most recent update timestamp from all tables."""
    try:
        # Check loan_summaries table
        loans_res = supabase.table("loan_summaries").select("updated_at").order("updated_at", desc=True).limit(1).execute()
        
        # Check deals table  
        deals_res = supabase.table("deals").select("updated_at").order("updated_at", desc=True).limit(1).execute()
        
        # Check loan_schedules table
        schedules_res = supabase.table("loan_schedules").select("updated_at").order("updated_at", desc=True).limit(1).execute()
        
        # Get the most recent timestamp
        timestamps = []
        if loans_res.data:
            timestamps.append(pd.to_datetime(loans_res.data[0]['updated_at']))
        if deals_res.data:
            timestamps.append(pd.to_datetime(deals_res.data[0]['updated_at']))
        if schedules_res.data:
            timestamps.append(pd.to_datetime(schedules_res.data[0]['updated_at']))
            
        if timestamps:
            last_updated = max(timestamps)
            return last_updated.strftime('%B %d, %Y at %I:%M %p')
    except:
        return "Unknown"
    
    return "Unknown"

# ------------------------------
# Data Processing Functions
# ------------------------------
def prepare_loan_data(loans_df, deals_df):
    """
    Merge and prepare loan and deal data.
    """
    # Merge dataframes if data exists
    if not loans_df.empty and not deals_df.empty:
        df = loans_df.merge(
            deals_df[["loan_id", "deal_name", "partner_source", "industry", "commission","fico", "tib"]], 
            on="loan_id", 
            how="left"
        )
    else:
        df = loans_df.copy()
    
    # Convert dates - handle timezone issues
    for date_col in ["funding_date", "maturity_date", "payoff_date"]:
        if date_col in df.columns:
            try:
                # Convert to datetime and handle timezone consistently
                df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
            except Exception as e:
                st.warning(f"Error converting {date_col}: {str(e)}")
                df[date_col] = pd.NaT
    
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
    
    # Calculate days since funding with timezone handling
    try:
        today = pd.Timestamp.today().tz_localize(None)
        df["days_since_funding"] = df["funding_date"].apply(
            lambda x: (today - pd.to_datetime(x).tz_localize(None)).days if pd.notnull(x) else 0
        )
    except Exception as e:
        st.warning(f"Error calculating days since funding: {str(e)}")
        df["days_since_funding"] = 0
    
    # Calculate remaining maturity in months (only for active loans)
    df["remaining_maturity_months"] = 0.0
    
    try:
        active_loans_mask = (df['loan_status'] != "Paid Off") & (df['maturity_date'] > pd.Timestamp.today())
        if 'maturity_date' in df.columns:
            today = pd.Timestamp.today().tz_localize(None)
            df.loc[active_loans_mask, "remaining_maturity_months"] = df.loc[active_loans_mask, 'maturity_date'].apply(
                lambda x: (pd.to_datetime(x).tz_localize(None) - today).days / 30 if pd.notnull(x) else 0
            )
    except Exception as e:
        st.warning(f"Error calculating remaining maturity: {str(e)}")
    
    # Add cohort information
    try:
        df['cohort'] = df['funding_date'].dt.to_period('Q').astype(str)
        df['funding_month'] = df['funding_date'].dt.to_period('M')
    except Exception as e:
        st.warning(f"Error calculating cohort information: {str(e)}")
        df['cohort'] = 'Unknown'
        df['funding_month'] = pd.NaT
    
    # Extract NAICS sector if industry data exists
    if 'industry' in df.columns:
        df['sector_code'] = df['industry'].astype(str).str[:2]
    
    return df

def calculate_irr(df):
    """
    Calculate IRR metrics for loans.
    """
    result_df = df.copy()
    
    # Calculate Realized IRR for paid-off loans using the actual payoff date
    def calc_realized_irr(row):
        if pd.isna(row['funding_date']) or pd.isna(row['payoff_date']) or row['total_invested'] <= 0:
            return None
            
        # Calculate days between funding and payoff - ensure consistent timezone handling
        try:
            # Convert to timezone-naive dates for consistent comparison
            funding_date = pd.to_datetime(row['funding_date']).tz_localize(None)
            payoff_date = pd.to_datetime(row['payoff_date']).tz_localize(None)
            
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
        except Exception as e:
            # Handle any errors in date conversion or comparison
            st.warning(f"Error calculating IRR: {str(e)}")
            return None
    
    # Calculate Expected IRR based on original maturity date and projected payment
    def calc_expected_irr(row):
        if pd.isna(row['funding_date']) or pd.isna(row['maturity_date']) or row['total_invested'] <= 0:
            return None
            
        # For paid off loans, we still calculate based on original maturity date
        try:
            # Calculate days between funding and original maturity date
            funding_date = pd.to_datetime(row['funding_date']).tz_localize(None)
            maturity_date = pd.to_datetime(row['maturity_date']).tz_localize(None)
            
            if maturity_date <= funding_date:
                # Can't calculate IRR if dates are invalid
                return None
                
            # Expected payment at original maturity (total RTR)
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
        except Exception as e:
            # Handle any errors in date conversion or comparison
            st.warning(f"Error calculating expected IRR: {str(e)}")
            return None
    
    # Apply IRR calculations with error handling
    try:
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
    except Exception as e:
        st.error(f"Error in IRR calculations: {str(e)}")
        # Add empty columns to avoid errors downstream
        result_df['realized_irr'] = None
        result_df['expected_irr'] = None
        result_df['realized_irr_pct'] = "N/A"
        result_df['expected_irr_pct'] = "N/A"
    
    return result_df

def calculate_risk_scores(df):
    """
    Calculate risk scores for active loans.
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

def calculate_expected_payment_to_date(row):
    """
    Calculate how much should have been paid by now based on:
    - Time elapsed since funding
    - Expected maturity date
    - Total expected RTR
    
    This assumes linear payment over time.
    """
    if pd.isna(row['funding_date']) or pd.isna(row['maturity_date']) or pd.isna(row['our_rtr']):
        return 0
        
    try:
        # Convert to timezone-naive dates for consistent comparison
        funding_date = pd.to_datetime(row['funding_date']).tz_localize(None)
        maturity_date = pd.to_datetime(row['maturity_date']).tz_localize(None)
        current_date = pd.Timestamp.today().tz_localize(None)
        
        # If loan is past maturity, expected payment is full RTR
        if current_date >= maturity_date:
            return row['our_rtr']
            
        # Calculate expected payment based on time elapsed
        total_days = (maturity_date - funding_date).days
        days_elapsed = (current_date - funding_date).days
        
        if total_days <= 0:
            return 0
            
        # Calculate expected percentage completion
        expected_pct = min(1.0, max(0.0, days_elapsed / total_days))
        
        # Calculate expected payment
        expected_payment = row['our_rtr'] * expected_pct
        
        return expected_payment
    except Exception as e:
        # Handle any timezone or conversion errors
        st.warning(f"Error calculating expected payment: {str(e)}")
        return 0

# ------------------------------
# Visualization Functions
# ------------------------------
def format_dataframe_for_display(df, columns=None, rename_map=None):
    """
    Format DataFrame for display by renaming columns and formatting values.
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
        elif any(term in col for term in ["Maturity", "Months"]):
            display_df[col] = display_df[col].map(lambda x: f"{x:.1f}" if pd.notnull(x) else "")
        elif any(term in col for term in ["Capital", "Invested", "Paid", "Balance", "Fees"]):
            display_df[col] = display_df[col].map(lambda x: f"${x:,.2f}" if pd.notnull(x) else "")
    
    # Format date columns safely
    try:
        # Get column mapping for safer column lookup
        if rename_map:
            # Create reverse mapping (display name -> original name)
            reverse_map = {v: k for k, v in rename_map.items() if k in df.columns}
        else:
            reverse_map = {}
            
        for col in display_df.columns:
            # Check if column looks like a date column
            if any(term in col for term in ["Date", "Funding", "Maturity"]):
                # Try to get the original column name
                original_col = reverse_map.get(col, col.replace(" ", "_").lower())
                
                # Check if original column exists in source dataframe
                if original_col in df.columns and pd.api.types.is_datetime64_dtype(df[original_col]):
                    # Format as date
                    display_df[col] = pd.to_datetime(df[original_col]).dt.strftime('%Y-%m-%d')
    except Exception as e:
        st.warning(f"Error formatting date columns: {str(e)}")
    
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
    
    # Add color to the DataFrame - using the improved color scheme
    status_summary["color"] = status_summary["status"].apply(
        lambda x: LOAN_STATUS_COLORS.get(x, "#808080")  # Default to gray if status not in mapping
    )
    
    # Create a lookup table of loan IDs grouped by status
    loan_ids_by_status = {}
    for status in active_df["loan_status"].unique():
        loans = active_df[active_df["loan_status"] == status]["loan_id"].tolist()
        loan_ids_by_status[status] = ", ".join(loans)
    
    # Add loan IDs to status summary
    status_summary["loan_ids"] = status_summary["status"].map(loan_ids_by_status)
    
    # Add note about excluding Paid Off loans
    st.caption("Note: 'Paid Off' loans are excluded from this chart")
    
    # Create pie chart with custom colors and loan IDs in tooltip
    pie_chart = alt.Chart(status_summary).mark_arc().encode(
        theta=alt.Theta(field="percentage", type="quantitative"),
        color=alt.Color(
            "status:N", 
            scale=alt.Scale(
                domain=list(status_summary["status"]),
                range=list(status_summary["color"])
            ),
            legend=alt.Legend(title="Loan Status", orient="right")
        ),
        tooltip=[
            alt.Tooltip("status:N", title="Loan Status"),
            alt.Tooltip("count:Q", title="Number of Loans"),
            alt.Tooltip("percentage:Q", title="% of Active Loans", format=".1%"),
            alt.Tooltip("balance:Q", title="Net Balance", format="$,.0f"),
            alt.Tooltip("loan_ids:N", title="Loan IDs")
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
    Create and display a line chart showing capital deployment vs returns over time,
    with a table showing days between significant milestones.
    """
    # Get loan schedules for actual payment dates
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
    if deploy_df.empty:
        st.info("Insufficient data to display capital flow chart.")
        return
    
    # SIMPLIFIED APPROACH: Create a single chart with normalized values
    # Create a combined dataset with both series
    combined_data = []
    
    # Add deployment data
    for _, row in deploy_df.iterrows():
        combined_data.append({
            'date': row['funding_date'],
            'amount': row['capital_deployed'],
            'series': 'Capital Deployed'
        })
    
    # Add return data if available
    if not return_df.empty:
        for _, row in return_df.iterrows():
            combined_data.append({
                'date': row['payment_date'],
                'amount': row['capital_returned'],
                'series': 'Capital Returned'
            })
    
    # Create dataframe
    combined_df = pd.DataFrame(combined_data)
    
    # Create a single chart with one y-axis
    chart = alt.Chart(combined_df).mark_line().encode(
        x=alt.X('date:T', title="Date", axis=alt.Axis(format="%b %Y")),
        y=alt.Y('amount:Q', title="Amount ($)", axis=alt.Axis(format="$,.0f")),
        color=alt.Color(
            'series:N', 
            scale=alt.Scale(
                domain=['Capital Deployed', 'Capital Returned'],
                range=['red', 'green']
            ),
            legend=alt.Legend(title="Capital Flow")
        ),
        tooltip=[
            alt.Tooltip('date:T', title="Date", format="%Y-%m-%d"),
            alt.Tooltip('amount:Q', title="Amount", format="$,.0f"),
            alt.Tooltip('series:N', title="Type")
        ]
    ).properties(
        width=800,
        height=400,
        title="Capital Deployed vs. Capital Returned Over Time"
    )
    
    # Add milestone markers
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
        # Convert milestone data to the same format
        milestone_points_data = []
        for _, row in milestone_df.iterrows():
            milestone_points_data.append({
                'date': row['funding_date'],
                'amount': row['capital_deployed'],
                'milestone': row['milestone']
            })
        
        milestone_points_df = pd.DataFrame(milestone_points_data)
        
        # Create milestone points
        milestone_layer = alt.Chart(milestone_points_df).mark_circle(size=80).encode(
            x='date:T',
            y='amount:Q',
            color=alt.value('red'),
            tooltip=[
                alt.Tooltip('milestone:N', title="Milestone"),
                alt.Tooltip('date:T', title="Date Reached", format="%Y-%m-%d")
            ]
        )
        
        # Combine charts
        chart = alt.layer(chart, milestone_layer)
    
    # Display chart
    st.altair_chart(chart, use_container_width=True)
    
    # NEW: Add milestone days table
    if not milestone_df.empty and len(milestone_df) > 1:
        st.subheader("Days Between Capital Milestones")
        
        # Sort milestone dataframe by date
        milestone_df_sorted = milestone_df.sort_values('funding_date')
        
        # Calculate days between consecutive milestones
        milestone_days = []
        prev_date = None
        prev_milestone = None
        
        for _, row in milestone_df_sorted.iterrows():
            current_date = row['funding_date']
            current_milestone = row['milestone']
            
            if prev_date is not None:
                days_between = (current_date - prev_date).days
                milestone_days.append({
                    'From': prev_milestone,
                    'To': current_milestone,
                    'Start Date': prev_date.strftime('%Y-%m-%d'),
                    'End Date': current_date.strftime('%Y-%m-%d'),
                    'Days': days_between
                })
            
            prev_date = current_date
            prev_milestone = current_milestone
        
        # Create and display milestone days table
        milestone_days_df = pd.DataFrame(milestone_days)
        st.table(milestone_days_df)
    
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
    """
    try:
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
            {"label": "Capital Deployed", "value": total_capital_deployed, "category": "start"},
            {"label": "Platform Fees", "value": platform_fees, "category": "info"},
            {"label": "Commission Fees", "value": commission_fees, "category": "info"},
            {"label": "Bad Debt Allowance", "value": bad_debt_allowance, "category": "info"},
            {"label": "Capital Returned", "value": capital_returned, "category": "end"},
            {"label": "Net Position", "value": net_position, "category": "net"}
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
            if row['category'] == "start":
                # Capital deployed (starting point)
                cumulative.append({
                    "label": row['label'],
                    "start": 0,
                    "end": row['value'],
                    "color": "#1f77b4",  # Blue
                    "value": row['value'],
                    "pct": row['pct_of_capital'],
                    "category": row['category']
                })
            elif row['category'] == "info":
                # Fees (informational only, doesn't affect waterfall)
                cumulative.append({
                    "label": row['label'],
                    "start": 0,
                    "end": row['value'],
                    "color": "#ff7f0e",  # Orange
                    "value": row['value'],
                    "pct": row['pct_of_capital'],
                    "category": row['category']
                })
            elif row['category'] == "end":
                # Capital returned 
                cumulative.append({
                    "label": row['label'],
                    "start": 0,
                    "end": row['value'],
                    "color": "#2ca02c",  # Green
                    "value": row['value'],
                    "pct": row['pct_of_capital'] if total_capital_deployed > 0 else 0,
                    "category": row['category']
                })
            elif row['category'] == "net":
                # Net position (gain or loss)
                cumulative.append({
                    "label": row['label'],
                    "start": total_capital_deployed,
                    "end": row['running_total'],
                    "color": "#2ca02c" if row['value'] >= 0 else "#d62728",  # Green if gain, red if loss
                    "value": row['value'],
                    "pct": row['pct_of_capital'],
                    "category": row['category']
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
        
        # Add percentage labels for fees and returns, but only for non-net categories
        # Using .transform_filter instead of .query to avoid pandas query errors
        pct_labels = alt.Chart(wf_df).transform_filter(
            alt.datum.category != 'net'
        ).mark_text(
            dy=10,
            size=10,
            color="white",
            fontWeight="bold"
        ).encode(
            x='label:N',
            y='end:Q',
            text=alt.Text('pct:Q', format=".1%")
        )
        
        st.altair_chart(bars + labels + pct_labels, use_container_width=True)
        
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
    except Exception as e:
        st.error(f"Error creating capital waterfall chart: {str(e)}")
        st.info("Unable to generate capital waterfall chart due to an error in data processing.")

def plot_investment_net_position(df):
    """
    Create a chart showing the net investment position over time (total invested - total returned).
    This represents the cumulative cash flow position of the portfolio.
    """
    st.subheader("Net Investment Position Over Time", 
                help="Shows the running total of capital deployed minus capital returned over time.")
    
    # Get loan schedules for actual payment dates
    loan_schedules = load_loan_schedules()
    
    # Convert funding_date to datetime if not already
    df['funding_date'] = pd.to_datetime(df['funding_date'], errors='coerce').dt.tz_localize(None)
    
    # Create daily timeline from first funding to today
    min_date = df['funding_date'].min()
    max_date = pd.Timestamp.today()
    
    if min_date is pd.NaT or max_date is pd.NaT:
        st.warning("Invalid date range for net position analysis.")
        return
        
    # Create date range for analysis
    date_range = pd.date_range(start=min_date, end=max_date, freq='D')
    position_df = pd.DataFrame({'date': date_range})
    
    # Capital deployment events
    deployments = df[['funding_date', 'csl_participation_amount']].copy()
    deployments.rename(columns={'funding_date': 'date', 'csl_participation_amount': 'amount'}, inplace=True)
    deployments['type'] = 'Investment'
    
    # Capital return events
    if not loan_schedules.empty and 'payment_date' in loan_schedules.columns:
        loan_schedules['payment_date'] = pd.to_datetime(loan_schedules['payment_date'], errors='coerce').dt.tz_localize(None)
        returns = loan_schedules[loan_schedules['actual_payment'] > 0].copy()
        returns = returns[['payment_date', 'actual_payment']].dropna()
        returns.rename(columns={'payment_date': 'date', 'actual_payment': 'amount'}, inplace=True)
        returns['type'] = 'Return'
    else:
        returns = pd.DataFrame(columns=['date', 'amount', 'type'])
    
    # Combine all cash flow events
    all_events = pd.concat([
        deployments,
        returns
    ]).sort_values('date')
    
    # Calculate daily cash flows
    daily_flows = all_events.groupby(['date', 'type'])['amount'].sum().reset_index()
    
    # Pivot to get investments and returns in separate columns
    daily_pivot = daily_flows.pivot_table(
        index='date', 
        columns='type', 
        values='amount',
        fill_value=0
    ).reset_index()
    
    # Ensure both columns exist
    if 'Investment' not in daily_pivot.columns:
        daily_pivot['Investment'] = 0
    if 'Return' not in daily_pivot.columns:
        daily_pivot['Return'] = 0
    
    # Calculate cumulative sums
    daily_pivot['cum_investment'] = daily_pivot['Investment'].cumsum()
    daily_pivot['cum_return'] = daily_pivot['Return'].cumsum()
    daily_pivot['net_position'] = daily_pivot['cum_investment'] - daily_pivot['cum_return']
    
    # Merge with the full date range to get a complete timeline
    position_timeline = position_df.merge(
        daily_pivot[['date', 'cum_investment', 'cum_return', 'net_position']],
        on='date',
        how='left'
    )
    
    # Forward fill to get position on days without transactions
    position_timeline = position_timeline.fillna(method='ffill')
    position_timeline = position_timeline.fillna(0)  # Fill any remaining NAs with zeros
    
    # Create the chart
    # Create data in long format for area chart
    chart_data = []
    for _, row in position_timeline.iterrows():
        chart_data.append({
            'Date': row['date'],
            'Amount': row['cum_investment'],
            'Type': 'Total Invested'
        })
        chart_data.append({
            'Date': row['date'],
            'Amount': row['cum_return'],
            'Type': 'Total Returned'
        })
        chart_data.append({
            'Date': row['date'],
            'Amount': row['net_position'],
            'Type': 'Net Position'
        })
    
    chart_df = pd.DataFrame(chart_data)
    
    # Create a stacked area chart
    area_chart = alt.Chart(chart_df).mark_line().encode(
        x=alt.X('Date:T', title='Date'),
        y=alt.Y('Amount:Q', title='Amount ($)', axis=alt.Axis(format='$,.0f')),
        color=alt.Color(
            'Type:N',
            scale=alt.Scale(
                domain=['Total Invested', 'Total Returned', 'Net Position'],
                range=['#ff7f0e', '#2ca02c', '#1f77b4']
            )
        ),
        tooltip=[
            alt.Tooltip('Date:T', title='Date', format='%Y-%m-%d'),
            alt.Tooltip('Amount:Q', title='Amount', format='$,.2f'),
            alt.Tooltip('Type:N', title='Metric')
        ]
    ).properties(
        width=800,
        height=500,
        title='Portfolio Net Position Over Time'
    )
    
    # Add reference line at $0 (break-even)
    zero_line = alt.Chart(pd.DataFrame({'y': [0]})).mark_rule(
        strokeDash=[2, 2],
        color='gray',
        strokeWidth=1
    ).encode(y='y:Q')
    
    st.altair_chart(area_chart + zero_line, use_container_width=True)
    
    # Add explanation
    st.caption(
        "**Net Investment Position:** This chart shows the cumulative investment (orange), " +
        "cumulative returns (green), and net position (blue) over time. The net position represents " +
        "the capital still deployed (positive values) or the profit after full recovery (negative values)."
    )
def plot_risk_scatter(risk_df, avg_payment_performance=0.75):
    """
    Create and display a scatter plot of risk factors.
    """
    # Filter out 'Paid Off' loans
    risk_df = risk_df[risk_df['loan_status'] != 'Paid Off'].copy()

    if risk_df.empty:
        st.info("No active loans to display in risk assessment chart.")
        return
    
    # Calculate average performance gap
    avg_performance_gap = 1 - avg_payment_performance
    
    # Use average performance gap (or fallback to 0.10) for horizontal threshold
    perf_threshold = max(0.05, min(0.25, avg_performance_gap))
    
    # Default age threshold is 90 days, but adjust based on data
    age_threshold = 90
    if not risk_df['days_since_funding'].empty:
        median_age = risk_df['days_since_funding'].median()
        age_threshold = max(60, min(120, median_age))

    st.subheader("Loan Risk Assessment", 
               help="Risk Score = 70% × Performance Gap + 30% × Age Weight. Performance Gap measures how far behind schedule payments are (1 - Payment Performance). Age Weight is normalized based on the oldest loan in the portfolio.")
    
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
            "text": f"Loan Risk Assessment: Performance Gap vs. Loan Age",
            "subtitle": f"Thresholds: Performance Gap {perf_threshold:.0%}, Age {age_threshold} days (based on portfolio averages)",
            "fontSize": 16
        }
    )
    
    # Add reference lines at key thresholds
    threshold_x = alt.Chart(pd.DataFrame({"x": [perf_threshold]})).mark_rule(
        strokeDash=[4, 4], 
        color="gray", 
        strokeWidth=1
    ).encode(
        x="x:Q",
        tooltip=alt.Tooltip("x:Q", title="Performance Gap Threshold", format=".0%")
    )
    
    threshold_y = alt.Chart(pd.DataFrame({"y": [age_threshold]})).mark_rule(
        strokeDash=[4, 4], 
        color="gray",
        strokeWidth=1
    ).encode(
        y="y:Q",
        tooltip=alt.Tooltip("y:Q", title="Age Threshold (Days)")
    )
    
    # Add annotations - adjusted based on thresholds
    annotations = pd.DataFrame([
        {"x": perf_threshold/2, "y": age_threshold/2, "text": "Low Risk"},
        {"x": perf_threshold*1.5, "y": age_threshold/2, "text": "Moderate Risk"},
        {"x": perf_threshold/2, "y": age_threshold*1.5, "text": "Moderate Risk"},
        {"x": perf_threshold*1.5, "y": age_threshold*1.5, "text": "High Risk"}
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
        "Quadrants are defined by the portfolio's average payment performance and loan age. " +
        "Higher values in either dimension increase risk."
    )

def plot_sector_risk_by_dollars(df):
    """
    Create and display visualizations of industry risk by dollar amount of outstanding principal.
    """
    try:
        # Only include active loans
        active_df = df[df['loan_status'] != 'Paid Off'].copy()
        
        if 'industry' not in active_df.columns or active_df['industry'].isna().all():
            st.warning("Industry data not available for sector risk analysis.")
            return
            
        # Get sector risk data
        sector_risk_df = load_naics_sector_risk()
        
        # Join with loan data
        df_with_risk = active_df.merge(
            sector_risk_df,
            on='sector_code',
            how='left'
        )
        
        # Summary by sector
        sector_summary = df_with_risk.groupby(['sector_name', 'risk_score']).agg(
            loan_count=('loan_id', 'count'),
            total_deployed=('csl_participation_amount', 'sum'),
            net_balance=('net_balance', 'sum'),
            avg_payment_performance=('payment_performance', 'mean')
        ).reset_index()
        
        sector_summary = sector_summary.sort_values('net_balance', ascending=False)
        
        if sector_summary.empty:
            st.info("No sector data available with sufficient risk information.")
            return
            
        # Display data table with both loan count and dollar exposure
        st.subheader("Industry Exposure by Dollar Amount")
        
        # Format for display
        display_df = sector_summary.copy()
        display_df['total_deployed'] = display_df['total_deployed'].map(lambda x: f"${x:,.0f}")
        display_df['net_balance'] = display_df['net_balance'].map(lambda x: f"${x:,.0f}")
        display_df['avg_payment_performance'] = display_df['avg_payment_performance'].map(lambda x: f"{x:.2%}")
        
        # Rename columns
        display_df.columns = [
            'Industry Sector', 'Risk Score', 'Loan Count', 
            'Capital Deployed', 'Outstanding Balance', 'Avg Payment Performance'
        ]
        
        # Only show top 5 sectors by outstanding balance for focused analysis
        top_5_sectors = display_df.head(5)
        st.dataframe(top_5_sectors, use_container_width=True)
        
        # NEW: Create scatter plot showing capital at risk vs risk score
        # Size of dots representing number of deals
        st.subheader("Industry Risk Analysis: Outstanding Balance vs. Risk Score")
        
        scatter_data = sector_summary.copy()
        scatter_data['sector_name_count'] = scatter_data['sector_name'] + ' (' + scatter_data['loan_count'].astype(str) + ' loans)'
        
        risk_scatter = alt.Chart(scatter_data).mark_circle().encode(
            x=alt.X('risk_score:Q', title='Risk Score', axis=alt.Axis(format='.1f')),
            y=alt.Y('net_balance:Q', title='Outstanding Balance ($)', axis=alt.Axis(format='$,.0f')),
            size=alt.Size('loan_count:Q', 
                title='Number of Loans',
                scale=alt.Scale(range=[50, 500])
            ),
            color=alt.Color('risk_score:Q', 
                title='Risk Score',
                scale=alt.Scale(scheme='orangered', domain=[0, 1])
            ),
            tooltip=[
                alt.Tooltip('sector_name:N', title='Industry Sector'),
                alt.Tooltip('risk_score:Q', title='Risk Score', format='.2f'),
                alt.Tooltip('loan_count:Q', title='Number of Loans'),
                alt.Tooltip('net_balance:Q', title='Outstanding Balance', format='$,.0f'),
                alt.Tooltip('avg_payment_performance:Q', title='Avg Payment Performance', format='.2%')
            ]
        ).properties(
            width=800,
            height=500,
            title={
                "text": "Industry Risk Analysis",
                "subtitle": "Bubble size indicates number of loans in each sector",
                "fontSize": 16
            }
        )
        
        # Add labels for the bubbles
        text = alt.Chart(scatter_data).mark_text(
            align='center',
            baseline='middle',
            fontSize=12,
            dy=-15
        ).encode(
            x='risk_score:Q',
            y='net_balance:Q',
            text='sector_name:N'
        )
        
        st.altair_chart(risk_scatter + text, use_container_width=True)
        
        # Create chart showing dollar exposure by industry
        chart = alt.Chart(sector_summary).mark_bar().encode(
            x=alt.X('net_balance:Q', title="Outstanding Balance ($)", axis=alt.Axis(format="$,.0f")),
            y=alt.Y('sector_name:N', title="Industry Sector", sort='-x'),
            color=alt.Color('risk_score:Q', 
                title="Risk Score", 
                scale=alt.Scale(scheme="orangered"),
                legend=alt.Legend(format=".0f")
            ),
            tooltip=[
                alt.Tooltip('sector_name:N', title="Industry Sector"),
                alt.Tooltip('loan_count:Q', title="Loans"),
                alt.Tooltip('net_balance:Q', title="Outstanding Balance", format="$,.0f"),
                alt.Tooltip('total_deployed:Q', title="Capital Deployed", format="$,.0f"),
                alt.Tooltip('avg_payment_performance:Q', title="Avg Payment Performance", format=".2%")
            ]
        ).properties(
            width=800,
            height=400,
            title={
                "text": "Outstanding Balance by Industry Sector",
                "subtitle": "Color indicates risk level of each sector",
                "fontSize": 16
            }
        )
        
        # Add count labels
        text = alt.Chart(sector_summary).mark_text(
            align='left',
            baseline='middle',
            dx=5,
            fontSize=10
        ).encode(
            x='net_balance:Q',
            y='sector_name:N',
            text=alt.Text('loan_count:Q', format='d', title="Loan Count")
        )

        st.altair_chart(chart + text, use_container_width=True)
        
    except Exception as e:
        st.error(f"Error creating industry risk visualization: {str(e)}")
        st.info("Unable to generate industry risk chart due to an error in data processing.")

def display_capital_at_risk(df):
    """
    Display the Capital at Risk analysis section.
    """
    st.header("Capital at Risk Analysis")
    
    # Filter out paid-off loans
    active_df = df[df['loan_status'] != "Paid Off"].copy()
    
    if active_df.empty:
        st.info("No active loans to analyze for capital at risk.")
        return
    
    # Calculate remaining principal by loan status
    status_principal = active_df.groupby('loan_status').agg(
        remaining_principal=('csl_participation_amount', 'sum'),
        total_paid=('total_paid', 'sum'),
        count=('loan_id', 'count')
    ).reset_index()
    
    # Calculate principal recovery percentage
    status_principal['recovery_pct'] = status_principal['total_paid'] / status_principal['remaining_principal']
    
    # Calculate total remaining principal
    total_principal = status_principal['remaining_principal'].sum()
    total_recovery = status_principal['total_paid'].sum()
    overall_recovery_pct = total_recovery / total_principal if total_principal > 0 else 0
    
    # Display metrics with tooltips
    st.subheader("Remaining Principal Exposure")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric(
            "Total Principal at Risk", 
            f"${total_principal:,.2f}",
            help="Total principal amount for all active loans (excluding paid off loans)"
        )
    with col2:
        st.metric(
            "Total Recovery to Date", 
            f"${total_recovery:,.2f}",
            help="Total amount recovered so far from active loans"
        )
    with col3:
        st.metric(
            "Overall Recovery Percentage", 
            f"{overall_recovery_pct:.2%}",
            help="Percentage of principal that has been recovered from active loans"
        )
        
    # Create stacked bar chart of principal by status
    st.subheader("Principal at Risk by Loan Status")
    
    # Plot principal at risk by status
    principal_chart = alt.Chart(status_principal).mark_bar().encode(
        x=alt.X('loan_status:N', title="Loan Status", axis=alt.Axis(labelAngle=-45)),
        y=alt.Y('remaining_principal:Q', title="Principal Amount ($)", axis=alt.Axis(format="$,.0f")),
        color=alt.Color('loan_status:N', 
            scale=alt.Scale(
                domain=list(status_principal['loan_status']),
                range=[LOAN_STATUS_COLORS.get(status, "#808080") for status in status_principal['loan_status']]
            ),
            legend=alt.Legend(title="Loan Status")
        ),
        tooltip=[
            alt.Tooltip('loan_status:N', title="Status"),
            alt.Tooltip('remaining_principal:Q', title="Principal Amount", format="$,.2f"),
            alt.Tooltip('count:Q', title="Number of Loans"),
            alt.Tooltip('recovery_pct:Q', title="Recovery Percentage", format=".1%")
        ]
    ).properties(
        width=600,
        height=400,
        title="Principal Balance Remaining by Loan Status"
    )
    
    st.altair_chart(principal_chart, use_container_width=True)
    
    # Expected vs Actual Return Analysis  
    # Information tooltip for expected vs actual calculation
    st.subheader("Expected vs. Actual Return Analysis", 
            help="Expected payments are calculated based on linear payment over time from funding date to maturity date. Actual payments are the amount received to date.")
    
    # Calculate expected and actual RTR and differences
    active_df['expected_rtr'] = active_df['our_rtr']
    active_df['actual_paid'] = active_df['total_paid']
    active_df['expected_paid_to_date'] = active_df.apply(
        lambda x: calculate_expected_payment_to_date(x), axis=1
    )
    active_df['payment_difference'] = active_df['actual_paid'] - active_df['expected_paid_to_date']
    
    # Handle division by zero for difference percentage
    active_df['difference_pct'] = 0.0
    mask = active_df['expected_paid_to_date'] > 0
    active_df.loc[mask, 'difference_pct'] = active_df.loc[mask, 'payment_difference'] / active_df.loc[mask, 'expected_paid_to_date']
    
    # Aggregate for visualization
    payment_summary = pd.DataFrame({
        'Category': ['Expected Payment to Date', 'Actual Payment to Date', 'Difference'],
        'Amount': [
            active_df['expected_paid_to_date'].sum(),
            active_df['actual_paid'].sum(),
            active_df['payment_difference'].sum()
        ]
    })
    
    # Add percentage of difference
    expected_total = payment_summary.loc[0, 'Amount']
    difference = payment_summary.loc[2, 'Amount']
    difference_pct = difference / expected_total if expected_total > 0 else 0
    
    # Display metrics for expected vs actual - with tooltips
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric(
            "Expected Payment to Date", 
            f"${expected_total:,.2f}",
            help="Total amount expected to be paid by now based on loan terms and elapsed time"
        )
    with col2:
        st.metric(
            "Actual Payment to Date", 
            f"${payment_summary.loc[1, 'Amount']:,.2f}",
            help="Total amount actually received from borrowers"
        )
    with col3:
        st.metric(
            "Difference", 
            f"${difference:,.2f}", 
            delta=f"{difference_pct:.2%}",
            delta_color="normal",
            help="Difference between actual and expected payments (positive means ahead of schedule)"
        )
    
    # Create section for visualizations
    st.markdown("#### Payment Performance Visualization")
    
    # Create bar chart for expected vs actual
    bar_chart = alt.Chart(payment_summary[:2]).mark_bar().encode(
        x=alt.X('Category:N', title=None),
        y=alt.Y('Amount:Q', title="Amount ($)", axis=alt.Axis(format="$,.0f")),
        color=alt.Color('Category:N', 
            scale=alt.Scale(
                domain=['Expected Payment to Date', 'Actual Payment to Date'],
                range=['#1f77b4', '#2ca02c']
            ),
            legend=alt.Legend(title="Payment Type")
        ),
        tooltip=[
            alt.Tooltip('Category:N', title="Type"),
            alt.Tooltip('Amount:Q', title="Amount", format="$,.2f")
        ]
    ).properties(
        width=400,
        height=300,
        title="Expected vs. Actual Payments to Date"
    )
    
    # Distribution of payment differences by loan
    active_df['payment_diff_category'] = pd.cut(
        active_df['difference_pct'],
        bins=[-float('inf'), -0.2, -0.1, -0.05, 0.05, 0.1, 0.2, float('inf')],
        labels=[
            "Severely Underperforming (<-20%)", 
            "Underperforming (-20% to -10%)",
            "Slightly Underperforming (-10% to -5%)",
            "On Target (-5% to +5%)",
            "Slightly Overperforming (5% to 10%)",
            "Overperforming (10% to 20%)",
            "Strongly Overperforming (>20%)"
        ]
    )
    
    diff_distribution = active_df.groupby('payment_diff_category').agg(
        loan_count=('loan_id', 'count'),
        total_difference=('payment_difference', 'sum'),
        avg_difference_pct=('difference_pct', 'mean')
    ).reset_index()

    loan_ids_by_category = {}
    for category in active_df["payment_diff_category"].unique():
        if pd.notna(category):  # Ensure we're not processing NaN categories
            loans = active_df[active_df["payment_diff_category"] == category]["loan_id"].tolist()
            loan_ids_by_category[category] = ", ".join(loans)
    
    # Add loan IDs to diff_distribution
    diff_distribution["loan_ids"] = diff_distribution["payment_diff_category"].map(
        lambda x: loan_ids_by_category.get(x, "") if pd.notna(x) else ""
    )
    
    diff_chart = alt.Chart(diff_distribution).mark_bar().encode(
        x=alt.X(
            'payment_diff_category:N', 
            title="Performance Category",
            sort=list(diff_distribution['payment_diff_category']),
            axis=alt.Axis(labelAngle=-45)
        ),
        y=alt.Y(
            'loan_count:Q', 
            title="Number of Loans"
        ),
        color=alt.Color(
            'avg_difference_pct:Q',
            scale=alt.Scale(
                domain=[-0.3, 0, 0.3],
                range=['#d62728', '#ffbb78', '#2ca02c']
            ),
            legend=alt.Legend(title="Avg. Difference")
        ),
        tooltip=[
            alt.Tooltip('payment_diff_category:N', title="Category"),
            alt.Tooltip('loan_count:Q', title="Loan Count"),
            alt.Tooltip('total_difference:Q', title="Total Difference", format="$,.2f"),
            alt.Tooltip('avg_difference_pct:Q', title="Avg Difference %", format=".1%"),
            alt.Tooltip('loan_ids:N', title="Loan IDs")
        ]
    ).properties(
        width=700,
        height=400,
        title="Distribution of Loans by Payment Performance"
    )
    
    # Layout charts side by side
    col1, col2 = st.columns([1, 1])
    with col1:
        st.altair_chart(bar_chart, use_container_width=True)
    with col2:
        st.altair_chart(diff_chart, use_container_width=True)

def plot_payment_performance_over_time(df):
    """
    Create and display charts showing expected vs. actual payment performance over time and by vintage.
    """
    # Filter out paid-off loans
    active_df = df[df['loan_status'] != "Paid Off"].copy()
    
    if active_df.empty:
        st.info("No active loans to analyze for payment performance over time.")
        return
    
    # Calculate expected and actual payments
    active_df['expected_paid_to_date'] = active_df.apply(
        lambda x: calculate_expected_payment_to_date(x), axis=1
    )
    active_df['actual_paid'] = active_df['total_paid']
    active_df['payment_difference'] = active_df['actual_paid'] - active_df['expected_paid_to_date']
    active_df['performance_ratio'] = active_df.apply(
        lambda x: x['actual_paid'] / x['expected_paid_to_date'] if x['expected_paid_to_date'] > 0 else 1.0, 
        axis=1
    )
    
    # Add date fields for time series analysis
    active_df['funding_month'] = pd.to_datetime(active_df['funding_date']).dt.to_period('M')
    active_df['cohort'] = pd.to_datetime(active_df['funding_date']).dt.to_period('Q').astype(str)
    
    # Chart 1: Performance Over Time
    st.subheader("Payment Performance Over Time", 
                help="Shows how expected vs. actual payments trend over time for the entire portfolio.")
    
    # Aggregate by month
    monthly_performance = active_df.groupby('funding_month').agg(
        expected_payment=('expected_paid_to_date', 'sum'),
        actual_payment=('actual_paid', 'sum'),
        loan_count=('loan_id', 'count')
    ).reset_index()
    
    # Convert period to datetime for chart
    monthly_performance['month'] = monthly_performance['funding_month'].dt.to_timestamp()
    
    # Calculate performance ratio
    monthly_performance['performance_ratio'] = monthly_performance['actual_payment'] / monthly_performance['expected_payment']
    
    # Create chart data in long format
    chart_data = []
    for _, row in monthly_performance.iterrows():
        chart_data.append({
            'Month': row['month'],
            'Amount': row['expected_payment'],
            'Type': 'Expected Payment',
            'Performance Ratio': row['performance_ratio'],
            'Loan Count': row['loan_count']
        })
        chart_data.append({
            'Month': row['month'],
            'Amount': row['actual_payment'],
            'Type': 'Actual Payment',
            'Performance Ratio': row['performance_ratio'],
            'Loan Count': row['loan_count']
        })
    
    chart_df = pd.DataFrame(chart_data)
    
    # Create time series chart
    time_chart = alt.Chart(chart_df).mark_line(point=True).encode(
        x=alt.X('Month:T', title='Month'),
        y=alt.Y('Amount:Q', title='Amount ($)', axis=alt.Axis(format='$,.0f')),
        color=alt.Color(
            'Type:N',
            scale=alt.Scale(
                domain=['Expected Payment', 'Actual Payment'],
                range=['#1f77b4', '#2ca02c']
            )
        ),
        tooltip=[
            alt.Tooltip('Month:T', title='Month', format='%b %Y'),
            alt.Tooltip('Amount:Q', title='Amount', format='$,.0f'),
            alt.Tooltip('Type:N', title='Payment Type'),
            alt.Tooltip('Performance Ratio:Q', title='Performance Ratio', format='.2f'),
            alt.Tooltip('Loan Count:Q', title='Number of Loans')
        ]
    ).properties(
        width=700,
        height=400,
        title='Expected vs. Actual Payments Over Time'
    )
    
    st.altair_chart(time_chart, use_container_width=True)
    
    # Chart 2: Performance by Cohort (Vintage)
    st.subheader("Payment Performance by Cohort", 
                help="Shows how different funding cohorts (quarters) are performing relative to expectations.")
    
    # Aggregate by cohort
    cohort_performance = active_df.groupby('cohort').agg(
        expected_payment=('expected_paid_to_date', 'sum'),
        actual_payment=('actual_paid', 'sum'),
        loan_count=('loan_id', 'count')
    ).reset_index()
    
    # Calculate performance ratio
    cohort_performance['performance_ratio'] = cohort_performance['actual_payment'] / cohort_performance['expected_payment']
    
    # Sort cohorts chronologically
    cohort_performance = cohort_performance.sort_values('cohort')
    
    # Create cohort ratio chart
    ratio_chart = alt.Chart(cohort_performance).mark_bar().encode(
        x=alt.X('cohort:N', title='Funding Quarter', sort=None),
        y=alt.Y('performance_ratio:Q', 
                title='Payment Performance Ratio (Actual/Expected)',
                axis=alt.Axis(format='.0%')),
        color=alt.Color(
            'performance_ratio:Q',
            scale=alt.Scale(
                domain=[0.8, 1.0, 1.2],
                range=['#d62728', '#ffbb78', '#2ca02c']
            ),
            legend=None
        ),
        tooltip=[
            alt.Tooltip('cohort:N', title='Cohort'),
            alt.Tooltip('expected_payment:Q', title='Expected Payment', format='$,.0f'),
            alt.Tooltip('actual_payment:Q', title='Actual Payment', format='$,.0f'),
            alt.Tooltip('performance_ratio:Q', title='Performance Ratio', format='.2f'),
            alt.Tooltip('loan_count:Q', title='Number of Loans')
        ]
    ).properties(
        width=700,
        height=400,
        title='Payment Performance Ratio by Cohort'
    )
    
    # Add reference line at 100%
    ref_line = alt.Chart(pd.DataFrame({'y': [1.0]})).mark_rule(
        strokeDash=[4, 4],
        color='gray',
        strokeWidth=1
    ).encode(y='y:Q')
    
    st.altair_chart(ratio_chart + ref_line, use_container_width=True)
    
    # Add explanation
    st.caption(
        "**Payment Performance Ratio:** A ratio of 1.0 (100%) means loans are performing exactly as expected. " +
        "Values above 1.0 indicate better than expected performance, while values below 1.0 indicate underperformance."
    )

def plot_fico_distribution(df):
    """
    Create and display visualizations of loan distribution by FICO score.
    """
    try:
        # Check if FICO score data exists (renamed from fico_score to fico based on schema)
        if 'fico' not in df.columns or df['fico'].isna().all():
            st.warning("FICO score data not available for analysis.")
            return
            
        # Define FICO score bands
        fico_bins = [0, 580, 620, 660, 700, 740, 850]
        fico_labels = ['<580', '580-619', '620-659', '660-699', '700-739', '740+']
        
        # Create a copy to avoid modifying the original
        fico_df = df.copy()
        
        # Convert FICO to numeric and handle any errors
        fico_df['fico'] = pd.to_numeric(fico_df['fico'], errors='coerce')
        
        # Add FICO band column
        fico_df['fico_band'] = pd.cut(
            fico_df['fico'], 
            bins=fico_bins, 
            labels=fico_labels, 
            right=False
        )
        
        # Summary by FICO band - count of deals
        fico_count = fico_df.groupby('fico_band').size().reset_index(name='count')
        
        # Summary by FICO band - outstanding balance
        fico_balance = fico_df.groupby('fico_band')['net_balance'].sum().reset_index()
        
        # Create chart for count of deals by FICO
        count_chart = alt.Chart(fico_count).mark_bar().encode(
            x=alt.X('fico_band:N', title="FICO Score Band", sort=fico_labels),
            y=alt.Y('count:Q', title="Number of Deals"),
            color=alt.Color('fico_band:N', 
                scale=alt.Scale(
                    domain=fico_labels,
                    range=['#d62728', '#ff7f0e', '#ffbb78', '#2ca02c', '#98df8a', '#1f77b4']
                ),
                legend=None
            ),
            tooltip=[
                alt.Tooltip('fico_band:N', title="FICO Score Band"),
                alt.Tooltip('count:Q', title="Number of Deals")
            ]
        ).properties(
            width=600,
            height=300,
            title="Count of Deals by FICO Score Band"
        )
        
        # Create chart for outstanding balance by FICO
        balance_chart = alt.Chart(fico_balance).mark_bar().encode(
            x=alt.X('fico_band:N', title="FICO Score Band", sort=fico_labels),
            y=alt.Y('net_balance:Q', title="Outstanding Balance ($)", axis=alt.Axis(format="$,.0f")),
            color=alt.Color('fico_band:N', 
                scale=alt.Scale(
                    domain=fico_labels,
                    range=['#d62728', '#ff7f0e', '#ffbb78', '#2ca02c', '#98df8a', '#1f77b4']
                ),
                legend=None
            ),
            tooltip=[
                alt.Tooltip('fico_band:N', title="FICO Score Band"),
                alt.Tooltip('net_balance:Q', title="Outstanding Balance", format="$,.0f")
            ]
        ).properties(
            width=600,
            height=300,
            title="CSL Principal Outstanding by FICO Score Band"
        )
        
        # Display charts
        st.altair_chart(count_chart, use_container_width=True)
        st.altair_chart(balance_chart, use_container_width=True)
        
        # Add summary table
        st.subheader("FICO Score Band Summary")
        fico_summary = fico_df.groupby('fico_band').agg(
            deal_count=('loan_id', 'count'),
            capital_deployed=('csl_participation_amount', 'sum'),
            outstanding_balance=('net_balance', 'sum'),
        ).reset_index()
        
        # Format for display
        display_df = fico_summary.copy()
        display_df['capital_deployed'] = display_df['capital_deployed'].map(lambda x: f"${x:,.0f}")
        display_df['outstanding_balance'] = display_df['outstanding_balance'].map(lambda x: f"${x:,.0f}")
        
        # Rename columns
        display_df.columns = [
            'FICO Score Band', 'Deal Count', 'Capital Deployed', 'Outstanding Balance'
        ]
        
        # Display table
        st.dataframe(display_df, use_container_width=True)
        
    except Exception as e:
        st.error(f"Error creating FICO distribution visualization: {str(e)}")
        st.info("Unable to generate FICO distribution chart due to an error in data processing.")

def plot_tib_distribution(df):
    """
    Create and display visualizations of capital exposure by Time in Business.
    """
    try:
        # Check if Time in Business data exists (renamed from time_in_business to tib based on schema)
        if 'tib' not in df.columns or df['tib'].isna().all():
            st.warning("Time in Business data not available for analysis.")
            return
            
        # Define TIB bands
        tib_bins = [0, 5, 10, 15, 20, 25, 100]
        tib_labels = ['≤5', '5-10', '10-15', '15-20', '20-25', '25+']
        
        # Create a copy to avoid modifying the original
        tib_df = df.copy()
        
        # Convert TIB to numeric and handle any errors
        tib_df['tib'] = pd.to_numeric(tib_df['tib'], errors='coerce')
        
        # Add TIB band column
        tib_df['tib_band'] = pd.cut(
            tib_df['tib'], 
            bins=tib_bins, 
            labels=tib_labels, 
            right=False
        )
        
        # Summary by TIB band
        tib_summary = tib_df.groupby('tib_band').agg(
            deal_count=('loan_id', 'count'),
            capital_deployed=('csl_participation_amount', 'sum'),
            outstanding_balance=('net_balance', 'sum'),
        ).reset_index()
        
        # Create chart for outstanding balance by TIB
        tib_chart = alt.Chart(tib_summary).mark_bar().encode(
            x=alt.X('tib_band:N', title="Time in Business (Years)", sort=tib_labels),
            y=alt.Y('outstanding_balance:Q', title="CSL Principal Outstanding ($)", axis=alt.Axis(format="$,.0f")),
            color=alt.Color('tib_band:N', 
                scale=alt.Scale(
                    domain=tib_labels,
                    range=['#fee08b', '#fc8d59', '#ffcc80', '#fdae61', '#d9ef8b', '#91cf60']
                ),
                legend=None
            ),
            tooltip=[
                alt.Tooltip('tib_band:N', title="Time in Business"),
                alt.Tooltip('outstanding_balance:Q', title="Outstanding Balance", format="$,.0f"),
                alt.Tooltip('deal_count:Q', title="Number of Deals")
            ]
        ).properties(
            width=800,
            height=400,
            title="CSL Principal Outstanding by Time in Business"
        )
        
        # Display chart
        st.altair_chart(tib_chart, use_container_width=True)
        
        # Add summary table
        st.subheader("Time in Business Summary")
        
        # Format for display
        display_df = tib_summary.copy()
        display_df['capital_deployed'] = display_df['capital_deployed'].map(lambda x: f"${x:,.0f}")
        display_df['outstanding_balance'] = display_df['outstanding_balance'].map(lambda x: f"${x:,.0f}")
        
        # Rename columns
        display_df.columns = [
            'TIB Band', 'Deal Count', 'CSL Capital Deployed', 'CSL Principal Outstanding'
        ]
        
        # Display table
        st.dataframe(display_df, use_container_width=True)
        
    except Exception as e:
        st.error(f"Error creating Time in Business visualization: {str(e)}")
        st.info("Unable to generate Time in Business chart due to an error in data processing.")

def plot_industry_risk_summary(df):
    """
    Create and display a comprehensive industry risk summary visualization.
    """
    try:
        # Only include active loans
        active_df = df[df['loan_status'] != 'Paid Off'].copy()
        
        if 'industry' not in active_df.columns or active_df['industry'].isna().all():
            st.warning("Industry data not available for sector risk analysis.")
            return
            
        # Get sector risk data
        sector_risk_df = load_naics_sector_risk()
        
        # Join with loan data
        df_with_risk = active_df.merge(
            sector_risk_df,
            on='sector_code',
            how='left'
        )
        
        # Summary by sector
        sector_summary = df_with_risk.groupby(['sector_name', 'risk_score']).agg(
            loan_count=('loan_id', 'count'),
            total_deployed=('csl_participation_amount', 'sum'),
            net_balance=('net_balance', 'sum'),
            avg_payment_performance=('payment_performance', 'mean')
        ).reset_index()
        
        # Create a summary by risk score
        risk_summary = df_with_risk.groupby('risk_score').agg(
            loan_count=('loan_id', 'count'),
            net_balance=('net_balance', 'sum')
        ).reset_index()
        
        # Risk score summary chart
        risk_chart = alt.Chart(risk_summary).mark_bar().encode(
            x=alt.X('risk_score:O', title="Risk Score"),
            y=alt.Y('net_balance:Q', title="CSL Principal Outstanding ($)", axis=alt.Axis(format="$,.0f")),
            color=alt.Color('risk_score:O', 
                scale=alt.Scale(scheme="orangered"),
                legend=None
            ),
            tooltip=[
                alt.Tooltip('risk_score:O', title="Industry Risk Score"),
                alt.Tooltip('net_balance:Q', title="Outstanding Balance", format="$,.0f"),
                alt.Tooltip('loan_count:Q', title="Number of Loans")
            ]
        ).properties(
            width=800,
            height=400,
            title="CSL Principal Outstanding by Risk Score"
        )
        
        # Display chart
        st.subheader("CSL Capital Exposure by Industry")
        st.caption("Risk scores are based on industry sector risk profiles from NAICS data. Risk Score 5 represents the highest industry risk (darkest red).")
        st.altair_chart(risk_chart, use_container_width=True)
        
        # Display table
        st.subheader("Portfolio Summary by Industry Sector")
        
        # Sort sector summary by net balance
        sector_table = sector_summary.sort_values('net_balance', ascending=False)
        
        # Format for display
        display_df = sector_table.copy()
        display_df['total_deployed'] = display_df['total_deployed'].map(lambda x: f"${x:,.0f}")
        display_df['net_balance'] = display_df['net_balance'].map(lambda x: f"${x:,.0f}")
        display_df['avg_payment_performance'] = display_df['avg_payment_performance'].map(lambda x: f"{x:.2%}")
        
        # Calculate percentage of total
        total_balance = display_df['net_balance'].sum() if 'net_balance' in display_df.columns else 0
        display_df['pct_of_total'] = sector_table['net_balance'] / sector_table['net_balance'].sum() * 100 if sector_table['net_balance'].sum() > 0 else 0
        display_df['pct_of_total'] = display_df['pct_of_total'].map(lambda x: f"{x:.1f}%")
        
        # Rename columns
        display_df.columns = [
            'Industry Sector', 'Risk Score', 'Count of Deals', 'Total Deployed', 
            'Principal Outstanding', 'Avg Payment Performance', '% of Total'
        ]
        
        # Display table
        st.dataframe(display_df, use_container_width=True)
        
    except Exception as e:
        st.error(f"Error creating industry risk summary: {str(e)}")
        st.info("Unable to generate industry risk summary due to an error in data processing.")

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
    """Display portfolio overview metrics in a 4x3 matrix layout."""
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
    
    # New 4x3 matrix layout
    # Row 1: Counts
    row1_col1, row1_col2, row1_col3 = st.columns(3)
    with row1_col1:
        st.metric("Total Positions", f"{total_positions}")
    with row1_col2:
        st.metric("Total Paid Off", f"{total_paid_off}")
    with row1_col3:
        st.metric("Total Outstanding", f"{total_active}")
    
    # Row 2: Capital metrics
    row2_col1, row2_col2, row2_col3 = st.columns(3)
    with row2_col1:
        st.metric("Total Capital Deployed", f"${total_capital_deployed:,.2f}")
    with row2_col2:
        st.metric("Total Capital Returned", f"${total_capital_returned:,.2f}")
    with row2_col3:
        st.metric("Total Capital Outstanding", f"${net_balance:,.2f}")
    
    # Row 3: Fee metrics
    row3_col1, row3_col2, row3_col3 = st.columns(3)
    with row3_col1:
        st.metric("Total Commission Fees", f"${total_commission_fees:,.2f}")
    with row3_col2:
        st.metric("Total Platform Fees", f"${total_platform_fees:,.2f}")
    with row3_col3:
        st.metric("Total Bad Debt Allowance", f"${total_bad_debt_allowance:,.2f}")
    
    # Row 4: Average metrics
    row4_col1, row4_col2, row4_col3 = st.columns(3)
    with row4_col1:
        st.metric("Average Total Paid", f"${avg_total_paid:,.2f}")
    with row4_col2:
        st.metric(
            "Average Payment Performance", 
            f"{avg_payment_performance:.2%}", 
            help="Payment Performance measures the ratio of actual payments to expected payments. 100% means payments are on schedule."
        )
    with row4_col3:
        st.metric("Average Remaining Maturity", f"{avg_remaining_maturity:.1f} months")

def display_top_positions(df):
    """Display top outstanding positions."""
    st.subheader("Top 5 Largest Outstanding Positions")
    
    try:
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
            
            # Format for display - using only columns that exist
            available_columns = [
                'loan_id', 'deal_name', 'loan_status', 'total_invested', 
                'total_paid', 'net_balance', 'remaining_maturity_months'
            ]
            display_columns = [col for col in available_columns if col in top_positions.columns]
            
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
    except Exception as e:
        st.error(f"Error displaying top positions: {str(e)}")
        st.info("Unable to display top positions due to an error in data processing.")

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

        # Create a copy of paid_df for display purposes
    irr_display_df = paid_df.copy()
    
    # Pre-format the IRR columns as percentages
    irr_display_df['realized_irr_formatted'] = irr_display_df['realized_irr'].apply(
        lambda x: f"{x:.2%}" if pd.notnull(x) else "N/A"
    )
    irr_display_df['expected_irr_formatted'] = irr_display_df['expected_irr'].apply(
        lambda x: f"{x:.2%}" if pd.notnull(x) else "N/A"
    )
    
    # Calculate duration in days
    irr_display_df['duration_days'] = (
        pd.to_datetime(irr_display_df['payoff_date']).dt.tz_localize(None) - 
        pd.to_datetime(irr_display_df['funding_date']).dt.tz_localize(None)
    ).dt.days
    
    # Calculate working days (excluding weekends) - approximate method
    irr_display_df['working_days'] = irr_display_df['duration_days'].apply(
        lambda x: max(0, round(x * 5/7)) if pd.notnull(x) else None  # Approximation: 5/7 of days are working days
    )
    
    # Select columns for display   
    irr_columns = [
        'loan_id', 'deal_name', 'partner_source', 'funding_date', 'payoff_date',
        'duration_days', 'working_days', 'total_invested', 'total_paid', 
        'realized_irr_formatted', 'expected_irr_formatted'
    ]
    
    # Rename columns for display
    column_rename = {
        'loan_id': 'Loan ID',
        'deal_name': 'Deal Name',
        'partner_source': 'Partner Source',
        'funding_date': 'Funding Date',
        'payoff_date': 'Payoff Date',
        'duration_days': 'Duration (Days)',
        'working_days': 'Working Days',
        'total_invested': 'Total Invested',
        'total_paid': 'Total Paid',
        'realized_irr_formatted': 'Realized IRR',
        'expected_irr_formatted': 'Expected IRR'
    }
    
    # Format dates and numeric columns
    irr_display = format_dataframe_for_display(
        irr_display_df, 
        columns=irr_columns,
        rename_map=column_rename
    )
    
    # Display the dataframe
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
    """
    Display risk analytics section with comprehensive industry, FICO, and time in business analysis.
    """
    st.header("Portfolio Risk Analytics")
    
    # Calculate risk scores for active loans
    risk_df = calculate_risk_scores(df)
    
    if risk_df.empty:
        st.info("No active loans to display risk analytics.")
        return
    
    # Display top risk loans
    st.subheader("Top 10 Underperforming Loans by Risk Score", 
            help="Risk Score = 70% × Performance Gap + 30% × Age Weight. Performance Gap measures how far behind schedule payments are (1 - Payment Performance). Age Weight is normalized based on the oldest loan in the portfolio."
    )
    
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
    
    # Risk scatter plot with tooltip and quadrant adjustment    
    plot_risk_scatter(risk_df, df['payment_performance'].mean())
    
    # Industry risk analysis
    st.markdown("---")
    plot_industry_risk_summary(df)
    
    # FICO Score Distribution
    st.markdown("---")
    st.header("Portfolio Distribution by FICO Score")
    plot_fico_distribution(df)
    
    # Time in Business Distribution
    st.markdown("---")
    st.header("Capital Exposure by Time in Business")
    plot_tib_distribution(df)

# ------------------------------
# Main Application
# ------------------------------
def main():
    """Main application entry point."""
    st.title("Loan Tape Dashboard")

    # Display last updated time
    last_updated = get_last_updated()
    st.caption(f"Data last updated: {last_updated}")
    
    # Load data
    loans_df = load_loan_summaries()
    deals_df = load_deals()
    
    # Process data
    df = prepare_loan_data(loans_df, deals_df)
    
    # Add IRR calculations
    df = calculate_irr(df)
    
    # Display filters
    filtered_df = display_filters(df)
    
    # Create tabs for main sections
    tabs = st.tabs(["Summary", "Visualizations", "Analytics", "Capital at Risk"])
    
    with tabs[0]:
        # Display portfolio summary
        display_portfolio_metrics(filtered_df)
        
        # Display top positions
        display_top_positions(filtered_df)
        
        # Display loan tape
        display_loan_tape(filtered_df)
    
    with tabs[1]:
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

        st.header("Payment Performance Analysis Over Time")
        plot_payment_performance_over_time(filtered_df)
    
        st.header("Investment Position Analysis")
        plot_investment_net_position(filtered_df)
    
    with tabs[2]:
        # Risk analytics section with enhanced industry, FICO, and TIB analysis
        display_risk_analytics(filtered_df)
        
    with tabs[3]:
        # Capital at Risk section
        display_capital_at_risk(filtered_df)

        st.header("Payment Performance Analysis Over Time")
        plot_payment_performance_over_time(filtered_df)


# Run the application
if __name__ == "__main__":
    main()
