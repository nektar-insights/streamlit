# Risk band distribution
            st.subheader("Risk Score Distribution")
            band_summary = risk_df.groupby("risk_band").agg(
                loan_count=("loan_id", "count"),
                net_balance=("net_balance", "sum")
            ).reset_index()
            
            if not band_summary.empty:
                # Define proper order for risk bands
                risk_band_order = ["Low (0-0.5)", "Moderate (0.5-1.0)", "Elevated (1.0-1.5)", 
                                  "High (1.5-2.0)", "Severe (2.0+)"]
                
                risk_bar = alt.Chart(band_summary).mark_bar().encode(
                    x=alt.X("risk_band:# pages/loan_tape.py
"""
Loan Tape Dashboard - Enhanced Version
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

# Status-based risk multipliers
STATUS_RISK_MULTIPLIERS = {
    "Active": 1.0,
    "Active - Frequently Late": 1.3,
    "Minor Delinquency": 1.5,
    "Past Delinquency": 1.2,
    "Moderate Delinquency": 2.0,
    "Late": 2.5,
    "Severe Delinquency": 3.0,
    "Default": 4.0,
    "Bankrupt": 5.0,
    "Severe": 5.0,
    "Paid Off": 0.0
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
        timestamps = []
        
        # Check loan_summaries - try both updated_at and created_at
        try:
            loans_res = supabase.table("loan_summaries").select("updated_at").order("updated_at", desc=True).limit(1).execute()
            if loans_res.data and loans_res.data[0].get('updated_at'):
                timestamps.append(pd.to_datetime(loans_res.data[0]['updated_at']))
        except:
            # Fallback to created_at if updated_at doesn't exist
            try:
                loans_res = supabase.table("loan_summaries").select("created_at").order("created_at", desc=True).limit(1).execute()
                if loans_res.data and loans_res.data[0].get('created_at'):
                    timestamps.append(pd.to_datetime(loans_res.data[0]['created_at']))
            except:
                pass
        
        # Check deals table
        try:
            deals_res = supabase.table("deals").select("updated_at").order("updated_at", desc=True).limit(1).execute()
            if deals_res.data and deals_res.data[0].get('updated_at'):
                timestamps.append(pd.to_datetime(deals_res.data[0]['updated_at']))
        except:
            try:
                deals_res = supabase.table("deals").select("created_at").order("created_at", desc=True).limit(1).execute()
                if deals_res.data and deals_res.data[0].get('created_at'):
                    timestamps.append(pd.to_datetime(deals_res.data[0]['created_at']))
            except:
                pass
        
        # Check loan_schedules table
        try:
            schedules_res = supabase.table("loan_schedules").select("updated_at").order("updated_at", desc=True).limit(1).execute()
            if schedules_res.data and schedules_res.data[0].get('updated_at'):
                timestamps.append(pd.to_datetime(schedules_res.data[0]['updated_at']))
        except:
            try:
                schedules_res = supabase.table("loan_schedules").select("created_at").order("created_at", desc=True).limit(1).execute()
                if schedules_res.data and schedules_res.data[0].get('created_at'):
                    timestamps.append(pd.to_datetime(schedules_res.data[0]['created_at']))
            except:
                pass
            
        if timestamps:
            last_updated = max(timestamps)
            return last_updated.strftime('%B %d, %Y at %I:%M %p')
        else:
            return "Unable to determine (check table schemas)"
    except Exception as e:
        return f"Error: {str(e)}"

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
    """Calculate IRR metrics for loans."""
    result_df = df.copy()
    
    def calc_realized_irr(row):
        if pd.isna(row['funding_date']) or pd.isna(row['payoff_date']) or row['total_invested'] <= 0:
            return None
            
        try:
            funding_date = pd.to_datetime(row['funding_date']).tz_localize(None)
            payoff_date = pd.to_datetime(row['payoff_date']).tz_localize(None)
            
            if payoff_date <= funding_date:
                return None
                
            days_to_payoff = (payoff_date - funding_date).days
            years_to_payoff = days_to_payoff / 365.0
            
            if years_to_payoff < 0.01:
                simple_return = (row['total_paid'] / row['total_invested']) - 1
                annualized = (1 + simple_return) ** (1 / years_to_payoff) - 1
                return annualized
                
            try:
                irr = npf.irr([-row['total_invested'], row['total_paid']])
                
                if irr < -1 or irr > 10:
                    simple_return = (row['total_paid'] / row['total_invested']) - 1
                    annualized = (1 + simple_return) ** (1 / years_to_payoff) - 1
                    return annualized
                    
                return irr
            except:
                simple_return = (row['total_paid'] / row['total_invested']) - 1
                annualized = (1 + simple_return) ** (1 / years_to_payoff) - 1
                return annualized
        except Exception as e:
            return None
    
    def calc_expected_irr(row):
        if pd.isna(row['funding_date']) or pd.isna(row['maturity_date']) or row['total_invested'] <= 0:
            return None
            
        try:
            funding_date = pd.to_datetime(row['funding_date']).tz_localize(None)
            maturity_date = pd.to_datetime(row['maturity_date']).tz_localize(None)
            
            if maturity_date <= funding_date:
                return None
                
            expected_payment = row['our_rtr'] if 'our_rtr' in row and pd.notnull(row['our_rtr']) else row['total_invested'] * (1 + row['roi'])
            
            try:
                days_to_maturity = (maturity_date - funding_date).days
                years_to_maturity = days_to_maturity / 365.0
                
                if years_to_maturity < 0.01:
                    simple_return = (expected_payment / row['total_invested']) - 1
                    annualized = (1 + simple_return) ** (1 / years_to_maturity) - 1
                    return annualized
                    
                irr = npf.irr([-row['total_invested'], expected_payment])
                
                if irr < -1 or irr > 10:
                    simple_return = (expected_payment / row['total_invested']) - 1
                    annualized = (1 + simple_return) ** (1 / years_to_maturity) - 1
                    return annualized
                    
                return irr
            except:
                simple_return = (expected_payment / row['total_invested']) - 1
                annualized = (1 + simple_return) ** (1 / years_to_maturity) - 1
                return annualized
        except Exception as e:
            return None
    
    try:
        result_df['realized_irr'] = result_df.apply(calc_realized_irr, axis=1)
        result_df['expected_irr'] = result_df.apply(calc_expected_irr, axis=1)
        
        result_df['realized_irr_pct'] = result_df['realized_irr'].apply(
            lambda x: f"{x:.2%}" if pd.notnull(x) else "N/A"
        )
        result_df['expected_irr_pct'] = result_df['expected_irr'].apply(
            lambda x: f"{x:.2%}" if pd.notnull(x) else "N/A"
        )
    except Exception as e:
        st.error(f"Error in IRR calculations: {str(e)}")
        result_df['realized_irr'] = None
        result_df['expected_irr'] = None
        result_df['realized_irr_pct'] = "N/A"
        result_df['expected_irr_pct'] = "N/A"
    
    return result_df

def calculate_risk_scores(df):
    """
    Calculate enhanced risk scores for active loans using status multipliers and age factor.
    """
    # Only include active loans (exclude paid off loans)
    risk_df = df[df['loan_status'] != 'Paid Off'].copy()
    
    if risk_df.empty:
        return risk_df
    
    # Calculate performance gap
    risk_df['performance_gap'] = 1 - risk_df['payment_performance'].clip(upper=1.0)
    
    # Get status multiplier
    risk_df['status_multiplier'] = risk_df['loan_status'].map(STATUS_RISK_MULTIPLIERS).fillna(1.0)
    
    # Calculate days past maturity (overdue factor)
    today = pd.Timestamp.today().tz_localize(None)
    risk_df['days_past_maturity'] = risk_df['maturity_date'].apply(
        lambda x: max(0, (today - pd.to_datetime(x).tz_localize(None)).days) if pd.notnull(x) else 0
    )
    risk_df['overdue_factor'] = (risk_df['days_past_maturity'] / 30).clip(upper=12) / 12  # Max 1 year overdue
    
    # New risk score formula:
    # Risk Score = Performance Gap × (1 + Status Multiplier) × (1 + Overdue Factor)
    risk_df['risk_score'] = (
        risk_df['performance_gap'] * 
        risk_df['status_multiplier'] * 
        (1 + risk_df['overdue_factor'])
    ).clip(upper=5.0)  # Cap at 5.0
    
    # Create risk bands
    risk_bins = [0, 0.5, 1.0, 1.5, 2.0, 5.0]
    risk_labels = ["Low (0-0.5)", "Moderate (0.5-1.0)", "Elevated (1.0-1.5)", "High (1.5-2.0)", "Severe (2.0+)"]
    risk_df["risk_band"] = pd.cut(risk_df["risk_score"], bins=risk_bins, labels=risk_labels)
    
    return risk_df

def calculate_expected_payment_to_date(row):
    """
    Calculate how much should have been paid by now based on linear payment over time.
    """
    if pd.isna(row['funding_date']) or pd.isna(row['maturity_date']) or pd.isna(row['our_rtr']):
        return 0
        
    try:
        funding_date = pd.to_datetime(row['funding_date']).tz_localize(None)
        maturity_date = pd.to_datetime(row['maturity_date']).tz_localize(None)
        current_date = pd.Timestamp.today().tz_localize(None)
        
        if current_date >= maturity_date:
            return row['our_rtr']
            
        total_days = (maturity_date - funding_date).days
        days_elapsed = (current_date - funding_date).days
        
        if total_days <= 0:
            return 0
            
        expected_pct = min(1.0, max(0.0, days_elapsed / total_days))
        expected_payment = row['our_rtr'] * expected_pct
        
        return expected_payment
    except Exception as e:
        return 0

def format_dataframe_for_display(df, columns=None, rename_map=None):
    """Format DataFrame for display."""
    if columns:
        display_columns = [col for col in columns if col in df.columns]
        display_df = df[display_columns].copy()
    else:
        display_df = df.copy()
    
    if rename_map:
        display_df.rename(
            columns={k: v for k, v in rename_map.items() if k in display_df.columns}, 
            inplace=True
        )
    
    for col in display_df.select_dtypes(include=['float64', 'float32']).columns:
        if any(term in col for term in ["ROI", "Rate", "Percentage", "Performance"]):
            display_df[col] = display_df[col].map(lambda x: f"{x:.2%}" if pd.notnull(x) else "")
        elif any(term in col for term in ["Maturity", "Months"]):
            display_df[col] = display_df[col].map(lambda x: f"{x:.1f}" if pd.notnull(x) else "")
        elif any(term in col for term in ["Capital", "Invested", "Paid", "Balance", "Fees"]):
            display_df[col] = display_df[col].map(lambda x: f"${x:,.2f}" if pd.notnull(x) else "")
    
    try:
        if rename_map:
            reverse_map = {v: k for k, v in rename_map.items() if k in df.columns}
        else:
            reverse_map = {}
            
        for col in display_df.columns:
            if any(term in col for term in ["Date", "Funding", "Maturity"]):
                original_col = reverse_map.get(col, col.replace(" ", "_").lower())
                
                if original_col in df.columns and pd.api.types.is_datetime64_dtype(df[original_col]):
                    display_df[col] = pd.to_datetime(df[original_col]).dt.strftime('%Y-%m-%d')
    except Exception as e:
        st.warning(f"Error formatting date columns: {str(e)}")
    
    return display_df

# FIXED: Capital Flow Chart
def plot_capital_flow(df):
    """
    Create and display a line chart showing capital deployment vs returns over time.
    FIXED: Uses unified dataframe and aligned timeline. Shows actual totals that match summary metrics.
    """
    st.subheader("Capital Flow: Deployment vs. Returns")
    
    # Get loan schedules for actual payment dates
    loan_schedules = load_loan_schedules()
    
    # Ensure dates are properly formatted
    df_copy = df.copy()
    df_copy['funding_date'] = pd.to_datetime(df_copy['funding_date'], errors='coerce').dt.tz_localize(None)
    
    # Show what the totals should be for verification
    total_deployed = df_copy['csl_participation_amount'].sum()
    total_returned = df_copy['total_paid'].sum()
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Total Capital Deployed (Expected)", f"${total_deployed:,.0f}")
    with col2:
        st.metric("Total Capital Returned (Expected)", f"${total_returned:,.0f}")
    
    # Create deployment timeline (cumulative)
    deploy_data = df_copy[['funding_date', 'csl_participation_amount']].dropna()
    deploy_timeline = deploy_data.groupby('funding_date')['csl_participation_amount'].sum().sort_index().cumsum()
    
    # Create return timeline (cumulative) from loan_schedules
    if not loan_schedules.empty and 'payment_date' in loan_schedules.columns:
        loan_schedules['payment_date'] = pd.to_datetime(loan_schedules['payment_date'], errors='coerce').dt.tz_localize(None)
        
        payment_data = loan_schedules[
            (loan_schedules['actual_payment'].notna()) & 
            (loan_schedules['actual_payment'] > 0) &
            (loan_schedules['payment_date'].notna())
        ]
        
        return_timeline = payment_data.groupby('payment_date')['actual_payment'].sum().sort_index().cumsum()
    else:
        return_timeline = pd.Series(dtype=float)
    
    # Create unified date range
    if not deploy_timeline.empty:
        min_date = deploy_timeline.index.min()
        max_date = pd.Timestamp.today().normalize()
        
        date_range = pd.date_range(start=min_date, end=max_date, freq='D')
        
        # Create unified dataframe
        unified_df = pd.DataFrame(index=date_range)
        unified_df['capital_deployed'] = deploy_timeline.reindex(date_range).ffill().fillna(0)
        unified_df['capital_returned'] = return_timeline.reindex(date_range).ffill().fillna(0)
        unified_df['date'] = unified_df.index
        
        # Verify final values match
        final_deployed = unified_df['capital_deployed'].iloc[-1]
        final_returned = unified_df['capital_returned'].iloc[-1]
        
        st.caption(f"Chart shows: Deployed ${final_deployed:,.0f} | Returned ${final_returned:,.0f}")
        
        # Reshape for plotting
        plot_data = []
        for date, row in unified_df.iterrows():
            plot_data.append({
                'date': date,
                'amount': row['capital_deployed'],
                'series': 'Capital Deployed'
            })
            plot_data.append({
                'date': date,
                'amount': row['capital_returned'],
                'series': 'Capital Returned'
            })
        
        plot_df = pd.DataFrame(plot_data)
        
        # Create chart
        chart = alt.Chart(plot_df).mark_line().encode(
            x=alt.X('date:T', title="Date", axis=alt.Axis(format="%b %Y")),
            y=alt.Y('amount:Q', title="Cumulative Amount ($)", axis=alt.Axis(format="$,.0f")),
            color=alt.Color(
                'series:N',
                scale=alt.Scale(
                    domain=['Capital Deployed', 'Capital Returned'],
                    range=['#ff7f0e', '#2ca02c']
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
            title="Capital Deployed vs. Capital Returned Over Time (Cumulative)"
        )
        
        st.altair_chart(chart, use_container_width=True)
        
        # Add milestone analysis
        milestones = [500_000, 1_000_000, 2_000_000, 3_000_000]
        milestone_data = []
        
        for milestone in milestones:
            deploy_date = unified_df[unified_df['capital_deployed'] >= milestone].index.min()
            if pd.notna(deploy_date):
                milestone_data.append({
                    'Milestone': f"${milestone:,.0f}",
                    'Date Reached': deploy_date.strftime('%Y-%m-%d'),
                    'Days from Start': (deploy_date - min_date).days
                })
        
        if milestone_data:
            st.subheader("Capital Deployment Milestones")
            milestone_df = pd.DataFrame(milestone_data)
            
            # Calculate days between milestones
            milestone_df['Days Between'] = milestone_df['Days from Start'].diff().fillna(0).astype(int)
            st.dataframe(milestone_df, use_container_width=True, hide_index=True)
    else:
        st.info("Insufficient data to display capital flow chart.")

# FIXED: Net Investment Position
def plot_investment_net_position(df):
    """
    FIXED: Create accurate net investment position chart.
    Net Position = Cumulative Deployed - Cumulative Returned
    """
    st.subheader("Net Investment Position Over Time",
                help="Shows capital at work: cumulative deployed minus cumulative returned")
    
    loan_schedules = load_loan_schedules()
    
    df_copy = df.copy()
    df_copy['funding_date'] = pd.to_datetime(df_copy['funding_date'], errors='coerce').dt.tz_localize(None)
    
    # Build deployment timeline
    deploy_data = df_copy[['funding_date', 'csl_participation_amount']].dropna()
    deploy_timeline = deploy_data.groupby('funding_date')['csl_participation_amount'].sum().sort_index().cumsum()
    
    # Build return timeline
    if not loan_schedules.empty and 'payment_date' in loan_schedules.columns:
        loan_schedules['payment_date'] = pd.to_datetime(loan_schedules['payment_date'], errors='coerce').dt.tz_localize(None)
        
        payment_data = loan_schedules[
            (loan_schedules['actual_payment'].notna()) & 
            (loan_schedules['actual_payment'] > 0) &
            (loan_schedules['payment_date'].notna())
        ]
        
        return_timeline = payment_data.groupby('payment_date')['actual_payment'].sum().sort_index().cumsum()
    else:
        return_timeline = pd.Series(dtype=float)
    
    if not deploy_timeline.empty:
        min_date = deploy_timeline.index.min()
        max_date = pd.Timestamp.today().normalize()
        
        date_range = pd.date_range(start=min_date, end=max_date, freq='D')
        
        # Create unified timeline
        unified_df = pd.DataFrame(index=date_range)
        unified_df['cum_deployed'] = deploy_timeline.reindex(date_range).ffill().fillna(0)
        unified_df['cum_returned'] = return_timeline.reindex(date_range).ffill().fillna(0)
        unified_df['net_position'] = unified_df['cum_deployed'] - unified_df['cum_returned']
        unified_df['date'] = unified_df.index
        
        # Reshape for plotting
        plot_data = []
        for date, row in unified_df.iterrows():
            plot_data.append({'date': date, 'amount': row['cum_deployed'], 'Type': 'Cumulative Deployed'})
            plot_data.append({'date': date, 'amount': row['cum_returned'], 'Type': 'Cumulative Returned'})
            plot_data.append({'date': date, 'amount': row['net_position'], 'Type': 'Net Position'})
        
        plot_df = pd.DataFrame(plot_data)
        
        chart = alt.Chart(plot_df).mark_line().encode(
            x=alt.X('date:T', title='Date'),
            y=alt.Y('amount:Q', title='Amount ($)', axis=alt.Axis(format='$,.0f')),
            color=alt.Color(
                'Type:N',
                scale=alt.Scale(
                    domain=['Cumulative Deployed', 'Cumulative Returned', 'Net Position'],
                    range=['#ff7f0e', '#2ca02c', '#1f77b4']
                )
            ),
            tooltip=[
                alt.Tooltip('date:T', title='Date', format='%Y-%m-%d'),
                alt.Tooltip('amount:Q', title='Amount', format='$,.2f'),
                alt.Tooltip('Type:N', title='Metric')
            ]
        ).properties(
            width=800,
            height=500,
            title='Portfolio Net Position Over Time'
        )
        
        # Add zero reference line
        zero_line = alt.Chart(pd.DataFrame({'y': [0]})).mark_rule(
            strokeDash=[2, 2], color='gray', strokeWidth=1
        ).encode(y='y:Q')
        
        st.altair_chart(chart + zero_line, use_container_width=True)
        
        st.caption(
            "**Net Position:** Capital still deployed (positive) or profit after recovery (negative). "
            "When net position reaches zero or negative, the fund has recovered all capital."
        )
    else:
        st.info("Insufficient data for net position analysis.")

# FIXED: Payment Performance by Cohort - Show actual percentage difference
def plot_payment_performance_by_cohort(df):
    """
    FIXED: Show percentage difference from expected with proper formatting.
    """
    active_df = df[df['loan_status'] != "Paid Off"].copy()
    
    if active_df.empty:
        st.info("No active loans to analyze for payment performance over time.")
        return
    
    # Calculate expected and actual payments
    active_df['expected_paid_to_date'] = active_df.apply(
        lambda x: calculate_expected_payment_to_date(x), axis=1
    )
    active_df['actual_paid'] = active_df['total_paid']
    
    # Calculate performance as percentage difference from expected
    # Formula: ((Actual / Expected) - 1) * 100 = % difference
    active_df['performance_pct_diff'] = active_df.apply(
        lambda x: ((x['actual_paid'] / x['expected_paid_to_date']) - 1) if x['expected_paid_to_date'] > 0 else 0,
        axis=1
    )
    
    active_df['cohort'] = pd.to_datetime(active_df['funding_date']).dt.to_period('Q').astype(str)
    
    # Aggregate by cohort
    cohort_performance = active_df.groupby('cohort').agg(
        expected_payment=('expected_paid_to_date', 'sum'),
        actual_payment=('actual_paid', 'sum'),
        loan_count=('loan_id', 'count')
    ).reset_index()
    
    # Calculate percentage difference from expected for the cohort
    cohort_performance['performance_pct_diff'] = (
        (cohort_performance['actual_payment'] / cohort_performance['expected_payment']) - 1
    )
    
    # Add text labels showing the actual percentage
    cohort_performance['perf_label'] = cohort_performance['performance_pct_diff'].apply(
        lambda x: f"{x:+.1%}"
    )
    
    cohort_performance = cohort_performance.sort_values('cohort')
    
    # Add color classification
    def classify_performance(pct_diff):
        if pct_diff >= -0.05:
            return 'On/Above Target'
        elif pct_diff >= -0.15:
            return 'Slightly Below'
        else:
            return 'Significantly Below'
    
    cohort_performance['performance_category'] = cohort_performance['performance_pct_diff'].apply(classify_performance)
    
    # Create chart
    bars = alt.Chart(cohort_performance).mark_bar().encode(
        x=alt.X('cohort:N', title='Funding Quarter', sort=None),
        y=alt.Y('performance_pct_diff:Q',
                title='Performance Difference from Expected',
                axis=alt.Axis(format='.0%')),
        color=alt.Color('performance_category:N',
            scale=alt.Scale(
                domain=['On/Above Target', 'Slightly Below', 'Significantly Below'],
                range=['#2ca02c', '#ffbb78', '#d62728']
            ),
            legend=alt.Legend(title='Performance')
        ),
        tooltip=[
            alt.Tooltip('cohort:N', title='Cohort'),
            alt.Tooltip('expected_payment:Q', title='Expected Payment', format='$,.0f'),
            alt.Tooltip('actual_payment:Q', title='Actual Payment', format='$,.0f'),
            alt.Tooltip('performance_pct_diff:Q', title='Performance Difference', format='+.1%'),
            alt.Tooltip('loan_count:Q', title='Number of Loans')
        ]
    ).properties(
        width=700,
        height=400,
        title='Payment Performance by Cohort (Difference from Expected)'
    )
    
    # Add text labels on bars
    text = alt.Chart(cohort_performance).mark_text(
        align='center',
        baseline='bottom',
        dy=-5,
        fontSize=11,
        fontWeight='bold'
    ).encode(
        x=alt.X('cohort:N', sort=None),
        y=alt.Y('performance_pct_diff:Q'),
        text='perf_label:N'
    )
    
    # Add reference line at 0% (on target)
    ref_line = alt.Chart(pd.DataFrame({'y': [0]})).mark_rule(
        strokeDash=[4, 4],
        color='gray',
        strokeWidth=2
    ).encode(y='y:Q')
    
    # Add on-target zone (-5% to +5%)
    target_zone = alt.Chart(pd.DataFrame({
        'y': [-0.05],
        'y2': [0.05]
    })).mark_rect(opacity=0.2, color='green').encode(
        y='y:Q',
        y2='y2:Q'
    )
    
    st.altair_chart(target_zone + bars + text + ref_line, use_container_width=True)
    
    st.caption(
        "On-Target Zone (Green Area): -5% to +5% from expected. "
        "Positive values mean ahead of schedule, negative values mean behind schedule."
    )

# Enhanced FICO Analysis - Use Payment Performance instead of ROI
def plot_fico_performance_analysis(df):
    """
    ENHANCED: FICO score analysis with payment performance metrics (better than ROI for active loans).
    """
    st.header("FICO Score Performance Analysis")
    
    if 'fico' not in df.columns or df['fico'].isna().all():
        st.warning("FICO score data not available for analysis.")
        return
    
    fico_bins = [0, 580, 620, 660, 700, 740, 850]
    fico_labels = ['<580', '580-619', '620-659', '660-699', '700-739', '740+']
    
    fico_df = df.copy()
    fico_df['fico'] = pd.to_numeric(fico_df['fico'], errors='coerce')
    fico_df['fico_band'] = pd.cut(fico_df['fico'], bins=fico_bins, labels=fico_labels, right=False)
    
    # Calculate performance metrics by FICO band
    fico_metrics = fico_df.groupby('fico_band').agg(
        deal_count=('loan_id', 'count'),
        capital_deployed=('csl_participation_amount', 'sum'),
        outstanding_balance=('net_balance', 'sum'),
        avg_payment_performance=('payment_performance', 'mean'),
        total_paid=('total_paid', 'sum'),
        total_invested=('total_invested', 'sum')
    ).reset_index()
    
    # Calculate actual return rate (useful for comparison)
    fico_metrics['actual_return_rate'] = fico_metrics['total_paid'] / fico_metrics['total_invested']
    
    # Calculate default/late rates
    status_by_fico = fico_df.groupby(['fico_band', 'loan_status']).size().reset_index(name='count')
    total_by_fico = fico_df.groupby('fico_band').size().reset_index(name='total')
    status_by_fico = status_by_fico.merge(total_by_fico, on='fico_band')
    status_by_fico['pct'] = status_by_fico['count'] / status_by_fico['total']
    
    # Calculate problem loan rate (Late, Default, Bankrupt, Severe, Delinquencies)
    problem_statuses = ['Late', 'Default', 'Bankrupt', 'Severe', 'Severe Delinquency', 'Moderate Delinquency']
    problem_loans = status_by_fico[status_by_fico['loan_status'].isin(problem_statuses)]
    problem_rate = problem_loans.groupby('fico_band')['pct'].sum().reset_index(name='problem_rate')
    
    fico_metrics = fico_metrics.merge(problem_rate, on='fico_band', how='left')
    fico_metrics['problem_rate'] = fico_metrics['problem_rate'].fillna(0)
    
    # Chart 1: Payment Performance by FICO
    col1, col2 = st.columns(2)
    
    with col1:
        perf_chart = alt.Chart(fico_metrics).mark_bar().encode(
            x=alt.X('fico_band:N', title='FICO Score Band', sort=fico_labels),
            y=alt.Y('avg_payment_performance:Q', title='Avg Payment Performance', 
                    axis=alt.Axis(format='.0%')),
            color=alt.Color('avg_payment_performance:Q',
                scale=alt.Scale(domain=[0.5, 0.8, 1.0], range=['#d62728', '#ffbb78', '#2ca02c']),
                legend=None),
            tooltip=[
                alt.Tooltip('fico_band:N', title='FICO Band'),
                alt.Tooltip('avg_payment_performance:Q', title='Avg Payment Performance', format='.1%'),
                alt.Tooltip('deal_count:Q', title='Loan Count')
            ]
        ).properties(
            width=350,
            height=300,
            title='Payment Performance by FICO Score'
        )
        st.altair_chart(perf_chart, use_container_width=True)
    
    with col2:
        # Chart 2: Problem Loan Rate by FICO
        problem_chart = alt.Chart(fico_metrics).mark_bar().encode(
            x=alt.X('fico_band:N', title='FICO Score Band', sort=fico_labels),
            y=alt.Y('problem_rate:Q', title='Problem Loan Rate', 
                    axis=alt.Axis(format='.0%')),
            color=alt.Color('problem_rate:Q',
                scale=alt.Scale(domain=[0, 0.2, 0.4], range=['#2ca02c', '#ffbb78', '#d62728']),
                legend=None),
            tooltip=[
                alt.Tooltip('fico_band:N', title='FICO Band'),
                alt.Tooltip('problem_rate:Q', title='Problem Loan Rate', format='.1%'),
                alt.Tooltip('deal_count:Q', title='Total Loans')
            ]
        ).properties(
            width=350,
            height=300,
            title='Problem Loan Rate by FICO Score'
        )
        st.altair_chart(problem_chart, use_container_width=True)
    
    # Chart 3: Actual Return Rate by FICO (better than ROI for mixed maturity loans)
    return_chart = alt.Chart(fico_metrics).mark_bar().encode(
        x=alt.X('fico_band:N', title='FICO Score Band', sort=fico_labels),
        y=alt.Y('actual_return_rate:Q', title='Actual Return Rate (Paid/Invested)', 
                axis=alt.Axis(format='.0%')),
        color=alt.Color('actual_return_rate:Q',
            scale=alt.Scale(domain=[0.5, 1.0, 1.3], range=['#d62728', '#ffbb78', '#2ca02c']),
            legend=None),
        tooltip=[
            alt.Tooltip('fico_band:N', title='FICO Band'),
            alt.Tooltip('actual_return_rate:Q', title='Return Rate', format='.2%'),
            alt.Tooltip('deal_count:Q', title='Loan Count'),
            alt.Tooltip('total_paid:Q', title='Total Paid', format='$,.0f'),
            alt.Tooltip('total_invested:Q', title='Total Invested', format='$,.0f')
        ]
    ).properties(
        width=700,
        height=300,
        title='Actual Return Rate by FICO Score (Total Paid / Total Invested)'
    )
    st.altair_chart(return_chart, use_container_width=True)
    
    # Summary table
    st.subheader("FICO Performance Summary")
    display_df = fico_metrics.copy()
    display_df['capital_deployed'] = display_df['capital_deployed'].map(lambda x: f"${x:,.0f}")
    display_df['outstanding_balance'] = display_df['outstanding_balance'].map(lambda x: f"${x:,.0f}")
    display_df['avg_payment_performance'] = display_df['avg_payment_performance'].map(lambda x: f"{x:.1%}")
    display_df['actual_return_rate'] = display_df['actual_return_rate'].map(lambda x: f"{x:.2%}")
    display_df['problem_rate'] = display_df['problem_rate'].map(lambda x: f"{x:.1%}")
    
    display_df.columns = [
        'FICO Band', 'Loan Count', 'Capital Deployed', 'Outstanding Balance',
        'Avg Payment Performance', 'Total Paid', 'Total Invested', 'Actual Return Rate', 'Problem Loan Rate'
    ]
    
    st.dataframe(display_df[['FICO Band', 'Loan Count', 'Outstanding Balance', 
                              'Avg Payment Performance', 'Actual Return Rate', 'Problem Loan Rate']], 
                 use_container_width=True, hide_index=True)
    
    st.caption(
        "Payment Performance: Ratio of actual payments to expected payments. "
        "Actual Return Rate: Total paid divided by total invested (better metric than ROI for mixed-maturity portfolios). "
        "Problem Loan Rate: Percentage in Late/Default/Bankrupt/Delinquency status."
    )
    """
    ENHANCED: FICO score analysis with performance metrics.
    """
    st.header("FICO Score Performance Analysis")
    
    if 'fico' not in df.columns or df['fico'].isna().all():
        st.warning("FICO score data not available for analysis.")
        return
    
    fico_bins = [0, 580, 620, 660, 700, 740, 850]
    fico_labels = ['<580', '580-619', '620-659', '660-699', '700-739', '740+']
    
    fico_df = df.copy()
    fico_df['fico'] = pd.to_numeric(fico_df['fico'], errors='coerce')
    fico_df['fico_band'] = pd.cut(fico_df['fico'], bins=fico_bins, labels=fico_labels, right=False)
    
    # Calculate performance metrics by FICO band
    fico_metrics = fico_df.groupby('fico_band').agg(
        deal_count=('loan_id', 'count'),
        capital_deployed=('csl_participation_amount', 'sum'),
        outstanding_balance=('net_balance', 'sum'),
        avg_payment_performance=('payment_performance', 'mean'),
        avg_roi=('current_roi', 'mean'),
        total_paid=('total_paid', 'sum'),
        total_invested=('total_invested', 'sum')
    ).reset_index()
    
    # Calculate default/late rates
    status_by_fico = fico_df.groupby(['fico_band', 'loan_status']).size().reset_index(name='count')
    total_by_fico = fico_df.groupby('fico_band').size().reset_index(name='total')
    status_by_fico = status_by_fico.merge(total_by_fico, on='fico_band')
    status_by_fico['pct'] = status_by_fico['count'] / status_by_fico['total']
    
    # Calculate problem loan rate (Late, Default, Bankrupt, Severe, Delinquencies)
    problem_statuses = ['Late', 'Default', 'Bankrupt', 'Severe', 'Severe Delinquency', 'Moderate Delinquency']
    problem_loans = status_by_fico[status_by_fico['loan_status'].isin(problem_statuses)]
    problem_rate = problem_loans.groupby('fico_band')['pct'].sum().reset_index(name='problem_rate')
    
    fico_metrics = fico_metrics.merge(problem_rate, on='fico_band', how='left')
    fico_metrics['problem_rate'] = fico_metrics['problem_rate'].fillna(0)
    
    # Chart 1: Payment Performance by FICO
    col1, col2 = st.columns(2)
    
    with col1:
        perf_chart = alt.Chart(fico_metrics).mark_bar().encode(
            x=alt.X('fico_band:N', title='FICO Score Band', sort=fico_labels),
            y=alt.Y('avg_payment_performance:Q', title='Avg Payment Performance', 
                    axis=alt.Axis(format='.0%')),
            color=alt.Color('avg_payment_performance:Q',
                scale=alt.Scale(domain=[0.5, 0.8, 1.0], range=['#d62728', '#ffbb78', '#2ca02c']),
                legend=None),
            tooltip=[
                alt.Tooltip('fico_band:N', title='FICO Band'),
                alt.Tooltip('avg_payment_performance:Q', title='Avg Payment Performance', format='.1%'),
                alt.Tooltip('deal_count:Q', title='Loan Count')
            ]
        ).properties(
            width=350,
            height=300,
            title='Payment Performance by FICO Score'
        )
        st.altair_chart(perf_chart, use_container_width=True)
    
    with col2:
        # Chart 2: Problem Loan Rate by FICO
        problem_chart = alt.Chart(fico_metrics).mark_bar().encode(
            x=alt.X('fico_band:N', title='FICO Score Band', sort=fico_labels),
            y=alt.Y('problem_rate:Q', title='Problem Loan Rate', 
                    axis=alt.Axis(format='.0%')),
            color=alt.Color('problem_rate:Q',
                scale=alt.Scale(domain=[0, 0.2, 0.4], range=['#2ca02c', '#ffbb78', '#d62728']),
                legend=None),
            tooltip=[
                alt.Tooltip('fico_band:N', title='FICO Band'),
                alt.Tooltip('problem_rate:Q', title='Problem Loan Rate', format='.1%'),
                alt.Tooltip('deal_count:Q', title='Total Loans')
            ]
        ).properties(
            width=350,
            height=300,
            title='Problem Loan Rate by FICO Score'
        )
        st.altair_chart(problem_chart, use_container_width=True)
    
    # Chart 3: Average ROI by FICO
    roi_chart = alt.Chart(fico_metrics).mark_bar().encode(
        x=alt.X('fico_band:N', title='FICO Score Band', sort=fico_labels),
        y=alt.Y('avg_roi:Q', title='Average ROI', axis=alt.Axis(format='.0%')),
        color=alt.Color('avg_roi:Q',
            scale=alt.Scale(domain=[-0.2, 0, 0.3], range=['#d62728', '#ffbb78', '#2ca02c']),
            legend=None),
        tooltip=[
            alt.Tooltip('fico_band:N', title='FICO Band'),
            alt.Tooltip('avg_roi:Q', title='Avg ROI', format='.2%'),
            alt.Tooltip('deal_count:Q', title='Loan Count')
        ]
    ).properties(
        width=700,
        height=300,
        title='Average ROI by FICO Score'
    )
    st.altair_chart(roi_chart, use_container_width=True)
    
    # Summary table
    st.subheader("FICO Performance Summary")
    display_df = fico_metrics.copy()
    display_df['capital_deployed'] = display_df['capital_deployed'].map(lambda x: f"${x:,.0f}")
    display_df['outstanding_balance'] = display_df['outstanding_balance'].map(lambda x: f"${x:,.0f}")
    display_df['avg_payment_performance'] = display_df['avg_payment_performance'].map(lambda x: f"{x:.1%}")
    display_df['avg_roi'] = display_df['avg_roi'].map(lambda x: f"{x:.2%}")
    display_df['problem_rate'] = display_df['problem_rate'].map(lambda x: f"{x:.1%}")
    
    display_df.columns = [
        'FICO Band', 'Loan Count', 'Capital Deployed', 'Outstanding Balance',
        'Avg Payment Performance', 'Avg ROI', 'Total Paid', 'Total Invested', 'Problem Loan Rate'
    ]
    
    st.dataframe(display_df[['FICO Band', 'Loan Count', 'Outstanding Balance', 
                              'Avg Payment Performance', 'Avg ROI', 'Problem Loan Rate']], 
                 use_container_width=True, hide_index=True)
    
    st.caption(
        "**Problem Loan Rate:** Percentage of loans in Late, Default, Bankrupt, or Delinquency status. "
        "Lower is better."
    )

# Enhanced TIB Analysis - Use Payment Performance instead of ROI
def plot_tib_performance_analysis(df):
    """
    ENHANCED: Time in Business analysis with payment performance metrics (better than ROI).
    """
    st.header("Time in Business Performance Analysis")
    
    if 'tib' not in df.columns or df['tib'].isna().all():
        st.warning("Time in Business data not available for analysis.")
        return
    
    tib_bins = [0, 5, 10, 15, 20, 25, 100]
    tib_labels = ['≤5', '5-10', '10-15', '15-20', '20-25', '25+']
    
    tib_df = df.copy()
    tib_df['tib'] = pd.to_numeric(tib_df['tib'], errors='coerce')
    tib_df['tib_band'] = pd.cut(tib_df['tib'], bins=tib_bins, labels=tib_labels, right=False)
    
    # Calculate performance metrics by TIB band
    tib_metrics = tib_df.groupby('tib_band').agg(
        deal_count=('loan_id', 'count'),
        capital_deployed=('csl_participation_amount', 'sum'),
        outstanding_balance=('net_balance', 'sum'),
        avg_payment_performance=('payment_performance', 'mean'),
        total_paid=('total_paid', 'sum'),
        total_invested=('total_invested', 'sum')
    ).reset_index()
    
    # Calculate actual return rate
    tib_metrics['actual_return_rate'] = tib_metrics['total_paid'] / tib_metrics['total_invested']
    
    # Calculate problem loan rate
    status_by_tib = tib_df.groupby(['tib_band', 'loan_status']).size().reset_index(name='count')
    total_by_tib = tib_df.groupby('tib_band').size().reset_index(name='total')
    status_by_tib = status_by_tib.merge(total_by_tib, on='tib_band')
    status_by_tib['pct'] = status_by_tib['count'] / status_by_tib['total']
    
    problem_statuses = ['Late', 'Default', 'Bankrupt', 'Severe', 'Severe Delinquency', 'Moderate Delinquency']
    problem_loans = status_by_tib[status_by_tib['loan_status'].isin(problem_statuses)]
    problem_rate = problem_loans.groupby('tib_band')['pct'].sum().reset_index(name='problem_rate')
    
    tib_metrics = tib_metrics.merge(problem_rate, on='tib_band', how='left')
    tib_metrics['problem_rate'] = tib_metrics['problem_rate'].fillna(0)
    
    # Chart 1: Payment Performance by TIB
    col1, col2 = st.columns(2)
    
    with col1:
        perf_chart = alt.Chart(tib_metrics).mark_bar().encode(
            x=alt.X('tib_band:N', title='Time in Business (Years)', sort=tib_labels),
            y=alt.Y('avg_payment_performance:Q', title='Avg Payment Performance',
                    axis=alt.Axis(format='.0%')),
            color=alt.Color('avg_payment_performance:Q',
                scale=alt.Scale(domain=[0.5, 0.8, 1.0], range=['#d62728', '#ffbb78', '#2ca02c']),
                legend=None),
            tooltip=[
                alt.Tooltip('tib_band:N', title='TIB Band'),
                alt.Tooltip('avg_payment_performance:Q', title='Avg Payment Performance', format='.1%'),
                alt.Tooltip('deal_count:Q', title='Loan Count')
            ]
        ).properties(
            width=350,
            height=300,
            title='Payment Performance by Time in Business'
        )
        st.altair_chart(perf_chart, use_container_width=True)
    
    with col2:
        # Chart 2: Problem Loan Rate by TIB
        problem_chart = alt.Chart(tib_metrics).mark_bar().encode(
            x=alt.X('tib_band:N', title='Time in Business (Years)', sort=tib_labels),
            y=alt.Y('problem_rate:Q', title='Problem Loan Rate',
                    axis=alt.Axis(format='.0%')),
            color=alt.Color('problem_rate:Q',
                scale=alt.Scale(domain=[0, 0.2, 0.4], range=['#2ca02c', '#ffbb78', '#d62728']),
                legend=None),
            tooltip=[
                alt.Tooltip('tib_band:N', title='TIB Band'),
                alt.Tooltip('problem_rate:Q', title='Problem Loan Rate', format='.1%'),
                alt.Tooltip('deal_count:Q', title='Total Loans')
            ]
        ).properties(
            width=350,
            height=300,
            title='Problem Loan Rate by Time in Business'
        )
        st.altair_chart(problem_chart, use_container_width=True)
    
    # Chart 3: Actual Return Rate by TIB
    return_chart = alt.Chart(tib_metrics).mark_bar().encode(
        x=alt.X('tib_band:N', title='Time in Business (Years)', sort=tib_labels),
        y=alt.Y('actual_return_rate:Q', title='Actual Return Rate (Paid/Invested)', 
                axis=alt.Axis(format='.0%')),
        color=alt.Color('actual_return_rate:Q',
            scale=alt.Scale(domain=[0.5, 1.0, 1.3], range=['#d62728', '#ffbb78', '#2ca02c']),
            legend=None),
        tooltip=[
            alt.Tooltip('tib_band:N', title='TIB Band'),
            alt.Tooltip('actual_return_rate:Q', title='Return Rate', format='.2%'),
            alt.Tooltip('deal_count:Q', title='Loan Count')
        ]
    ).properties(
        width=700,
        height=300,
        title='Actual Return Rate by Time in Business'
    )
    st.altair_chart(return_chart, use_container_width=True)
    
    # Summary table
    st.subheader("Time in Business Performance Summary")
    display_df = tib_metrics.copy()
    display_df['capital_deployed'] = display_df['capital_deployed'].map(lambda x: f"${x:,.0f}")
    display_df['outstanding_balance'] = display_df['outstanding_balance'].map(lambda x: f"${x:,.0f}")
    display_df['avg_payment_performance'] = display_df['avg_payment_performance'].map(lambda x: f"{x:.1%}")
    display_df['actual_return_rate'] = display_df['actual_return_rate'].map(lambda x: f"{x:.2%}")
    display_df['problem_rate'] = display_df['problem_rate'].map(lambda x: f"{x:.1%}")
    
    display_df.columns = [
        'TIB Band', 'Loan Count', 'Capital Deployed', 'Outstanding Balance',
        'Avg Payment Performance', 'Total Paid', 'Total Invested', 'Actual Return Rate', 'Problem Loan Rate'
    ]
    
    st.dataframe(display_df[['TIB Band', 'Loan Count', 'Outstanding Balance',
                              'Avg Payment Performance', 'Actual Return Rate', 'Problem Loan Rate']],
                 use_container_width=True, hide_index=True)

# Enhanced Industry Performance Analysis - Use Payment Performance instead of ROI
def plot_industry_performance_analysis(df):
    """
    ENHANCED: Industry performance with scatter and bar charts plus actual return rate.
    """
    st.header("Industry Performance Analysis")
    
    active_df = df[df['loan_status'] != 'Paid Off'].copy()
    
    if 'industry' not in active_df.columns or active_df['industry'].isna().all():
        st.warning("Industry data not available for analysis.")
        return
    
    sector_risk_df = load_naics_sector_risk()
    df_with_risk = active_df.merge(sector_risk_df, on='sector_code', how='left')
    
    # Calculate metrics by sector
    sector_metrics = df_with_risk.groupby(['sector_name', 'risk_score']).agg(
        loan_count=('loan_id', 'count'),
        net_balance=('net_balance', 'sum'),
        avg_payment_performance=('payment_performance', 'mean'),
        total_paid=('total_paid', 'sum'),
        total_invested=('total_invested', 'sum')
    ).reset_index()
    
    # Calculate actual return rate
    sector_metrics['actual_return_rate'] = sector_metrics['total_paid'] / sector_metrics['total_invested']
    
    # Calculate problem loan rate by sector
    status_by_sector = df_with_risk.groupby(['sector_name', 'loan_status']).size().reset_index(name='count')
    total_by_sector = df_with_risk.groupby('sector_name').size().reset_index(name='total')
    status_by_sector = status_by_sector.merge(total_by_sector, on='sector_name')
    status_by_sector['pct'] = status_by_sector['count'] / status_by_sector['total']
    
    problem_statuses = ['Late', 'Default', 'Bankrupt', 'Severe', 'Severe Delinquency', 'Moderate Delinquency']
    problem_loans = status_by_sector[status_by_sector['loan_status'].isin(problem_statuses)]
    problem_rate = problem_loans.groupby('sector_name')['pct'].sum().reset_index(name='problem_rate')
    
    sector_metrics = sector_metrics.merge(problem_rate, on='sector_name', how='left')
    sector_metrics['problem_rate'] = sector_metrics['problem_rate'].fillna(0)
    
    # Chart 1: Scatter - Risk Score vs Payment Performance
    st.subheader("Industry Risk vs Performance")
    scatter = alt.Chart(sector_metrics).mark_circle(size=200).encode(
        x=alt.X('risk_score:Q', title='Industry Risk Score', axis=alt.Axis(format='.1f')),
        y=alt.Y('avg_payment_performance:Q', title='Avg Payment Performance', 
                axis=alt.Axis(format='.0%')),
        size=alt.Size('net_balance:Q', title='Outstanding Balance'),
        color=alt.Color('avg_payment_performance:Q',
            scale=alt.Scale(domain=[0.6, 0.8, 1.0], range=['#d62728', '#ffbb78', '#2ca02c']),
            legend=alt.Legend(title='Performance', format='.0%')),
        tooltip=[
            alt.Tooltip('sector_name:N', title='Industry'),
            alt.Tooltip('risk_score:Q', title='Risk Score', format='.1f'),
            alt.Tooltip('avg_payment_performance:Q', title='Avg Performance', format='.1%'),
            alt.Tooltip('loan_count:Q', title='Loan Count'),
            alt.Tooltip('net_balance:Q', title='Outstanding Balance', format='$,.0f')
        ]
    ).properties(
        width=700,
        height=400,
        title='Industry Risk Score vs Payment Performance (Bubble size = Outstanding Balance)'
    )
    st.altair_chart(scatter, use_container_width=True)
    
    # Chart 2: Bar - Problem Loan Rate by Industry
    st.subheader("Problem Loan Rate by Industry")
    top_sectors = sector_metrics.nlargest(10, 'net_balance')
    
    problem_bar = alt.Chart(top_sectors).mark_bar().encode(
        x=alt.X('problem_rate:Q', title='Problem Loan Rate', axis=alt.Axis(format='.0%')),
        y=alt.Y('sector_name:N', title='Industry Sector', sort='-x'),
        color=alt.Color('problem_rate:Q',
            scale=alt.Scale(domain=[0, 0.2, 0.4], range=['#2ca02c', '#ffbb78', '#d62728']),
            legend=None),
        tooltip=[
            alt.Tooltip('sector_name:N', title='Industry'),
            alt.Tooltip('problem_rate:Q', title='Problem Loan Rate', format='.1%'),
            alt.Tooltip('loan_count:Q', title='Total Loans')
        ]
    ).properties(
        width=700,
        height=400,
        title='Problem Loan Rate by Industry (Top 10 by Balance)'
    )
    st.altair_chart(problem_bar, use_container_width=True)
    
    # Chart 3: Actual Return Rate by Industry
    st.subheader("Actual Return Rate by Industry")
    return_bar = alt.Chart(top_sectors).mark_bar().encode(
        x=alt.X('actual_return_rate:Q', title='Actual Return Rate (Paid/Invested)', 
                axis=alt.Axis(format='.0%')),
        y=alt.Y('sector_name:N', title='Industry Sector', sort='-x'),
        color=alt.Color('actual_return_rate:Q',
            scale=alt.Scale(domain=[0.5, 1.0, 1.3], range=['#d62728', '#ffbb78', '#2ca02c']),
            legend=None),
        tooltip=[
            alt.Tooltip('sector_name:N', title='Industry'),
            alt.Tooltip('actual_return_rate:Q', title='Return Rate', format='.2%'),
            alt.Tooltip('loan_count:Q', title='Loan Count')
        ]
    ).properties(
        width=700,
        height=400,
        title='Actual Return Rate by Industry (Top 10 by Balance)'
    )
    st.altair_chart(return_bar, use_container_width=True)
        y=alt.Y('sector_name:N', title='Industry Sector', sort='-x'),
        color=alt.Color('avg_roi:Q',
            scale=alt.Scale(domain=[-0.2, 0, 0.3], range=['#d62728', '#ffbb78', '#2ca02c']),
            legend=None),
        tooltip=[
            alt.Tooltip('sector_name:N', title='Industry'),
            alt.Tooltip('avg_roi:Q', title='Avg ROI', format='.2%'),
            alt.Tooltip('loan_count:Q', title='Loan Count')
        ]
    ).properties(
        width=700,
        height=400,
        title='Average ROI by Industry (Top 10 by Balance)'
    )
    st.altair_chart(roi_bar, use_container_width=True)

# Additional existing visualization functions

def plot_status_distribution(df):
    """Create and display a pie chart of loan status distribution (excluding Paid Off)."""
    active_df = df[df["loan_status"] != "Paid Off"].copy()
    
    if active_df.empty:
        st.info("No active loans to display in status distribution chart.")
        return
        
    status_counts = active_df["loan_status"].value_counts(normalize=True)
    status_summary = pd.DataFrame({
        "status": status_counts.index.astype(str),
        "percentage": status_counts.values,
        "count": active_df["loan_status"].value_counts().values,
        "balance": active_df.groupby("loan_status")["net_balance"].sum().reindex(status_counts.index).values
    })
    
    status_summary["color"] = status_summary["status"].apply(
        lambda x: LOAN_STATUS_COLORS.get(x, "#808080")
    )
    
    loan_ids_by_status = {}
    for status in active_df["loan_status"].unique():
        loans = active_df[active_df["loan_status"] == status]["loan_id"].tolist()
        loan_ids_by_status[status] = ", ".join(loans)
    
    status_summary["loan_ids"] = status_summary["status"].map(loan_ids_by_status)
    
    st.caption("Note: 'Paid Off' loans are excluded from this chart")
    
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
        title="Distribution of Active Loan Status"
    )
    
    st.altair_chart(pie_chart, use_container_width=True)

def plot_roi_distribution(df):
    """Create and display a bar chart showing ROI distribution by loan."""
    roi_df = df[df['total_invested'] > 0].copy()
    roi_df = roi_df.sort_values('current_roi', ascending=False)
    
    if roi_df.empty:
        st.info("No loans with investment data to display ROI distribution.")
        return
        
    roi_chart = alt.Chart(roi_df).mark_bar().encode(
        x=alt.X("loan_id:N", title="Loan ID", sort="-y", 
                axis=alt.Axis(labelAngle=-90, labelLimit=150)),
        y=alt.Y("current_roi:Q", title="Return on Investment (ROI)",
                axis=alt.Axis(format=".0%", grid=True)),
        color=alt.Color("current_roi:Q",
            scale=alt.Scale(domain=[-0.5, 0, 0.5], range=["#ff0505", "#ffc302", "#2ca02c"]),
            legend=alt.Legend(title="ROI", format=".0%")),
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
        title="Return on Investment by Loan"
    )

    st.altair_chart(roi_chart, use_container_width=True)

def plot_irr_by_partner(df):
    """Create and display a bar chart showing average IRR by partner."""
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
        x=alt.X('avg_irr:Q', title="Average IRR", 
                axis=alt.Axis(format=".0%", grid=True)),
        y=alt.Y('partner_source:N', title="Partner", sort='-x',
                axis=alt.Axis(labelLimit=150)),
        color=alt.Color('avg_irr:Q',
            scale=alt.Scale(domain=[-0.1, 0, 0.5], range=["#d62728", "#ffc302", "#2ca02c"]),
            legend=alt.Legend(title="IRR", format=".0%")),
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
        title="Average IRR by Partner (Paid-off Loans)"
    )
    
    text = alt.Chart(irr_by_partner).mark_text(
        align='left', baseline='middle', dx=5, fontSize=12
    ).encode(
        x='avg_irr:Q',
        y='partner_source:N',
        text=alt.Text('deal_count:Q', format='d')
    )
    
    st.altair_chart(irr_chart + text, use_container_width=True)

def display_irr_analysis(df):
    """Display IRR analysis for paid-off loans."""
    st.subheader("IRR Analysis for Paid-Off Loans")
    
    paid_df = df[df['loan_status'] == "Paid Off"].copy()
    
    if paid_df.empty:
        st.info("No paid-off loans to analyze for IRR.")
        return
    
    weighted_realized_irr = (paid_df['realized_irr'] * paid_df['total_invested']).sum() / paid_df['total_invested'].sum()
    weighted_expected_irr = (paid_df['expected_irr'] * paid_df['total_invested']).sum() / paid_df['total_invested'].sum()
    
    avg_realized_irr = paid_df['realized_irr'].mean()
    avg_expected_irr = paid_df['expected_irr'].mean()
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Weighted Avg Realized IRR", 
                 f"{weighted_realized_irr:.2%}" if pd.notnull(weighted_realized_irr) else "N/A",
                 help="Weighted by investment amount")
        st.metric("Simple Avg Realized IRR", 
                 f"{avg_realized_irr:.2%}" if pd.notnull(avg_realized_irr) else "N/A",
                 help="Equal weight per loan")
    with col2:
        st.metric("Weighted Avg Expected IRR", 
                 f"{weighted_expected_irr:.2%}" if pd.notnull(weighted_expected_irr) else "N/A",
                 help="Based on expected returns")
        st.metric("Simple Avg Expected IRR", 
                 f"{avg_expected_irr:.2%}" if pd.notnull(avg_expected_irr) else "N/A",
                 help="Equal weight per loan")
    
    with st.expander("How IRR is Calculated"):
        st.markdown("""
        **Internal Rate of Return (IRR)** is the annualized rate of return.
        
        - **Realized IRR**: Uses actual cash flows (funding to payoff)
        - **Expected IRR**: Uses projected returns at maturity
        - Weighted average gives more importance to larger investments
        """)
    
    st.subheader("IRR by Loan")
    irr_display_df = paid_df.copy()
    irr_display_df['realized_irr_formatted'] = irr_display_df['realized_irr'].apply(
        lambda x: f"{x:.2%}" if pd.notnull(x) else "N/A"
    )
    irr_display_df['expected_irr_formatted'] = irr_display_df['expected_irr'].apply(
        lambda x: f"{x:.2%}" if pd.notnull(x) else "N/A"
    )
    
    irr_display_df['duration_days'] = (
        pd.to_datetime(irr_display_df['payoff_date']).dt.tz_localize(None) - 
        pd.to_datetime(irr_display_df['funding_date']).dt.tz_localize(None)
    ).dt.days
    
    irr_display_df['working_days'] = irr_display_df['duration_days'].apply(
        lambda x: max(0, round(x * 5/7)) if pd.notnull(x) else None
    )
    
    irr_columns = [
        'loan_id', 'deal_name', 'partner_source', 'funding_date', 'payoff_date',
        'duration_days', 'working_days', 'total_invested', 'total_paid', 
        'realized_irr_formatted', 'expected_irr_formatted'
    ]
    
    column_rename = {
        'loan_id': 'Loan ID',
        'deal_name': 'Deal Name',
        'partner_source': 'Partner',
        'funding_date': 'Funded',
        'payoff_date': 'Paid Off',
        'duration_days': 'Days',
        'working_days': 'Working Days',
        'total_invested': 'Invested',
        'total_paid': 'Paid',
        'realized_irr_formatted': 'Realized IRR',
        'expected_irr_formatted': 'Expected IRR'
    }
    
    irr_display = format_dataframe_for_display(
        irr_display_df, 
        columns=irr_columns,
        rename_map=column_rename
    )
    
    st.dataframe(irr_display, use_container_width=True, hide_index=True)

# Main application
def main():
    """Main application entry point."""
    st.title("Loan Tape Dashboard")
    
    last_updated = get_last_updated()
    st.caption(f"Data last updated: {last_updated}")
    
    # Load data
    loans_df = load_loan_summaries()
    deals_df = load_deals()
    
    # Process data
    df = prepare_loan_data(loans_df, deals_df)
    df = calculate_irr(df)
    
    # Sidebar filters
    st.sidebar.header("Filters")
    
    # Date filter
    if 'funding_date' in df.columns and not df['funding_date'].isna().all():
        min_date = df["funding_date"].min().date()
        max_date = df["funding_date"].max().date()
        
        use_date_filter = st.sidebar.checkbox("Filter by Funding Date", value=False)
        
        if use_date_filter:
            date_range = st.sidebar.date_input(
                "Select Date Range",
                [min_date, max_date],
                min_value=min_date,
                max_value=max_date
            )
            if len(date_range) == 2:
                filtered_df = df[(df["funding_date"].dt.date >= date_range[0]) & 
                                (df["funding_date"].dt.date <= date_range[1])]
            else:
                filtered_df = df.copy()
        else:
            filtered_df = df.copy()
    else:
        filtered_df = df.copy()
    
    # Status filter
    all_statuses = ["All"] + sorted(df["loan_status"].unique().tolist())
    selected_status = st.sidebar.selectbox("Filter by Status", all_statuses, index=0)
    
    if selected_status != "All":
        filtered_df = filtered_df[filtered_df["loan_status"] == selected_status]
    
    # Show filter summary
    st.sidebar.markdown("---")
    st.sidebar.write(f"**Showing:** {len(filtered_df)} of {len(df)} loans")
    
    # Create tabs
    tabs = st.tabs([
        "Summary", 
        "Capital Flow", 
        "Performance Analysis", 
        "Risk Analytics",
        "Loan Tape"
    ])
    
    with tabs[0]:  # Summary
        st.header("Portfolio Overview")
        
        # Top metrics in 4 columns
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            total_positions = len(filtered_df)
            paid_off = (filtered_df['loan_status'] == 'Paid Off').sum()
            st.metric("Total Positions", f"{total_positions}")
            st.caption(f"({paid_off} paid off)")
        with col2:
            total_deployed = filtered_df['csl_participation_amount'].sum()
            st.metric("Capital Deployed", f"${total_deployed:,.0f}")
        with col3:
            total_returned = filtered_df['total_paid'].sum()
            st.metric("Capital Returned", f"${total_returned:,.0f}")
        with col4:
            net_balance = filtered_df['net_balance'].sum()
            st.metric("Net Outstanding", f"${net_balance:,.0f}")
        
        st.markdown("---")
        
        # Status distribution
        st.subheader("Loan Status Distribution")
        plot_status_distribution(filtered_df)
        
        st.markdown("---")
        
        # ROI distribution
        st.subheader("ROI Distribution by Loan")
        plot_roi_distribution(filtered_df)
        
        st.markdown("---")
        
        # Top 5 positions
        st.subheader("Top 5 Outstanding Positions")
        top_positions = (
            filtered_df[filtered_df['is_unpaid']]
            .sort_values('net_balance', ascending=False)
            .head(5)
        )
        
        if not top_positions.empty:
            display_cols = ['loan_id', 'deal_name', 'loan_status', 'net_balance', 
                           'payment_performance', 'remaining_maturity_months']
            col_rename = {
                'loan_id': 'Loan ID',
                'deal_name': 'Deal Name',
                'loan_status': 'Status',
                'net_balance': 'Net Balance',
                'payment_performance': 'Payment Performance',
                'remaining_maturity_months': 'Months to Maturity'
            }
            top_display = format_dataframe_for_display(top_positions, display_cols, col_rename)
            st.dataframe(top_display, use_container_width=True, hide_index=True)
    
    with tabs[1]:  # Capital Flow
        st.header("Capital Flow Analysis")
        
        plot_capital_flow(filtered_df)
        
        st.markdown("---")
        
        plot_investment_net_position(filtered_df)
        
        st.markdown("---")
        
        st.subheader("Payment Performance by Cohort")
        plot_payment_performance_by_cohort(filtered_df)
        
        st.markdown("---")
        
        # IRR Analysis
        display_irr_analysis(filtered_df)
        
        st.markdown("---")
        
        st.subheader("Average IRR by Partner")
        plot_irr_by_partner(filtered_df)
    
    with tabs[2]:  # Performance Analysis
        st.header("Performance Analysis")
        
        plot_industry_performance_analysis(filtered_df)
        
        st.markdown("---")
        
        plot_fico_performance_analysis(filtered_df)
        
        st.markdown("---")
        
        plot_tib_performance_analysis(filtered_df)
    
    with tabs[3]:  # Risk Analytics
        st.header("Risk Analytics")
        
        # Calculate risk scores
        risk_df = calculate_risk_scores(filtered_df)
        
        if not risk_df.empty:
            # Explain risk formula
            with st.expander("How Risk Scores are Calculated"):
                st.markdown("""
                **Risk Score Formula:**
                ```
                Risk Score = Performance Gap × Status Multiplier × (1 + Overdue Factor)
                ```
                
                **Components:**
                - **Performance Gap**: 1 - Payment Performance (higher = worse)
                - **Status Multipliers**: 
                  - Active: 1.0
                  - Active - Frequently Late: 1.3
                  - Minor Delinquency: 1.5
                  - Past Delinquency: 1.2
                  - Moderate Delinquency: 2.0
                  - Late: 2.5
                  - Severe Delinquency: 3.0
                  - Default: 4.0
                  - Bankrupt/Severe: 5.0
                - **Overdue Factor**: Months past maturity / 12 (capped at 1.0)
                
                **Risk Bands:**
                - Low: 0-0.5
                - Moderate: 0.5-1.0
                - Elevated: 1.0-1.5
                - High: 1.5-2.0
                - Severe: 2.0+
                """)
            
            # Risk summary metrics
            col1, col2, col3 = st.columns(3)
            with col1:
                avg_risk = risk_df['risk_score'].mean()
                st.metric("Average Risk Score", f"{avg_risk:.2f}")
            with col2:
                high_risk_count = (risk_df['risk_score'] >= 1.5).sum()
                st.metric("High/Severe Risk Loans", f"{high_risk_count}")
            with col3:
                high_risk_balance = risk_df[risk_df['risk_score'] >= 1.5]['net_balance'].sum()
                st.metric("High Risk Balance", f"${high_risk_balance:,.0f}")
            
            st.markdown("---")
            
            # Top 10 highest risk loans
            st.subheader("Top 10 Highest Risk Loans")
            top_risk = risk_df.nlargest(10, 'risk_score')[
                ['loan_id', 'deal_name', 'loan_status', 'payment_performance',
                 'days_since_funding', 'days_past_maturity', 'status_multiplier',
                 'risk_score', 'net_balance']
            ].copy()
            
            # Format for display
            top_risk_display = top_risk.copy()
            top_risk_display['payment_performance'] = top_risk_display['payment_performance'].map(
                lambda x: f"{x:.1%}" if pd.notnull(x) else "N/A"
            )
            top_risk_display['risk_score'] = top_risk_display['risk_score'].map(
                lambda x: f"{x:.2f}" if pd.notnull(x) else "N/A"
            )
            top_risk_display['status_multiplier'] = top_risk_display['status_multiplier'].map(
                lambda x: f"{x:.1f}x" if pd.notnull(x) else "N/A"
            )
            top_risk_display['net_balance'] = top_risk_display['net_balance'].map(
                lambda x: f"${x:,.0f}" if pd.notnull(x) else "N/A"
            )
            
            top_risk_display.columns = [
                'Loan ID', 'Deal Name', 'Status', 'Payment Perf',
                'Days Funded', 'Days Overdue', 'Status Mult',
                'Risk Score', 'Net Balance'
            ]
            
            st.dataframe(top_risk_display, use_container_width=True, hide_index=True)
            
            st.markdown("---")
            
            # Risk band distribution
            st.subheader("Risk Score Distribution")
            band_summary = risk_df.groupby("risk_band").agg(
                loan_count=("loan_id", "count"),
                net_balance=("net_balance", "sum")
            ).reset_index()
            
            if not band_summary.empty:
                # Define proper order for risk bands
                risk_band_order = ["Low (0-0.5)", "Moderate (0.5-1.0)", "Elevated (1.0-1.5)", 
                                  "High (1.5-2.0)", "Severe (2.0+)"]
                
                risk_bar = alt.Chart(band_summary).mark_bar().encode(
                    x=alt.X("risk_band:N", title="Risk Band", 
                           sort=risk_band_order),
                    y=alt.Y("loan_count:Q", title="Number of Loans"),
                    color=alt.Color("risk_band:N",
                        scale=alt.Scale(
                            domain=risk_band_order,
                            range=["#2ca02c", "#98df8a", "#ffbb78", "#ff7f0e", "#d62728"]
                        ),
                        legend=alt.Legend(title="Risk Level", orient="right"),
                        sort=risk_band_order
                    ),
                    tooltip=[
                        alt.Tooltip("risk_band:N", title="Risk Band"),
                        alt.Tooltip("loan_count:Q", title="Loan Count"),
                        alt.Tooltip("net_balance:Q", title="Net Balance", format="$,.0f")
                    ]
                ).properties(
                    width=700,
                    height=350,
                    title="Loan Count by Risk Band (Active Loans Only)"
                )
                
                st.altair_chart(risk_bar, use_container_width=True)
        else:
            st.info("No active loans to calculate risk scores.")
    
    with tabs[4]:  # Loan Tape
        st.header("Complete Loan Tape")
        
        # Select columns for display
        display_columns = ["loan_id", "deal_name", "partner_source", "loan_status",
                          "funding_date", "maturity_date", "csl_participation_amount",
                          "total_invested", "total_paid", "net_balance", "current_roi",
                          "payment_performance", "remaining_maturity_months"]
        
        column_rename = {
            "loan_id": "Loan ID",
            "deal_name": "Deal Name",
            "partner_source": "Partner",
            "loan_status": "Status",
            "funding_date": "Funded",
            "maturity_date": "Maturity",
            "csl_participation_amount": "Capital Deployed",
            "total_invested": "Total Invested",
            "total_paid": "Total Paid",
            "net_balance": "Net Balance",
            "current_roi": "ROI",
            "payment_performance": "Payment Perf",
            "remaining_maturity_months": "Months Left"
        }
        
        loan_tape = format_dataframe_for_display(filtered_df, display_columns, column_rename)
        
        st.dataframe(loan_tape, use_container_width=True, hide_index=True)
        
        # Export functionality
        csv = loan_tape.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="Download Loan Tape as CSV",
            data=csv,
            file_name=f"loan_tape_{pd.Timestamp.today().strftime('%Y%m%d')}.csv",
            mime="text/csv"
        )

if __name__ == "__main__":
    main()
