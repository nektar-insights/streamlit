# pages/loan_tape.py
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

# Page Configuration
st.set_page_config(
    page_title="CSL Capital | Loan Tape",
    layout="wide",
)
inject_global_styles()
inject_logo()

# Constants
PLATFORM_FEE = 0.03
LOAN_STATUS_COLORS = {
    "Active": "#2ca02c",
    "Late": "#ffbb78",
    "Default": "#ff7f0e",
    "Bankrupt": "#d62728",
    "Severe": "#990000",
    "Minor Delinquency": "#88c999",
    "Moderate Delinquency": "#ffcc88",
    "Past Delinquency": "#aaaaaa",
    "Severe Delinquency": "#cc4444",
    "Active - Frequently Late": "#66aa66"
}

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

# Data Loading Functions
supabase = get_supabase_client()

@st.cache_data(ttl=3600)
def load_loan_summaries():
    res = supabase.table("loan_summaries").select("*").execute()
    return pd.DataFrame(res.data)

@st.cache_data(ttl=3600)
def load_deals():
    res = supabase.table("deals").select("*").execute()
    return pd.DataFrame(res.data)

@st.cache_data(ttl=3600)
def load_naics_sector_risk():
    res = supabase.table("naics_sector_risk_profile").select("*").execute()
    return pd.DataFrame(res.data)

@st.cache_data(ttl=3600)
def load_loan_schedules():
    res = supabase.table("loan_schedules").select("*").execute()
    return pd.DataFrame(res.data)

@st.cache_data(ttl=3600)
def get_last_updated():
    try:
        timestamps = []
        
        # Try loan_summaries
        try:
            res = supabase.table("loan_summaries").select("updated_at").order("updated_at", desc=True).limit(1).execute()
            if res.data and res.data[0].get('updated_at'):
                timestamps.append(pd.to_datetime(res.data[0]['updated_at']))
        except:
            try:
                res = supabase.table("loan_summaries").select("created_at").order("created_at", desc=True).limit(1).execute()
                if res.data and res.data[0].get('created_at'):
                    timestamps.append(pd.to_datetime(res.data[0]['created_at']))
            except:
                pass
        
        # Try deals
        try:
            res = supabase.table("deals").select("updated_at").order("updated_at", desc=True).limit(1).execute()
            if res.data and res.data[0].get('updated_at'):
                timestamps.append(pd.to_datetime(res.data[0]['updated_at']))
        except:
            try:
                res = supabase.table("deals").select("created_at").order("created_at", desc=True).limit(1).execute()
                if res.data and res.data[0].get('created_at'):
                    timestamps.append(pd.to_datetime(res.data[0]['created_at']))
            except:
                pass
        
        # Try loan_schedules
        try:
            res = supabase.table("loan_schedules").select("updated_at").order("updated_at", desc=True).limit(1).execute()
            if res.data and res.data[0].get('updated_at'):
                timestamps.append(pd.to_datetime(res.data[0]['updated_at']))
        except:
            try:
                res = supabase.table("loan_schedules").select("created_at").order("created_at", desc=True).limit(1).execute()
                if res.data and res.data[0].get('created_at'):
                    timestamps.append(pd.to_datetime(res.data[0]['created_at']))
            except:
                pass
            
        if timestamps:
            last_updated = max(timestamps)
            return last_updated.strftime('%B %d, %Y at %I:%M %p')
        else:
            return "Unable to determine"
    except Exception as e:
        return f"Error: {str(e)}"

def prepare_loan_data(loans_df, deals_df):
    if not loans_df.empty and not deals_df.empty:
        df = loans_df.merge(
            deals_df[["loan_id", "deal_name", "partner_source", "industry", "commission","fico", "tib"]], 
            on="loan_id", 
            how="left"
        )
    else:
        df = loans_df.copy()
    
    for date_col in ["funding_date", "maturity_date", "payoff_date"]:
        if date_col in df.columns:
            try:
                df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
            except:
                df[date_col] = pd.NaT
    
    df['commission'] = pd.to_numeric(df['commission'], errors='coerce').fillna(0)
    
    df['total_invested'] = (
        df['csl_participation_amount'] + 
        (df['csl_participation_amount'] * PLATFORM_FEE) +
        (df['csl_participation_amount'] * df['commission_fee'])
    )
    
    df['commission_fees'] = df['csl_participation_amount'] * df['commission_fee']
    df['platform_fees'] = df['csl_participation_amount'] * PLATFORM_FEE
    df['net_balance'] = df['total_invested'] - df['total_paid']
    
    df['current_roi'] = df.apply(
        lambda x: (x['total_paid'] / x['total_invested']) - 1 if x['total_invested'] > 0 else 0, 
        axis=1
    )
    
    df['is_unpaid'] = df['loan_status'] != "Paid Off"
    
    try:
        today = pd.Timestamp.today().tz_localize(None)
        df["days_since_funding"] = df["funding_date"].apply(
            lambda x: (today - pd.to_datetime(x).tz_localize(None)).days if pd.notnull(x) else 0
        )
    except:
        df["days_since_funding"] = 0
    
    df["remaining_maturity_months"] = 0.0
    
    try:
        active_loans_mask = (df['loan_status'] != "Paid Off") & (df['maturity_date'] > pd.Timestamp.today())
        if 'maturity_date' in df.columns:
            today = pd.Timestamp.today().tz_localize(None)
            df.loc[active_loans_mask, "remaining_maturity_months"] = df.loc[active_loans_mask, 'maturity_date'].apply(
                lambda x: (pd.to_datetime(x).tz_localize(None) - today).days / 30 if pd.notnull(x) else 0
            )
    except:
        pass
    
    try:
        df['cohort'] = df['funding_date'].dt.to_period('Q').astype(str)
        df['funding_month'] = df['funding_date'].dt.to_period('M')
    except:
        df['cohort'] = 'Unknown'
        df['funding_month'] = pd.NaT
    
    if 'industry' in df.columns:
        df['sector_code'] = df['industry'].astype(str).str[:2]
    
    return df

def calculate_irr(df):
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
        except:
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
        except:
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
    except:
        result_df['realized_irr'] = None
        result_df['expected_irr'] = None
        result_df['realized_irr_pct'] = "N/A"
        result_df['expected_irr_pct'] = "N/A"
    
    return result_df

def calculate_risk_scores(df):
    risk_df = df[df['loan_status'] != 'Paid Off'].copy()
    
    if risk_df.empty:
        return risk_df
    
    risk_df['performance_gap'] = 1 - risk_df['payment_performance'].clip(upper=1.0)
    risk_df['status_multiplier'] = risk_df['loan_status'].map(STATUS_RISK_MULTIPLIERS).fillna(1.0)
    
    today = pd.Timestamp.today().tz_localize(None)
    risk_df['days_past_maturity'] = risk_df['maturity_date'].apply(
        lambda x: max(0, (today - pd.to_datetime(x).tz_localize(None)).days) if pd.notnull(x) else 0
    )
    risk_df['overdue_factor'] = (risk_df['days_past_maturity'] / 30).clip(upper=12) / 12
    
    risk_df['risk_score'] = (
        risk_df['performance_gap'] * 
        risk_df['status_multiplier'] * 
        (1 + risk_df['overdue_factor'])
    ).clip(upper=5.0)
    
    risk_bins = [0, 0.5, 1.0, 1.5, 2.0, 5.0]
    risk_labels = ["Low (0-0.5)", "Moderate (0.5-1.0)", "Elevated (1.0-1.5)", "High (1.5-2.0)", "Severe (2.0+)"]
    risk_df["risk_band"] = pd.cut(risk_df["risk_score"], bins=risk_bins, labels=risk_labels)
    
    return risk_df

def calculate_expected_payment_to_date(row):
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
    except:
        return 0

# Main application
def main():
    st.title("Loan Tape Dashboard")
    
    last_updated = get_last_updated()
    st.caption(f"Data last updated: {last_updated}")
    
    loans_df = load_loan_summaries()
    deals_df = load_deals()
    
    df = prepare_loan_data(loans_df, deals_df)
    df = calculate_irr(df)
    
    st.sidebar.header("Filters")
    
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
    
    all_statuses = ["All"] + sorted(df["loan_status"].unique().tolist())
    selected_status = st.sidebar.selectbox("Filter by Status", all_statuses, index=0)
    
    if selected_status != "All":
        filtered_df = filtered_df[filtered_df["loan_status"] == selected_status]
    
    st.sidebar.markdown("---")
    st.sidebar.write(f"**Showing:** {len(filtered_df)} of {len(df)} loans")
    
    tabs = st.tabs([
        "Summary", 
        "Capital Flow", 
        "Performance Analysis", 
        "Risk Analytics",
        "Loan Tape"
    ])
    
    with tabs[0]:
        st.header("Portfolio Overview")
        
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
    
    with tabs[3]:
        st.header("Risk Analytics")
        
        risk_df = calculate_risk_scores(filtered_df)
        
        if not risk_df.empty:
            with st.expander("How Risk Scores are Calculated"):
                st.markdown("""
                **Risk Score Formula:**
                ```
                Risk Score = Performance Gap × Status Multiplier × (1 + Overdue Factor)
                ```
                
                **Components:**
                - **Performance Gap**: 1 - Payment Performance
                - **Status Multipliers**: Active=1.0, Late=2.5, Default=4.0, Bankrupt=5.0
                - **Overdue Factor**: Months past maturity / 12
                
                **Risk Bands:**
                - Low: 0-0.5
                - Moderate: 0.5-1.0
                - Elevated: 1.0-1.5
                - High: 1.5-2.0
                - Severe: 2.0+
                """)
            
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
            
            st.subheader("Top 10 Highest Risk Loans")
            top_risk = risk_df.nlargest(10, 'risk_score')[
                ['loan_id', 'deal_name', 'loan_status', 'payment_performance',
                 'days_since_funding', 'days_past_maturity', 'status_multiplier',
                 'risk_score', 'net_balance']
            ].copy()
            
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
            
            st.subheader("Risk Score Distribution")
            band_summary = risk_df.groupby("risk_band").agg(
                loan_count=("loan_id", "count"),
                net_balance=("net_balance", "sum")
            ).reset_index()
            
            if not band_summary.empty:
                risk_band_order = ["Low (0-0.5)", "Moderate (0.5-1.0)", "Elevated (1.0-1.5)", 
                                  "High (1.5-2.0)", "Severe (2.0+)"]
                
                risk_bar = alt.Chart(band_summary).mark_bar().encode(
                    x=alt.X("risk_band:N", title="Risk Band", sort=risk_band_order),
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

if __name__ == "__main__":
    main()
