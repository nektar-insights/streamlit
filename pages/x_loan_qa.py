"""
Loan QA - Data Quality Checks for Loan Portfolio
=================================================
This page performs quality assurance checks on loan data to identify potential data issues:
- Validates that maturity dates are after funding dates for active loans
- Highlights loans expiring in the next 30 days
- Provides actionable insights for data correction
"""

from utils.imports import *
from utils.config import setup_page, PRIMARY_COLOR, COLOR_PALETTE
from utils.data_loader import load_loan_summaries, load_deals
from utils.loan_tape_data import prepare_loan_data
from utils.display_components import (
    create_date_range_filter,
    create_partner_source_filter,
    create_status_filter,
)

# Setup page
setup_page("CSL Capital | Loan QA")

# Load and prepare data
@st.cache_data(ttl=3600)
def load_qa_data():
    """Load and prepare loan data for QA checks"""
    loans_df = load_loan_summaries()
    deals_df = load_deals()
    df = prepare_loan_data(loans_df, deals_df)
    return df

# Main QA Logic
def identify_date_issues(df):
    """
    Identify loans where maturity_date <= funding_date
    Only checks non-matured loans (not "Paid Off")
    """
    # Filter to non-matured loans
    active_loans = df[df['loan_status'] != 'Paid Off'].copy()

    # Find loans with invalid date logic
    invalid_dates = active_loans[
        (active_loans['maturity_date'].notna()) &
        (active_loans['funding_date'].notna()) &
        (active_loans['maturity_date'] <= active_loans['funding_date'])
    ].copy()

    return invalid_dates

def identify_expiring_soon(df, days=30):
    """
    Identify loans expiring in the next X days that are not matured
    """
    today = pd.Timestamp.now().normalize()
    cutoff_date = today + pd.Timedelta(days=days)

    # Filter to non-matured loans
    active_loans = df[df['loan_status'] != 'Paid Off'].copy()

    # Find loans expiring soon
    expiring_soon = active_loans[
        (active_loans['maturity_date'].notna()) &
        (active_loans['maturity_date'] >= today) &
        (active_loans['maturity_date'] <= cutoff_date)
    ].copy()

    # Calculate days until maturity
    expiring_soon['days_until_maturity'] = (
        expiring_soon['maturity_date'] - today
    ).dt.days

    return expiring_soon.sort_values('days_until_maturity')

def identify_matured_not_paid(df):
    """
    Identify loans that are past maturity date but not marked as paid off
    """
    today = pd.Timestamp.now().normalize()

    # Filter to non-matured loans that are past maturity
    overdue_loans = df[
        (df['loan_status'] != 'Paid Off') &
        (df['maturity_date'].notna()) &
        (df['maturity_date'] < today)
    ].copy()

    # Calculate days overdue
    overdue_loans['days_overdue'] = (
        today - overdue_loans['maturity_date']
    ).dt.days

    return overdue_loans.sort_values('days_overdue', ascending=False)

def format_qa_display_dataframe(df, columns_to_include):
    """Format dataframe for display with proper number and date formatting"""
    if df.empty:
        return df

    display_df = df[columns_to_include].copy()

    # Format currency columns
    currency_cols = ['csl_participation_amount', 'total_invested', 'total_paid',
                     'net_balance', 'total_funded_amount']
    for col in currency_cols:
        if col in display_df.columns:
            display_df[col] = display_df[col].apply(
                lambda x: f"${x:,.0f}" if pd.notna(x) else "-"
            )

    # Format percentage columns
    pct_cols = ['payment_performance', 'current_roi', 'commission_fee']
    for col in pct_cols:
        if col in display_df.columns:
            display_df[col] = display_df[col].apply(
                lambda x: f"{x:.1%}" if pd.notna(x) else "-"
            )

    # Format date columns
    date_cols = ['funding_date', 'maturity_date', 'payoff_date', 'created_date']
    for col in date_cols:
        if col in display_df.columns:
            display_df[col] = display_df[col].apply(
                lambda x: x.strftime('%Y-%m-%d') if pd.notna(x) else "-"
            )

    # Format integer columns
    int_cols = ['days_until_maturity', 'days_overdue']
    for col in int_cols:
        if col in display_df.columns:
            display_df[col] = display_df[col].apply(
                lambda x: f"{int(x)}" if pd.notna(x) else "-"
            )

    return display_df

# Load data
try:
    df = load_qa_data()

    # Sidebar filters
    st.sidebar.header("Filters")
    with st.sidebar:
        filtered_df, _ = create_date_range_filter(
            df, date_col="funding_date", label="Funding Date Range",
            key_prefix="loan_qa_date"
        )
        filtered_df, _ = create_partner_source_filter(
            filtered_df, key_prefix="loan_qa_partner"
        )
        filtered_df, _ = create_status_filter(
            filtered_df, status_col="loan_status", key_prefix="loan_qa_status"
        )

    st.sidebar.markdown("---")
    st.sidebar.write(f"**Total Loans:** {len(df):,}")
    st.sidebar.write(f"**Filtered Loans:** {len(filtered_df):,}")
    st.sidebar.write(f"**Active Loans:** {len(filtered_df[filtered_df['loan_status'] != 'Paid Off']):,}")

    # Main content
    st.title("🔍 Loan QA Dashboard")
    st.markdown("Quality assurance checks for loan portfolio data")
    st.markdown("---")

    # Run QA checks
    invalid_dates_df = identify_date_issues(filtered_df)
    expiring_soon_df = identify_expiring_soon(filtered_df, days=30)
    overdue_df = identify_matured_not_paid(filtered_df)

    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        active_count = len(filtered_df[filtered_df['loan_status'] != 'Paid Off'])
        st.metric(
            "Active Loans",
            f"{active_count:,}",
            help="Loans that are not marked as Paid Off"
        )

    with col2:
        st.metric(
            "Invalid Date Logic",
            f"{len(invalid_dates_df):,}",
            delta=f"-{len(invalid_dates_df)}" if len(invalid_dates_df) > 0 else "✓",
            delta_color="inverse",
            help="Loans where maturity date ≤ funding date"
        )

    with col3:
        st.metric(
            "Expiring in 30 Days",
            f"{len(expiring_soon_df):,}",
            help="Active loans with maturity date in next 30 days"
        )

    with col4:
        st.metric(
            "Past Maturity",
            f"{len(overdue_df):,}",
            help="Active loans past their maturity date"
        )

    st.markdown("---")

    # Create tabs for different QA checks
    tabs = st.tabs([
        "❌ Invalid Date Logic",
        "⚠️ Expiring Soon (30 Days)",
        "🔴 Past Maturity",
        "📊 Summary Statistics"
    ])

    # Tab 1: Invalid Date Logic
    with tabs[0]:
        st.subheader("Invalid Date Logic")
        st.markdown("""
        **Issue:** Loans where maturity date is on or before funding date
        **Impact:** Data integrity issue - loans should have maturity dates after funding
        **Action Required:** Review and correct maturity dates in source system
        """)

        if len(invalid_dates_df) > 0:
            st.error(f"⚠️ Found {len(invalid_dates_df)} loan(s) with invalid date logic")

            # Display columns
            display_cols = [
                'loan_id', 'partner_source', 'loan_status',
                'funding_date', 'maturity_date', 'created_date',
                'csl_participation_amount', 'payment_performance', 'current_roi'
            ]

            # Filter to available columns
            display_cols = [col for col in display_cols if col in invalid_dates_df.columns]

            # Format and display
            display_df = format_qa_display_dataframe(invalid_dates_df, display_cols)

            st.dataframe(
                display_df,
                use_container_width=True,
                height=400
            )

            # Export option
            csv = invalid_dates_df.to_csv(index=False)
            st.download_button(
                label="📥 Download Invalid Date Issues (CSV)",
                data=csv,
                file_name=f"invalid_date_loans_{pd.Timestamp.now().strftime('%Y%m%d')}.csv",
                mime="text/csv"
            )
        else:
            st.success("✅ No loans found with invalid date logic!")

    # Tab 2: Expiring Soon
    with tabs[1]:
        st.subheader("Loans Expiring in Next 30 Days")
        st.markdown("""
        **Status:** Active loans approaching maturity
        **Purpose:** Monitor loans that need attention for renewal or payoff
        **Action:** Contact borrowers for payoff or renewal discussions
        """)

        if len(expiring_soon_df) > 0:
            st.warning(f"⏰ {len(expiring_soon_df)} loan(s) expiring in next 30 days")

            # Display columns
            display_cols = [
                'loan_id', 'partner_source', 'loan_status',
                'days_until_maturity', 'funding_date', 'maturity_date',
                'csl_participation_amount', 'net_balance', 'payment_performance'
            ]

            # Filter to available columns
            display_cols = [col for col in display_cols if col in expiring_soon_df.columns]

            # Format and display
            display_df = format_qa_display_dataframe(expiring_soon_df, display_cols)

            st.dataframe(
                display_df,
                use_container_width=True,
                height=400
            )

            # Visual breakdown
            st.markdown("#### Days Until Maturity Breakdown")

            # Create bins for visualization
            expiring_soon_df['urgency'] = pd.cut(
                expiring_soon_df['days_until_maturity'],
                bins=[0, 7, 14, 30],
                labels=['0-7 days', '8-14 days', '15-30 days'],
                include_lowest=True
            )

            urgency_counts = expiring_soon_df['urgency'].value_counts().sort_index()

            col1, col2, col3 = st.columns(3)
            for i, (urgency, count) in enumerate(urgency_counts.items()):
                with [col1, col2, col3][i]:
                    color = ["🔴", "🟡", "🟢"][i]
                    st.metric(f"{color} {urgency}", f"{count}")

            # Export option
            csv = expiring_soon_df.to_csv(index=False)
            st.download_button(
                label="📥 Download Expiring Loans (CSV)",
                data=csv,
                file_name=f"expiring_loans_{pd.Timestamp.now().strftime('%Y%m%d')}.csv",
                mime="text/csv"
            )
        else:
            st.success("✅ No loans expiring in the next 30 days")

    # Tab 3: Past Maturity
    with tabs[2]:
        st.subheader("Loans Past Maturity Date")
        st.markdown("""
        **Status:** Active loans that are past their maturity date
        **Issue:** Loans may be delinquent or data may need updating
        **Action:** Review loan status, update payment status, or mark as paid off if applicable
        """)

        if len(overdue_df) > 0:
            st.error(f"🔴 {len(overdue_df)} loan(s) past maturity date")

            # Display columns
            display_cols = [
                'loan_id', 'partner_source', 'loan_status',
                'days_overdue', 'funding_date', 'maturity_date',
                'csl_participation_amount', 'total_paid', 'net_balance',
                'payment_performance'
            ]

            # Filter to available columns
            display_cols = [col for col in display_cols if col in overdue_df.columns]

            # Format and display
            display_df = format_qa_display_dataframe(overdue_df, display_cols)

            st.dataframe(
                display_df,
                use_container_width=True,
                height=400
            )

            # Severity breakdown
            st.markdown("#### Overdue Severity Breakdown")

            overdue_df['severity'] = pd.cut(
                overdue_df['days_overdue'],
                bins=[0, 30, 90, 180, float('inf')],
                labels=['0-30 days', '31-90 days', '91-180 days', '180+ days'],
                include_lowest=True
            )

            severity_counts = overdue_df['severity'].value_counts().sort_index()

            cols = st.columns(len(severity_counts))
            for i, (severity, count) in enumerate(severity_counts.items()):
                with cols[i]:
                    st.metric(f"{severity}", f"{count}")

            # Export option
            csv = overdue_df.to_csv(index=False)
            st.download_button(
                label="📥 Download Past Maturity Loans (CSV)",
                data=csv,
                file_name=f"past_maturity_loans_{pd.Timestamp.now().strftime('%Y%m%d')}.csv",
                mime="text/csv"
            )
        else:
            st.success("✅ No active loans past their maturity date")

    # Tab 4: Summary Statistics
    with tabs[3]:
        st.subheader("QA Summary Statistics")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("#### Data Quality Score")

            total_active = len(filtered_df[filtered_df['loan_status'] != 'Paid Off'])
            if total_active > 0:
                issues_count = len(invalid_dates_df)
                quality_score = max(0, 100 - (issues_count / total_active * 100))

                # Display quality score with color
                if quality_score >= 95:
                    color = "green"
                    status = "Excellent"
                elif quality_score >= 85:
                    color = "orange"
                    status = "Good"
                else:
                    color = "red"
                    status = "Needs Attention"

                st.markdown(f"### :{color}[{quality_score:.1f}%]")
                st.markdown(f"**Status:** {status}")
                st.markdown(f"**Issues Found:** {issues_count} of {total_active} active loans")
            else:
                st.info("No active loans to evaluate")

        with col2:
            st.markdown("#### Key Metrics")

            metrics_data = {
                "Metric": [
                    "Total Loans (Filtered)",
                    "Active Loans",
                    "Paid Off Loans",
                    "Invalid Date Issues",
                    "Expiring in 30 Days",
                    "Past Maturity",
                    "Data Quality Score"
                ],
                "Value": [
                    f"{len(filtered_df):,}",
                    f"{len(filtered_df[filtered_df['loan_status'] != 'Paid Off']):,}",
                    f"{len(filtered_df[filtered_df['loan_status'] == 'Paid Off']):,}",
                    f"{len(invalid_dates_df):,}",
                    f"{len(expiring_soon_df):,}",
                    f"{len(overdue_df):,}",
                    f"{quality_score:.1f}%" if total_active > 0 else "N/A"
                ]
            }

            st.dataframe(
                pd.DataFrame(metrics_data),
                use_container_width=True,
                hide_index=True
            )

        st.markdown("---")
        st.markdown("#### Portfolio Health Overview")

        # Create a summary chart
        if total_active > 0:
            summary_data = pd.DataFrame({
                'Category': [
                    'Valid Active Loans',
                    'Invalid Date Logic',
                    'Expiring Soon (30d)',
                    'Past Maturity'
                ],
                'Count': [
                    total_active - len(invalid_dates_df),
                    len(invalid_dates_df),
                    len(expiring_soon_df),
                    len(overdue_df)
                ],
                'Type': ['Good', 'Issue', 'Warning', 'Issue']
            })

            # Color mapping
            color_map = {
                'Good': PRIMARY_COLOR,
                'Issue': '#ea4335',
                'Warning': '#fbbc04'
            }

            chart = alt.Chart(summary_data).mark_bar().encode(
                x=alt.X('Count:Q', title='Number of Loans'),
                y=alt.Y('Category:N', title='', sort='-x'),
                color=alt.Color('Type:N',
                              scale=alt.Scale(domain=list(color_map.keys()),
                                            range=list(color_map.values())),
                              legend=None),
                tooltip=['Category', 'Count']
            ).properties(
                height=300
            )

            st.altair_chart(chart, use_container_width=True)

except Exception as e:
    st.error(f"Error loading data: {str(e)}")
    st.exception(e)

# Footer
st.markdown("---")
st.caption(f"Last updated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
