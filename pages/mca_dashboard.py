import streamlit as st
import pandas as pd
import plotly.express as px
from datetime import datetime, timedelta

# Import your existing Supabase connection
from utils import get_supabase_client  # or however you import it

st.set_page_config(page_title="MCA Portfolio", page_icon="💰", layout="wide")

def load_mca_data():
    """Load MCA portfolio data"""
    supabase = get_supabase_client()
    
    # Get latest extraction
    result = supabase.table('mca_deals').select('*').order('extracted_at', desc=True).execute()
    
    if result.data:
        return pd.DataFrame(result.data)
    return pd.DataFrame()

def render_mca_overview():
    """Render MCA portfolio overview"""
    st.title("💰 MCA Portfolio Dashboard")
    
    df = load_mca_data()
    
    if df.empty:
        st.warning("No MCA data available. Scraper may not have run yet.")
        return
    
    # Get latest data only
    latest_extraction = df['extracted_at'].max()
    latest_df = df[df['extracted_at'] == latest_extraction]
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Deals", f"{len(latest_df):,}")
    
    with col2:
        total_value = latest_df['purchase_price'].sum()
        st.metric("Portfolio Value", f"${total_value:,.0f}")
    
    with col3:
        current_balance = latest_df['current_balance'].sum()
        collection_rate = ((total_value - current_balance) / total_value * 100) if total_value > 0 else 0
        st.metric("Collection Rate", f"{collection_rate:.1f}%")
    
    with col4:
        past_due = latest_df['past_due_amount'].sum()
        st.metric("Past Due", f"${past_due:,.0f}")
    
    # Charts
    col1, col2 = st.columns(2)
    
    with col1:
        # Status distribution
        status_counts = latest_df['status'].value_counts()
        fig = px.pie(values=status_counts.values, names=status_counts.index, 
                    title="Deal Status Distribution")
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Sales rep performance
        rep_data = latest_df.groupby('sales_rep')['purchase_price'].sum().sort_values(ascending=False).head(10)
        fig = px.bar(x=rep_data.values, y=rep_data.index, orientation='h',
                    title="Top Sales Reps by Portfolio Value")
        st.plotly_chart(fig, use_container_width=True)
    
    # Recent deals table
    st.subheader("Recent Deals")
    display_cols = ['deal_id', 'dba', 'status', 'purchase_price', 'current_balance', 'sales_rep']
    st.dataframe(latest_df[display_cols].head(20), use_container_width=True)

if __name__ == "__main__":
    render_mca_overview()
