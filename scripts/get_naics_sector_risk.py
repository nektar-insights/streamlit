# scripts/get_naics_sector_risk.py
from utils.imports import *

def get_naics_sector_risk():
    """
    Fetch NAICS sector risk profile data from Supabase
    Returns a DataFrame with sector codes, names, and risk profiles
    """
    supabase = get_supabase_client()
    
    try:
        res = supabase.table("naics_sector_risk_profile").select("*").execute()
        df = pd.DataFrame(res.data)
        
        # Ensure sector_code is string for joining
        if 'sector_code' in df.columns:
            df['sector_code'] = df['sector_code'].astype(str).str.zfill(2)
        
        return df
    except Exception as e:
        st.error(f"Error fetching NAICS sector risk data: {e}")
        return pd.DataFrame()
