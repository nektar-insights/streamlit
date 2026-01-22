# pages/investment_rules.py
"""
Investment Committee Rules of Engagement

Reference page displaying CSL Capital's investment criteria, voting requirements,
and placement sizing guidelines for the Investment Committee.
"""

import streamlit as st
from utils.config import setup_page, PRIMARY_COLOR

# Page setup
setup_page("CSL Capital | Investment Rules")

st.title("Investment Committee Rules of Engagement")
st.markdown("Reference guide for deal evaluation criteria, voting requirements, and placement sizing.")

st.markdown("---")

# =============================================================================
# HARD FAIL CRITERIA
# =============================================================================
st.header("Hard Fail Criteria")
st.markdown("""
The following criteria result in **automatic deal decline**. No exceptions.
""")

hard_fail_data = """
| Criteria | Threshold |
|----------|-----------|
| Time in Business (TIB) | < 5 years |
| FICO Score | < 620 |
| Lien Position | 3rd or worse |
| Debt Leverage | > 25% |
| Net MOIC | < 1.15x |
"""

st.markdown(hard_fail_data)

st.markdown("---")

# =============================================================================
# UNANIMOUS VOTE REQUIREMENTS
# =============================================================================
st.header("Unanimous Vote Requirements")
st.markdown("""
Voting requirements vary based on placement size.
""")

vote_data = """
| Placement Size | Requirement |
|----------------|-------------|
| Below $10K | No unanimous vote required |
| $10K - $15K | Standard committee review |
| Above $15K | Unanimous written agreement required |
| $100K+ | Video or in-person meeting required |
"""

st.markdown(vote_data)

st.markdown("---")

# =============================================================================
# PLACEMENT SIZING GUIDELINES
# =============================================================================
st.header("Placement Sizing Guidelines")
st.markdown("""
Maximum allocation and approval thresholds.
""")

sizing_data = """
| Parameter | Rule |
|-----------|------|
| Maximum Allocation | $40K or 10% of deal (whichever is lower) |
| Below $5K | No committee approval needed |
| Above $15K | Requires unanimous agreement |
| $100K+ | Requires video/in-person meeting |
"""

st.markdown(sizing_data)

st.markdown("---")

# =============================================================================
# DEAL SIZE PREFERENCE
# =============================================================================
st.header("Deal Size Preference")

st.markdown("""
- Avoid smaller deals
- Focus on opportunities where CSL participation adds meaningful value
- Prioritize quality over quantity in deal selection
""")

st.markdown("---")

# =============================================================================
# QUICK REFERENCE CHECKLIST
# =============================================================================
st.header("Quick Reference Checklist")
st.markdown("""
**Before Approving Any Deal:**
""")

# Create columns for a nicer checklist layout
col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    **Eligibility Criteria:**
    - [ ] TIB ≥ 5 years
    - [ ] FICO ≥ 620
    - [ ] Position is 1st or 2nd
    - [ ] Debt leverage < 25%
    - [ ] Net MOIC ≥ 1.15x
    """)

with col2:
    st.markdown("""
    **Sizing & Approval:**
    - [ ] Allocation ≤ $40K and ≤ 10% of deal
    - [ ] Unanimous written agreement (if > $15K)
    - [ ] Video/in-person meeting scheduled (if ≥ $100K)
    """)

st.markdown("---")

# =============================================================================
# SUMMARY CARD
# =============================================================================
st.markdown(f"""
<div style="
    background: linear-gradient(135deg, {PRIMARY_COLOR}11, {PRIMARY_COLOR}22);
    border-left: 4px solid {PRIMARY_COLOR};
    padding: 20px;
    border-radius: 8px;
    margin: 20px 0;
">
    <h3 style="margin-top: 0; color: {PRIMARY_COLOR};">Key Thresholds Summary</h3>
    <table style="width: 100%; border-collapse: collapse; color: #1a1a1a;">
        <tr>
            <td style="padding: 8px 0; color: #1a1a1a;"><strong>Min TIB:</strong></td>
            <td style="padding: 8px 0; color: #1a1a1a;">5 years</td>
            <td style="padding: 8px 0; color: #1a1a1a;"><strong>Min FICO:</strong></td>
            <td style="padding: 8px 0; color: #1a1a1a;">620</td>
        </tr>
        <tr>
            <td style="padding: 8px 0; color: #1a1a1a;"><strong>Max Lien:</strong></td>
            <td style="padding: 8px 0; color: #1a1a1a;">2nd position</td>
            <td style="padding: 8px 0; color: #1a1a1a;"><strong>Max Debt Leverage:</strong></td>
            <td style="padding: 8px 0; color: #1a1a1a;">25%</td>
        </tr>
        <tr>
            <td style="padding: 8px 0; color: #1a1a1a;"><strong>Min Net MOIC:</strong></td>
            <td style="padding: 8px 0; color: #1a1a1a;">1.15x</td>
            <td style="padding: 8px 0; color: #1a1a1a;"><strong>Max Allocation:</strong></td>
            <td style="padding: 8px 0; color: #1a1a1a;">$40K or 10%</td>
        </tr>
    </table>
</div>
""", unsafe_allow_html=True)

# Footer
st.markdown("---")
st.caption("*Last Updated: January 2026*")
