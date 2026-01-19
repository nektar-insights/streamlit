# ============================================
# CSL CAPITAL - ALTAIR CHART THEME
# ============================================

import altair as alt

def register_csl_theme():
    """
    Register and enable the CSL Capital Altair theme.
    Call this once at the top of your Streamlit app.
    """
    
    # CSL Color Palette
    CSL_COLORS = {
        'green': '#00C853',
        'green_dark': '#00A344',
        'green_light': '#69F0AE',
        'black': '#1a1a1a',
        'white': '#ffffff',
        'gray': '#6b7280',
        'light_gray': '#f5f7fa',
        'positive': '#00C853',
        'negative': '#ef4444',
        'warning': '#f59e0b',
        'blue': '#3b82f6',
        'purple': '#8b5cf6',
        'orange': '#f97316',
    }
    
    # Sequential color scheme (for gradients, heatmaps)
    CSL_SEQUENTIAL = ['#e8f5e9', '#a5d6a7', '#69f0ae', '#00c853', '#00a344']
    
    # Categorical color scheme (for multiple series)
    CSL_CATEGORICAL = [
        '#00C853',  # CSL Green
        '#3b82f6',  # Blue
        '#f97316',  # Orange
        '#8b5cf6',  # Purple
        '#ef4444',  # Red
        '#06b6d4',  # Cyan
        '#f59e0b',  # Amber
        '#ec4899',  # Pink
    ]
    
    def csl_theme():
        return {
            'config': {
                # Background
                'background': '#ffffff',
                
                # Title styling
                'title': {
                    'fontSize': 16,
                    'fontWeight': 600,
                    'color': '#1a1a1a',
                    'anchor': 'start',
                    'offset': 10,
                },
                
                # Axis styling
                'axis': {
                    'labelFontSize': 11,
                    'labelColor': '#6b7280',
                    'titleFontSize': 12,
                    'titleColor': '#1a1a1a',
                    'titleFontWeight': 500,
                    'gridColor': '#e5e7eb',
                    'gridOpacity': 0.5,
                    'domainColor': '#e5e7eb',
                    'tickColor': '#e5e7eb',
                },
                
                'axisX': {
                    'labelAngle': 0,
                    'grid': False,
                },
                
                'axisY': {
                    'grid': True,
                    'domain': False,
                    'ticks': False,
                },
                
                # Legend styling
                'legend': {
                    'labelFontSize': 11,
                    'labelColor': '#6b7280',
                    'titleFontSize': 12,
                    'titleColor': '#1a1a1a',
                    'symbolSize': 100,
                    'orient': 'top',
                    'direction': 'horizontal',
                },
                
                # View / chart area
                'view': {
                    'strokeWidth': 0,
                    'continuousHeight': 300,
                    'continuousWidth': 400,
                },
                
                # Mark defaults
                'bar': {
                    'color': '#00C853',
                    'cornerRadiusTopLeft': 4,
                    'cornerRadiusTopRight': 4,
                },
                
                'line': {
                    'color': '#00C853',
                    'strokeWidth': 2.5,
                },
                
                'point': {
                    'color': '#00C853',
                    'filled': True,
                    'size': 60,
                },
                
                'area': {
                    'color': '#00C853',
                    'opacity': 0.3,
                    'line': True,
                },
                
                'arc': {
                    'stroke': '#ffffff',
                    'strokeWidth': 2,
                },
                
                # Color schemes
                'range': {
                    'category': CSL_CATEGORICAL,
                    'ordinal': CSL_SEQUENTIAL,
                    'ramp': CSL_SEQUENTIAL,
                },
            }
        }
    
    # Register the theme
    alt.themes.register('csl_capital', csl_theme)
    alt.themes.enable('csl_capital')
    
    return CSL_COLORS, CSL_CATEGORICAL


# ============================================
# EXAMPLE CHART FUNCTIONS
# ============================================

def create_bar_chart(data, x, y, title="", color_field=None, horizontal=False):
    """
    Create a styled bar chart.
    
    Parameters:
    - data: pandas DataFrame
    - x: column name for x-axis
    - y: column name for y-axis
    - title: chart title
    - color_field: optional column for color encoding
    - horizontal: if True, creates horizontal bars
    """
    if horizontal:
        x, y = y, x
    
    chart = alt.Chart(data).mark_bar().encode(
        x=alt.X(x, title=x.replace('_', ' ').title()),
        y=alt.Y(y, title=y.replace('_', ' ').title()),
        tooltip=[x, y]
    ).properties(
        title=title,
        height=350
    )
    
    if color_field:
        chart = chart.encode(color=alt.Color(color_field, legend=alt.Legend(title=color_field)))
    
    return chart


def create_line_chart(data, x, y, title="", color_field=None, include_points=True):
    """
    Create a styled line chart.
    
    Parameters:
    - data: pandas DataFrame
    - x: column name for x-axis
    - y: column name for y-axis
    - title: chart title
    - color_field: optional column for multiple lines
    - include_points: if True, adds points to the line
    """
    base = alt.Chart(data).encode(
        x=alt.X(x, title=x.replace('_', ' ').title()),
        y=alt.Y(y, title=y.replace('_', ' ').title()),
        tooltip=[x, y]
    )
    
    if color_field:
        base = base.encode(color=alt.Color(color_field, legend=alt.Legend(title=color_field)))
    
    line = base.mark_line()
    
    if include_points:
        points = base.mark_point(filled=True, size=50)
        chart = line + points
    else:
        chart = line
    
    return chart.properties(title=title, height=350)


def create_area_chart(data, x, y, title="", color_field=None):
    """
    Create a styled area chart with gradient fill.
    """
    chart = alt.Chart(data).mark_area(
        line=True,
        opacity=0.3
    ).encode(
        x=alt.X(x, title=x.replace('_', ' ').title()),
        y=alt.Y(y, title=y.replace('_', ' ').title()),
        tooltip=[x, y]
    ).properties(
        title=title,
        height=350
    )
    
    if color_field:
        chart = chart.encode(color=alt.Color(color_field))
    
    return chart


def create_donut_chart(data, theta, color, title=""):
    """
    Create a styled donut/pie chart.
    """
    return alt.Chart(data).mark_arc(innerRadius=60).encode(
        theta=alt.Theta(theta, type='quantitative'),
        color=alt.Color(color, type='nominal'),
        tooltip=[color, theta]
    ).properties(
        title=title,
        height=300,
        width=300
    )


def create_positive_negative_bar(data, x, y, title=""):
    """
    Create a bar chart with positive (green) and negative (red) coloring.
    Useful for P&L, delta values, etc.
    """
    return alt.Chart(data).mark_bar().encode(
        x=alt.X(x, title=x.replace('_', ' ').title()),
        y=alt.Y(y, title=y.replace('_', ' ').title()),
        color=alt.condition(
            alt.datum[y] > 0,
            alt.value('#00C853'),  # Positive = green
            alt.value('#ef4444')   # Negative = red
        ),
        tooltip=[x, y]
    ).properties(
        title=title,
        height=350
    )


# ============================================
# USAGE EXAMPLE
# ============================================
"""
# In your Streamlit app:

import streamlit as st
import pandas as pd
from altair_theme import register_csl_theme, create_bar_chart, create_line_chart

# Register theme (do this once at app startup)
CSL_COLORS, CSL_CATEGORICAL = register_csl_theme()

# Example data
df = pd.DataFrame({
    'month': ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun'],
    'deals': [12, 18, 15, 22, 19, 25],
    'volume': [150000, 220000, 180000, 290000, 240000, 310000]
})

# Create charts
bar = create_bar_chart(df, 'month', 'deals', title='Deals by Month')
line = create_line_chart(df, 'month', 'volume', title='Volume Trend')

# Display in Streamlit
st.altair_chart(bar, use_container_width=True)
st.altair_chart(line, use_container_width=True)
"""
