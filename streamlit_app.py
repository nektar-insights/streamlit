# ----------------------------
# Monthly trend charts - IMPROVED VERSION
# ----------------------------
st.subheader("Total Funded Amount by Month")
funded_chart = alt.Chart(monthly_funded).mark_bar(
    size=40,  # Reduced bar size to prevent overlap
    color=PRIMARY_COLOR,
    cornerRadiusTopLeft=3,
    cornerRadiusTopRight=3
).encode(
    x=alt.X("month_date:T", 
            axis=alt.Axis(labelAngle=-45, title="Month", format="%b %Y", labelPadding=10)),
    y=alt.Y("total_funded_amount:Q", 
            title="Total Funded ($)", 
            axis=alt.Axis(format="$.2s", titlePadding=20, labelPadding=5)),
    tooltip=[
        alt.Tooltip("month_date:T", title="Month", format="%B %Y"),
        alt.Tooltip("total_funded_amount:Q", title="Total Funded Amount", format="$,.0f")
    ]
).resolve_scale(
    x='independent'
)

funded_avg = alt.Chart(monthly_funded).mark_rule(
    color="gray", 
    strokeWidth=2, 
    strokeDash=[4, 2],
    opacity=0.7
).encode(
    y=alt.Y("mean(total_funded_amount):Q")
)

# Add regression line
funded_regression = alt.Chart(monthly_funded).mark_line(
    color="#1f77b4", 
    strokeWidth=3
).transform_regression(
    'month_date', 'total_funded_amount'
).encode(
    x='month_date:T',
    y='total_funded_amount:Q'
)

st.altair_chart(
    (funded_chart + funded_avg + funded_regression).properties(
        height=400,
        width=800,
        padding={"left": 80, "top": 20, "right": 20, "bottom": 60}
    ).resolve_scale(x='independent'), 
    use_container_width=True
)

st.subheader("Total Deal Count by Month")
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

st.altair_chart(
    (deal_chart + deal_avg + deal_regression).properties(
        height=400,
        width=800,
        padding={"left": 60, "top": 20, "right": 20, "bottom": 60}
    ), 
    use_container_width=True
)

st.subheader("Participation Trends by Month")
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

st.altair_chart(
    (participation_chart + participation_avg + participation_regression).properties(
        height=400,
        width=800,
        padding={"left": 60, "top": 20, "right": 20, "bottom": 60}
    ), 
    use_container_width=True
)

st.subheader("Participation Amount by Month")
amount_chart = alt.Chart(monthly_participation).mark_bar(
    size=40,  # Reduced bar size
    color=PRIMARY_COLOR,
    cornerRadiusTopLeft=3,
    cornerRadiusTopRight=3
).encode(
    x=alt.X("month_date:T", 
            title="Month", 
            axis=alt.Axis(labelAngle=-45, format="%b %Y", labelPadding=10)),
    y=alt.Y("total_amount:Q", 
            title="Participation Amount ($)", 
            axis=alt.Axis(format="$.2s", titlePadding=20, labelPadding=5)),
    tooltip=[
        alt.Tooltip("month_date:T", title="Month", format="%B %Y"),
        alt.Tooltip("total_amount:Q", title="Participation Amount", format="$,.0f")
    ]
)

amount_avg = alt.Chart(monthly_participation).mark_rule(
    color="gray", 
    strokeWidth=2, 
    strokeDash=[4, 2],
    opacity=0.7
).encode(
    y=alt.Y("mean(total_amount):Q")
)

# Add regression line
amount_regression = alt.Chart(monthly_participation).mark_line(
    color="#17a2b8", 
    strokeWidth=3
).transform_regression(
    'month_date', 'total_amount'
).encode(
    x='month_date:T',
    y='total_amount:Q'
)

st.altair_chart(
    (amount_chart + amount_avg + amount_regression).properties(
        height=400,
        width=800,
        padding={"left": 80, "top": 20, "right": 20, "bottom": 60}
    ), 
    use_container_width=True
)

st.subheader("Monthly Participation Rate")
rate_line = alt.Chart(monthly_participation_ratio).mark_line(
    color="#e45756", 
    strokeWidth=4,
    point=alt.OverlayMarkDef(color="#e45756", size=80, filled=True)
).encode(
    x=alt.X("month_date:T", 
            title="Month", 
            axis=alt.Axis(labelAngle=-45, format="%b %Y", labelPadding=10)),
    y=alt.Y("participation_pct:Q", 
            title="Participation Rate", 
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
