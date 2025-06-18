# pages/qbo_dashboard.py
from utils.imports import *

# -------------------------
# Setup: Supabase Connection
# -------------------------
supabase = get_supabase_client()

# -------------------------
# Load and Prepare Data
# -------------------------
@st.cache_data(ttl=3600)
def load_qbo_data():
    df_txn = pd.DataFrame(supabase.table("qbo_invoice_payments").select("*").execute().data)
    df_gl = pd.DataFrame(supabase.table("qbo_general_ledger").select("*").execute().data)
    return df_txn, df_gl

df, gl_df = load_qbo_data()

# Preprocess Transactions
for d in [df, gl_df]:
    if "total_amount" in d.columns:
        d["total_amount"] = pd.to_numeric(d["total_amount"], errors="coerce")
    if "balance" in d.columns:
        d["balance"] = pd.to_numeric(d["balance"], errors="coerce")
    if "amount" in d.columns:
        d["amount"] = pd.to_numeric(d["amount"], errors="coerce")
    if "date" in d.columns:
        d["date"] = pd.to_datetime(d["date"], errors="coerce")
    if "txn_date" in d.columns:
        d["txn_date"] = pd.to_datetime(d["txn_date"], errors="coerce")

st.title("QBO Dashboard")

# -------------------------
# Loan Performance by Deal (Updated for new schema)
# -------------------------
# -------------------------
# Loan Performance by Deal (Updated for new schema)
# -------------------------
st.subheader("Loan Performance by Deal")

# Debug information
with st.expander("Debug: Data Overview", expanded=False):
    st.write(f"Total records in qbo_invoice_payments: {len(df)}")
    if len(df) > 0:
        st.write("Transaction types available:")
        st.write(df['transaction_type'].value_counts())
        st.write("Sample data:")
        st.dataframe(df.head(3))
    else:
        st.warning("No data available in qbo_invoice_payments table")

if len(df) == 0:
    st.warning("No data available for loan performance analysis")
else:
    filtered_df = df[df["transaction_type"].isin(["Invoice", "Payment"])].copy()
    filtered_df = filtered_df[~filtered_df["customer_name"].isin(["CSL", "VEEM"])]
    
    if len(filtered_df) == 0:
        st.warning("No Invoice or Payment transactions found after filtering")
        st.info("Available transaction types: " + ", ".join(df["transaction_type"].unique()) if len(df) > 0 else "None")
    else:
        filtered_df["total_amount"] = filtered_df["total_amount"].abs()

        pivot = filtered_df.pivot_table(
            index="customer_name",
            columns="transaction_type",
            values="total_amount",
            aggfunc="sum",
            fill_value=0
        ).reset_index()

        # Ensure Invoice and Payment columns exist even if no data
        if "Invoice" not in pivot.columns:
            pivot["Invoice"] = 0
        if "Payment" not in pivot.columns:
            pivot["Payment"] = 0

        pivot["outstanding_balance"] = pivot.get("Invoice", 0) - pivot.get("Payment", 0)
        pivot["balance_ratio"] = pivot["outstanding_balance"] / pivot["Invoice"].where(pivot["Invoice"] > 0, 1)  # Avoid division by zero
        pivot["indicator"] = pivot["balance_ratio"].apply(
            lambda x: "🔴" if x >= 0.25 else ("🟡" if x >= 0.10 else "🟢")
        )

        pivot_display = pivot.copy()
        # Safely format columns that may or may not exist
        if "Invoice" in pivot_display.columns:
            pivot_display["Invoice"] = pivot_display["Invoice"].map("${:,.2f}".format)
        if "Payment" in pivot_display.columns:
            pivot_display["Payment"] = pivot_display["Payment"].map("${:,.2f}".format)
        else:
            pivot_display["Payment"] = "$0.00"  # Add Payment column with zeros if it doesn't exist

        pivot_display["outstanding_balance"] = pivot_display["outstanding_balance"].map("${:,.2f}".format)
        pivot_display["Customer Name"] = pivot_display["customer_name"]
        pivot_display["Balance (with Risk)"] = pivot_display["indicator"] + " " + pivot_display["outstanding_balance"]

        st.dataframe(
            pivot_display[["Customer Name", "Invoice", "Payment", "Balance (with Risk)"]]
            .sort_values("Balance (with Risk)", ascending=False),
            use_container_width=True
        )

# -------------------------
# Top Outstanding Balances
# -------------------------
top_balances = pivot.sort_values("outstanding_balance", ascending=False).head(15)

bar_chart = alt.Chart(top_balances).mark_bar().encode(
    x=alt.X("outstanding_balance:Q", title="Outstanding Balance ($)", axis=alt.Axis(format="$,.0f")),
    y=alt.Y("customer_name:N", sort="-x", title="Customer Name"),
    tooltip=["customer_name", alt.Tooltip("outstanding_balance:Q", format="$,.2f")]
).properties(
    width=800,
    height=400,
    title="Top 15 Deals by Outstanding Balance"
)

st.altair_chart(bar_chart, use_container_width=True)

# -------------------------
# Problem Loan Ratios
# -------------------------
problem_loans = pivot[pivot["Invoice"] > 0].copy()

if len(problem_loans) > 0:
    problem_loans["percentage"] = (problem_loans["outstanding_balance"] / problem_loans["Invoice"]) * 100
    problem_loans = problem_loans.sort_values("percentage", ascending=False).head(15)

    ratio_chart = (
        alt.Chart(problem_loans)
        .transform_calculate(
            risk_color="""
            datum.percentage >= 25 ? '#e45756' :
            datum.percentage >= 10 ? '#ffcc00' :
            '#34a853'
            """
        )
        .mark_bar()
        .encode(
            x=alt.X("percentage:Q", title="Balance as % of Invoice", axis=alt.Axis(format=".1f")),
            y=alt.Y("customer_name:N", title="Customer Name", sort="-x"),
            tooltip=["customer_name", alt.Tooltip("percentage:Q", format=".2f")],
            color=alt.Color("risk_color:N", scale=None, legend=None)
        )
        .properties(
            width=800,
            height=400,
            title="Problem Loan Ratios (Top 15 by Balance %)"
        )
    )

    st.altair_chart(ratio_chart, use_container_width=True)
else:
    st.info("No invoices found to calculate problem loan ratios")

# -------------------------
# Monthly Payment Trends (Using General Ledger)
# -------------------------
st.subheader("Monthly Payments Received")

# Try invoice/payment data first, fall back to GL data
payments_found = False
payment_source = ""

# First try: Invoice/Payment data
if not df.empty and "txn_date" in df.columns:
    payments_df = df[df["transaction_type"] == "Payment"].copy()
    
    if not payments_df.empty:
        payments_df["total_amount"] = payments_df["total_amount"].abs()
        payments_df["month"] = payments_df["txn_date"].dt.strftime("%Y-%m")
        payments_df["month_name"] = payments_df["txn_date"].dt.strftime("%b %Y")
        
        monthly_payments = payments_df.groupby(["month", "month_name"])["total_amount"].sum().reset_index()
        monthly_payments = monthly_payments.sort_values("month")
        
        payment_trend = alt.Chart(monthly_payments).mark_line(point=True).encode(
            x=alt.X("month_name:N", title="Month", sort=alt.SortField("month")),
            y=alt.Y("total_amount:Q", title="Total Payments ($)", axis=alt.Axis(format="$,.0f")),
            tooltip=["month_name", alt.Tooltip("total_amount", format="$,.2f")]
        ).properties(width=800, height=300, title="Monthly Payments (from Invoice/Payment Data)")
        
        st.altair_chart(payment_trend, use_container_width=True)
        
        # Show summary statistics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Payments", f"${payments_df['total_amount'].sum():,.2f}")
        with col2:
            st.metric("Average Monthly Payment", f"${monthly_payments['total_amount'].mean():,.2f}")
        with col3:
            st.metric("Payment Transactions Count", len(payments_df))
        
        payments_found = True
        payment_source = "Invoice/Payment Data"

# Second try: General Ledger data (if no payments found above)
if not payments_found and not gl_df.empty:
# -------------------------
# Outstanding Balance Summary from Invoice/Payment Data
# -------------------------
st.subheader("Outstanding Balance Summary")

if not df.empty:
    # Calculate outstanding balances using the balance field from the new schema
    outstanding_invoices = df[df["transaction_type"] == "Invoice"].copy()
    
    if not outstanding_invoices.empty:
        # Use the balance field which represents the remaining unpaid amount
        outstanding_invoices["balance"] = pd.to_numeric(outstanding_invoices["balance"], errors="coerce")
        
        # Filter for invoices with outstanding balances
        unpaid_invoices = outstanding_invoices[outstanding_invoices["balance"] > 0]
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            total_outstanding = unpaid_invoices["balance"].sum()
            st.metric("Total Outstanding", f"${total_outstanding:,.2f}")
        
        with col2:
            num_unpaid = len(unpaid_invoices)
            st.metric("Unpaid Invoices", num_unpaid)
        
        with col3:
            if num_unpaid > 0:
                avg_outstanding = total_outstanding / num_unpaid
                st.metric("Avg Outstanding per Invoice", f"${avg_outstanding:,.2f}")
            else:
                st.metric("Avg Outstanding per Invoice", "$0.00")
        
        with col4:
            total_invoiced = outstanding_invoices["total_amount"].sum()
            if total_invoiced > 0:
                collection_rate = ((total_invoiced - total_outstanding) / total_invoiced) * 100
                st.metric("Collection Rate", f"{collection_rate:.1f}%")
            else:
                st.metric("Collection Rate", "N/A")
        
        # Show overdue invoices (past due date)
        today = pd.Timestamp.now().date()
        unpaid_invoices["due_date"] = pd.to_datetime(unpaid_invoices["due_date"], errors="coerce")
        overdue_invoices = unpaid_invoices[
            (unpaid_invoices["due_date"].notna()) & 
            (unpaid_invoices["due_date"].dt.date < today)
        ]
        
        if not overdue_invoices.empty:
            st.subheader("⚠️ Overdue Invoices")
            overdue_display = overdue_invoices[["customer_name", "doc_number", "due_date", "balance"]].copy()
            overdue_display["due_date"] = overdue_display["due_date"].dt.strftime("%Y-%m-%d")
            overdue_display["balance"] = overdue_display["balance"].apply(lambda x: f"${x:,.2f}")
            overdue_display.columns = ["Customer", "Invoice #", "Due Date", "Outstanding Balance"]
            
            st.dataframe(
                overdue_display.sort_values("Due Date"),
                use_container_width=True,
                hide_index=True
            )
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Total Overdue", f"${overdue_invoices['balance'].sum():,.2f}")
            with col2:
                st.metric("Overdue Count", len(overdue_invoices))

# -------------------------
# Enhanced Cash Flow Stats (Using General Ledger)
# -------------------------
st.subheader("Cash Flow Averages (from General Ledger)")

if not gl_df.empty and "txn_date" in gl_df.columns:
    # Exclude voided transactions from all cash flow calculations
    gl_df_active = gl_df[~gl_df["description"].str.contains("Voided", case=False, na=False)]
    
    # Calculate date range from GL data (excluding voided)
    valid_dates = gl_df_active["txn_date"].dropna()
    if len(valid_dates) > 0:
        first_day = valid_dates.min().date()
        last_day = valid_dates.max().date()
        today = pd.Timestamp.today().date()
        
        # Use the actual last transaction date instead of today for more accurate calculations
        end_date = min(last_day, today)
        
        # Calculate working days
        working_days = busday_count(first_day, end_date)
        total_days = (end_date - first_day).days
        
        # Identify inflows and outflows from GL using txn_type and amount (excluding voided)
        # Inflows: Payments, deposits, receipts - money coming in
        inflow_conditions = (
            gl_df_active["txn_type"].str.contains("Payment|Deposit|Receipt|Invoice", case=False, na=False) &
            (gl_df_active["amount"] > 0)  # Positive amounts for inflows
        )
        
        # Outflows: Bills, expenses, checks - money going out  
        outflow_conditions = (
            gl_df_active["txn_type"].str.contains("Bill|Expense|Check|Transfer", case=False, na=False) |
            (gl_df_active["amount"] < 0)  # Negative amounts are typically outflows
        )
        
        # Calculate totals using amount field (from active transactions only)
        inflows_gl = gl_df_active[inflow_conditions]
        outflows_gl = gl_df_active[outflow_conditions]
        
        # Sum amounts (take absolute value for outflows since they might be negative)
        inflows_total = inflows_gl["amount"].abs().sum()
        outflows_total = outflows_gl["amount"].abs().sum()
        
        # Calculate averages
        avg_inflow_day = inflows_total / working_days if working_days > 0 else 0
        avg_outflow_day = outflows_total / working_days if working_days > 0 else 0
        
        avg_inflow_week = avg_inflow_day * 5
        avg_outflow_week = avg_outflow_day * 5
        avg_inflow_month = avg_inflow_day * 21  # ~21 working days per month
        avg_outflow_month = avg_outflow_day * 21
        
        # Display metrics in columns
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 📈 Inflows")
            st.metric("Total Inflows", f"${inflows_total:,.2f}")
            st.metric("Avg Inflow/Day", f"${avg_inflow_day:,.2f}")
            st.metric("Avg Inflow/Week", f"${avg_inflow_week:,.2f}")
            st.metric("Avg Inflow/Month", f"${avg_inflow_month:,.2f}")
        
        with col2:
            st.markdown("### 📉 Outflows")
            st.metric("Total Outflows", f"${outflows_total:,.2f}")
            st.metric("Avg Outflow/Day", f"${avg_outflow_day:,.2f}")
            st.metric("Avg Outflow/Week", f"${avg_outflow_week:,.2f}")
            st.metric("Avg Outflow/Month", f"${avg_outflow_month:,.2f}")
        
        # Summary statistics
        st.markdown("### 📊 Summary")
        col3, col4, col5 = st.columns(3)
        
        with col3:
            st.metric("Date Range", f"{first_day} to {end_date}")
        with col4:
            st.metric("Total Days", total_days)
        with col5:
            st.metric("Working Days", working_days)
        
        # Net cash flow
        net_daily = avg_inflow_day - avg_outflow_day
        net_weekly = avg_inflow_week - avg_outflow_week
        net_monthly = avg_inflow_month - avg_outflow_month
        
        st.markdown("### 💰 Net Cash Flow")
        col6, col7, col8 = st.columns(3)
        
        with col6:
            color = "normal" if net_daily >= 0 else "inverse"
            st.metric("Net Daily", f"${net_daily:,.2f}", delta_color=color)
        with col7:
            color = "normal" if net_weekly >= 0 else "inverse"
            st.metric("Net Weekly", f"${net_weekly:,.2f}", delta_color=color)
        with col8:
            color = "normal" if net_monthly >= 0 else "inverse"
            st.metric("Net Monthly", f"${net_monthly:,.2f}", delta_color=color)
        
        # -------------------------
        # Enhanced Net Burn Forecast
        # -------------------------
        st.subheader("Cash Flow Forecast (GL-Based)")
        
        forecast_periods = [30, 60, 90, 180, 365]
        forecast_data = []
        
        for days in forecast_periods:
            net_burn = net_daily * days
            forecast_data.append({
                "Period": f"{days} days",
                "Projected Net Cash Flow": net_burn,
                "Formatted": f"${net_burn:,.2f}"
            })
        
        forecast_df = pd.DataFrame(forecast_data)
        
        # Display as metrics
        cols = st.columns(len(forecast_periods))
        for i, (_, row) in enumerate(forecast_df.iterrows()):
            with cols[i]:
                color = "normal" if row["Projected Net Cash Flow"] >= 0 else "inverse"
                st.metric(
                    row["Period"], 
                    row["Formatted"],
                    delta_color=color
                )
        
        # Chart of forecast
        forecast_chart = alt.Chart(forecast_df).mark_line(point=True).encode(
            x=alt.X("Period:N", title="Forecast Period"),
            y=alt.Y("Projected Net Cash Flow:Q", title="Projected Net Cash Flow ($)", axis=alt.Axis(format="$,.0f")),
            tooltip=["Period", alt.Tooltip("Projected Net Cash Flow:Q", format="$,.2f")]
        ).properties(
            width=600,
            height=300,
            title="Cash Flow Forecast"
        )
        
        st.altair_chart(forecast_chart, use_container_width=True)
        
    else:
        st.error("No valid transaction dates found in General Ledger")
else:
    st.error("General Ledger data not available or missing transaction dates")
