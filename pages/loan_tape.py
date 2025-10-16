# pages/loan_tape.py
"""
Loan Tape Dashboard - Complete Enhanced Version
"""

import streamlit as st
import pandas as pd
import altair as alt
import numpy as np
import numpy_financial as npf
# ML & stats
from sklearn.model_selection import StratifiedKFold, KFold, cross_val_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score, precision_score, recall_score, roc_curve
from scipy.stats import pearsonr, spearmanr, pointbiserialr

from utils.config import (
    inject_global_styles,
    inject_logo,
    get_supabase_client,
    PRIMARY_COLOR,      # not directly used but kept for consistency
    COLOR_PALETTE,      # not directly used but kept for consistency
    PLATFORM_FEE_RATE,  # not directly used; we set PLATFORM_FEE below
)

# ---------------------------
# Page Configuration & Styles
# ---------------------------
st.set_page_config(page_title="CSL Capital | Loan Tape", layout="wide")
inject_global_styles()
inject_logo()

# -------------
# Constants
# -------------
PLATFORM_FEE = 0.03  # keep as requested
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
    "Active - Frequently Late": "#66aa66",
    "Paid Off": "#1f77b4",
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
    "Paid Off": 0.0,
}

PROBLEM_STATUSES = {"Late","Default","Bankrupt","Severe","Severe Delinquency","Moderate Delinquency","Active - Frequently Late"}

# -------------
# Supabase
# -------------
supabase = get_supabase_client()

@st.cache_data(ttl=3600)
def load_loan_summaries() -> pd.DataFrame:
    res = supabase.table("loan_summaries").select("*").execute()
    return pd.DataFrame(res.data or [])

@st.cache_data(ttl=3600)
def load_deals() -> pd.DataFrame:
    res = supabase.table("deals").select("*").execute()
    return pd.DataFrame(res.data or [])

@st.cache_data(ttl=3600)
def load_naics_sector_risk() -> pd.DataFrame:
    res = supabase.table("naics_sector_risk_profile").select("*").execute()
    return pd.DataFrame(res.data or [])

@st.cache_data(ttl=3600)
def load_loan_schedules() -> pd.DataFrame:
    res = supabase.table("loan_schedules").select("*").execute()
    return pd.DataFrame(res.data or [])

@st.cache_data(ttl=3600)
def get_last_updated() -> str:
    try:
        timestamps = []
        for table in ["loan_summaries", "deals", "loan_schedules"]:
            try:
                res = supabase.table(table).select("updated_at").order("updated_at", desc=True).limit(1).execute()
                if res.data and res.data[0].get("updated_at"):
                    timestamps.append(pd.to_datetime(res.data[0]["updated_at"]))
            except:
                try:
                    res = supabase.table(table).select("created_at").order("created_at", desc=True).limit(1).execute()
                    if res.data and res.data[0].get("created_at"):
                        timestamps.append(pd.to_datetime(res.data[0]["created_at"]))
                except:
                    pass
        if timestamps:
            return max(timestamps).strftime("%B %d, %Y at %I:%M %p")
        return "Unable to determine"
    except Exception as e:
        return f"Error: {str(e)}"

# ----------------------
# Data Preparation Utils
# ----------------------
def prepare_loan_data(loans_df: pd.DataFrame, deals_df: pd.DataFrame) -> pd.DataFrame:
    """
    Merge and derive dashboard-calculated fields.
    Expects deals_df to include commission_fee, fico, tib, industry, etc.
    """
    if not loans_df.empty and not deals_df.empty:
        # include commission_fee explicitly (used below)
        merge_cols = ["loan_id", "deal_name", "partner_source", "industry", "commission_fee", "fico", "tib"]
        merge_cols = [c for c in merge_cols if c in deals_df.columns]
        df = loans_df.merge(deals_df[merge_cols], on="loan_id", how="left")
    else:
        df = loans_df.copy()

    # normalize date fields
    for date_col in ["funding_date", "maturity_date", "payoff_date"]:
        if date_col in df.columns:
            df[date_col] = pd.to_datetime(df[date_col], errors="coerce")

    # fees & totals
    df["commission_fee"] = pd.to_numeric(df.get("commission_fee", 0), errors="coerce").fillna(0.0)
    df["csl_participation_amount"] = pd.to_numeric(df.get("csl_participation_amount", 0), errors="coerce").fillna(0.0)
    df["total_paid"] = pd.to_numeric(df.get("total_paid", 0), errors="coerce").fillna(0.0)

    df["commission_fees"] = df["csl_participation_amount"] * df["commission_fee"]
    df["platform_fees"]  = df["csl_participation_amount"] * PLATFORM_FEE

    df["total_invested"] = df["csl_participation_amount"] + df["platform_fees"] + df["commission_fees"]
    df["net_balance"] = df["total_invested"] - df["total_paid"]

    # ROI (simple)
    df["current_roi"] = np.where(
        df["total_invested"] > 0,
        (df["total_paid"] / df["total_invested"]) - 1,
        0.0
    )

    # flags
    df["is_unpaid"] = df.get("loan_status", "").ne("Paid Off")

    # days since funding
    try:
        today = pd.Timestamp.today().tz_localize(None)
        df["days_since_funding"] = df["funding_date"].apply(
            lambda x: (today - pd.to_datetime(x).tz_localize(None)).days if pd.notnull(x) else 0
        )
    except:
        df["days_since_funding"] = 0

    # months left to maturity
    df["remaining_maturity_months"] = 0.0
    try:
        if "maturity_date" in df.columns:
            today = pd.Timestamp.today().tz_localize(None)
            active_mask = (df.get("loan_status", "") != "Paid Off") & (df["maturity_date"] > today)
            df.loc[active_mask, "remaining_maturity_months"] = df.loc[active_mask, "maturity_date"].apply(
                lambda x: (pd.to_datetime(x).tz_localize(None) - today).days / 30 if pd.notnull(x) else 0
            )
    except:
        pass

    # cohorts
    try:
        df["cohort"] = df["funding_date"].dt.to_period("Q").astype(str)
        df["funding_month"] = df["funding_date"].dt.to_period("M")
    except:
        df["cohort"] = "Unknown"
        df["funding_month"] = pd.NaT

    # sector_code (first two digits of NAICS-like string)
    if "industry" in df.columns:
        df["sector_code"] = df["industry"].astype(str).str[:2]

    # payment_performance can be derived if you want, but we assume it's present.
    # If not, default to realized ratio (clamped 0..1)
    if "payment_performance" not in df.columns or df["payment_performance"].isna().all():
        with np.errstate(divide="ignore", invalid="ignore"):
            perf = np.where(df["total_invested"] > 0, df["total_paid"] / df["total_invested"], np.nan)
            df["payment_performance"] = np.clip(perf, 0, 1)

    return df

def calculate_irr(df: pd.DataFrame) -> pd.DataFrame:
    result_df = df.copy()

    def calc_realized_irr(row):
        if pd.isna(row.get("funding_date")) or pd.isna(row.get("payoff_date")) or row.get("total_invested", 0) <= 0:
            return None
        try:
            funding_date = pd.to_datetime(row["funding_date"]).tz_localize(None)
            payoff_date  = pd.to_datetime(row["payoff_date"]).tz_localize(None)
            if payoff_date <= funding_date:
                return None
            days = (payoff_date - funding_date).days
            years = days / 365.0
            if years < 0.01:
                simple = (row["total_paid"] / row["total_invested"]) - 1
                return (1 + simple) ** (1 / max(years, 1e-6)) - 1
            try:
                irr = npf.irr([-row["total_invested"], row["total_paid"]])
                if irr is None or irr < -1 or irr > 10:
                    simple = (row["total_paid"] / row["total_invested"]) - 1
                    return (1 + simple) ** (1 / years) - 1
                return irr
            except:
                simple = (row["total_paid"] / row["total_invested"]) - 1
                return (1 + simple) ** (1 / years) - 1
        except:
            return None

    def calc_expected_irr(row):
        if pd.isna(row.get("funding_date")) or pd.isna(row.get("maturity_date")) or row.get("total_invested", 0) <= 0:
            return None
        try:
            funding_date  = pd.to_datetime(row["funding_date"]).tz_localize(None)
            maturity_date = pd.to_datetime(row["maturity_date"]).tz_localize(None)
            if maturity_date <= funding_date:
                return None

            # Expected payment proxy
            if "our_rtr" in row and pd.notnull(row["our_rtr"]):
                expected_payment = row["our_rtr"]
            elif "roi" in row and pd.notnull(row["roi"]):
                expected_payment = row["total_invested"] * (1 + row["roi"])
            else:
                # fall back: assume 1.2x if ROI missing
                expected_payment = row["total_invested"] * 1.2

            days = (maturity_date - funding_date).days
            years = days / 365.0
            if years < 0.01:
                simple = (expected_payment / row["total_invested"]) - 1
                return (1 + simple) ** (1 / max(years, 1e-6)) - 1
            try:
                irr = npf.irr([-row["total_invested"], expected_payment])
                if irr is None or irr < -1 or irr > 10:
                    simple = (expected_payment / row["total_invested"]) - 1
                    return (1 + simple) ** (1 / years) - 1
                return irr
            except:
                simple = (expected_payment / row["total_invested"]) - 1
                return (1 + simple) ** (1 / years) - 1
        except:
            return None

    try:
        result_df["realized_irr"] = result_df.apply(calc_realized_irr, axis=1)
        result_df["expected_irr"] = result_df.apply(calc_expected_irr, axis=1)

        result_df["realized_irr_pct"] = result_df["realized_irr"].apply(lambda x: f"{x:.2%}" if pd.notnull(x) else "N/A")
        result_df["expected_irr_pct"] = result_df["expected_irr"].apply(lambda x: f"{x:.2%}" if pd.notnull(x) else "N/A")
    except:
        result_df["realized_irr"] = None
        result_df["expected_irr"] = None
        result_df["realized_irr_pct"] = "N/A"
        result_df["expected_irr_pct"] = "N/A"

    return result_df

def calculate_risk_scores(df: pd.DataFrame) -> pd.DataFrame:
    risk_df = df[df.get("loan_status", "") != "Paid Off"].copy()
    if risk_df.empty:
        return risk_df

    risk_df["payment_performance"] = pd.to_numeric(risk_df["payment_performance"], errors="coerce").clip(upper=1.0)
    risk_df["performance_gap"] = 1 - risk_df["payment_performance"]

    risk_df["status_multiplier"] = risk_df["loan_status"].map(STATUS_RISK_MULTIPLIERS).fillna(1.0)

    today = pd.Timestamp.today().tz_localize(None)
    risk_df["days_past_maturity"] = risk_df["maturity_date"].apply(
        lambda x: max(0, (today - pd.to_datetime(x)).days) if pd.notnull(x) else 0
    )
    # clamp at 12 months past due in factor
    risk_df["overdue_factor"] = (risk_df["days_past_maturity"] / 30).clip(upper=12) / 12

    risk_df["risk_score"] = (
        risk_df["performance_gap"] *
        risk_df["status_multiplier"] *
        (1 + risk_df["overdue_factor"])
    ).clip(upper=5.0)

    risk_bins   = [0, 0.5, 1.0, 1.5, 2.0, 5.0]
    risk_labels = ["Low (0-0.5)", "Moderate (0.5-1.0)", "Elevated (1.0-1.5)", "High (1.5-2.0)", "Severe (2.0+)"]
    risk_df["risk_band"] = pd.cut(risk_df["risk_score"], bins=risk_bins, labels=risk_labels)

    return risk_df

def calculate_expected_payment_to_date(row) -> float:
    if pd.isna(row.get("funding_date")) or pd.isna(row.get("maturity_date")) or pd.isna(row.get("our_rtr")):
        return 0.0
    try:
        funding_date  = pd.to_datetime(row["funding_date"]).tz_localize(None)
        maturity_date = pd.to_datetime(row["maturity_date"]).tz_localize(None)
        today = pd.Timestamp.today().tz_localize(None)

        if today >= maturity_date:
            return float(row["our_rtr"])

        total_days = (maturity_date - funding_date).days
        days_elapsed = (today - funding_date).days

        if total_days <= 0:
            return 0.0

        expected_pct = min(1.0, max(0.0, days_elapsed / total_days))
        return float(row["our_rtr"]) * expected_pct
    except:
        return 0.0

def format_dataframe_for_display(df: pd.DataFrame, columns=None, rename_map=None) -> pd.DataFrame:
    if columns:
        display_columns = [c for c in columns if c in df.columns]
        display_df = df[display_columns].copy()
    else:
        display_df = df.copy()

    if rename_map:
        display_df.rename(columns={k: v for k, v in rename_map.items() if k in display_df.columns}, inplace=True)

    # Pretty formats
    for col in display_df.select_dtypes(include=["float64", "float32"]).columns:
        up = col.upper()
        if any(term in up for term in ["ROI", "RATE", "PERCENTAGE", "PERFORMANCE"]):
            display_df[col] = display_df[col].map(lambda x: f"{x:.2%}" if pd.notnull(x) else "")
        elif any(term in up for term in ["MATURITY", "MONTHS"]):
            display_df[col] = display_df[col].map(lambda x: f"{x:.1f}" if pd.notnull(x) else "")
        elif any(term in up for term in ["CAPITAL", "INVESTED", "PAID", "BALANCE", "FEES"]):
            display_df[col] = display_df[col].map(lambda x: f"${x:,.2f}" if pd.notnull(x) else "")

    # Date string formats (preserve original datetime internally)
    try:
        reverse_map = {v: k for k, v in rename_map.items()} if rename_map else {}
        for col in display_df.columns:
            if any(term in col for term in ["Date", "Funding", "Maturity", "Funded"]):
                original_col = reverse_map.get(col, col.replace(" ", "_").lower())
                if original_col in df.columns and pd.api.types.is_datetime64_any_dtype(df[original_col]):
                    display_df[col] = pd.to_datetime(df[original_col]).dt.strftime("%Y-%m-%d")
    except:
        pass

    return display_df

def make_problem_label(df: pd.DataFrame, perf_cutoff: float = 0.90) -> pd.Series:
    status_bad = df.get("loan_status", "").isin(PROBLEM_STATUSES)
    perf_bad = pd.to_numeric(df.get("payment_performance", np.nan), errors="coerce") < perf_cutoff
    return (status_bad | perf_bad).astype(int)

def safe_kfold(n_items: int, preferred: int = 5) -> int:
    # ensure at least 2 folds and at least 2 items per fold
    return max(2, min(preferred, n_items if n_items >= preferred else max(2, n_items // 2)))

def compute_correlations(base_df: pd.DataFrame) -> pd.DataFrame:
    df = base_df.copy()
    # attach risk score from sector table if available
    try:
        sr = load_naics_sector_risk()
        if "sector_code" not in df.columns and "industry" in df.columns:
            df["sector_code"] = df["industry"].astype(str).str[:2]
        if not sr.empty and "sector_code" in df.columns:
            df = df.merge(sr[["sector_code","risk_score"]], on="sector_code", how="left")
    except:
        pass

    # numeric candidates
    num_cols = []
    for c in ["fico","tib","risk_score","total_invested","total_paid","net_balance","commission_fee","remaining_maturity_months","days_since_funding"]:
        if c in df.columns:
            num_cols.append(c)

    y_perf = pd.to_numeric(df.get("payment_performance", np.nan), errors="coerce")
    y_clf = make_problem_label(df)

    rows = []
    for col in num_cols:
        x = pd.to_numeric(df[col], errors="coerce")
        mask_perf = x.notna() & y_perf.notna()
        mask_clf  = x.notna() & y_clf.notna()
        if mask_perf.sum() >= 8:
            try:
                pr, pp = pearsonr(x[mask_perf], y_perf[mask_perf])
            except:
                pr, pp = np.nan, np.nan
            try:
                srho, sp = spearmanr(x[mask_perf], y_perf[mask_perf])
            except:
                srho, sp = np.nan, np.nan
        else:
            pr=pp=srho=sp=np.nan
        if mask_clf.sum() >= 8 and y_clf[mask_clf].nunique() > 1:
            try:
                pbr, pbr_p = pointbiserialr(y_clf[mask_clf], x[mask_clf])
            except:
                pbr, pbr_p = np.nan, np.nan
        else:
            pbr=pbr_p=np.nan

        rows.append({
            "feature": col,
            "pearson_r_vs_performance": pr,
            "pearson_p": pp,
            "spearman_rho_vs_performance": srho,
            "spearman_p": sp,
            "pointbiserial_vs_problem": pbr,
            "pointbiserial_p": pbr_p
        })

    out = pd.DataFrame(rows).sort_values("spearman_rho_vs_performance", ascending=False)
    return out

def build_feature_matrix(df: pd.DataFrame):
    # minimal, robust set
    X = pd.DataFrame({
        "fico": pd.to_numeric(df.get("fico"), errors="coerce"),
        "tib": pd.to_numeric(df.get("tib"), errors="coerce"),
        "total_invested": pd.to_numeric(df.get("total_invested"), errors="coerce"),
        "net_balance": pd.to_numeric(df.get("net_balance"), errors="coerce"),
    })
    # join sector risk if available
    try:
        sr = load_naics_sector_risk()
        if "sector_code" not in df.columns and "industry" in df.columns:
            sec = df["industry"].astype(str).str[:2]
        else:
            sec = df.get("sector_code")
        if sr is not None and not sr.empty and sec is not None:
            risk_map = dict(zip(sr["sector_code"], sr["risk_score"]))
            X["sector_risk"] = sec.map(risk_map)
    except:
        pass

    # categorical
    cat_df = pd.DataFrame()
    for c in ["industry", "partner_source"]:
        if c in df.columns:
            cat_df[c] = df[c].astype(str)
    if not cat_df.empty:
        X = pd.concat([X, cat_df], axis=1)
    return X

def train_classification_small(df: pd.DataFrame):
    y = make_problem_label(df)
    X = build_feature_matrix(df)

    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = [c for c in X.columns if c not in num_cols]

    pre = ColumnTransformer([
        ("num", Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())]), num_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore", min_frequency=0.05), cat_cols)
    ])

    model = Pipeline([
        ("pre", pre),
        ("clf", LogisticRegression(max_iter=2000, class_weight="balanced", C=1.0, solver="lbfgs"))
    ])

    # CV folds
    min_class = int(pd.Series(y).value_counts().min()) if y.nunique() == 2 else len(df)
    n_splits = max(2, min(safe_kfold(len(df)), min_class))
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    Xy = df.index  # dummy to align lengths

    # fit and CV
    aucs, precs, recs = [], [], []
    for tr, te in skf.split(X, y):
        model.fit(X.iloc[tr], y.iloc[tr])
        p = model.predict_proba(X.iloc[te])[:,1]
        yhat = (p >= 0.5).astype(int)
        try: aucs.append(roc_auc_score(y.iloc[te], p))
        except: pass
        precs.append(precision_score(y.iloc[te], yhat, zero_division=0))
        recs.append(recall_score(y.iloc[te], yhat, zero_division=0))

    # final fit on all data for coefficients
    model.fit(X, y)

    # extract top coefficients
    try:
        # get feature names after preprocessing
        ohe = model.named_steps["pre"].named_transformers_["cat"]
        num_names = num_cols
        cat_names = ohe.get_feature_names_out(cat_cols).tolist() if cat_cols else []
        feat_names = num_names + cat_names
        coefs = model.named_steps["clf"].coef_.ravel()
        coef_df = pd.DataFrame({"feature": feat_names, "coef": coefs}).sort_values("coef", ascending=False)
        top_pos = coef_df.head(10)
        top_neg = coef_df.tail(10).iloc[::-1]
    except Exception as e:
        top_pos = pd.DataFrame(columns=["feature","coef"])
        top_neg = pd.DataFrame(columns=["feature","coef"])

    metrics = {
        "ROC AUC": (np.mean(aucs) if aucs else np.nan, np.std(aucs) if aucs else np.nan),
        "Precision": (np.mean(precs), np.std(precs)),
        "Recall": (np.mean(recs), np.std(recs)),
        "n_samples": len(df),
        "pos_rate": float(y.mean())
    }
    return model, metrics, top_pos, top_neg

def train_regression_small(df: pd.DataFrame):
    y = pd.to_numeric(df.get("payment_performance"), errors="coerce")
    mask = y.notna()
    X = build_feature_matrix(df.loc[mask])

    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = [c for c in X.columns if c not in num_cols]

    pre = ColumnTransformer([
        ("num", Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())]), num_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore", min_frequency=0.05), cat_cols)
    ])

    model = Pipeline([
        ("pre", pre),
        ("reg", Ridge(alpha=1.0))
    ])

    kf = KFold(n_splits=safe_kfold(mask.sum()), shuffle=True, random_state=42)
    r2s, rmses = [], []
    for tr, te in kf.split(X):
        model.fit(X.iloc[tr], y.iloc[mask].iloc[tr])
        yhat = model.predict(X.iloc[te])
        ytrue = y.iloc[mask].iloc[te]
        r2s.append(1 - np.sum((ytrue - yhat)**2)/np.sum((ytrue - ytrue.mean())**2) if ytrue.nunique()>1 else np.nan)
        rmses.append(np.sqrt(np.mean((ytrue - yhat)**2)))

    model.fit(X, y.loc[mask])

    return model, {
        "R2": (np.nanmean(r2s), np.nanstd(r2s)),
        "RMSE": (np.mean(rmses), np.std(rmses)),
        "n_samples": int(mask.sum())
    }

def render_corr_outputs(df: pd.DataFrame):
    st.subheader("Correlation Snapshot")
    corr_df = compute_correlations(df)
    if corr_df.empty:
        st.info("Not enough numeric data for correlations.")
        return

    # table
    disp = corr_df.copy()
    for c in ["pearson_r_vs_performance","spearman_rho_vs_performance","pointbiserial_vs_problem"]:
        if c in disp.columns:
            disp[c] = disp[c].map(lambda x: f"{x:.3f}" if pd.notnull(x) else "")
    st.dataframe(disp, width="stretch", hide_index=True)

    # heatmap vs performance (Spearman)
    hm = corr_df[["feature","spearman_rho_vs_performance"]].dropna()
    if not hm.empty:
        chart = alt.Chart(hm).mark_rect().encode(
            x=alt.X("feature:N", sort=None, title="Feature"),
            y=alt.Y("var:N", sort=None, title="Target", axis=alt.Axis(labels=True)),
            color=alt.Color("value:Q", scale=alt.Scale(scheme="redblue", domain=[-1,1]), title="Spearman ρ"),
            tooltip=[alt.Tooltip("feature:N"), alt.Tooltip("value:Q", title="ρ", format=".3f")]
        ).transform_calculate(var="'payment_performance'").transform_calculate(value="datum.spearman_rho_vs_performance"
        ).properties(width=700, height=150, title="Correlation with Payment Performance")
        st.altair_chart(chart, width="stretch")

    st.download_button(
        "Download correlations (CSV)",
        corr_df.to_csv(index=False).encode("utf-8"),
        file_name="correlations.csv",
        mime="text/csv"
    )

def render_fico_tib_heatmap(df: pd.DataFrame):
    st.subheader("FICO × TIB: Avg Payment Performance")
    if "fico" not in df.columns or "tib" not in df.columns:
        st.info("Need both FICO and TIB for this view.")
        return
    d = df.copy()
    d["fico"] = pd.to_numeric(d["fico"], errors="coerce")
    d["tib"]  = pd.to_numeric(d["tib"], errors="coerce")
    fico_bins = [0, 580, 620, 660, 700, 740, 850]
    tib_bins  = [0, 5, 10, 15, 20, 100]
    d["fico_band"] = pd.cut(d["fico"], bins=fico_bins, labels=["<580","580-619","620-659","660-699","700-739","740+"], right=False)
    d["tib_band"]  = pd.cut(d["tib"],  bins=tib_bins,  labels=["≤5","5-10","10-15","15-20","20+"], right=False)

    m = d.groupby(["fico_band","tib_band"], observed=True).agg(
        avg_perf=("payment_performance","mean"),
        n=("loan_id","count")
    ).reset_index().dropna()

    if m.empty:
        st.info("Insufficient data to render heatmap.")
        return

    heat = alt.Chart(m).mark_rect().encode(
        x=alt.X("fico_band:N", title="FICO Band"),
        y=alt.Y("tib_band:N", title="TIB Band"),
        color=alt.Color("avg_perf:Q", scale=alt.Scale(domain=[0.6, 1.0], scheme="greens"), title="Avg Perf"),
        tooltip=[
            alt.Tooltip("fico_band:N", title="FICO"),
            alt.Tooltip("tib_band:N", title="TIB"),
            alt.Tooltip("avg_perf:Q", title="Avg Performance", format=".1%"),
            alt.Tooltip("n:Q", title="Loans")
        ]
    ).properties(width=600, height=300, title="Avg Payment Performance by FICO × TIB")
    st.altair_chart(heat, width="stretch")

# -------------------
# Charts & Visuals
# -------------------
def plot_capital_flow(df: pd.DataFrame):
    st.subheader("Capital Flow: Deployment vs. Returns")

    schedules = load_loan_schedules()

    d = df.copy()
    d["funding_date"] = pd.to_datetime(d["funding_date"], errors="coerce").dt.tz_localize(None)

    total_deployed = d["csl_participation_amount"].sum()
    total_returned = d["total_paid"].sum()

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Total Capital Deployed (Expected)", f"${total_deployed:,.0f}")
    with col2:
        st.metric("Total Capital Returned (Expected)", f"${total_returned:,.0f}")

    deploy_data = d[["funding_date", "csl_participation_amount"]].dropna()
    deploy_timeline = deploy_data.groupby("funding_date")["csl_participation_amount"].sum().sort_index().cumsum()

    if not schedules.empty and "payment_date" in schedules.columns:
        schedules["payment_date"] = pd.to_datetime(schedules["payment_date"], errors="coerce").dt.tz_localize(None)
        payment_data = schedules[
            schedules["actual_payment"].notna() &
            (schedules["actual_payment"] > 0) &
            schedules["payment_date"].notna()
        ]
        return_timeline = payment_data.groupby("payment_date")["actual_payment"].sum().sort_index().cumsum()
    else:
        return_timeline = pd.Series(dtype=float)

    if not deploy_timeline.empty:
        min_date = deploy_timeline.index.min()
        max_date = pd.Timestamp.today().normalize()

        date_range = pd.date_range(start=min_date, end=max_date, freq="D")

        unified = pd.DataFrame(index=date_range)
        unified["capital_deployed"] = deploy_timeline.reindex(date_range).ffill().fillna(0)
        unified["capital_returned"] = return_timeline.reindex(date_range).ffill().fillna(0)
        unified["date"] = unified.index

        st.caption(
            f"Chart shows: Deployed ${unified['capital_deployed'].iloc[-1]:,.0f} | "
            f"Returned ${unified['capital_returned'].iloc[-1]:,.0f}"
        )

        plot_df = pd.concat([
            pd.DataFrame({"date": unified.index, "amount": unified["capital_deployed"].values, "series": "Capital Deployed"}),
            pd.DataFrame({"date": unified.index, "amount": unified["capital_returned"].values, "series": "Capital Returned"})
        ], ignore_index=True)

        chart = alt.Chart(plot_df).mark_line().encode(
            x=alt.X("date:T", title="Date", axis=alt.Axis(format="%b %Y")),
            y=alt.Y("amount:Q", title="Cumulative Amount ($)", axis=alt.Axis(format="$,.0f")),
            color=alt.Color(
                "series:N",
                scale=alt.Scale(domain=["Capital Deployed", "Capital Returned"], range=["#ff7f0e", "#2ca02c"]),
                legend=alt.Legend(title="Capital Flow")
            ),
            tooltip=[
                alt.Tooltip("date:T", title="Date", format="%Y-%m-%d"),
                alt.Tooltip("amount:Q", title="Amount", format="$,.0f"),
                alt.Tooltip("series:N", title="Type"),
            ],
        ).properties(width=800, height=400, title="Capital Deployed vs. Capital Returned Over Time")

        st.altair_chart(chart, width="stretch")
    else:
        st.info("Insufficient data to display capital flow chart.")

def plot_investment_net_position(df: pd.DataFrame):
    st.subheader("Net Investment Position Over Time", help="Shows capital at work: cumulative deployed minus cumulative returned")

    schedules = load_loan_schedules()

    d = df.copy()
    d["funding_date"] = pd.to_datetime(d["funding_date"], errors="coerce").dt.tz_localize(None)

    deploy_data = d[["funding_date", "csl_participation_amount"]].dropna()
    deploy_timeline = deploy_data.groupby("funding_date")["csl_participation_amount"].sum().sort_index().cumsum()

    if not schedules.empty and "payment_date" in schedules.columns:
        schedules["payment_date"] = pd.to_datetime(schedules["payment_date"], errors="coerce").dt.tz_localize(None)
        payment_data = schedules[
            schedules["actual_payment"].notna() &
            (schedules["actual_payment"] > 0) &
            schedules["payment_date"].notna()
        ]
        return_timeline = payment_data.groupby("payment_date")["actual_payment"].sum().sort_index().cumsum()
    else:
        return_timeline = pd.Series(dtype=float)

    if not deploy_timeline.empty:
        min_date = deploy_timeline.index.min()
        max_date = pd.Timestamp.today().normalize()
        date_range = pd.date_range(start=min_date, end=max_date, freq="D")

        unified = pd.DataFrame(index=date_range)
        unified["cum_deployed"] = deploy_timeline.reindex(date_range).ffill().fillna(0)
        unified["cum_returned"] = return_timeline.reindex(date_range).ffill().fillna(0)
        unified["net_position"] = unified["cum_deployed"] - unified["cum_returned"]
        unified["date"] = unified.index

        plot_df = pd.concat([
            pd.DataFrame({"date": unified.index, "amount": unified["cum_deployed"].values, "Type": "Cumulative Deployed"}),
            pd.DataFrame({"date": unified.index, "amount": unified["cum_returned"].values, "Type": "Cumulative Returned"}),
            pd.DataFrame({"date": unified.index, "amount": unified["net_position"].values, "Type": "Net Position"}),
        ], ignore_index=True)

        chart = alt.Chart(plot_df).mark_line().encode(
            x=alt.X("date:T", title="Date"),
            y=alt.Y("amount:Q", title="Amount ($)", axis=alt.Axis(format="$,.0f")),
            color=alt.Color(
                "Type:N",
                scale=alt.Scale(
                    domain=["Cumulative Deployed", "Cumulative Returned", "Net Position"],
                    range=["#ff7f0e", "#2ca02c", "#1f77b4"]
                ),
            ),
            tooltip=[
                alt.Tooltip("date:T", title="Date", format="%Y-%m-%d"),
                alt.Tooltip("amount:Q", title="Amount", format="$,.2f"),
                alt.Tooltip("Type:N", title="Metric"),
            ],
        ).properties(width=800, height=500, title="Portfolio Net Position Over Time")

        zero_line = alt.Chart(pd.DataFrame({"y": [0]})).mark_rule(strokeDash=[2, 2], color="gray", strokeWidth=1).encode(y="y:Q")

        st.altair_chart(chart + zero_line, width="stretch")
        st.caption("Net Position: Capital still deployed (positive) or profit after recovery (negative).")
    else:
        st.info("Insufficient data for net position analysis.")

def plot_payment_performance_by_cohort(df: pd.DataFrame):
    active_df = df[df.get("loan_status", "") != "Paid Off"].copy()
    if active_df.empty:
        st.info("No active loans to analyze.")
        return

    active_df["expected_paid_to_date"] = active_df.apply(calculate_expected_payment_to_date, axis=1)
    active_df["actual_paid"] = active_df["total_paid"]

    active_df["performance_pct_diff"] = active_df.apply(
        lambda x: ((x["actual_paid"] / x["expected_paid_to_date"]) - 1) if x["expected_paid_to_date"] > 0 else 0,
        axis=1
    )
    active_df["cohort"] = pd.to_datetime(active_df["funding_date"]).dt.to_period("Q").astype(str)

    cohort_perf = active_df.groupby("cohort").agg(
        expected_payment=("expected_paid_to_date", "sum"),
        actual_payment=("actual_paid", "sum"),
        loan_count=("loan_id", "count"),
    ).reset_index()

    cohort_perf["performance_pct_diff"] = (cohort_perf["actual_payment"] / cohort_perf["expected_payment"]) - 1
    cohort_perf["perf_label"] = cohort_perf["performance_pct_diff"].apply(lambda x: f"{x:+.1%}")
    cohort_perf = cohort_perf.sort_values("cohort")

    def classify(p):
        if p >= -0.05:
            return "On/Above Target"
        elif p >= -0.15:
            return "Slightly Below"
        else:
            return "Significantly Below"

    cohort_perf["performance_category"] = cohort_perf["performance_pct_diff"].apply(classify)

    bars = alt.Chart(cohort_perf).mark_bar().encode(
        x=alt.X("cohort:N", title="Funding Quarter", sort=None),
        y=alt.Y("performance_pct_diff:Q", title="Performance Difference from Expected", axis=alt.Axis(format=".0%")),
        color=alt.Color(
            "performance_category:N",
            scale=alt.Scale(
                domain=["On/Above Target", "Slightly Below", "Significantly Below"],
                range=["#2ca02c", "#ffbb78", "#d62728"]
            ),
            legend=alt.Legend(title="Performance"),
        ),
        tooltip=[
            alt.Tooltip("cohort:N", title="Cohort"),
            alt.Tooltip("expected_payment:Q", title="Expected Payment", format="$,.0f"),
            alt.Tooltip("actual_payment:Q", title="Actual Payment", format="$,.0f"),
            alt.Tooltip("performance_pct_diff:Q", title="Performance Difference", format="+.1%"),
            alt.Tooltip("loan_count:Q", title="Number of Loans"),
        ],
    ).properties(width=700, height=400, title="Payment Performance by Cohort")

    text = alt.Chart(cohort_perf).mark_text(align="center", baseline="bottom", dy=-5, fontSize=11, fontWeight="bold").encode(
        x=alt.X("cohort:N", sort=None),
        y=alt.Y("performance_pct_diff:Q"),
        text="perf_label:N",
    )

    ref_line = alt.Chart(pd.DataFrame({"y": [0]})).mark_rule(strokeDash=[4, 4], color="gray", strokeWidth=2).encode(y="y:Q")

    target_zone = alt.Chart(pd.DataFrame({"y": [-0.05], "y2": [0.05]})).mark_rect(opacity=0.2, color="green").encode(
        y="y:Q", y2="y2:Q"
    )

    st.altair_chart(target_zone + bars + text + ref_line, width="stretch")
    st.caption("On-Target Zone: -5% to +5%. Positive = ahead of schedule, negative = behind schedule.")

def plot_fico_performance_analysis(df: pd.DataFrame):
    st.header("FICO Score Performance Analysis")

    if "fico" not in df.columns or df["fico"].isna().all():
        st.warning("FICO score data not available.")
        return

    fico_bins = [0, 580, 620, 660, 700, 740, 850]
    fico_labels = ["<580", "580-619", "620-659", "660-699", "700-739", "740+"]

    fico_df = df.copy()
    fico_df["fico"] = pd.to_numeric(fico_df["fico"], errors="coerce")
    fico_df["fico_band"] = pd.cut(fico_df["fico"], bins=fico_bins, labels=fico_labels, right=False)

    fico_metrics = fico_df.groupby("fico_band", observed=True).agg(
        deal_count=("loan_id", "count"),
        capital_deployed=("csl_participation_amount", "sum"),
        outstanding_balance=("net_balance", "sum"),
        avg_payment_performance=("payment_performance", "mean"),
        total_paid=("total_paid", "sum"),
        total_invested=("total_invested", "sum"),
    ).reset_index()

    fico_metrics["actual_return_rate"] = fico_metrics["total_paid"] / fico_metrics["total_invested"]

    status_by_fico = fico_df.groupby(["fico_band", "loan_status"], observed=True).size().reset_index(name="count")
    total_by_fico = fico_df.groupby("fico_band", observed=True).size().reset_index(name="total")
    status_by_fico = status_by_fico.merge(total_by_fico, on="fico_band")
    status_by_fico["pct"] = status_by_fico["count"] / status_by_fico["total"]

    problem_statuses = ["Late", "Default", "Bankrupt", "Severe", "Severe Delinquency", "Moderate Delinquency"]
    problem_loans = status_by_fico[status_by_fico["loan_status"].isin(problem_statuses)]
    problem_rate = problem_loans.groupby("fico_band", observed=True)["pct"].sum().reset_index(name="problem_rate")

    fico_metrics = fico_metrics.merge(problem_rate, on="fico_band", how="left")
    fico_metrics["problem_rate"] = fico_metrics["problem_rate"].fillna(0)

    col1, col2 = st.columns(2)

    with col1:
        perf_chart = alt.Chart(fico_metrics).mark_bar().encode(
            x=alt.X("fico_band:N", title="FICO Score Band", sort=fico_labels),
            y=alt.Y("avg_payment_performance:Q", title="Avg Payment Performance", axis=alt.Axis(format=".0%")),
            color=alt.Color(
                "avg_payment_performance:Q",
                scale=alt.Scale(domain=[0.5, 0.8, 1.0], range=["#d62728", "#ffbb78", "#2ca02c"]),
                legend=None,
            ),
            tooltip=[
                alt.Tooltip("fico_band:N", title="FICO Band"),
                alt.Tooltip("avg_payment_performance:Q", title="Avg Payment Performance", format=".1%"),
                alt.Tooltip("deal_count:Q", title="Loan Count"),
            ],
        ).properties(width=350, height=300, title="Payment Performance by FICO Score")
        st.altair_chart(perf_chart, width="stretch")

    with col2:
        problem_chart = alt.Chart(fico_metrics).mark_bar().encode(
            x=alt.X("fico_band:N", title="FICO Score Band", sort=fico_labels),
            y=alt.Y("problem_rate:Q", title="Problem Loan Rate", axis=alt.Axis(format=".0%")),
            color=alt.Color(
                "problem_rate:Q",
                scale=alt.Scale(domain=[0, 0.2, 0.4], range=["#2ca02c", "#ffbb78", "#d62728"]),
                legend=None,
            ),
            tooltip=[
                alt.Tooltip("fico_band:N", title="FICO Band"),
                alt.Tooltip("problem_rate:Q", title="Problem Loan Rate", format=".1%"),
                alt.Tooltip("deal_count:Q", title="Total Loans"),
            ],
        ).properties(width=350, height=300, title="Problem Loan Rate by FICO Score")
        st.altair_chart(problem_chart, width="stretch")

    return_chart = alt.Chart(fico_metrics).mark_bar().encode(
        x=alt.X("fico_band:N", title="FICO Score Band", sort=fico_labels),
        y=alt.Y("actual_return_rate:Q", title="Actual Return Rate", axis=alt.Axis(format=".0%")),
        color=alt.Color(
            "actual_return_rate:Q",
            scale=alt.Scale(domain=[0.5, 1.0, 1.3], range=["#d62728", "#ffbb78", "#2ca02c"]),
            legend=None,
        ),
        tooltip=[
            alt.Tooltip("fico_band:N", title="FICO Band"),
            alt.Tooltip("actual_return_rate:Q", title="Return Rate", format=".2%"),
            alt.Tooltip("deal_count:Q", title="Loan Count"),
        ],
    ).properties(width=700, height=300, title="Actual Return Rate by FICO Score")
    st.altair_chart(return_chart, width="stretch")

    st.subheader("FICO Performance Summary")
    display_df = fico_metrics.copy()
    display_df["outstanding_balance"] = display_df["outstanding_balance"].map(lambda x: f"${x:,.0f}")
    display_df["avg_payment_performance"] = display_df["avg_payment_performance"].map(lambda x: f"{x:.1%}")
    display_df["actual_return_rate"] = display_df["actual_return_rate"].map(lambda x: f"{x:.2%}")
    display_df["problem_rate"] = display_df["problem_rate"].map(lambda x: f"{x:.1%}")
    display_df = display_df[[
        "fico_band", "deal_count", "outstanding_balance",
        "avg_payment_performance", "actual_return_rate", "problem_rate"
    ]]
    display_df.columns = ["FICO Band", "Loan Count", "Outstanding Balance", "Avg Payment Performance", "Actual Return Rate", "Problem Loan Rate"]
    st.dataframe(display_df, width="stretch", hide_index=True)

def plot_tib_performance_analysis(df: pd.DataFrame):
    st.header("Time in Business Performance Analysis")

    if "tib" not in df.columns or df["tib"].isna().all():
        st.warning("Time in Business data not available.")
        return

    tib_bins = [0, 5, 10, 15, 20, 25, 100]
    tib_labels = ["≤5", "5-10", "10-15", "15-20", "20-25", "25+"]

    tib_df = df.copy()
    tib_df["tib"] = pd.to_numeric(tib_df["tib"], errors="coerce")
    tib_df["tib_band"] = pd.cut(tib_df["tib"], bins=tib_bins, labels=tib_labels, right=False)

    tib_metrics = tib_df.groupby("tib_band", observed=True).agg(
        deal_count=("loan_id", "count"),
        capital_deployed=("csl_participation_amount", "sum"),
        outstanding_balance=("net_balance", "sum"),
        avg_payment_performance=("payment_performance", "mean"),
        total_paid=("total_paid", "sum"),
        total_invested=("total_invested", "sum"),
    ).reset_index()

    tib_metrics["actual_return_rate"] = tib_metrics["total_paid"] / tib_metrics["total_invested"]

    status_by_tib = tib_df.groupby(["tib_band", "loan_status"], observed=True).size().reset_index(name="count")
    total_by_tib = tib_df.groupby("tib_band", observed=True).size().reset_index(name="total")
    status_by_tib = status_by_tib.merge(total_by_tib, on="tib_band")
    status_by_tib["pct"] = status_by_tib["count"] / status_by_tib["total"]

    problem_statuses = ["Late", "Default", "Bankrupt", "Severe", "Severe Delinquency", "Moderate Delinquency"]
    problem_loans = status_by_tib[status_by_tib["loan_status"].isin(problem_statuses)]
    problem_rate = problem_loans.groupby("tib_band", observed=True)["pct"].sum().reset_index(name="problem_rate")

    tib_metrics = tib_metrics.merge(problem_rate, on="tib_band", how="left")
    tib_metrics["problem_rate"] = tib_metrics["problem_rate"].fillna(0)

    col1, col2 = st.columns(2)

    with col1:
        perf_chart = alt.Chart(tib_metrics).mark_bar().encode(
            x=alt.X("tib_band:N", title="Time in Business (Years)", sort=tib_labels),
            y=alt.Y("avg_payment_performance:Q", title="Avg Payment Performance", axis=alt.Axis(format=".0%")),
            color=alt.Color(
                "avg_payment_performance:Q",
                scale=alt.Scale(domain=[0.5, 0.8, 1.0], range=["#d62728", "#ffbb78", "#2ca02c"]),
                legend=None,
            ),
            tooltip=[
                alt.Tooltip("tib_band:N", title="TIB Band"),
                alt.Tooltip("avg_payment_performance:Q", title="Avg Payment Performance", format=".1%"),
                alt.Tooltip("deal_count:Q", title="Loan Count"),
            ],
        ).properties(width=350, height=300, title="Payment Performance by Time in Business")
        st.altair_chart(perf_chart, width="stretch")

    with col2:
        problem_chart = alt.Chart(tib_metrics).mark_bar().encode(
            x=alt.X("tib_band:N", title="Time in Business (Years)", sort=tib_labels),
            y=alt.Y("problem_rate:Q", title="Problem Loan Rate", axis=alt.Axis(format=".0%")),
            color=alt.Color(
                "problem_rate:Q",
                scale=alt.Scale(domain=[0, 0.2, 0.4], range=["#2ca02c", "#ffbb78", "#d62728"]),
                legend=None,
            ),
            tooltip=[
                alt.Tooltip("tib_band:N", title="TIB Band"),
                alt.Tooltip("problem_rate:Q", title="Problem Loan Rate", format=".1%"),
                alt.Tooltip("deal_count:Q", title="Total Loans"),
            ],
        ).properties(width=350, height=300, title="Problem Loan Rate by Time in Business")
        st.altair_chart(problem_chart, width="stretch")

    return_chart = alt.Chart(tib_metrics).mark_bar().encode(
        x=alt.X("tib_band:N", title="Time in Business (Years)", sort=tib_labels),
        y=alt.Y("actual_return_rate:Q", title="Actual Return Rate", axis=alt.Axis(format=".0%")),
        color=alt.Color(
            "actual_return_rate:Q",
            scale=alt.Scale(domain=[0.5, 1.0, 1.3], range=["#d62728", "#ffbb78", "#2ca02c"]),
            legend=None,
        ),
        tooltip=[
            alt.Tooltip("tib_band:N", title="TIB Band"),
            alt.Tooltip("actual_return_rate:Q", title="Return Rate", format=".2%"),
            alt.Tooltip("deal_count:Q", title="Loan Count"),
        ],
    ).properties(width=700, height=300, title="Actual Return Rate by Time in Business")
    st.altair_chart(return_chart, width="stretch")

    st.subheader("Time in Business Performance Summary")
    display_df = tib_metrics.copy()
    display_df["outstanding_balance"] = display_df["outstanding_balance"].map(lambda x: f"${x:,.0f}")
    display_df["avg_payment_performance"] = display_df["avg_payment_performance"].map(lambda x: f"{x:.1%}")
    display_df["actual_return_rate"] = display_df["actual_return_rate"].map(lambda x: f"{x:.2%}")
    display_df["problem_rate"] = display_df["problem_rate"].map(lambda x: f"{x:.1%}")
    display_df = display_df[[
        "tib_band", "deal_count", "outstanding_balance",
        "avg_payment_performance", "actual_return_rate", "problem_rate"
    ]]
    display_df.columns = ["TIB Band", "Loan Count", "Outstanding Balance", "Avg Payment Performance", "Actual Return Rate", "Problem Loan Rate"]
    st.dataframe(display_df, width="stretch", hide_index=True)

def plot_industry_performance_analysis(df: pd.DataFrame):
    st.header("Industry Performance Analysis")

    active_df = df[df.get("loan_status", "") != "Paid Off"].copy()
    if active_df.empty or "industry" not in active_df.columns or active_df["industry"].isna().all():
        st.warning("Industry data not available.")
        return

    sector_risk_df = load_naics_sector_risk()
    # Ensure sector_code exists
    if "sector_code" not in active_df.columns:
        active_df["sector_code"] = active_df["industry"].astype(str).str[:2]

    df_with_risk = active_df.merge(sector_risk_df, on="sector_code", how="left")

    sector_metrics = df_with_risk.groupby(["sector_name", "risk_score"]).agg(
        loan_count=("loan_id", "count"),
        net_balance=("net_balance", "sum"),
        avg_payment_performance=("payment_performance", "mean"),
        total_paid=("total_paid", "sum"),
        total_invested=("total_invested", "sum"),
    ).reset_index()

    sector_metrics["actual_return_rate"] = sector_metrics["total_paid"] / sector_metrics["total_invested"]

    status_by_sector = df_with_risk.groupby(["sector_name", "loan_status"]).size().reset_index(name="count")
    total_by_sector = df_with_risk.groupby("sector_name").size().reset_index(name="total")
    status_by_sector = status_by_sector.merge(total_by_sector, on="sector_name")
    status_by_sector["pct"] = status_by_sector["count"] / status_by_sector["total"]

    problem_statuses = ["Late", "Default", "Bankrupt", "Severe", "Severe Delinquency", "Moderate Delinquency"]
    problem_loans = status_by_sector[status_by_sector["loan_status"].isin(problem_statuses)]
    problem_rate = problem_loans.groupby("sector_name")["pct"].sum().reset_index(name="problem_rate")

    sector_metrics = sector_metrics.merge(problem_rate, on="sector_name", how="left")
    sector_metrics["problem_rate"] = sector_metrics["problem_rate"].fillna(0)

    st.subheader("Industry Risk vs Performance")
    scatter = alt.Chart(sector_metrics).mark_circle(size=200).encode(
        x=alt.X("risk_score:Q", title="Industry Risk Score", axis=alt.Axis(format=".1f")),
        y=alt.Y("avg_payment_performance:Q", title="Avg Payment Performance", axis=alt.Axis(format=".0%")),
        size=alt.Size("net_balance:Q", title="Outstanding Balance"),
        color=alt.Color(
            "avg_payment_performance:Q",
            scale=alt.Scale(domain=[0.6, 0.8, 1.0], range=["#d62728", "#ffbb78", "#2ca02c"]),
            legend=alt.Legend(title="Performance", format=".0%"),
        ),
        tooltip=[
            alt.Tooltip("sector_name:N", title="Industry"),
            alt.Tooltip("risk_score:Q", title="Risk Score", format=".1f"),
            alt.Tooltip("avg_payment_performance:Q", title="Avg Performance", format=".1%"),
            alt.Tooltip("loan_count:Q", title="Loan Count"),
            alt.Tooltip("net_balance:Q", title="Outstanding Balance", format="$,.0f"),
        ],
    ).properties(width=700, height=400, title="Industry Risk Score vs Payment Performance")
    st.altair_chart(scatter, width="stretch")

    st.subheader("Problem Loan Rate by Industry")
    top_sectors = sector_metrics.nlargest(10, "net_balance")

    problem_bar = alt.Chart(top_sectors).mark_bar().encode(
        x=alt.X("problem_rate:Q", title="Problem Loan Rate", axis=alt.Axis(format=".0%")),
        y=alt.Y("sector_name:N", title="Industry Sector", sort="-x"),
        color=alt.Color(
            "problem_rate:Q",
            scale=alt.Scale(domain=[0, 0.2, 0.4], range=["#2ca02c", "#ffbb78", "#d62728"]),
            legend=None,
        ),
        tooltip=[
            alt.Tooltip("sector_name:N", title="Industry"),
            alt.Tooltip("problem_rate:Q", title="Problem Loan Rate", format=".1%"),
            alt.Tooltip("loan_count:Q", title="Total Loans"),
        ],
    ).properties(width=700, height=400, title="Problem Loan Rate by Industry (Top 10)")
    st.altair_chart(problem_bar, width="stretch")

    st.subheader("Actual Return Rate by Industry")
    return_bar = alt.Chart(top_sectors).mark_bar().encode(
        x=alt.X("actual_return_rate:Q", title="Actual Return Rate", axis=alt.Axis(format=".0%")),
        y=alt.Y("sector_name:N", title="Industry Sector", sort="-x"),
        color=alt.Color(
            "actual_return_rate:Q",
            scale=alt.Scale(domain=[0.5, 1.0, 1.3], range=["#d62728", "#ffbb78", "#2ca02c"]),
            legend=None,
        ),
        tooltip=[
            alt.Tooltip("sector_name:N", title="Industry"),
            alt.Tooltip("actual_return_rate:Q", title="Return Rate", format=".2%"),
            alt.Tooltip("loan_count:Q", title="Loan Count"),
        ],
    ).properties(width=700, height=400, title="Actual Return Rate by Industry (Top 10)")
    st.altair_chart(return_bar, width="stretch")

def plot_status_distribution(df: pd.DataFrame):
    active_df = df[df.get("loan_status", "") != "Paid Off"].copy()
    if active_df.empty:
        st.info("No active loans to display.")
        return

    status_counts = active_df["loan_status"].value_counts(normalize=True)
    counts_abs = active_df["loan_status"].value_counts()

    status_summary = pd.DataFrame({
        "status": status_counts.index.astype(str),
        "percentage": status_counts.values,
        "count": counts_abs.reindex(status_counts.index).values,
        "balance": active_df.groupby("loan_status")["net_balance"].sum().reindex(status_counts.index).values,
    })

    status_summary["color"] = status_summary["status"].apply(lambda x: LOAN_STATUS_COLORS.get(x, "#808080"))

    st.caption("Note: 'Paid Off' loans are excluded")

    pie_chart = alt.Chart(status_summary).mark_arc().encode(
        theta=alt.Theta(field="percentage", type="quantitative"),
        color=alt.Color(
            "status:N",
            scale=alt.Scale(
                domain=list(status_summary["status"]),
                range=list(status_summary["color"])
            ),
            legend=alt.Legend(title="Loan Status", orient="right"),
        ),
        tooltip=[
            alt.Tooltip("status:N", title="Loan Status"),
            alt.Tooltip("count:Q", title="Number of Loans"),
            alt.Tooltip("percentage:Q", title="% of Active Loans", format=".1%"),
            alt.Tooltip("balance:Q", title="Net Balance", format="$,.0f"),
        ],
    ).properties(width=600, height=400, title="Distribution of Active Loan Status")

    st.altair_chart(pie_chart, width="stretch")

def plot_roi_distribution(df: pd.DataFrame):
    roi_df = df[df["total_invested"] > 0].copy()
    roi_df = roi_df.sort_values("current_roi", ascending=False)

    if roi_df.empty:
        st.info("No loans with investment data.")
        return

    roi_chart = alt.Chart(roi_df).mark_bar().encode(
        x=alt.X("loan_id:N", title="Loan ID", sort="-y", axis=alt.Axis(labelAngle=-90, labelLimit=150)),
        y=alt.Y("current_roi:Q", title="Return on Investment (ROI)", axis=alt.Axis(format=".0%", grid=True)),
        color=alt.Color(
            "current_roi:Q",
            scale=alt.Scale(domain=[-0.5, 0, 0.5], range=["#ff0505", "#ffc302", "#2ca02c"]),
            legend=alt.Legend(title="ROI", format=".0%"),
        ),
        tooltip=[
            alt.Tooltip("loan_id:N", title="Loan ID"),
            alt.Tooltip("deal_name:N", title="Deal Name"),
            alt.Tooltip("loan_status:N", title="Status"),
            alt.Tooltip("current_roi:Q", title="Current ROI", format=".2%"),
            alt.Tooltip("total_invested:Q", title="Total Invested", format="$,.2f"),
            alt.Tooltip("total_paid:Q", title="Total Paid", format="$,.2f"),
        ],
    ).properties(width=800, height=400, title="Return on Investment by Loan")

    st.altair_chart(roi_chart, width="stretch")

def plot_irr_by_partner(df: pd.DataFrame):
    paid_df = df[df.get("loan_status", "") == "Paid Off"].copy()
    if paid_df.empty or "realized_irr" not in paid_df.columns:
        st.info("No paid-off loans with IRR data.")
        return

    irr_by_partner = paid_df.groupby("partner_source").agg(
        avg_irr=("realized_irr", "mean"),
        deal_count=("loan_id", "count"),
        total_invested=("total_invested", "sum"),
        total_returned=("total_paid", "sum"),
    ).dropna().reset_index()

    if irr_by_partner.empty:
        st.info("No partner data available.")
        return

    irr_chart = alt.Chart(irr_by_partner).mark_bar().encode(
        x=alt.X("avg_irr:Q", title="Average IRR", axis=alt.Axis(format=".0%", grid=True)),
        y=alt.Y("partner_source:N", title="Partner", sort="-x"),
        color=alt.Color(
            "avg_irr:Q",
            scale=alt.Scale(domain=[-0.1, 0, 0.5], range=["#d62728", "#ffc302", "#2ca02c"]),
            legend=alt.Legend(title="IRR", format=".0%"),
        ),
        tooltip=[
            alt.Tooltip("partner_source:N", title="Partner"),
            alt.Tooltip("avg_irr:Q", title="Average IRR", format=".2%"),
            alt.Tooltip("deal_count:Q", title="Number of Deals"),
        ],
    ).properties(width=700, height=400, title="Average IRR by Partner")

    st.altair_chart(irr_chart, width="stretch")

def display_irr_analysis(df: pd.DataFrame):
    st.subheader("IRR Analysis for Paid-Off Loans")

    paid_df = df[df.get("loan_status", "") == "Paid Off"].copy()
    if paid_df.empty:
        st.info("No paid-off loans to analyze.")
        return

    weighted_realized_irr = (paid_df["realized_irr"] * paid_df["total_invested"]).sum() / paid_df["total_invested"].sum()
    avg_realized_irr = paid_df["realized_irr"].mean()

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Weighted Avg Realized IRR", f"{weighted_realized_irr:.2%}" if pd.notnull(weighted_realized_irr) else "N/A")
    with col2:
        st.metric("Simple Avg Realized IRR", f"{avg_realized_irr:.2%}" if pd.notnull(avg_realized_irr) else "N/A")

# -----------
# Main Page
# -----------
def main():
    st.title("Loan Tape Dashboard")

    last_updated = get_last_updated()
    st.caption(f"Data last updated: {last_updated}")

    loans_df = load_loan_summaries()
    deals_df = load_deals()

    df = prepare_loan_data(loans_df, deals_df)
    df = calculate_irr(df)

    # -------------
    # Sidebar Filters
    # -------------
    st.sidebar.header("Filters")

    if "funding_date" in df.columns and not df["funding_date"].isna().all():
        min_date = df["funding_date"].min().date()
        max_date = df["funding_date"].max().date()

        use_date_filter = st.sidebar.checkbox("Filter by Funding Date", value=False)

        if use_date_filter:
            date_range = st.sidebar.date_input(
                "Select Date Range", [min_date, max_date], min_value=min_date, max_value=max_date
            )
            if isinstance(date_range, (list, tuple)) and len(date_range) == 2:
                filtered_df = df[
                    (df["funding_date"].dt.date >= date_range[0]) &
                    (df["funding_date"].dt.date <= date_range[1])
                ].copy()
            else:
                filtered_df = df.copy()
        else:
            filtered_df = df.copy()
    else:
        filtered_df = df.copy()

    all_statuses = ["All"] + sorted(df["loan_status"].dropna().unique().tolist())
    selected_status = st.sidebar.selectbox("Filter by Status", all_statuses, index=0)

    if selected_status != "All":
        filtered_df = filtered_df[filtered_df["loan_status"] == selected_status]

    st.sidebar.markdown("---")
    st.sidebar.write(f"**Showing:** {len(filtered_df)} of {len(df)} loans")

    # -----
    # Tabs
    # -----
    tabs = st.tabs(["Summary", "Capital Flow", "Performance Analysis", "Risk Analytics", "Loan Tape", "Diagnostics & ML"])

    with tabs[0]:
        st.header("Portfolio Overview")

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            total_positions = len(filtered_df)
            paid_off = (filtered_df["loan_status"] == "Paid Off").sum()
            st.metric("Total Positions", f"{total_positions}")
            st.caption(f"({paid_off} paid off)")
        with col2:
            total_deployed = filtered_df["csl_participation_amount"].sum()
            st.metric("Capital Deployed", f"${total_deployed:,.0f}")
        with col3:
            total_returned = filtered_df["total_paid"].sum()
            st.metric("Capital Returned", f"${total_returned:,.0f}")
        with col4:
            net_balance = filtered_df["net_balance"].sum()
            st.metric("Net Outstanding", f"${net_balance:,.0f}")

        st.markdown("---")
        st.subheader("Loan Status Distribution")
        plot_status_distribution(filtered_df)

        st.markdown("---")
        st.subheader("ROI Distribution by Loan")
        plot_roi_distribution(filtered_df)

    with tabs[1]:
        st.header("Capital Flow Analysis")
        plot_capital_flow(filtered_df)

        st.markdown("---")
        plot_investment_net_position(filtered_df)

        st.markdown("---")
        st.subheader("Payment Performance by Cohort")
        plot_payment_performance_by_cohort(filtered_df)

        st.markdown("---")
        display_irr_analysis(filtered_df)

        st.markdown("---")
        st.subheader("Average IRR by Partner")
        plot_irr_by_partner(filtered_df)

    with tabs[2]:
        st.header("Performance Analysis")

        plot_industry_performance_analysis(filtered_df)

        st.markdown("---")
        plot_fico_performance_analysis(filtered_df)

        st.markdown("---")
        plot_tib_performance_analysis(filtered_df)

    with tabs[3]:
        st.header("Risk Analytics")

        risk_df = calculate_risk_scores(filtered_df)

        if not risk_df.empty:
            with st.expander("How Risk Scores are Calculated"):
                st.markdown(
                    """
**Risk Score Formula:**
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
"""
                )

            col1, col2, col3 = st.columns(3)
            with col1:
                avg_risk = risk_df["risk_score"].mean()
                st.metric("Average Risk Score", f"{avg_risk:.2f}")
            with col2:
                high_risk_count = (risk_df["risk_score"] >= 1.5).sum()
                st.metric("High/Severe Risk Loans", f"{high_risk_count}")
            with col3:
                high_risk_balance = risk_df[risk_df["risk_score"] >= 1.5]["net_balance"].sum()
                st.metric("High Risk Balance", f"${high_risk_balance:,.0f}")

            st.markdown("---")
            st.subheader("Top 10 Highest Risk Loans")

            top_risk = risk_df.nlargest(10, "risk_score")[
                [
                    "loan_id", "deal_name", "loan_status", "payment_performance",
                    "days_since_funding", "days_past_maturity", "status_multiplier",
                    "risk_score", "net_balance",
                ]
            ].copy()

            top_risk_display = top_risk.copy()
            top_risk_display["payment_performance"] = top_risk_display["payment_performance"].map(lambda x: f"{x:.1%}" if pd.notnull(x) else "N/A")
            top_risk_display["risk_score"] = top_risk_display["risk_score"].map(lambda x: f"{x:.2f}" if pd.notnull(x) else "N/A")
            top_risk_display["status_multiplier"] = top_risk_display["status_multiplier"].map(lambda x: f"{x:.1f}x" if pd.notnull(x) else "N/A")
            top_risk_display["net_balance"] = top_risk_display["net_balance"].map(lambda x: f"${x:,.0f}" if pd.notnull(x) else "N/A")

            top_risk_display.columns = [
                "Loan ID", "Deal Name", "Status", "Payment Perf",
                "Days Funded", "Days Overdue", "Status Mult",
                "Risk Score", "Net Balance",
            ]
            st.dataframe(top_risk_display, width="stretch", hide_index=True)

            st.markdown("---")
            st.subheader("Risk Score Distribution")

            band_summary = risk_df.groupby("risk_band", observed=True).agg(
                loan_count=("loan_id", "count"),
                net_balance=("net_balance", "sum"),
            ).reset_index()

            if not band_summary.empty:
                risk_band_order = ["Low (0-0.5)", "Moderate (0.5-1.0)", "Elevated (1.0-1.5)", "High (1.5-2.0)", "Severe (2.0+)"]

                risk_bar = alt.Chart(band_summary).mark_bar().encode(
                    x=alt.X("risk_band:N", title="Risk Band", sort=risk_band_order),
                    y=alt.Y("loan_count:Q", title="Number of Loans"),
                    color=alt.Color(
                        "risk_band:N",
                        scale=alt.Scale(
                            domain=risk_band_order,
                            range=["#2ca02c", "#98df8a", "#ffbb78", "#ff7f0e", "#d62728"],
                        ),
                        legend=alt.Legend(title="Risk Level", orient="right"),
                        sort=risk_band_order,
                    ),
                    tooltip=[
                        alt.Tooltip("risk_band:N", title="Risk Band"),
                        alt.Tooltip("loan_count:Q", title="Loan Count"),
                        alt.Tooltip("net_balance:Q", title="Net Balance", format="$,.0f"),
                    ],
                ).properties(width=700, height=350, title="Loan Count by Risk Band (Active Loans Only)")

                st.altair_chart(risk_bar, width="stretch")
        else:
            st.info("No active loans to calculate risk scores.")

    with tabs[4]:
        st.header("Complete Loan Tape")

        display_columns = [
            "loan_id", "deal_name", "partner_source", "loan_status",
            "funding_date", "maturity_date",
            "csl_participation_amount", "total_invested", "total_paid", "net_balance",
            "current_roi", "payment_performance", "remaining_maturity_months",
        ]

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
            "remaining_maturity_months": "Months Left",
        }

        loan_tape = format_dataframe_for_display(filtered_df, display_columns, column_rename)
        st.dataframe(loan_tape, width="stretch", hide_index=True)

        csv = loan_tape.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="Download Loan Tape as CSV",
            data=csv,
            file_name=f"loan_tape_{pd.Timestamp.today().strftime('%Y%m%d')}.csv",
            mime="text/csv",
        )

    with tabs[5]:
        st.header("Diagnostics & ML")
    
        st.markdown("##### Correlations")
        render_corr_outputs(filtered_df)
    
        st.markdown("---")
        render_fico_tib_heatmap(filtered_df)
    
        st.markdown("---")
        st.subheader("Small Classification Model: Predict Problem Loans")
        try:
            model, metrics, top_pos, top_neg = train_classification_small(filtered_df)
            col1, col2, col3, col4 = st.columns(4)
            with col1: st.metric("ROC AUC (CV)", f"{metrics['ROC AUC'][0]:.3f}" if pd.notnull(metrics['ROC AUC'][0]) else "N/A")
            with col2: st.metric("Precision (CV)", f"{metrics['Precision'][0]:.3f}")
            with col3: st.metric("Recall (CV)", f"{metrics['Recall'][0]:.3f}")
            with col4: st.caption(f"n={metrics['n_samples']}, positive rate={metrics['pos_rate']:.2f}")
    
            st.write("**Top Risk-Increasing Signals (coefficients)**")
            st.dataframe(top_pos.assign(coef=lambda s: s["coef"].map(lambda x: f"{x:.3f}")), width="stretch", hide_index=True)
            st.write("**Top Risk-Decreasing Signals (coefficients)**")
            st.dataframe(top_neg.assign(coef=lambda s: s["coef"].map(lambda x: f"{x:.3f}")), width="stretch", hide_index=True)
        except ImportError:
            st.warning("scikit-learn or scipy not installed. `pip install scikit-learn scipy` to enable modeling.")
        except Exception as e:
            st.warning(f"Classification model could not run: {e}")
    
        st.markdown("---")
        st.subheader("Small Regression Model: Predict Payment Performance")
        try:
            r_model, r_metrics = train_regression_small(filtered_df)
            c1, c2 = st.columns(2)
            with c1: st.metric("R² (CV)", f"{r_metrics['R2'][0]:.3f}" if pd.notnull(r_metrics['R2'][0]) else "N/A")
            with c2: st.metric("RMSE (CV)", f"{r_metrics['RMSE'][0]:.3f}")
            st.caption(f"n={r_metrics['n_samples']} (rows with non-null payment_performance)")
        except ImportError:
            st.warning("scikit-learn not installed. `pip install scikit-learn`.")
        except Exception as e:
            st.warning(f"Regression model could not run: {e}")

if __name__ == "__main__":
    main()
