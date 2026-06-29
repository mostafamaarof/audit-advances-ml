"""
Audit Analytics Script - Open Advances to Suppliers / Staff
=============================================================
Performs a complete exploratory audit analysis on an open-advances
extract (SAP-style open item report) and produces a risk-ranked set
of audit findings, an Excel workbook, charts, and a Markdown report.

Author: Generated for audit data analytics review.
Run:    python audit_analysis.py
"""

import os
import re
import warnings
from datetime import datetime

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")  # headless rendering - no display available
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.cluster import KMeans, DBSCAN

warnings.filterwarnings("ignore")

# =====================================================================
# CONFIG
# =====================================================================
INPUT_PATH = r"C:\Users\IT\Desktop\Adances_Py.xlsx"
OUTPUT_DIR = r"C:\Users\IT\Desktop\Adances_Py_Audit_Output"
CHART_DIR = os.path.join(OUTPUT_DIR, "charts")
DATA_DIR = os.path.join(OUTPUT_DIR, "data")
REPORT_DIR = os.path.join(OUTPUT_DIR, "reports")

RANDOM_STATE = 42
TODAY = pd.Timestamp(datetime.now().date())

ROUND_NUMBER_THRESHOLDS = [100, 500, 1000, 5000, 10000]

plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.grid": True,
    "grid.alpha": 0.3,
    "font.size": 9,
})

# Collected throughout the run; each entry becomes a candidate for the
# "Top 20 Audit Issues" list at the end.
ISSUES = []


def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")


def add_issue(title, category, risk_score, description, recommendation, evidence=""):
    """Register an audit finding. risk_score is 0-100 (higher = more urgent)."""
    ISSUES.append({
        "Title": title,
        "Category": category,
        "Risk Score": round(float(risk_score), 1),
        "Description": description,
        "Recommendation": recommendation,
        "Evidence": evidence,
    })


def ensure_dirs():
    for d in (OUTPUT_DIR, CHART_DIR, DATA_DIR, REPORT_DIR):
        os.makedirs(d, exist_ok=True)


def safe_sheet_name(name):
    return name[:31]


# =====================================================================
# 1. DATA LOADING
# =====================================================================
def load_data(path):
    """Detect file type/encoding automatically and load into a DataFrame.

    Accepts either a filesystem path (str) or a file-like object with a
    '.name' attribute (e.g. a Streamlit UploadedFile), so the same function
    serves both the CLI script and the web app.
    """
    filename = path if isinstance(path, str) else getattr(path, "name", "")
    ext = os.path.splitext(filename)[1].lower()
    log(f"Detected file extension: {ext}")

    if ext in (".xlsx", ".xls", ".xlsm"):
        xls = pd.ExcelFile(path)
        sheet = xls.sheet_names[0]
        if len(xls.sheet_names) > 1:
            log(f"Workbook has multiple sheets {xls.sheet_names}; using first sheet '{sheet}'.")
        df = xls.parse(sheet)
        filetype_info = f"Excel workbook (sheet: {sheet})"
    elif ext == ".csv":
        encoding = "utf-8"
        try:
            df = pd.read_csv(path, encoding=encoding)
        except UnicodeDecodeError:
            encoding = "latin1"
            log("UTF-8 decode failed, retrying with latin1 encoding.")
            df = pd.read_csv(path, encoding=encoding)
        filetype_info = f"CSV (encoding: {encoding})"
    elif ext == ".json":
        df = pd.read_json(path)
        filetype_info = "JSON"
    else:
        raise ValueError(f"Unsupported file type: {ext}")

    log(f"Loaded {filetype_info}: {df.shape[0]} rows x {df.shape[1]} columns")
    return df, filetype_info


# =====================================================================
# 2. INTELLIGENT COLUMN MAPPING
# =====================================================================
# Each canonical field maps to a list of substrings searched (case-insensitive)
# against actual column names, in priority order. This lets the script adapt
# to datasets with different column naming conventions.
CANDIDATES = {
    "doc_number": ["document number", "doc no", "invoice number", "invoice no", "voucher"],
    "doc_type": ["document type", "doc type"],
    "doc_date": ["document date", "transaction date", "advance date", "doc date"],
    "posting_date": ["posting date"],
    "due_date": ["due date", "net due date"],
    "amount": ["amount in local currency", "amount", "net amount", "value"],
    "currency": ["currency key", "currency", "curr.", "ccy"],
    "exchange_rate": ["exchange rate", "fx rate"],
    "text": ["text", "description", "narrative"],
    "doc_status": ["doc.status", "document status", "status"],
    "fiscal_year": ["fiscal year", "fy"],
    "account_type": ["account type"],
    "supplier_id": ["vendor id", "vendor no", "supplier id", "supplier no", "account"],
    "supplier_name": ["supplier name", "vendor name", "name 1", "name"],
    "debit_credit": ["debit/credit", "dr/cr", "d/c ind"],
    "special_gl": ["special g/l", "special gl"],
    "sp_gl_trans_type": ["sp.g/l trans", "special gl trans"],
    "gl_account": ["g/l account", "gl account"],
    "user_responsible": ["user responsible", "responsible"],
    "user_name": ["user name"],
    "due_flag": ["due net"],
    "age_days": ["age (days)", "age in days", "days outstanding", "aging days"],
    "aging_bucket": ["aging bucket", "age bucket"],
    "payee_type": ["payee type", "vendor type"],
    "assignment": ["assignment", "reference"],
    "po_number": ["purchase order", "po number", "po no"],
    "contract_number": ["contract"],
    "project_code": ["project code", "wbs"],
    "country": ["country"],
    "approver": ["approved by", "approver"],
}


def map_columns(df):
    """Map canonical audit fields to actual dataframe columns."""
    colmap = {}
    cols_lower = {c: c.lower().strip() for c in df.columns}
    used = set()

    for canon, substrings in CANDIDATES.items():
        match = None
        for sub in substrings:
            for col, low in cols_lower.items():
                if col in used:
                    continue
                if low == sub or sub in low:
                    match = col
                    break
            if match:
                break
        if match:
            colmap[canon] = match
            used.add(match)

    # Validate date-like canonical fields actually contain dates
    for date_field in ("doc_date", "posting_date", "due_date"):
        col = colmap.get(date_field)
        if col is not None:
            converted = pd.to_datetime(df[col], errors="coerce")
            valid_ratio = converted.notna().mean()
            if valid_ratio < 0.5:
                log(f"Column '{col}' mapped to '{date_field}' but only {valid_ratio:.0%} "
                    f"parse as dates -> dropping this mapping.")
                del colmap[date_field]

    log("Column mapping resolved:")
    for canon, col in colmap.items():
        log(f"  {canon:18s} -> '{col}'")

    missing = [c for c in CANDIDATES if c not in colmap]
    if missing:
        log(f"Fields not found in source data (related analyses will be skipped): {missing}")

    return colmap


# =====================================================================
# 3. DATA CLEANING
# =====================================================================
def clean_data(df, colmap):
    """Type-convert dates/numerics, standardize text, return a cleaned copy."""
    cdf = df.copy()

    for date_field in ("doc_date", "posting_date", "due_date"):
        col = colmap.get(date_field)
        if col is not None:
            cdf[col] = pd.to_datetime(cdf[col], errors="coerce")

    amt_col = colmap.get("amount")
    if amt_col is not None:
        cdf[amt_col] = pd.to_numeric(cdf[amt_col], errors="coerce")

    age_col = colmap.get("age_days")
    if age_col is not None:
        cdf[age_col] = pd.to_numeric(cdf[age_col], errors="coerce")

    # Standardize text fields: strip whitespace, collapse internal spaces
    text_fields = ["supplier_name", "text", "assignment", "user_responsible", "user_name"]
    for field in text_fields:
        col = colmap.get(field)
        if col is not None and cdf[col].dtype == object:
            cdf[col] = cdf[col].astype(str).str.strip().str.replace(r"\s+", " ", regex=True)
            cdf.loc[cdf[col].isin(["nan", "None", ""]), col] = np.nan

    return cdf


# =====================================================================
# 4. DATA QUALITY ASSESSMENT
# =====================================================================
def data_quality_report(df, colmap):
    log("Running data quality assessment...")
    rows = []

    n = len(df)
    rows.append(("Total records", n, ""))
    rows.append(("Total columns", df.shape[1], ""))
    rows.append(("Fully duplicated rows", int(df.duplicated().sum()), ""))

    amt_col = colmap.get("amount")
    date_col = colmap.get("doc_date")
    name_col = colmap.get("supplier_name")
    id_col = colmap.get("supplier_id")
    doc_col = colmap.get("doc_number")
    curr_col = colmap.get("currency")
    fx_col = colmap.get("exchange_rate")

    for canon, col in colmap.items():
        if df[col].isna().any():
            rows.append((f"Missing values - {col}", int(df[col].isna().sum()),
                         f"{df[col].isna().mean():.1%} of records"))

    if name_col:
        blanks = df[name_col].isna().sum()
        rows.append(("Blank supplier/payee names", int(blanks), ""))

    if doc_col:
        dup_docs = df[doc_col].duplicated(keep=False).sum()
        rows.append(("Records sharing a duplicated document number", int(dup_docs),
                     "May be normal offsetting line items - reviewed separately"))

    if id_col and amt_col and date_col:
        dup_key = df[[id_col, amt_col, date_col]].duplicated(keep=False)
        rows.append(("Potential duplicate advances (same payee+amount+date)", int(dup_key.sum()), ""))

    if date_col:
        invalid_dates = df[date_col].isna().sum()
        future_dates = (df[date_col] > TODAY).sum()
        rows.append(("Invalid/unparseable dates", int(invalid_dates), ""))
        rows.append(("Future-dated transactions", int(future_dates), ""))

    if amt_col:
        rows.append(("Negative amounts (credit-side open items)", int((df[amt_col] < 0).sum()), ""))
        rows.append(("Zero amounts", int((df[amt_col] == 0).sum()), ""))

    if curr_col is None:
        rows.append(("Currency field present?", "No", "Single local-currency dataset - currency "
                     "analysis (#8) skipped"))
    if fx_col is None:
        rows.append(("Exchange rate field present?", "No", "Exchange-rate validation skipped"))

    for missing_field, label in [("po_number", "Purchase order"), ("contract_number", "Contract"),
                                   ("project_code", "Project code"), ("country", "Country")]:
        if colmap.get(missing_field) is None:
            rows.append((f"{label} field present?", "No", f"{label} reference checks skipped - "
                         "recommend requesting this data from the source system"))

    if id_col and doc_col:
        suspicious = df[doc_col].astype(str).str.len().value_counts()
        rows.append(("Distinct document-number lengths observed", len(suspicious),
                     str(dict(suspicious))))

    if name_col:
        names = df[name_col].dropna().astype(str)
        norm = names.str.lower().str.replace(r"[^a-z0-9]", "", regex=True)
        n_raw = names.nunique()
        n_norm = norm.nunique()
        if n_raw != n_norm:
            rows.append(("Possible inconsistent supplier name spellings",
                         int(n_raw - n_norm), "Distinct raw names vs normalized names differ"))

    dq_df = pd.DataFrame(rows, columns=["Check", "Result", "Notes"])

    dup_doc_count = int(df[doc_col].duplicated(keep=False).sum()) if doc_col else 0
    if dup_doc_count > 0:
        add_issue(
            "Document numbers shared by multiple open line items",
            "Data Quality", 35,
            f"{dup_doc_count} records share a document number with at least one other record. "
            "These are mostly offsetting debit/credit pairs from transfer postings (doc type AB), "
            "but each pairing should be confirmed as a genuine partial clearing rather than a "
            "duplicate entry.",
            "Pull the full document for each duplicated document number and confirm the debit and "
            "credit lines net to the expected clearing amount.",
            evidence=f"{dup_doc_count} records affected"
        )

    neg_count = int((df[amt_col] < 0).sum()) if amt_col else 0
    if amt_col and neg_count > 0:
        add_issue(
            "Negative (credit-side) open advance balances",
            "Data Quality", 25,
            f"{neg_count} of {n} open items carry a negative balance, meaning a credit memo or "
            "reversal is sitting open against the advance account rather than being cleared.",
            "Investigate why these credit balances remain open; confirm they net correctly against "
            "the related debit advance and are not indicative of an unresolved dispute.",
            evidence=f"{neg_count} negative-balance records"
        )

    log(f"Data quality checks completed: {len(dq_df)} checks recorded.")
    return dq_df


# =====================================================================
# 5. DESCRIPTIVE STATISTICS & OUTLIERS
# =====================================================================
def descriptive_stats(df, colmap):
    log("Computing descriptive statistics...")
    numeric_fields = {k: colmap[k] for k in ("amount", "age_days") if k in colmap}
    if not numeric_fields:
        log("No numeric fields available for descriptive statistics - skipping.")
        return pd.DataFrame()

    stats_rows = []
    for canon, col in numeric_fields.items():
        s = df[col].dropna()
        stats_rows.append({
            "Field": col,
            "Count": s.count(),
            "Mean": s.mean(),
            "Median": s.median(),
            "Std Dev": s.std(),
            "Min": s.min(),
            "Max": s.max(),
            "P25": s.quantile(0.25),
            "P75": s.quantile(0.75),
            "P90": s.quantile(0.90),
            "P95": s.quantile(0.95),
            "P99": s.quantile(0.99),
        })
    return pd.DataFrame(stats_rows)


def detect_outliers(df, colmap):
    """Flag outliers via IQR, Z-score, Isolation Forest, and DBSCAN."""
    log("Running outlier detection (IQR, Z-score, Isolation Forest, DBSCAN)...")
    amt_col = colmap.get("amount")
    age_col = colmap.get("age_days")
    if amt_col is None:
        log("No amount field - skipping outlier detection.")
        return pd.DataFrame(index=df.index)

    out = pd.DataFrame(index=df.index)
    amt = df[amt_col]

    q1, q3 = amt.quantile(0.25), amt.quantile(0.75)
    iqr = q3 - q1
    out["outlier_iqr"] = (amt < q1 - 1.5 * iqr) | (amt > q3 + 1.5 * iqr)

    z = stats.zscore(amt.fillna(amt.mean()))
    out["outlier_zscore"] = np.abs(z) > 3

    features = [amt_col] + ([age_col] if age_col else [])
    feat_df = df[features].fillna(df[features].median())
    scaled = StandardScaler().fit_transform(feat_df)

    iso = IsolationForest(contamination=0.05, random_state=RANDOM_STATE)
    out["outlier_isoforest"] = iso.fit_predict(scaled) == -1

    db = DBSCAN(eps=0.6, min_samples=5)
    db_labels = db.fit_predict(scaled)
    out["outlier_dbscan"] = db_labels == -1

    out["outlier_vote_count"] = out[["outlier_iqr", "outlier_zscore",
                                      "outlier_isoforest", "outlier_dbscan"]].sum(axis=1)
    out["is_outlier"] = out["outlier_vote_count"] >= 2

    n_out = int(out["is_outlier"].sum())
    log(f"Outlier detection complete: {n_out} records flagged by >=2 methods.")

    if n_out > 0:
        top_outliers = df.loc[out["is_outlier"], amt_col].abs().sort_values(ascending=False)
        add_issue(
            "Statistical outliers in advance amounts",
            "Outlier Detection", 55,
            f"{n_out} advances were flagged as statistical outliers by at least two independent "
            f"methods (IQR, Z-score, Isolation Forest, DBSCAN). The largest flagged amount is "
            f"{top_outliers.iloc[0]:,.2f}.",
            "Review each flagged outlier individually for business justification (e.g. legitimate "
            "large mobilization advance vs. data entry error or unusual/unauthorized payment).",
            evidence=f"{n_out} records, largest = {top_outliers.iloc[0]:,.2f}"
        )

    return out


# =====================================================================
# 6. SUPPLIER / PAYEE ANALYSIS
# =====================================================================
def supplier_analysis(df, colmap):
    log("Running supplier/payee analysis...")
    id_col = colmap.get("supplier_id")
    name_col = colmap.get("supplier_name")
    amt_col = colmap.get("amount")
    age_col = colmap.get("age_days")
    payee_col = colmap.get("payee_type")

    if not (id_col and amt_col):
        log("Missing supplier id or amount field - skipping supplier analysis.")
        return pd.DataFrame()

    group_cols = [id_col] + ([name_col] if name_col else [])
    g = df.groupby(group_cols)

    summary = g.agg(
        num_advances=(amt_col, "count"),
        total_amount=(amt_col, "sum"),
        gross_debit=(amt_col, lambda s: s[s > 0].sum()),
        gross_credit=(amt_col, lambda s: s[s < 0].sum()),
        avg_advance=(amt_col, "mean"),
        largest_advance=(amt_col, "max"),
    ).reset_index()

    if age_col:
        age_g = df.groupby(group_cols)[age_col].agg(oldest_age_days="max",
                                                       overdue_2y=lambda s: (s > 730).sum())
        summary = summary.merge(age_g, on=group_cols)

    if payee_col:
        ptype = df.groupby(group_cols)[payee_col].first().reset_index()
        summary = summary.merge(ptype, on=group_cols)

    summary["abs_total_amount"] = summary["total_amount"].abs()
    summary = summary.sort_values("abs_total_amount", ascending=False).reset_index(drop=True)

    # Simple 0-100 risk rank per supplier: concentration + age + multiplicity
    amt_rank = summary["abs_total_amount"].rank(pct=True)
    cnt_rank = summary["num_advances"].rank(pct=True)
    age_rank = summary["oldest_age_days"].rank(pct=True) if age_col else 0
    summary["supplier_risk_score"] = (0.5 * amt_rank + 0.2 * cnt_rank + 0.3 * age_rank) * 100
    summary = summary.sort_values("supplier_risk_score", ascending=False).reset_index(drop=True)

    total_outstanding = df[amt_col].sum()
    top5_share = summary["total_amount"].abs().head(5).sum() / summary["abs_total_amount"].sum()
    add_issue(
        "Supplier/payee concentration",
        "Supplier Analysis", 45 + min(40, top5_share * 50),
        f"The top 5 payees by absolute open balance account for {top5_share:.1%} of total "
        f"outstanding advance value across {len(summary)} distinct payees.",
        "Confirm the largest balances are supported by valid, approved underlying transactions "
        "and an active liquidation plan; concentration in a few payees raises both credit and "
        "fraud/collusion risk.",
        evidence=f"Top 5 payees = {top5_share:.1%} of total |amount|"
    )

    multi_advance_suppliers = (summary["num_advances"] > summary["num_advances"].quantile(0.95)).sum()
    if multi_advance_suppliers:
        add_issue(
            "Payees with an unusually high number of simultaneous open advances",
            "Supplier Analysis", 40,
            f"{multi_advance_suppliers} payees sit above the 95th percentile for number of open "
            "advance line items outstanding at once.",
            "Assess whether multiple simultaneous advances to the same payee reflect normal "
            "operating cycles (e.g. recurring travel advances) or a failure to clear/liquidate "
            "prior advances before issuing new ones.",
            evidence=f"{multi_advance_suppliers} payees above P95 advance count"
        )

    log(f"Supplier analysis complete: {len(summary)} distinct payees ranked.")
    return summary


# =====================================================================
# 7. AGING ANALYSIS
# =====================================================================
AGING_BINS = [0, 30, 60, 90, 180, 365, 730, 1095, 1e9]
AGING_LABELS = ["0-30", "31-60", "61-90", "91-180", "181-365",
                "1-2 years", "2-3 years", "More than 3 years"]


def aging_analysis(df, colmap):
    log("Running aging analysis...")
    age_col = colmap.get("age_days")
    amt_col = colmap.get("amount")
    id_col = colmap.get("supplier_id")
    if not age_col:
        log("No age/aging field available - skipping aging analysis.")
        return pd.DataFrame(), None

    bucket_col = pd.cut(df[age_col], bins=AGING_BINS, labels=AGING_LABELS, right=True)
    tmp = df.assign(_bucket=bucket_col)

    agg = {"num_advances": (amt_col, "count")} if amt_col else {}
    g = tmp.groupby("_bucket", observed=False)
    summary = pd.DataFrame({
        "num_advances": g.size(),
    })
    if amt_col:
        summary["total_amount"] = g[amt_col].sum()
        summary["pct_of_total_amount"] = summary["total_amount"].abs() / df[amt_col].abs().sum() * 100
    summary["pct_of_count"] = summary["num_advances"] / len(df) * 100
    if id_col:
        summary["distinct_payees"] = g[id_col].nunique()
    summary = summary.reset_index().rename(columns={"_bucket": "Aging Bucket"})

    over3y = df[df[age_col] > 1095]
    very_old_n = len(over3y)
    very_old_amt = over3y[amt_col].sum() if amt_col else None
    if very_old_n > 0:
        max_age_row = df.loc[df[age_col].idxmax()]
        add_issue(
            "Advances outstanding for more than 3 years",
            "Aging Analysis", 90,
            f"{very_old_n} advances ({very_old_n/len(df):.1%} of records) have been open for "
            f"more than 3 years; total value {very_old_amt:,.2f}. The single oldest item has been "
            f"open {int(max_age_row[age_col])} days (~{max_age_row[age_col]/365:.1f} years).",
            "These are prime write-off/impairment candidates. Confirm recoverability, obtain "
            "management's basis for continuing to carry them, and assess compliance with the "
            "entity's advance-liquidation policy.",
            evidence=f"{very_old_n} records, {very_old_amt:,.2f} total, oldest = "
                      f"{int(max_age_row[age_col])} days"
        )

    over1y_pct = (df[age_col] > 365).mean()
    add_issue(
        "Entire population consists of long-outstanding advances",
        "Aging Analysis", 70,
        f"{over1y_pct:.1%} of all records in this extract are already aged over 1 year "
        f"(minimum age in the dataset is {int(df[age_col].min())} days). This dataset appears to "
        "be a pre-filtered report of stale advances rather than the full open-items population.",
        "Confirm with management whether this extract represents ALL open advances or only those "
        "already flagged as aged; request the full open-items population (including <1 year items) "
        "for a complete population-level audit conclusion.",
        evidence=f"Min age = {int(df[age_col].min())} days, max age = {int(df[age_col].max())} days"
    )

    log(f"Aging analysis complete. {very_old_n} advances older than 3 years.")
    return summary, bucket_col


# =====================================================================
# 8. HIGH-RISK TRANSACTIONS
# =====================================================================
def high_risk_transactions(df, colmap):
    log("Scanning for high-risk transaction patterns...")
    amt_col = colmap.get("amount")
    date_col = colmap.get("doc_date")
    id_col = colmap.get("supplier_id")
    age_col = colmap.get("age_days")

    flags = pd.DataFrame(index=df.index)

    if amt_col:
        flags["flag_largest_advance"] = df[amt_col].abs() >= df[amt_col].abs().quantile(0.99)
        flags["flag_round_number"] = df[amt_col].abs().apply(
            lambda v: any(v != 0 and v % t == 0 for t in ROUND_NUMBER_THRESHOLDS))
    if age_col:
        flags["flag_oldest_advance"] = df[age_col] >= df[age_col].quantile(0.99)
        if amt_col:
            flags["flag_old_small_balance"] = (df[age_col] > 1095) & \
                (df[amt_col].abs() < df[amt_col].abs().quantile(0.25))
    if date_col:
        flags["flag_weekend"] = df[date_col].dt.dayofweek >= 5
        flags["flag_month_end"] = df[date_col].dt.is_month_end
        flags["flag_year_end"] = (df[date_col].dt.month == 12) & (df[date_col].dt.day >= 28)

    if id_col and date_col:
        df_sorted = df.sort_values([id_col, date_col])
        gap = df_sorted.groupby(id_col)[date_col].diff().dt.days
        close_together = (gap <= 3) & (gap >= 0)
        flags["flag_clustered_with_prior"] = close_together.reindex(df.index, fill_value=False)

    if id_col:
        counts = df[id_col].value_counts()
        flags["flag_multiple_advances_same_payee"] = df[id_col].map(counts) >= counts.quantile(0.95)

    flags["high_risk_flag_count"] = flags.select_dtypes(bool).sum(axis=1)

    n_weekend = int(flags.get("flag_weekend", pd.Series(dtype=bool)).sum())
    if date_col and n_weekend > 0:
        add_issue(
            "Transactions dated on weekends",
            "High-Risk Transactions", 30,
            f"{n_weekend} advance documents carry a weekend document date, which is unusual for "
            "standard processing cycles.",
            "Confirm whether weekend postings reflect system back-dating, batch processing, or "
            "genuine urgent/emergency transactions; corroborate with approval timestamps.",
            evidence=f"{n_weekend} weekend-dated records"
        )

    n_round = int(flags.get("flag_round_number", pd.Series(dtype=bool)).sum())
    if amt_col and n_round > 0:
        add_issue(
            "Round-number advance amounts",
            "High-Risk Transactions", 25,
            f"{n_round} advances ({n_round/len(df):.1%}) are exact round numbers (multiples of "
            "100/500/1,000/5,000/10,000), which can indicate estimated/unsupported amounts rather "
            "than invoice-driven figures.",
            "Sample round-number advances and trace to supporting calculation/contract terms.",
            evidence=f"{n_round} round-number records"
        )

    n_old_small = int(flags.get("flag_old_small_balance", pd.Series(dtype=bool)).sum())
    if age_col and amt_col and n_old_small > 0:
        add_issue(
            "Very old advances with small residual balances",
            "High-Risk Transactions", 50,
            f"{n_old_small} advances are both more than 3 years old AND in the bottom quartile by "
            "amount - classic 'dormant small balance' candidates that are often left unreconciled.",
            "Consider a targeted write-off/clean-up exercise for small, immaterial, long-aged "
            "balances after confirming no recovery action is pending.",
            evidence=f"{n_old_small} records"
        )

    n_cluster = int(flags.get("flag_clustered_with_prior", pd.Series(dtype=bool)).sum())
    if n_cluster > 0:
        add_issue(
            "Multiple advances issued to the same payee within a short window",
            "High-Risk Transactions", 35,
            f"{n_cluster} advances were issued within 3 days of another advance to the same payee.",
            "Evaluate whether closely-spaced advances reflect legitimate operational need or "
            "potential splitting of a single requirement to bypass approval thresholds.",
            evidence=f"{n_cluster} records"
        )

    log("High-risk transaction scan complete.")
    return flags


# =====================================================================
# 9. TREND ANALYSIS
# =====================================================================
def trend_analysis(df, colmap):
    log("Running trend analysis...")
    date_col = colmap.get("doc_date")
    amt_col = colmap.get("amount")
    if not date_col:
        log("No date field - skipping trend analysis.")
        return {}

    s = df.set_index(date_col)[amt_col] if amt_col else df.set_index(date_col).iloc[:, 0]
    monthly = s.resample("ME").agg(["count", "sum"])
    quarterly = s.resample("QE").agg(["count", "sum"])
    yearly = s.resample("YE").agg(["count", "sum"])

    if len(monthly) > 3:
        mean_m, std_m = monthly["sum"].mean(), monthly["sum"].std()
        spikes = monthly[(monthly["sum"] - mean_m).abs() > 2 * std_m]
        if len(spikes):
            add_issue(
                "Unusual monthly spikes in advance volume",
                "Trend Analysis", 30,
                f"{len(spikes)} month(s) show total advance value more than 2 standard deviations "
                "from the monthly mean, indicating unusual concentration of activity in those periods.",
                "Investigate the business driver for spike month(s) (e.g. year-end push, project "
                "mobilization, or control breakdown allowing bulk unauthorized advances).",
                evidence=", ".join(spikes.index.strftime("%Y-%m").tolist())
            )

    log("Trend analysis complete.")
    return {"monthly": monthly, "quarterly": quarterly, "yearly": yearly}


# =====================================================================
# 10. CURRENCY ANALYSIS (graceful skip if not applicable)
# =====================================================================
def currency_analysis(df, colmap):
    curr_col = colmap.get("currency")
    if not curr_col:
        log("No currency column present - all amounts are in a single local currency. "
            "Currency analysis (#8) is not applicable and has been skipped.")
        return None
    log("Running currency analysis...")
    return df.groupby(curr_col)[colmap["amount"]].agg(["count", "sum"])


# =====================================================================
# 11. BUSINESS RULE TESTS
# =====================================================================
def business_rule_tests(df, colmap):
    log("Running business rule tests...")
    results = []
    amt_col = colmap.get("amount")
    id_col = colmap.get("supplier_id")
    age_col = colmap.get("age_days")
    assignment_col = colmap.get("assignment")
    doc_col = colmap.get("doc_number")

    if age_col:
        n = int((df[age_col] > 365).sum())
        results.append(("Advances open > 1 year", n, f"{n/len(df):.1%} of records"))
    if id_col and amt_col:
        per_payee = df.groupby(id_col)[amt_col].sum().abs()
        hi = per_payee[per_payee > per_payee.quantile(0.95)]
        results.append(("Payees with unusually high net balance (>P95)", len(hi), ""))
    if id_col:
        open_counts = df[id_col].value_counts()
        results.append(("Payees with multiple open advances simultaneously",
                         int((open_counts > 1).sum()), ""))
    if amt_col:
        repeated_amounts = df[amt_col].value_counts()
        repeated_amounts = repeated_amounts[repeated_amounts > 1]
        results.append(("Distinct amounts repeated 2+ times across records",
                         len(repeated_amounts), f"{repeated_amounts.sum()} total records involved"))
    if doc_col:
        nums = pd.to_numeric(df[doc_col], errors="coerce").dropna().sort_values()
        seq_gaps = nums.diff().eq(1).sum()
        results.append(("Sequential (+1) document numbers found", int(seq_gaps), ""))
    if assignment_col:
        missing_ref = df[assignment_col].isna().sum() + (df[assignment_col] == "00000000").sum()
        results.append(("Records with missing/placeholder reference (Assignment)",
                         int(missing_ref), "blank or '00000000' placeholder"))
    for field, label in [("po_number", "Missing purchase order reference"),
                          ("contract_number", "Missing contract reference"),
                          ("project_code", "Missing project code")]:
        if colmap.get(field) is None:
            results.append((label, "N/A", "Field not present in source data - cannot test"))

    rule_df = pd.DataFrame(results, columns=["Business Rule Test", "Result", "Notes"])

    if amt_col:
        repeated_amounts = df[amt_col].value_counts()
        repeated_amounts = repeated_amounts[repeated_amounts >= 5]
        if len(repeated_amounts):
            add_issue(
                "Same advance amount repeated many times",
                "Business Rules", 30,
                f"{len(repeated_amounts)} distinct amount value(s) each appear 5 or more times "
                "across different records, which can indicate templated/estimated advances rather "
                "than invoice-specific amounts, or potential duplicate processing.",
                "Review the most frequently repeated amounts and confirm each instance ties to a "
                "distinct, legitimate underlying transaction.",
                evidence=f"Top repeated amount = {repeated_amounts.index[0]:,.2f} "
                         f"({repeated_amounts.iloc[0]} occurrences)"
            )

    if assignment_col:
        missing_ref = df[assignment_col].isna().sum() + (df[assignment_col] == "00000000").sum()
        if missing_ref > 0:
            add_issue(
                "Missing or placeholder transaction references",
                "Business Rules", 20,
                f"{missing_ref} records have a blank or placeholder ('00000000') value in the "
                "Assignment/reference field, limiting traceability to the originating request.",
                "Request that all advance postings carry a meaningful assignment/reference value "
                "(e.g. travel request number, PO number) to support traceability.",
                evidence=f"{missing_ref} records"
            )

    log("Business rule tests complete.")
    return rule_df


# =====================================================================
# 12. BENFORD'S LAW
# =====================================================================
def benford_analysis(df, colmap):
    log("Running Benford's Law analysis...")
    amt_col = colmap.get("amount")
    if not amt_col:
        return None

    vals = df[amt_col].abs()
    vals = vals[vals >= 1]
    first_digits = vals.apply(lambda v: int(str(v).lstrip("0.").replace(".", "")[0]))
    observed = first_digits.value_counts(normalize=True).reindex(range(1, 10), fill_value=0).sort_index()
    expected = pd.Series({d: np.log10(1 + 1 / d) for d in range(1, 10)})

    chi_sq = ((observed - expected) ** 2 / expected).sum() * len(vals)
    mad = (observed - expected).abs().mean()

    result = pd.DataFrame({"Digit": range(1, 10), "Observed %": (observed * 100).round(2),
                            "Expected %": (expected * 100).round(2)})

    # MAD conformity thresholds per Nigrini
    if mad < 0.006:
        conformity = "Close conformity"
    elif mad < 0.012:
        conformity = "Acceptable conformity"
    elif mad < 0.015:
        conformity = "Marginally acceptable conformity"
    else:
        conformity = "Nonconformity"

    log(f"Benford's Law: Chi-sq={chi_sq:.2f}, MAD={mad:.4f} ({conformity})")

    if conformity in ("Marginally acceptable conformity", "Nonconformity"):
        worst_digit = (observed - expected).abs().idxmax()
        add_issue(
            "First-digit distribution deviates from Benford's Law",
            "Fraud Indicators", 50,
            f"The distribution of leading digits in advance amounts shows {conformity.lower()} "
            f"with Benford's Law (MAD={mad:.4f}, Chi-sq={chi_sq:.1f}). Digit {worst_digit} shows "
            "the largest deviation from expectation.",
            "Benford deviations alone are not proof of manipulation but warrant a closer look at "
            "how advance amounts are determined/estimated, especially for the over- or "
            "under-represented leading digit.",
            evidence=f"MAD={mad:.4f}, worst digit={worst_digit}"
        )

    return {"table": result, "chi_sq": chi_sq, "mad": mad, "conformity": conformity}


# =====================================================================
# 13. FRAUD INDICATORS
# =====================================================================
def fraud_indicators(df, colmap):
    log("Compiling fraud indicator flags...")
    amt_col = colmap.get("amount")
    id_col = colmap.get("supplier_id")
    date_col = colmap.get("doc_date")

    flags = pd.DataFrame(index=df.index)

    if amt_col and id_col:
        flags["dup_payee_amount"] = df.duplicated(subset=[id_col, amt_col], keep=False)
    if date_col:
        flags["dup_date"] = df.duplicated(subset=[date_col], keep=False)
    if amt_col and id_col and date_col:
        flags["dup_payee_amount_date"] = df.duplicated(subset=[id_col, amt_col, date_col], keep=False)
    if amt_col:
        flags["repeated_round_amount"] = df[amt_col].abs().apply(
            lambda v: v != 0 and v % 1000 == 0)

    flags["fraud_score"] = flags.select_dtypes(bool).sum(axis=1)

    if "dup_payee_amount_date" in flags and flags["dup_payee_amount_date"].sum() > 0:
        n = int(flags["dup_payee_amount_date"].sum())
        add_issue(
            "Exact duplicate payee + amount + date combinations",
            "Fraud Indicators", 65,
            f"{n} records share an identical payee, amount, and document date with at least one "
            "other record - a strong indicator of potential duplicate payment/advance issuance.",
            "Pull source documents for each matching set and confirm these are not duplicate "
            "advances issued in error or through circumvented controls.",
            evidence=f"{n} records in matching sets"
        )

    log("Fraud indicator compilation complete.")
    return flags


# =====================================================================
# 14. CORRELATION ANALYSIS
# =====================================================================
def correlation_analysis(df, colmap):
    log("Running correlation analysis...")
    numeric_cols = [colmap[k] for k in ("amount", "age_days", "fiscal_year") if k in colmap]
    numeric_cols = [c for c in numeric_cols if c in df.columns]
    if len(numeric_cols) < 2:
        log("Fewer than 2 numeric fields available - skipping correlation analysis.")
        return None
    corr = df[numeric_cols].corr(numeric_only=True)
    return corr


# =====================================================================
# 15. CLUSTER ANALYSIS
# =====================================================================
def cluster_analysis(supplier_summary):
    log("Running supplier cluster analysis...")
    if supplier_summary is None or supplier_summary.empty:
        log("No supplier summary available - skipping clustering.")
        return supplier_summary

    feature_cols = [c for c in ["abs_total_amount", "num_advances", "oldest_age_days"]
                     if c in supplier_summary.columns]
    if len(feature_cols) < 2:
        log("Not enough features for clustering - skipping.")
        return supplier_summary

    feats = supplier_summary[feature_cols].fillna(0)
    scaled = StandardScaler().fit_transform(feats)

    k = min(4, max(2, len(supplier_summary) // 50 + 2))
    km = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
    supplier_summary = supplier_summary.copy()
    supplier_summary["cluster"] = km.fit_predict(scaled)

    cluster_sizes = supplier_summary["cluster"].value_counts()
    small_unusual = cluster_sizes[cluster_sizes <= max(1, len(supplier_summary) * 0.03)]
    if len(small_unusual):
        for cl in small_unusual.index:
            members = supplier_summary[supplier_summary["cluster"] == cl]
            add_issue(
                f"Unusual small supplier cluster (cluster {cl})",
                "Cluster Analysis", 35,
                f"K-means clustering on amount/frequency/age isolated a small cluster of "
                f"{len(members)} payee(s) with a distinct profile from the rest of the population.",
                "Review members of this outlier cluster individually - small, distinct clusters "
                "often correspond to atypical, high-value, or otherwise unusual relationships.",
                evidence=f"Cluster size = {len(members)}"
            )

    log(f"Clustering complete: {k} clusters formed.")
    return supplier_summary


# =====================================================================
# 16. VISUALIZATIONS
# =====================================================================
def savefig(fig, name):
    path = os.path.join(CHART_DIR, name)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    log(f"Saved chart: {name}")


def simple_treemap(ax, sizes, labels):
    """Minimal slice-and-dice treemap (no external dependency)."""
    order = np.argsort(sizes)[::-1]
    sizes = np.array(sizes)[order]
    labels = np.array(labels)[order]
    total = sizes.sum()
    x, y, w, h = 0.0, 0.0, 1.0, 1.0
    horizontal = w >= h
    cursor = 0.0
    for s, lab in zip(sizes, labels):
        frac = s / total
        if horizontal:
            rect_w = w * frac
            ax.add_patch(plt.Rectangle((x + cursor, y), rect_w, h, edgecolor="white",
                                        facecolor=plt.cm.tab20(np.random.RandomState(hash(lab) % (2**32)).rand())))
            ax.text(x + cursor + rect_w / 2, y + h / 2, f"{lab}\n{s:,.0f}", ha="center",
                    va="center", fontsize=7, wrap=True)
            cursor += rect_w
        else:
            rect_h = h * frac
            ax.add_patch(plt.Rectangle((x, y + cursor), w, rect_h, edgecolor="white",
                                        facecolor=plt.cm.tab20(np.random.RandomState(hash(lab) % (2**32)).rand())))
            ax.text(x + w / 2, y + cursor + rect_h / 2, f"{lab}\n{s:,.0f}", ha="center",
                    va="center", fontsize=7, wrap=True)
            cursor += rect_h
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")


def make_visualizations(df, colmap, supplier_summary, aging_summary, trends, corr, benford):
    log("Generating visualizations...")
    amt_col = colmap.get("amount")
    age_col = colmap.get("age_days")
    payee_col = colmap.get("payee_type")
    doctype_col = colmap.get("doc_type")
    gl_col = colmap.get("gl_account")
    name_col = colmap.get("supplier_name")

    # 1. Bar: top 20 suppliers by absolute outstanding amount
    if supplier_summary is not None and not supplier_summary.empty and name_col:
        top20 = supplier_summary.head(20)
        fig, ax = plt.subplots(figsize=(10, 7))
        ax.barh(top20[name_col].astype(str).str[:30], top20["abs_total_amount"], color="#2c5f8a")
        ax.invert_yaxis()
        ax.set_xlabel("Absolute outstanding amount")
        ax.set_title("Top 20 Payees by Outstanding Advance Amount")
        savefig(fig, "01_bar_top_suppliers.png")

    # 2. Histogram of amounts
    if amt_col:
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.hist(df[amt_col].clip(df[amt_col].quantile(0.01), df[amt_col].quantile(0.99)),
                bins=40, color="#5b8fb9", edgecolor="white")
        ax.set_title("Distribution of Advance Amounts (1st-99th pct clipped)")
        ax.set_xlabel("Amount in local currency")
        savefig(fig, "02_histogram_amount.png")

    # 3. Pie: payee type by amount
    if payee_col and amt_col:
        by_type = df.groupby(payee_col)[amt_col].sum().abs()
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.pie(by_type, labels=by_type.index, autopct="%1.1f%%", startangle=90,
               colors=["#2c5f8a", "#e2a13a", "#7a9e7e"])
        ax.set_title("Outstanding Amount by Payee Type")
        savefig(fig, "03_pie_payee_type.png")

    # 4. Pareto chart of supplier concentration
    if supplier_summary is not None and not supplier_summary.empty:
        s = supplier_summary["abs_total_amount"].sort_values(ascending=False).reset_index(drop=True)
        cum_pct = s.cumsum() / s.sum() * 100
        fig, ax1 = plt.subplots(figsize=(10, 5))
        ax1.bar(range(1, min(51, len(s) + 1)), s.head(50), color="#2c5f8a")
        ax1.set_xlabel("Payee rank")
        ax1.set_ylabel("Amount")
        ax2 = ax1.twinx()
        ax2.plot(range(1, min(51, len(s) + 1)), cum_pct.head(50), color="#e2412a", marker="o", markersize=3)
        ax2.set_ylabel("Cumulative %")
        ax2.axhline(80, color="gray", linestyle="--", linewidth=1)
        ax1.set_title("Pareto Chart - Top 50 Payees by Outstanding Amount")
        savefig(fig, "04_pareto_suppliers.png")

    # 5. Boxplot of amount by aging bucket
    age_bucket_col_name = colmap.get("aging_bucket")
    if amt_col and age_col:
        bucket = pd.cut(df[age_col], bins=AGING_BINS, labels=AGING_LABELS)
        plot_df = pd.DataFrame({"bucket": bucket, "amount": df[amt_col]}).dropna()
        groups = [g["amount"].values for _, g in plot_df.groupby("bucket", observed=True)]
        labels = [str(k) for k, _ in plot_df.groupby("bucket", observed=True)]
        if groups:
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.boxplot(groups, labels=labels, showfliers=True)
            ax.set_title("Advance Amount Distribution by Aging Bucket")
            ax.set_ylabel("Amount")
            plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
            savefig(fig, "05_boxplot_amount_by_aging.png")

    # 6. Scatter: age vs amount
    if amt_col and age_col:
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.scatter(df[age_col], df[amt_col].abs(), alpha=0.4, s=15, color="#2c5f8a")
        ax.set_xlabel("Age (days)")
        ax.set_ylabel("|Amount|")
        ax.set_yscale("log")
        ax.set_title("Age vs. Amount (log scale)")
        savefig(fig, "06_scatter_age_amount.png")

    # 7. Correlation heatmap
    if corr is not None:
        fig, ax = plt.subplots(figsize=(5, 4))
        im = ax.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
        ax.set_xticks(range(len(corr.columns)))
        ax.set_yticks(range(len(corr.columns)))
        ax.set_xticklabels(corr.columns, rotation=45, ha="right")
        ax.set_yticklabels(corr.columns)
        for i in range(len(corr.columns)):
            for j in range(len(corr.columns)):
                ax.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center", fontsize=8)
        fig.colorbar(im, ax=ax, shrink=0.8)
        ax.set_title("Correlation Heatmap")
        savefig(fig, "07_correlation_heatmap.png")

    # 8. Time series: monthly count & amount
    if trends and "monthly" in trends and len(trends["monthly"]):
        m = trends["monthly"]
        fig, ax1 = plt.subplots(figsize=(11, 5))
        ax1.plot(m.index, m["sum"], color="#2c5f8a", marker="o", markersize=3, label="Total amount")
        ax1.set_ylabel("Total amount")
        ax2 = ax1.twinx()
        ax2.bar(m.index, m["count"], alpha=0.25, color="#e2a13a", width=20, label="Count")
        ax2.set_ylabel("Count")
        ax1.set_title("Monthly Advance Document Trend")
        fig.legend(loc="upper left")
        savefig(fig, "08_timeseries_monthly.png")

    # 9. Treemap of amount by G/L account
    if gl_col and amt_col:
        by_gl = df.groupby(gl_col)[amt_col].sum().abs().sort_values(ascending=False)
        fig, ax = plt.subplots(figsize=(9, 6))
        simple_treemap(ax, by_gl.values, [str(x) for x in by_gl.index])
        ax.set_title("Treemap - Outstanding Amount by G/L Account")
        savefig(fig, "09_treemap_gl_account.png")

    # 10. Bar: aging bucket distribution
    if aging_summary is not None and not aging_summary.empty:
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.bar(aging_summary["Aging Bucket"].astype(str), aging_summary["num_advances"], color="#7a9e7e")
        ax.set_title("Number of Advances by Aging Bucket")
        plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
        savefig(fig, "10_bar_aging_buckets.png")

    # 11. Bar: document type distribution
    if doctype_col:
        vc = df[doctype_col].value_counts()
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.bar(vc.index.astype(str), vc.values, color="#5b8fb9")
        ax.set_title("Record Count by Document Type")
        savefig(fig, "11_bar_doc_type.png")

    # 12. Benford digit distribution
    if benford:
        t = benford["table"]
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.bar(t["Digit"] - 0.15, t["Observed %"], width=0.3, label="Observed", color="#2c5f8a")
        ax.bar(t["Digit"] + 0.15, t["Expected %"], width=0.3, label="Benford expected", color="#e2a13a")
        ax.set_xticks(range(1, 10))
        ax.set_title(f"Benford's Law - First Digit Distribution ({benford['conformity']})")
        ax.legend()
        savefig(fig, "12_benford_digits.png")

    log("All visualizations generated.")


# =====================================================================
# 17. RISK SCORING ENGINE (0-100 PER ADVANCE)
# =====================================================================
def risk_scoring(df, colmap, outlier_flags, hr_flags, fraud_flags):
    log("Computing per-advance risk scores...")
    amt_col = colmap.get("amount")
    age_col = colmap.get("age_days")

    score = pd.Series(0.0, index=df.index)

    if age_col:
        age_pct = df[age_col].rank(pct=True)
        score += age_pct * 35

    if amt_col:
        amt_pct = df[amt_col].abs().rank(pct=True)
        score += amt_pct * 25

    if outlier_flags is not None and "is_outlier" in outlier_flags:
        score += outlier_flags["is_outlier"].astype(int) * 15

    if hr_flags is not None and "high_risk_flag_count" in hr_flags:
        max_flags = hr_flags["high_risk_flag_count"].max() or 1
        score += (hr_flags["high_risk_flag_count"] / max_flags) * 15

    if fraud_flags is not None and "fraud_score" in fraud_flags:
        max_fraud = fraud_flags["fraud_score"].max() or 1
        score += (fraud_flags["fraud_score"] / max_fraud) * 10

    score = score.clip(0, 100)
    result = df.copy()
    result["risk_score"] = score.round(1)
    result["risk_rank"] = result["risk_score"].rank(ascending=False, method="min").astype(int)
    result = result.sort_values("risk_score", ascending=False)
    log(f"Risk scoring complete. Highest score = {result['risk_score'].max()}, "
        f"mean = {result['risk_score'].mean():.1f}")
    return result


# =====================================================================
# 18-19. TOP 20 ISSUES + EXECUTIVE SUMMARY
# =====================================================================
def add_transaction_level_issues(risk_scored_df, colmap, top_n=8):
    """Surface the single highest-risk individual transactions as discrete issues."""
    name_col = colmap.get("supplier_name")
    amt_col = colmap.get("amount")
    age_col = colmap.get("age_days")
    doc_col = colmap.get("doc_number")

    top = risk_scored_df.head(top_n)
    for _, row in top.iterrows():
        payee = row[name_col] if name_col else "Unknown payee"
        amt = row[amt_col] if amt_col else None
        age = row[age_col] if age_col else None
        doc = row[doc_col] if doc_col else None
        desc = f"Advance to '{payee}'"
        if doc is not None:
            desc += f" (document {doc})"
        if amt is not None:
            desc += f", amount {amt:,.2f}"
        if age is not None:
            desc += f", open {int(age)} days (~{age/365:.1f} years)"
        add_issue(
            f"High-risk individual advance: {payee}",
            "Transaction-Level", float(row["risk_score"]),
            desc + ". This single record scored in the top tier of the composite risk model "
            "(age, amount, outlier and fraud-indicator weighted).",
            "Select for detailed substantive testing: confirm business purpose, approval, and "
            "current recoverability/liquidation status.",
            evidence=f"risk_score={row['risk_score']}"
        )


def build_top20():
    log("Ranking all findings to build the Top 20 Audit Issues list...")
    issues_df = pd.DataFrame(ISSUES).drop_duplicates(subset=["Title"])
    issues_df = issues_df.sort_values("Risk Score", ascending=False).reset_index(drop=True)
    top20 = issues_df.head(20).copy()
    top20.insert(0, "Rank", range(1, len(top20) + 1))
    return issues_df, top20


def executive_summary(df, colmap, dq_df, supplier_summary, aging_summary, top20, benford):
    amt_col = colmap.get("amount")
    age_col = colmap.get("age_days")
    name_col = colmap.get("supplier_name")

    lines = []
    lines.append(f"# Executive Audit Summary - Open Advances Review")
    lines.append(f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    lines.append(f"\n## Population")
    lines.append(f"- Records analyzed: {len(df):,}")
    if amt_col:
        lines.append(f"- Net outstanding amount: {df[amt_col].sum():,.2f}")
        lines.append(f"- Gross debit (advances issued): {df[df[amt_col] > 0][amt_col].sum():,.2f}")
        lines.append(f"- Gross credit (open reversals/credits): {df[df[amt_col] < 0][amt_col].sum():,.2f}")
    if age_col:
        lines.append(f"- Age range: {int(df[age_col].min())} to {int(df[age_col].max())} days "
                      f"({df[age_col].min()/365:.1f} to {df[age_col].max()/365:.1f} years)")
    if name_col:
        lines.append(f"- Distinct payees: {df[name_col].nunique():,}")

    if supplier_summary is not None and not supplier_summary.empty and name_col:
        top_payee = supplier_summary.iloc[0]
        lines.append(f"\n## Largest Exposure")
        lines.append(f"- Largest payee by absolute outstanding balance: "
                      f"**{top_payee.get(name_col, 'N/A')}** "
                      f"({top_payee['abs_total_amount']:,.2f} across {top_payee['num_advances']} "
                      f"advance(s))")

    lines.append(f"\n## Data Quality")
    n_dq_flags = len(dq_df)
    lines.append(f"- {n_dq_flags} data quality checks performed; see Data Quality Report sheet for detail.")

    if benford:
        lines.append(f"\n## Benford's Law")
        lines.append(f"- Conformity: {benford['conformity']} (MAD={benford['mad']:.4f})")

    lines.append(f"\n## Top Risk Areas")
    for _, row in top20.head(10).iterrows():
        lines.append(f"{row['Rank']}. **{row['Title']}** (risk score {row['Risk Score']:.0f}) - "
                      f"{row['Description']}")

    lines.append(f"\n## Overall Conclusion")
    lines.append(
        "This population is, by construction, already composed of long-aged advances "
        "(minimum age observed exceeds one year). The combination of extreme aging, "
        "concentration in a limited number of payees, and the data-quality/fraud indicators "
        "identified above warrant immediate substantive testing before year-end close, with a "
        "particular focus on recoverability assertions and potential impairment/write-off."
    )
    return "\n".join(lines)


# =====================================================================
# 20. EXPORT - EXCEL WORKBOOK + MARKDOWN REPORT
# =====================================================================
def export_excel(original_df, cleaned_df, dq_df, desc_stats, supplier_summary, aging_summary,
                  hr_flags, outlier_flags, fraud_flags, risk_scored_df, rule_df, benford,
                  issues_df, top20, exec_summary_text):
    path = os.path.join(REPORT_DIR, "Audit_Workbook.xlsx")
    log(f"Writing Excel workbook to {path} ...")

    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        original_df.to_excel(writer, sheet_name=safe_sheet_name("Original Data"), index=False)
        cleaned_df.to_excel(writer, sheet_name=safe_sheet_name("Cleaned Data"), index=False)
        dq_df.to_excel(writer, sheet_name=safe_sheet_name("Data Quality Report"), index=False)
        if not desc_stats.empty:
            desc_stats.to_excel(writer, sheet_name=safe_sheet_name("Descriptive Stats"), index=False)
        if supplier_summary is not None and not supplier_summary.empty:
            supplier_summary.to_excel(writer, sheet_name=safe_sheet_name("Supplier Summary"), index=False)
        if aging_summary is not None and not aging_summary.empty:
            aging_summary.to_excel(writer, sheet_name=safe_sheet_name("Aging Analysis"), index=False)
        if rule_df is not None and not rule_df.empty:
            rule_df.to_excel(writer, sheet_name=safe_sheet_name("Business Rule Tests"), index=False)
        if benford:
            benford["table"].to_excel(writer, sheet_name=safe_sheet_name("Benford Analysis"), index=False)

        if hr_flags is not None and not hr_flags.empty:
            high_risk_rows = cleaned_df.join(hr_flags)
            high_risk_rows = high_risk_rows[hr_flags.get("high_risk_flag_count", 0) > 0]
            high_risk_rows.to_excel(writer, sheet_name=safe_sheet_name("High-Risk Transactions"), index=False)

        if outlier_flags is not None and not outlier_flags.empty and "is_outlier" in outlier_flags:
            outlier_rows = cleaned_df.join(outlier_flags)
            outlier_rows = outlier_rows[outlier_flags["is_outlier"]]
            outlier_rows.to_excel(writer, sheet_name=safe_sheet_name("Outliers"), index=False)

        if fraud_flags is not None and not fraud_flags.empty:
            fraud_rows = cleaned_df.join(fraud_flags)
            fraud_rows = fraud_rows[fraud_flags.get("fraud_score", 0) > 0]
            fraud_rows.to_excel(writer, sheet_name=safe_sheet_name("Fraud Indicators"), index=False)

        risk_scored_df.to_excel(writer, sheet_name=safe_sheet_name("Risk Scoring (All Records)"), index=False)

        issues_df.to_excel(writer, sheet_name=safe_sheet_name("All Audit Issues"), index=False)
        top20.to_excel(writer, sheet_name=safe_sheet_name("Top 20 Issues"), index=False)

        pd.DataFrame({"Executive Summary": exec_summary_text.split("\n")}).to_excel(
            writer, sheet_name=safe_sheet_name("Executive Summary"), index=False)

    log("Excel workbook written successfully.")
    return path


def export_markdown(exec_summary_text, top20, dq_df, benford):
    path = os.path.join(REPORT_DIR, "Audit_Report.md")
    log(f"Writing Markdown report to {path} ...")

    lines = [exec_summary_text, "\n\n---\n\n## Top 20 Audit Issues Requiring Immediate Attention\n"]
    for _, row in top20.iterrows():
        lines.append(f"### {row['Rank']}. {row['Title']} (Risk Score: {row['Risk Score']:.0f}/100)")
        lines.append(f"**Category:** {row['Category']}\n")
        lines.append(f"**Why it matters:** {row['Description']}\n")
        lines.append(f"**Suggested audit procedure:** {row['Recommendation']}\n")
        if row["Evidence"]:
            lines.append(f"**Evidence:** {row['Evidence']}\n")

    lines.append("\n\n---\n\n## Data Quality Findings\n")
    lines.append(dq_df.to_markdown(index=False))

    if benford:
        lines.append("\n\n---\n\n## Benford's Law Analysis\n")
        lines.append(f"Conformity: **{benford['conformity']}** (MAD={benford['mad']:.4f}, "
                     f"Chi-sq={benford['chi_sq']:.1f})\n")
        lines.append(benford["table"].to_markdown(index=False))

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    log("Markdown report written successfully.")
    return path


# =====================================================================
# MAIN
# =====================================================================
def main():
    ensure_dirs()
    log("=" * 70)
    log("AUDIT ANALYTICS - OPEN ADVANCES TO SUPPLIERS/STAFF")
    log("=" * 70)

    log("STEP 1/13: Loading data")
    df, filetype_info = load_data(INPUT_PATH)

    log("STEP 2/13: Mapping columns")
    colmap = map_columns(df)

    log("STEP 3/13: Cleaning data")
    cleaned_df = clean_data(df, colmap)

    log("STEP 4/13: Data quality assessment")
    dq_df = data_quality_report(cleaned_df, colmap)

    log("STEP 5/13: Descriptive statistics & outliers")
    desc_stats = descriptive_stats(cleaned_df, colmap)
    outlier_flags = detect_outliers(cleaned_df, colmap)

    log("STEP 6/13: Supplier analysis & clustering")
    supplier_summary = supplier_analysis(cleaned_df, colmap)
    supplier_summary = cluster_analysis(supplier_summary)

    log("STEP 7/13: Aging analysis")
    aging_summary, _ = aging_analysis(cleaned_df, colmap)

    log("STEP 8/13: High-risk transaction scan & trend analysis")
    hr_flags = high_risk_transactions(cleaned_df, colmap)
    trends = trend_analysis(cleaned_df, colmap)
    currency_analysis(cleaned_df, colmap)

    log("STEP 9/13: Business rule tests, Benford's Law, fraud indicators")
    rule_df = business_rule_tests(cleaned_df, colmap)
    benford = benford_analysis(cleaned_df, colmap)
    fraud_flags = fraud_indicators(cleaned_df, colmap)

    log("STEP 10/13: Correlation analysis")
    corr = correlation_analysis(cleaned_df, colmap)

    log("STEP 11/13: Visualizations")
    make_visualizations(cleaned_df, colmap, supplier_summary, aging_summary, trends, corr, benford)

    log("STEP 12/13: Risk scoring & Top 20 issues")
    risk_scored_df = risk_scoring(cleaned_df, colmap, outlier_flags, hr_flags, fraud_flags)
    add_transaction_level_issues(risk_scored_df, colmap)
    issues_df, top20 = build_top20()
    exec_summary_text = executive_summary(cleaned_df, colmap, dq_df, supplier_summary,
                                           aging_summary, top20, benford)

    log("STEP 13/13: Exporting deliverables")
    cleaned_df.to_csv(os.path.join(DATA_DIR, "cleaned_data.csv"), index=False)
    cleaned_df.to_excel(os.path.join(DATA_DIR, "cleaned_data.xlsx"), index=False)
    excel_path = export_excel(df, cleaned_df, dq_df, desc_stats, supplier_summary, aging_summary,
                               hr_flags, outlier_flags, fraud_flags, risk_scored_df, rule_df,
                               benford, issues_df, top20, exec_summary_text)
    md_path = export_markdown(exec_summary_text, top20, dq_df, benford)

    log("=" * 70)
    log("ANALYSIS COMPLETE")
    log(f"Excel workbook : {excel_path}")
    log(f"Markdown report: {md_path}")
    log(f"Charts folder  : {CHART_DIR}")
    log(f"Cleaned data   : {DATA_DIR}")
    log("=" * 70)

    print("\n\n========== TOP 20 AUDIT ISSUES REQUIRING IMMEDIATE ATTENTION ==========\n")
    for _, row in top20.iterrows():
        print(f"{row['Rank']:2d}. [{row['Risk Score']:.0f}] {row['Title']}")
        print(f"     -> {row['Recommendation']}")
    print()


if __name__ == "__main__":
    main()
