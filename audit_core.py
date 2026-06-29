"""
Audit Core - thin reusable wrapper around audit_analysis.py
=============================================================
This module does NOT redo the analysis logic. It imports the existing,
already-tested functions from audit_analysis.py (column mapping, cleaning,
outlier detection, aging, business rules, Benford, fraud indicators, the
rule-based 0-100 risk_scoring engine, etc.) and arranges them into:

  1. run_pipeline()      - runs the full rule-based audit analysis on a
                            freshly uploaded file (reused as-is).
  2. build_feature_table() - turns the pipeline output into a flat numeric
                            feature table the ML model can consume.

Importing audit_analysis.py has no side effects: its functions only run
when explicitly called (the file's batch CLI logic lives behind
`if __name__ == "__main__":`).
"""

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import audit_analysis as core  # noqa: E402  (re-used, not reimplemented)

# Feature columns fed to the ML model - kept in one place so ml_model.py
# and app.py agree on the contract.
FEATURE_COLUMNS = [
    "amount",
    "age_days",
    "supplier_frequency",
    "open_advances_count",
    "duplicate_indicator",
    "round_amount_indicator",
    "weekend_transaction",
    "outlier_score",
    "benford_first_digit",
    "aging_bucket_code",
]

# Human-readable explanation templates for the ML "Top 3 reasons" output.
# Kept alongside the feature list they describe.
def _reason_amount(v):
    return f"Advance amount is large ({v:,.0f})"


def _reason_age(v):
    return f"Outstanding for {int(v):,} days (~{v / 365:.1f} years)"


def _reason_supplier_freq(v):
    return f"Supplier/payee has {int(v)} open advances"


def _reason_duplicate(v):
    return "Matches another advance on payee, amount and date (possible duplicate)"


def _reason_round(v):
    return "Amount is an exact round number (estimated rather than invoiced?)"


def _reason_weekend(v):
    return "Document is dated on a weekend"


def _reason_outlier(v):
    return "Amount is a statistical outlier vs. the rest of the population"


def _reason_benford(v):
    return f"Leading digit ({int(v)}) of the amount is unusual for this digit position"


def _reason_aging_bucket(v):
    idx = int(v)
    if 0 <= idx < len(core.AGING_LABELS):
        return f"Falls in an advanced aging bucket ('{core.AGING_LABELS[idx]}')"
    return "Falls in an advanced aging bucket"


REASON_TEMPLATES = {
    "amount": _reason_amount,
    "age_days": _reason_age,
    "supplier_frequency": _reason_supplier_freq,
    "open_advances_count": _reason_supplier_freq,
    "duplicate_indicator": _reason_duplicate,
    "round_amount_indicator": _reason_round,
    "weekend_transaction": _reason_weekend,
    "outlier_score": _reason_outlier,
    "benford_first_digit": _reason_benford,
    "aging_bucket_code": _reason_aging_bucket,
}

# Boolean/indicator-style features only deserve a "reason" when they fire (==1).
BOOLEAN_FEATURES = {"duplicate_indicator", "round_amount_indicator", "weekend_transaction"}

# Sparse/ordinal features need an explicit minimum raw value before they're
# allowed to surface as a "reason" - otherwise a value of 0 (e.g. "not an
# outlier", "0-30 day bucket") could still get phrased as if it were a risk
# driver once weighted by feature importance.
MIN_VALUE_TO_FIRE = {
    "outlier_score": 1,        # must be flagged by at least one outlier method
    "benford_first_digit": 1,  # 0 means amount was 0/invalid
    "aging_bucket_code": 4,    # only call out buckets from "181-365" upward
}


def first_significant_digit(value):
    """Return the leading digit of abs(value), or NaN if value is 0/invalid."""
    v = abs(value)
    if not np.isfinite(v) or v == 0:
        return np.nan
    s = f"{v:.10f}".lstrip("0").lstrip(".").lstrip("0")
    return int(s[0]) if s else np.nan


def run_pipeline(file_path_or_buffer):
    """Run the existing rule-based audit analysis end-to-end on one file.

    Returns a dict bundle with the cleaned dataframe, column map, and every
    intermediate result the existing audit_analysis.py functions produce.
    This is the single integration point reused by both the ML feature
    builder and the Streamlit reporting views.
    """
    core.ISSUES.clear()  # avoid unbounded growth across repeated Streamlit runs

    df, filetype_info = core.load_data(file_path_or_buffer)
    colmap = core.map_columns(df)
    cleaned = core.clean_data(df, colmap)

    outlier_flags = core.detect_outliers(cleaned, colmap)
    hr_flags = core.high_risk_transactions(cleaned, colmap)
    fraud_flags = core.fraud_indicators(cleaned, colmap)
    supplier_summary = core.supplier_analysis(cleaned, colmap)
    aging_summary, _ = core.aging_analysis(cleaned, colmap)
    benford = core.benford_analysis(cleaned, colmap)
    rule_df = core.business_rule_tests(cleaned, colmap)
    dq_df = core.data_quality_report(cleaned, colmap)
    risk_scored = core.risk_scoring(cleaned, colmap, outlier_flags, hr_flags, fraud_flags)

    issues_df, top20 = core.build_top20()
    exec_summary_text = core.executive_summary(cleaned, colmap, dq_df, supplier_summary,
                                                 aging_summary, top20, benford)

    return {
        "raw_df": df,
        "df": cleaned,
        "colmap": colmap,
        "outlier_flags": outlier_flags,
        "hr_flags": hr_flags,
        "fraud_flags": fraud_flags,
        "supplier_summary": supplier_summary,
        "aging_summary": aging_summary,
        "benford": benford,
        "rule_df": rule_df,
        "dq_df": dq_df,
        "risk_scored": risk_scored,
        "issues_df": issues_df,
        "top20": top20,
        "exec_summary_text": exec_summary_text,
    }


def build_feature_table(bundle):
    """Flatten the pipeline bundle into the numeric feature table for ML.

    Also returns `rule_risk_score` (0-100, from the existing rule-based
    engine) which is used as the weak-supervision label when no
    auditor-confirmed labels are available yet.
    """
    df = bundle["df"]
    colmap = bundle["colmap"]
    outlier_flags = bundle["outlier_flags"]
    hr_flags = bundle["hr_flags"]
    fraud_flags = bundle["fraud_flags"]

    amt_col = colmap.get("amount")
    age_col = colmap.get("age_days")
    id_col = colmap.get("supplier_id")

    feat = pd.DataFrame(index=df.index)
    feat["amount"] = df[amt_col].abs() if amt_col else 0.0
    feat["age_days"] = df[age_col] if age_col else 0.0

    if id_col:
        freq = df[id_col].map(df[id_col].value_counts())
    else:
        freq = pd.Series(1, index=df.index)
    feat["supplier_frequency"] = freq
    feat["open_advances_count"] = freq  # same concept in this open-items extract

    feat["duplicate_indicator"] = fraud_flags.get(
        "dup_payee_amount_date", pd.Series(False, index=df.index)).astype(int)
    feat["round_amount_indicator"] = hr_flags.get(
        "flag_round_number", pd.Series(False, index=df.index)).astype(int)
    feat["weekend_transaction"] = hr_flags.get(
        "flag_weekend", pd.Series(False, index=df.index)).astype(int)
    feat["outlier_score"] = outlier_flags.get(
        "outlier_vote_count", pd.Series(0, index=df.index))

    feat["benford_first_digit"] = feat["amount"].apply(first_significant_digit).fillna(0)

    if age_col:
        bucket = pd.cut(df[age_col], bins=core.AGING_BINS, labels=False)
        feat["aging_bucket_code"] = bucket.fillna(0)
    else:
        feat["aging_bucket_code"] = 0

    feat = feat.fillna(0)

    rule_risk_score = bundle["risk_scored"]["risk_score"].reindex(df.index)

    return feat[FEATURE_COLUMNS], rule_risk_score


def risk_score_to_level(score):
    """Map a 0-100 numeric risk score to a Low/Medium/High label."""
    if score >= 70:
        return "High"
    elif score >= 40:
        return "Medium"
    return "Low"


def find_label_column(df):
    """Look for a column the user might have used to provide confirmed labels."""
    candidates = ["confirmed risk level", "actual risk level", "risk level",
                  "label", "audit outcome", "confirmed label"]
    for col in df.columns:
        low = col.lower().strip()
        if any(c == low or c in low for c in candidates):
            return col
    return None


def normalize_label_value(v):
    """Accept Low/Medium/High text or 0/1/2 numeric labels and standardize."""
    if isinstance(v, str):
        v = v.strip().title()
        if v in ("Low", "Medium", "High"):
            return v
    try:
        n = float(v)
        if n <= 0.5:
            return "Low"
        elif n <= 1.5:
            return "Medium"
        return "High"
    except (TypeError, ValueError):
        return None
