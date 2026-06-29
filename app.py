"""
Streamlit App - Supplier Advances Audit Risk Analyzer
=======================================================
A no-code interface for auditors. Reuses the existing rule-based audit
pipeline (audit_analysis.py via audit_core.py) and layers a simple,
explainable ML risk model (ml_model.py) on top of it.

Run with:  streamlit run app.py

The UI exposes exactly four actions:
  1. Upload File
  2. Analyze
  3. Download Results
  4. Retrain Model
"""

from io import BytesIO

import pandas as pd
import streamlit as st
import plotly.express as px

import audit_core
import ml_model

st.set_page_config(page_title="Supplier Advances - Audit Risk Analyzer", layout="wide")
st.title("Supplier Advances Audit Risk Analyzer")
st.caption("Upload an open-advances extract, score every record for audit risk, and download the "
           "results - no coding required.")

# Persist results across Streamlit reruns within a session.
for key in ("bundle", "results", "model_bundle", "model_comparison"):
    st.session_state.setdefault(key, None)

existing_model = ml_model.load_model()
with st.expander("Model status", expanded=False):
    if existing_model is None:
        st.write("No trained model exists yet. One will be trained automatically the first time "
                 "you click **Analyze**, using the built-in rule-based risk engine as a starting "
                 "point.")
    else:
        st.write(f"Active model: **{existing_model['model_name']}**, "
                 f"trained {existing_model['trained_at']}, "
                 f"source: {existing_model['meta'].get('source', 'unknown')}, "
                 f"training samples: {existing_model['meta'].get('n_samples', 'n/a')}")

# =====================================================================
# 1. UPLOAD FILE
# =====================================================================
st.header("1. Upload File")
uploaded_file = st.file_uploader("Upload the open-advances Excel/CSV file", type=["xlsx", "xls", "csv"])

# =====================================================================
# 2. ANALYZE
# =====================================================================
st.header("2. Analyze")
analyze_clicked = st.button("Analyze", type="primary", disabled=uploaded_file is None)

if analyze_clicked:
    with st.spinner("Running audit analysis (cleaning, aging, outliers, fraud indicators, Benford)..."):
        bundle = audit_core.run_pipeline(uploaded_file)
        feat, rule_score = audit_core.build_feature_table(bundle)

    model_bundle = ml_model.load_model()
    if model_bundle is None:
        st.info("No trained model found. Training an initial model now using the existing "
                "rule-based risk engine's scores as starting labels (Low/Medium/High)...")
        y = rule_score.apply(audit_core.risk_score_to_level)
        best_model, best_name, comparison, _ = ml_model.train_and_select_best(feat, y)
        ml_model.save_model(best_model, best_name,
                             {"source": "rule-based bootstrap", "n_samples": len(y)})
        model_bundle = ml_model.load_model()
        st.session_state["model_comparison"] = comparison
        st.success(f"Initial model trained and saved as risk_model.pkl (best model: {best_name}).")

    preds = ml_model.predict(model_bundle, feat)

    results = bundle["df"].copy()
    results["rule_risk_score"] = rule_score.round(1)
    results["ml_risk_score"] = preds["risk_score"]
    results["ml_risk_level"] = preds["risk_level"]
    results["top_reasons"] = preds["top_reasons"].apply(lambda r: " | ".join(r))

    st.session_state["bundle"] = bundle
    st.session_state["results"] = results
    st.session_state["model_bundle"] = model_bundle
    st.success(f"Analysis complete: {len(results):,} records scored.")

# =====================================================================
# RESULTS DASHBOARD (shown once an analysis has run)
# =====================================================================
if st.session_state["results"] is not None:
    results = st.session_state["results"]
    bundle = st.session_state["bundle"]
    colmap = bundle["colmap"]
    name_col = colmap.get("supplier_name")
    amt_col = colmap.get("amount")
    age_col = colmap.get("age_days")

    st.header("Results Dashboard")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Records analyzed", f"{len(results):,}")
    if amt_col:
        c2.metric("Total outstanding", f"{results[amt_col].sum():,.0f}")
    n_high = int((results["ml_risk_level"] == "High").sum())
    c3.metric("High-risk records", f"{n_high:,}", f"{n_high/len(results):.1%}")
    c4.metric("Model in use", st.session_state["model_bundle"]["model_name"])

    col_a, col_b = st.columns(2)
    with col_a:
        level_counts = results["ml_risk_level"].value_counts().reindex(
            ["Low", "Medium", "High"]).fillna(0).reset_index()
        level_counts.columns = ["Risk Level", "Count"]
        fig = px.bar(level_counts, x="Risk Level", y="Count", color="Risk Level",
                     color_discrete_map={"Low": "#7a9e7e", "Medium": "#e2a13a", "High": "#c0392b"},
                     title="Risk Level Distribution")
        st.plotly_chart(fig, use_container_width=True)
    with col_b:
        fig2 = px.histogram(results, x="ml_risk_score", nbins=30, title="ML Risk Score Distribution")
        st.plotly_chart(fig2, use_container_width=True)

    if amt_col and age_col:
        fig3 = px.scatter(results, x=age_col, y=amt_col, color="ml_risk_level",
                           color_discrete_map={"Low": "#7a9e7e", "Medium": "#e2a13a", "High": "#c0392b"},
                           hover_data=[name_col] if name_col else None,
                           title="Age vs. Amount by Risk Level")
        st.plotly_chart(fig3, use_container_width=True)

    if name_col and amt_col:
        top_risk = results.sort_values("ml_risk_score", ascending=False).head(15)
        fig4 = px.bar(top_risk, x=amt_col, y=name_col, orientation="h",
                      color="ml_risk_score", color_continuous_scale="Reds",
                      title="Top 15 Highest-Risk Advances")
        fig4.update_yaxes(autorange="reversed")
        st.plotly_chart(fig4, use_container_width=True)

    st.subheader("High-Risk Transactions")
    high_risk_df = results[results["ml_risk_level"] == "High"].sort_values(
        "ml_risk_score", ascending=False)
    display_cols = [c for c in [name_col, amt_col, age_col, "ml_risk_score", "ml_risk_level",
                                  "top_reasons"] if c]
    st.dataframe(high_risk_df[display_cols], use_container_width=True, height=350)

    with st.expander("Executive Summary"):
        st.markdown(bundle["exec_summary_text"])

    if st.session_state["model_comparison"] is not None:
        with st.expander("Model comparison (most recent training run)"):
            st.dataframe(st.session_state["model_comparison"], use_container_width=True)

# =====================================================================
# 3. DOWNLOAD RESULTS
# =====================================================================
st.header("3. Download Results")


def build_excel_bytes():
    bundle = st.session_state["bundle"]
    results = st.session_state["results"]
    buf = BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        results.to_excel(writer, sheet_name="Cleaned Data + ML Scores", index=False)

        high_risk = results[results["ml_risk_level"] == "High"].sort_values(
            "ml_risk_score", ascending=False)
        high_risk.to_excel(writer, sheet_name="High-Risk Transactions", index=False)

        if bundle.get("aging_summary") is not None and not bundle["aging_summary"].empty:
            bundle["aging_summary"].to_excel(writer, sheet_name="Aging Summary", index=False)

        if bundle.get("supplier_summary") is not None and not bundle["supplier_summary"].empty:
            bundle["supplier_summary"].head(100).to_excel(
                writer, sheet_name="Supplier Summary (Top 100)", index=False)

        level_counts = results["ml_risk_level"].value_counts().reset_index()
        level_counts.columns = ["Risk Level", "Count"]
        level_counts.to_excel(writer, sheet_name="Summary Dashboard", index=False)

        pd.DataFrame({"Executive Summary": bundle["exec_summary_text"].split("\n")}).to_excel(
            writer, sheet_name="Executive Summary", index=False)
    buf.seek(0)
    return buf


results_ready = st.session_state["results"] is not None
excel_buffer = build_excel_bytes() if results_ready else BytesIO()
st.download_button(
    "Download Results",
    data=excel_buffer,
    file_name="Audit_ML_Results.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    disabled=not results_ready,
)

# =====================================================================
# 4. RETRAIN MODEL
# =====================================================================
st.header("4. Retrain Model")
with st.expander("Retrain with auditor-confirmed labels"):
    st.write(
        "Upload a file with the same columns as your advances extract, plus one extra column of "
        "confirmed risk labels (column name containing e.g. 'Risk Level', accepted values: "
        "Low / Medium / High, or 0/1/2). This replaces the current model with one trained on real "
        "audit conclusions instead of the rule-based starting point."
    )
    labeled_file = st.file_uploader("Upload labeled training file", type=["xlsx", "xls", "csv"],
                                     key="labeled_file")
    retrain_clicked = st.button("Retrain Model", disabled=labeled_file is None)

    if retrain_clicked:
        with st.spinner("Retraining on confirmed labels..."):
            bundle2 = audit_core.run_pipeline(labeled_file)
            label_col = audit_core.find_label_column(bundle2["raw_df"])

            if label_col is None:
                st.error("Could not find a label column. Expected a column named e.g. "
                         "'Risk Level', 'Confirmed Risk Level', or 'Label'.")
            else:
                feat2, _ = audit_core.build_feature_table(bundle2)
                y2 = bundle2["raw_df"][label_col].apply(audit_core.normalize_label_value)
                mask = y2.notna()
                feat2, y2 = feat2[mask], y2[mask]

                if y2.nunique() < 2:
                    st.error("Need at least two distinct risk levels in the labeled data to retrain.")
                else:
                    best_model, best_name, comparison, _ = ml_model.train_and_select_best(feat2, y2)
                    ml_model.save_model(
                        best_model, best_name,
                        {"source": "auditor-confirmed", "n_samples": int(mask.sum()),
                         "label_column": label_col})
                    st.session_state["model_comparison"] = comparison
                    st.session_state["model_bundle"] = ml_model.load_model()
                    st.success(f"Model retrained on {int(mask.sum())} confirmed labels. "
                               f"Best model: {best_name}. Saved to risk_model.pkl.")
                    st.dataframe(comparison, use_container_width=True)
