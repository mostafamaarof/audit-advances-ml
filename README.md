# Supplier Advances Audit Risk Analyzer

A no-code web app for auditors to upload an open-advances extract and get
back a risk-scored, explainable result set — combining the existing
rule-based audit engine with a simple, retrainable machine learning model.

## What this is

- `audit_analysis.py` — the original rule-based audit script (data quality,
  aging, outliers, Benford's Law, fraud indicators, the 0-100 rule-based
  `risk_scoring` engine, Excel/Markdown export). Unchanged in behavior; it
  still runs standalone with `python audit_analysis.py`.
- `audit_core.py` — thin wrapper that **reuses** the functions above and
  turns their output into a numeric feature table for ML.
- `ml_model.py` — trains/compares Random Forest, Logistic Regression, and
  (if installed) XGBoost; automatically keeps whichever scores best on a
  held-out test split; saves/loads the winner as `risk_model.pkl`; produces
  a 0-100 risk score, a Low/Medium/High level, and the top 3 plain-English
  reasons for each prediction.
- `app.py` — the Streamlit interface auditors actually use. Four actions:
  **Upload File**, **Analyze**, **Download Results**, **Retrain Model**.

## How the ML model is trained

There is no pre-existing "this advance was fraudulent" label in raw SAP
exports, so the app bootstraps itself:

1. The **first time** you click **Analyze** with no `risk_model.pkl`
   present, the app trains an initial model using the existing rule-based
   `risk_scoring` output (age + amount + outlier + fraud-flag composite,
   already in `audit_analysis.py`) as weak-supervision labels
   (score ≥70 → High, ≥40 → Medium, else Low).
2. As real audits conclude, use **Retrain Model** to upload a file
   containing the same columns plus one extra column of auditor-confirmed
   labels (a column named like `Risk Level`, accepted values
   `Low` / `Medium` / `High`, or `0`/`1`/`2`). The app retrains, compares
   Random Forest / Logistic Regression / XGBoost again, and overwrites
   `risk_model.pkl` with the best of the three.

Every analysis after that loads the saved model — no retraining needed
unless you explicitly click **Retrain Model**.

## Features used by the ML model

| Feature | Meaning |
|---|---|
| `amount` | Absolute advance amount |
| `age_days` | Days outstanding |
| `supplier_frequency` / `open_advances_count` | How many open items the same payee has in this file |
| `duplicate_indicator` | Same payee + amount + date as another record |
| `round_amount_indicator` | Amount is an exact multiple of 100/500/1,000/5,000/10,000 |
| `weekend_transaction` | Document date falls on a Saturday/Sunday |
| `outlier_score` | Vote count (0-4) across IQR, Z-score, Isolation Forest, DBSCAN |
| `benford_first_digit` | Leading digit of the amount |
| `aging_bucket_code` | Ordinal aging bucket (0-30 days ... >3 years) |

## Setup

```bash
pip install -r requirements.txt
streamlit run app.py
```

The app opens in your browser. No command-line or Python knowledge is
needed beyond these two lines.

## Using the app

1. **Upload File** — choose your advances extract (`.xlsx`, `.xls`, or `.csv`).
2. **Analyze** — runs the full audit pipeline and ML scoring, shows a
   dashboard (risk distribution, top high-risk advances, age vs. amount
   scatter) and a sortable high-risk transaction table.
3. **Download Results** — exports an Excel workbook: cleaned data with ML
   scores, high-risk transactions, aging/supplier summaries, and the
   executive summary.
4. **Retrain Model** — optional. Upload auditor-confirmed labels to improve
   the model over time; the new model is saved automatically.

## Files produced

- `risk_model.pkl` — the currently active trained model (created on first
  **Analyze**, overwritten by **Retrain Model**).
- Downloaded Excel workbook from the **Download Results** button.

## Notes / limitations

- The initial (bootstrap) model approximates the existing rule-based engine
  — it is a starting point, not independent ground truth. Treat its output
  as a triage aid, and prioritize **Retrain Model** with real confirmed
  outcomes as soon as they're available.
- XGBoost is optional; if not installed, the app still works and compares
  Random Forest vs. Logistic Regression only.
