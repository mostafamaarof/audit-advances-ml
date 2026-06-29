"""
ML Model - trains, compares, persists, and applies the advance risk model.

Random Forest is the primary model (accurate, explainable via feature
importances, handles mixed-scale tabular features well). Logistic
Regression and XGBoost (if installed) are trained alongside it and the
best performer on a held-out test split is kept automatically.
"""

import os
from datetime import datetime

import numpy as np
import pandas as pd
import joblib

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score, classification_report

try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

from audit_core import FEATURE_COLUMNS, REASON_TEMPLATES, BOOLEAN_FEATURES, MIN_VALUE_TO_FIRE

MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "risk_model.pkl")
LEVEL_ORDER = ["Low", "Medium", "High"]
LEVEL_ANCHOR_SCORE = {"Low": 12, "Medium": 50, "High": 90}
RANDOM_STATE = 42


def _candidate_models():
    models = {
        "RandomForest": RandomForestClassifier(
            n_estimators=300, max_depth=12, min_samples_leaf=3,
            class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1),
        "LogisticRegression": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=2000, class_weight="balanced",
                                        random_state=RANDOM_STATE)),
        ]),
    }
    if XGBOOST_AVAILABLE:
        models["XGBoost"] = XGBClassifier(
            n_estimators=200, max_depth=5, learning_rate=0.1,
            eval_metric="mlogloss", random_state=RANDOM_STATE)
    return models


def train_and_select_best(X, y, test_size=0.25):
    """Train every candidate model, evaluate on a held-out split, keep the best.

    Returns (best_model, best_name, comparison_table, classification_report_text).
    """
    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=RANDOM_STATE)

    rows = []
    fitted = {}
    for name, model in _candidate_models().items():
        # XGBoost needs integer-encoded labels
        if name == "XGBoost":
            mapping = {lvl: i for i, lvl in enumerate(LEVEL_ORDER)}
            inv_mapping = {i: lvl for lvl, i in mapping.items()}
            model.fit(Xtr, ytr.map(mapping))
            preds_int = model.predict(Xte)
            preds = pd.Series(preds_int).map(inv_mapping).values
        else:
            model.fit(Xtr, ytr)
            preds = model.predict(Xte)

        f1 = f1_score(yte, preds, average="weighted", zero_division=0)
        acc = accuracy_score(yte, preds)
        rows.append({"Model": name, "Accuracy": round(acc, 3), "F1 (weighted)": round(f1, 3)})
        fitted[name] = (model, f1)

    comparison = pd.DataFrame(rows).sort_values("F1 (weighted)", ascending=False).reset_index(drop=True)
    best_name = comparison.iloc[0]["Model"]
    best_model = fitted[best_name][0]

    # Refit the winning model on the FULL dataset before deployment.
    if best_name == "XGBoost":
        mapping = {lvl: i for i, lvl in enumerate(LEVEL_ORDER)}
        best_model.fit(X, y.map(mapping))
    else:
        best_model.fit(X, y)

    report_text = f"Best model: {best_name}\n\n{comparison.to_string(index=False)}"
    return best_model, best_name, comparison, report_text


def save_model(model, model_name, training_meta, path=MODEL_PATH):
    bundle = {
        "model": model,
        "model_name": model_name,
        "feature_columns": FEATURE_COLUMNS,
        "level_order": LEVEL_ORDER,
        "trained_at": datetime.now().isoformat(timespec="seconds"),
        "meta": training_meta,
    }
    joblib.dump(bundle, path)
    return path


def load_model(path=MODEL_PATH):
    if os.path.exists(path):
        return joblib.load(path)
    return None


def _predict_proba_by_level(bundle, X):
    """Return a (n_rows, 3) probability array ordered Low/Medium/High, regardless
    of which classes the underlying model happened to see during training."""
    model = bundle["model"]
    proba = model.predict_proba(X)
    classes = list(model.classes_)

    if bundle["model_name"] == "XGBoost":
        # XGBoost was trained on integer-coded classes 0/1/2
        classes = [LEVEL_ORDER[int(c)] for c in classes]

    out = np.zeros((len(X), 3))
    for i, lvl in enumerate(LEVEL_ORDER):
        if lvl in classes:
            out[:, i] = proba[:, classes.index(lvl)]
    return out


def predict(bundle, feature_df):
    """Score a feature table with a loaded model bundle.

    Returns a DataFrame with risk_score (0-100), risk_level, and reasons
    (list of up to 3 human-readable strings) per row.
    """
    X = feature_df[bundle["feature_columns"]].fillna(0)
    proba_by_level = _predict_proba_by_level(bundle, X)

    anchor = np.array([LEVEL_ANCHOR_SCORE[lvl] for lvl in LEVEL_ORDER])
    risk_score = (proba_by_level * anchor).sum(axis=1)
    risk_score = np.clip(risk_score, 0, 100)

    pred_idx = proba_by_level.argmax(axis=1)
    risk_level = [LEVEL_ORDER[i] for i in pred_idx]

    importances = _get_feature_importances(bundle["model"], bundle["feature_columns"])
    reasons = _generate_reasons(X, importances)

    return pd.DataFrame({
        "risk_score": np.round(risk_score, 1),
        "risk_level": risk_level,
        "top_reasons": reasons,
    }, index=feature_df.index)


def _get_feature_importances(model, feature_columns):
    inner = model
    if hasattr(model, "named_steps"):  # LogisticRegression pipeline
        inner = model.named_steps["clf"]
        coefs = np.abs(inner.coef_).mean(axis=0)
        return pd.Series(coefs / (coefs.sum() or 1), index=feature_columns)
    if hasattr(inner, "feature_importances_"):
        imp = inner.feature_importances_
        return pd.Series(imp / (imp.sum() or 1), index=feature_columns)
    return pd.Series(1.0 / len(feature_columns), index=feature_columns)


def _generate_reasons(X, importances, top_n=3):
    """Pick the top contributing features per row and phrase them in plain English.

    Contribution = global feature importance x the row's min-max normalized
    value for continuous features, or importance x 0/1 for boolean indicator
    features. Min-max (not percentile rank) is used deliberately: for sparse
    columns like outlier_score, percentile rank ties every zero-value row to
    a ~50th-percentile "contribution", which would wrongly surface "outlier"
    as a reason even when the row was never flagged as one.
    This stays simple and fast (no SHAP dependency) while still tying the
    explanation to what the trained model actually weighs most.
    """
    mins, maxs = X.min(), X.max()
    ranges = (maxs - mins).replace(0, 1)
    normed = (X - mins) / ranges

    contributions = pd.DataFrame(index=X.index)
    for col in X.columns:
        if col in BOOLEAN_FEATURES:
            contributions[col] = importances[col] * X[col]
        else:
            contributions[col] = importances[col] * normed[col]

    reasons_list = []
    for idx in X.index:
        row_contrib = contributions.loc[idx].sort_values(ascending=False)
        row_reasons = []
        for col in row_contrib.index:
            if row_contrib[col] <= 0:
                continue
            raw_value = X.loc[idx, col]
            if col in BOOLEAN_FEATURES and raw_value == 0:
                continue
            if col in MIN_VALUE_TO_FIRE and raw_value < MIN_VALUE_TO_FIRE[col]:
                continue
            template = REASON_TEMPLATES.get(col)
            if template:
                row_reasons.append(template(raw_value))
            if len(row_reasons) == top_n:
                break
        reasons_list.append(row_reasons if row_reasons else ["No single dominant risk driver identified"])
    return reasons_list
