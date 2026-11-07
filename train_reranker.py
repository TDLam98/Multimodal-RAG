"""
Train the reranker and test whether it improves retrieval.

Run (from the project folder, after rerank_data.py):
    python train_reranker.py
    python train_reranker.py --folds 5

Models
  logreg   Logistic regression on standardized features, class-balanced, with the
           regularization strength C chosen by an inner cross-validation.
  gbt      XGBoost if it is installed (pip install xgboost); otherwise scikit-learn's
           HistGradientBoostingClassifier. The output says which one ran.

Evaluation: grouped cross-validation BY COMPANY. Every question is scored by a model that never
saw its company during training. Within each configuration's top-K list, chunks are reordered by
the model's score, and recall@k and MRR are compared with the original embedding order.

Reranking only reorders the top K. A question whose right chunk never reached the top K is a
miss for every method, so recall@K is the ceiling for all of them.

Outputs
  data/results/rerank_summary.csv   recall@k and MRR per configuration and method, with fold spread
  data/results/rerank_details.csv   the first relevant rank for every question, configuration and method
  data/results/rerank_classifier.csv  ROC AUC and average precision of each model (out of fold)
  data/rerank/oof_scores.csv        held-out score of every pair; generate.py uses these to rerank
  data/models/                      both models refit on all the data, for reranking NEW questions only

Requires: pip install scikit-learn pandas joblib   (xgboost optional)
"""
import argparse
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import embed_index as ei
from rerank_data import FEATURES, OUT as PAIRS

try:
    from xgboost import XGBClassifier
    GBT_NAME = "xgboost"
except ImportError:
    from sklearn.ensemble import HistGradientBoostingClassifier
    GBT_NAME = "histgb"

KS = [1, 3, 5, 10]
C_GRID = [0.01, 0.1, 1.0, 10.0]
RESULTS = ei.DATA_DIR / "results"
MODELS = ei.DATA_DIR / "models"
SEED = 42


# ---------- models ----------
def make_logreg(C):
    return make_pipeline(StandardScaler(),
                         LogisticRegression(C=C, class_weight="balanced", max_iter=2000))


def make_gbt(y):
    if GBT_NAME == "xgboost":
        pos = max(int(y.sum()), 1)
        return XGBClassifier(n_estimators=300, max_depth=3, learning_rate=0.05, subsample=0.8,
                             colsample_bytree=0.8, scale_pos_weight=(len(y) - pos) / pos,
                             eval_metric="logloss", random_state=SEED)
    return HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05, max_iter=300,
                                          class_weight="balanced", random_state=SEED)


def choose_C(X, y, groups):
    """Pick C by inner grouped cross-validation on the training fold (mean average precision)."""
    n_inner = min(3, len(np.unique(groups)))
    if n_inner < 2 or y.sum() == 0:
        return 1.0
    best_C, best = 1.0, -1.0
    for C in C_GRID:
        scores = []
        for tr, va in GroupKFold(n_splits=n_inner).split(X, y, groups):
            if y[tr].min() == y[tr].max() or y[va].sum() == 0:
                continue
            m = make_logreg(C).fit(X[tr], y[tr])
            scores.append(average_precision_score(y[va], m.predict_proba(X[va])[:, 1]))
        if scores and np.mean(scores) > best:
            best_C, best = C, float(np.mean(scores))
    return best_C


# ---------- ranking metrics ----------
def first_rank(df, order_col, ascending):
    """1-based position of the first relevant chunk when df is sorted by order_col."""
    labels = df.sort_values(order_col, ascending=ascending, kind="stable")["label"].to_numpy()
    hits = np.flatnonzero(labels)
    return int(hits[0]) + 1 if len(hits) else None


def rank_rows(test, fold, methods):
    rows = []
    for cfg in ei.CONFIGS:
        col = f"rank_{cfg}"
        sub = test[test[col].notna()]
        for qid, g in sub.groupby("qid"):
            base = {"qid": qid, "group": g["group"].iat[0], "q_label": g["q_label"].iat[0],
                    "fold": fold, "config": cfg}
            rows.append({**base, "method": "embedding", "first_rank": first_rank(g, col, True)})
            for m in methods:
                rows.append({**base, "method": m, "first_rank": first_rank(g, f"score_{m}", False)})
    return rows


def summarize(details):
    out = []
    labels = ["all"] + sorted(details["q_label"].unique())
    for lab in labels:
        d = details if lab == "all" else details[details["q_label"] == lab]
        for (cfg, method), g in d.groupby(["config", "method"], sort=False):
            r = g["first_rank"]
            row = {"group": lab, "config": cfg, "method": method, "n": len(g)}
            for k in KS:
                row[f"R@{k}"] = float(((r <= k) & r.notna()).mean())
            rr = (1.0 / r).fillna(0.0)
            row["MRR"] = float(rr.mean())
            per_fold = rr.groupby(g["fold"]).mean()
            row["MRR_fold_sd"] = float(per_fold.std(ddof=0)) if len(per_fold) > 1 else 0.0
            out.append(row)
    return pd.DataFrame(out)


# ---------- main ----------
def run(folds=5):
    if not PAIRS.exists():
        raise SystemExit(f"{PAIRS} not found. Run rerank_data.py first.")
    df = pd.read_csv(PAIRS, keep_default_na=True)
    df["q_label"] = df["q_label"].fillna("unlabeled")
    for cfg in ei.CONFIGS:
        df[f"rank_{cfg}"] = pd.to_numeric(df[f"rank_{cfg}"], errors="coerce")
    X, y = df[FEATURES].to_numpy(dtype=float), df["label"].to_numpy(dtype=int)
    groups = df["group"].to_numpy()
    n_groups = len(np.unique(groups))
    if n_groups < 2:
        raise SystemExit("Need questions from at least 2 companies for grouped cross-validation.")
    if y.sum() == 0:
        raise SystemExit("No relevant pairs at all: check the questions' answers and row labels.")
    n_splits = min(folds, n_groups)
    methods = ["logreg", GBT_NAME]
    print(f"{len(df)} pairs, {df['qid'].nunique()} questions, {n_groups} companies, "
          f"{int(y.sum())} positives. {n_splits}-fold cross-validation grouped by company. "
          f"Gradient boosting: {GBT_NAME}.")

    for m in methods:
        df[f"score_{m}"] = np.nan
    details, chosen_C = [], []
    for fold, (tr, te) in enumerate(GroupKFold(n_splits=n_splits).split(X, y, groups), 1):
        if y[tr].min() == y[tr].max():
            print(f"  fold {fold}: training data has only one class; using embedding scores instead.")
            for m in methods:
                df.loc[df.index[te], f"score_{m}"] = df["sim"].to_numpy()[te]
        else:
            C = choose_C(X[tr], y[tr], groups[tr])
            chosen_C.append(C)
            lr = make_logreg(C).fit(X[tr], y[tr])
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                gb = make_gbt(y[tr]).fit(X[tr], y[tr])
            df.loc[df.index[te], "score_logreg"] = lr.predict_proba(X[te])[:, 1]
            df.loc[df.index[te], f"score_{GBT_NAME}"] = gb.predict_proba(X[te])[:, 1]
        details += rank_rows(df.iloc[te], fold, methods)

    details = pd.DataFrame(details)
    summary = summarize(details)
    clf = []
    for m in methods + ["embedding"]:
        s = df["sim"] if m == "embedding" else df[f"score_{m}"]
        clf.append({"model": m, "ROC_AUC": roc_auc_score(y, s) if 0 < y.sum() < len(y) else np.nan,
                    "avg_precision": average_precision_score(y, s)})
    clf = pd.DataFrame(clf)

    RESULTS.mkdir(parents=True, exist_ok=True)
    summary.to_csv(RESULTS / "rerank_summary.csv", index=False)
    details.to_csv(RESULTS / "rerank_details.csv", index=False)
    clf.to_csv(RESULTS / "rerank_classifier.csv", index=False)
    # Held-out (out-of-fold) scores for every pair, so generate.py can rerank each question with a
    # model that never saw that question's company. Never use the refit-on-all model for that.
    oof_cols = ["qid", "chunk_id"] + [f"score_{m}" for m in methods]
    df[oof_cols].to_csv(PAIRS.parent / "oof_scores.csv", index=False)

    # refit on everything for Phase 4
    MODELS.mkdir(parents=True, exist_ok=True)
    C_all = choose_C(X, y, groups)
    lr_all = make_logreg(C_all).fit(X, y)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        gb_all = make_gbt(y).fit(X, y)
    joblib.dump({"model": lr_all, "features": FEATURES}, MODELS / "reranker_logreg.joblib")
    joblib.dump({"model": gb_all, "features": FEATURES}, MODELS / f"reranker_{GBT_NAME}.joblib")

    # ---------- report ----------
    pd.set_option("display.width", 140)
    print("\nClassifier quality, out of fold (embedding = raw similarity, for reference):")
    print(clf.round(3).to_string(index=False))
    print(f"\nLogistic regression C chosen per fold: {chosen_C}; on all data: {C_all}")
    cols = ["config", "method", "n"] + [f"R@{k}" for k in KS] + ["MRR", "MRR_fold_sd"]
    for lab in summary["group"].unique():
        print(f"\n=== Reranking results: {lab} ===")
        print(summary[summary["group"] == lab][cols].round(3).to_string(index=False))

    coef = lr_all.named_steps["logisticregression"].coef_[0]
    top = sorted(zip(FEATURES, coef), key=lambda t: -abs(t[1]))[:8]
    print("\nLargest logistic regression weights (standardized features):")
    for name, w in top:
        print(f"  {name:<20}{w:+.2f}")
    if hasattr(gb_all, "feature_importances_"):
        imp = sorted(zip(FEATURES, gb_all.feature_importances_), key=lambda t: -t[1])[:8]
        print(f"\nLargest {GBT_NAME} feature importances:")
        for name, v in imp:
            print(f"  {name:<20}{v:.3f}")
    print(f"\nSaved results to {RESULTS} and models to {MODELS}")
    return summary, clf


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Train and evaluate the reranker.")
    ap.add_argument("--folds", type=int, default=5, help="cross-validation folds (default 5)")
    args = ap.parse_args()
    run(args.folds)