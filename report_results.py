"""
Final results: 95% confidence intervals, paired comparisons, and charts for the report and slides.

Run (from the project folder, after the pipeline):
    python report_results.py
    python report_results.py --configs html_only,html+naive_pdf     # choose configurations
    python report_results.py --boot 5000                            # more bootstrap samples

It reads what the pipeline already wrote (it never re-runs anything):
    data/results/rerank_details.csv       retrieval: first relevant rank per question (Phase 3)
    data/generation/answers_scored.csv    answers: correct or not per question (Phase 4), if present
    data/results/retrieval_details.csv    answer coverage per source, if present

Statistics
  Every number gets a 95% bootstrap confidence interval: questions are resampled with replacement
  2,000 times and the 2.5th and 97.5th percentiles are reported. Comparisons are PAIRED: the same
  resampled questions are used for both sides, so "reranker minus embedding" or "naive PDF minus
  HTML only" is measured on identical questions. A difference whose interval excludes 0 is marked
  as clearly different. Caveat for the report: questions from the same company are not independent,
  so with few companies these intervals are, if anything, too narrow.

Configurations whose results are identical to html_only (because their PDF chunks are not in the
index yet) are left out automatically, with a note; --configs overrides this.

Outputs in data/results/report/:
    results_with_ci.csv       every metric with its interval
    paired_differences.csv    every comparison with its interval
    summary.md                tables ready to paste into the report
    retrieval_mrr.png, answer_accuracy.png, answer_coverage.png

Requires: pip install pandas matplotlib numpy
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import embed_index as ei

RESULTS = ei.DATA_DIR / "results"
SCORED = ei.DATA_DIR / "generation" / "answers_scored.csv"
OUT = RESULTS / "report"
SEED = 42

# Reference palette, light mode (validated: first three slots pass all-pairs CVD checks).
COLORS = {"embedding": "#2a78d6", "logreg": "#eb6834", "gbt": "#1baf7a"}
MARKERS = {"embedding": "o", "logreg": "s", "gbt": "D"}     # shape = second cue besides color
INK, INK2, MUTED, GRID, AXIS, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"
METHOD_NAMES = {"embedding": "Embedding only", "logreg": "+ Logistic regression reranker",
                "xgboost": "+ XGBoost reranker", "histgb": "+ Gradient-boosting reranker"}


# ---------- loading ----------
def load_retrieval():
    path = RESULTS / "rerank_details.csv"
    if not path.exists():
        raise SystemExit(f"{path} not found. Run train_reranker.py first.")
    d = pd.read_csv(path)
    d["rr"] = (1.0 / d["first_rank"]).fillna(0.0)
    d["r1"] = (d["first_rank"] == 1).astype(float)
    return d


def load_answers():
    if not SCORED.exists():
        print(f"No {SCORED.name} yet (run generate.py and evaluate_answers.py); answer results skipped.")
        return None
    return pd.read_csv(SCORED)


def choose_configs(ret, requested):
    if requested:
        return [c.strip() for c in requested.split(",")]
    keep, dropped = [], []
    base = ret[ret["config"] == "html_only"].set_index(["qid", "method"])["first_rank"]
    for cfg in ret["config"].unique():
        if cfg == "html_only":
            keep.append(cfg)
            continue
        other = ret[ret["config"] == cfg].set_index(["qid", "method"])["first_rank"]
        same = other.reindex(base.index)
        if same.fillna(-1).equals(base.fillna(-1)):
            dropped.append(cfg)
        else:
            keep.append(cfg)
    if dropped:
        print(f"Left out (identical to html_only, so their PDF chunks are probably not indexed yet): {dropped}")
    return keep


# ---------- bootstrap ----------
class Boot:
    def __init__(self, qids, n, seed=SEED):
        self.qids = np.array(sorted(qids))
        rng = np.random.default_rng(seed)
        self.samples = rng.integers(0, len(self.qids), size=(n, len(self.qids)))

    def ci(self, values_by_qid):
        """values_by_qid: Series indexed by qid. Returns (mean, low, high)."""
        v = values_by_qid.reindex(self.qids).to_numpy(dtype=float)
        means = np.nanmean(v[self.samples], axis=1)
        return float(np.nanmean(v)), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def per_question(df, cfg, method, col):
    s = df[(df["config"] == cfg) & (df["method"] == method)].set_index("qid")[col]
    return s[~s.index.duplicated()]


def tables(df, metrics, configs, boot, part):
    methods = [m for m in ["embedding", "logreg", "xgboost", "histgb"] if m in set(df["method"])]
    rows, diffs = [], []
    for cfg in configs:
        for m in methods:
            for col, name in metrics:
                mean, lo, hi = boot.ci(per_question(df, cfg, m, col))
                rows.append({"part": part, "metric": name, "config": cfg, "method": m,
                             "value": mean, "ci_low": lo, "ci_high": hi, "n": int(per_question(df, cfg, m, col).notna().sum())})
    for col, name in metrics:
        for cfg in configs:                               # reranker vs embedding, same configuration
            base = per_question(df, cfg, "embedding", col)
            for m in methods:
                if m != "embedding":
                    diffs.append(diff_row(part, name, f"{cfg}: {m} - embedding",
                                          per_question(df, cfg, m, col) - base, boot))
        for m in methods:                                 # each configuration vs html_only, same method
            base = per_question(df, "html_only", m, col)
            for cfg in configs:
                if cfg != "html_only":
                    diffs.append(diff_row(part, name, f"{m}: {cfg} - html_only",
                                          per_question(df, cfg, m, col) - base, boot))
    return rows, diffs


def diff_row(part, metric, comparison, delta, boot):
    mean, lo, hi = boot.ci(delta)
    clear = lo > 0 or hi < 0
    return {"part": part, "metric": metric, "comparison": comparison, "difference": mean,
            "ci_low": lo, "ci_high": hi, "clearly_different": "yes" if clear else "no"}


# ---------- charts ----------
def dot_plot(rows, metric, title, xlabel, path, configs):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sel = [r for r in rows if r["metric"] == metric]
    methods = list(dict.fromkeys(r["method"] for r in sel))
    n_cfg = len(configs)
    fig_h = 1.0 + 0.62 * n_cfg * max(len(methods), 1) / 2 + 0.6
    fig, ax = plt.subplots(figsize=(8.0, max(2.6, fig_h)), dpi=200)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    step = 0.26
    offsets = {m: (i - (len(methods) - 1) / 2) * step for i, m in enumerate(methods)}
    for yi, cfg in enumerate(configs):
        y0 = n_cfg - 1 - yi
        for m in methods:
            r = next((r for r in sel if r["config"] == cfg and r["method"] == m), None)
            if r is None:
                continue
            key = "gbt" if m in ("xgboost", "histgb") else m
            y = y0 - offsets[m]
            ax.plot([r["ci_low"], r["ci_high"]], [y, y], color=COLORS[key], lw=2, solid_capstyle="round", zorder=2)
            ax.scatter([r["value"]], [y], s=64, marker=MARKERS[key], color=COLORS[key],
                       edgecolor=SURFACE, linewidth=2, zorder=3)
            ax.text(r["ci_high"] + 0.012, y, f"{r['value']:.2f}", va="center", ha="left", fontsize=8, color=INK2)
    ax.set_yticks(range(n_cfg))
    ax.set_yticklabels(list(reversed(configs)), fontsize=9, color=INK)
    ax.set_xlim(-0.03, 1.08)
    ax.set_ylim(-0.6, n_cfg - 0.4)
    ax.set_xlabel(xlabel, fontsize=9, color=INK2)
    ax.tick_params(axis="x", colors=MUTED, labelsize=8, length=0)
    ax.tick_params(axis="y", length=0)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(AXIS)
    handles = [plt.Line2D([], [], color=COLORS["gbt" if m in ("xgboost", "histgb") else m],
                          marker=MARKERS["gbt" if m in ("xgboost", "histgb") else m], lw=2, markersize=7,
                          markeredgecolor=SURFACE, label=METHOD_NAMES.get(m, m)) for m in methods]
    ax.legend(handles=handles, loc="lower left", bbox_to_anchor=(0, 1.0), ncol=len(methods),
              frameon=False, fontsize=8, labelcolor=INK2, borderaxespad=0.2, handlelength=2.2)
    ax.set_title(title, loc="left", fontsize=11, color=INK, pad=26)
    fig.text(0.02, 0.005, "Dots: estimate. Lines: 95% bootstrap confidence interval.", fontsize=7, color=MUTED)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)


def coverage_plot(cov, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.0, 0.5 * len(cov) + 1.3), dpi=200)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    names = list(cov.index)
    ys = np.arange(len(names))[::-1]
    ax.barh(ys, cov.values, height=0.42, color=COLORS["embedding"], zorder=2)
    for y, v in zip(ys, cov.values):
        ax.text(v + 0.012, y, f"{v:.0%}", va="center", fontsize=8, color=INK2)
    ax.set_yticks(ys)
    ax.set_yticklabels(names, fontsize=9, color=INK)
    ax.set_xlim(0, 1.1)
    ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.tick_params(axis="x", colors=MUTED, labelsize=8, length=0)
    ax.tick_params(axis="y", length=0)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(AXIS)
    fig.suptitle("Share of questions whose answer appears in each source", x=0.02, ha="left", fontsize=11, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)


# ---------- markdown ----------
def md_table(rows, metric):
    sel = [r for r in rows if r["metric"] == metric]
    methods = list(dict.fromkeys(r["method"] for r in sel))
    configs = list(dict.fromkeys(r["config"] for r in sel))
    head = "| Configuration | " + " | ".join(METHOD_NAMES.get(m, m) for m in methods) + " |"
    lines = [head, "|" + "---|" * (len(methods) + 1)]
    for cfg in configs:
        cells = []
        for m in methods:
            r = next(r for r in sel if r["config"] == cfg and r["method"] == m)
            cells.append(f"{r['value']:.2f} [{r['ci_low']:.2f}, {r['ci_high']:.2f}]")
        lines.append(f"| {cfg} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def md_diffs(diffs, metric):
    sel = [d for d in diffs if d["metric"] == metric]
    lines = ["| Comparison | Difference [95% CI] | Clearly different? |", "|---|---|---|"]
    for d in sel:
        lines.append(f"| {d['comparison']} | {d['difference']:+.2f} [{d['ci_low']:+.2f}, {d['ci_high']:+.2f}] "
                     f"| {d['clearly_different']} |")
    return "\n".join(lines)


# ---------- main ----------
def main(configs=None, n_boot=2000):
    OUT.mkdir(parents=True, exist_ok=True)
    ret = load_retrieval()
    configs = choose_configs(ret, configs)
    ret = ret[ret["config"].isin(configs)]
    boot = Boot(ret["qid"].unique(), n_boot)
    rows, diffs = tables(ret, [("rr", "MRR"), ("r1", "Recall@1")], configs, boot, "retrieval")

    ans = load_answers()
    if ans is not None:
        ans = ans[ans["config"].isin(configs)]
        a_configs = [c for c in configs if c in set(ans["config"])]
        a_boot = Boot(ans["qid"].unique(), n_boot)
        r2, d2 = tables(ans, [("correct", "Answer accuracy")], a_configs, a_boot, "answers")
        rows += r2
        diffs += d2

    pd.DataFrame(rows).to_csv(OUT / "results_with_ci.csv", index=False)
    pd.DataFrame(diffs).to_csv(OUT / "paired_differences.csv", index=False)

    n_q, n_co = ret["qid"].nunique(), ret["group"].nunique()
    md = [f"# Results ({n_q} questions, {n_co} companies)", "",
          "Values are estimates with 95% bootstrap confidence intervals over questions. Questions from the same "
          "company are not independent, so with few companies these intervals are, if anything, too narrow.", "",
          "## Retrieval (MRR)", md_table(rows, "MRR"), "", "## Retrieval (top-1 accuracy)", md_table(rows, "Recall@1"),
          "", "## Paired differences, retrieval MRR", md_diffs(diffs, "MRR")]
    dot_plot(rows, "MRR", "Retrieval quality by configuration and reranker", "Mean reciprocal rank (MRR)",
             OUT / "retrieval_mrr.png", configs)
    if ans is not None and len(ans):
        md += ["", "## Answer accuracy", md_table(rows, "Answer accuracy"), "",
               "## Paired differences, answer accuracy", md_diffs(diffs, "Answer accuracy")]
        dot_plot(rows, "Answer accuracy", "Answer accuracy by configuration and reranker",
                 "Share of questions answered correctly", OUT / "answer_accuracy.png", a_configs)

    cov_path = RESULTS / "retrieval_details.csv"
    if cov_path.exists():
        rd = pd.read_csv(cov_path).drop_duplicates("id")
        cov = rd[[c for c in rd.columns if c.startswith("in_")]].mean()
        cov.index = [c[3:] for c in cov.index]
        coverage_plot(cov.sort_values(ascending=False), OUT / "answer_coverage.png")
        md += ["", "## Answer coverage (share of questions whose answer appears in each source)", "",
               "| Source | Coverage |", "|---|---|"] + [f"| {s} | {v:.0%} |" for s, v in cov.items()]

    (OUT / "summary.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md))
    print(f"\nSaved tables, summary.md and charts to {OUT}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Confidence intervals, paired comparisons and charts.")
    ap.add_argument("--configs", help="comma-separated configurations to include")
    ap.add_argument("--boot", type=int, default=2000, help="bootstrap samples (default 2000)")
    a = ap.parse_args()
    main(a.configs, a.boot)