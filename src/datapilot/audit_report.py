"""Stage 8 — Explainability / audit report.

Pure Markdown formatting of objects the earlier stages already computed.
Nothing is recomputed or invented here.
"""
from __future__ import annotations

from datetime import datetime


def _esc(text) -> str:
    """Keep table cells on one line and pipe-safe."""
    return str(text).replace("|", "\\|").replace("\n", " ")


def _fmt(v, nd: int = 4) -> str:
    if v is None:
        return "n/a"
    try:
        return f"{float(v):.{nd}f}"
    except (TypeError, ValueError):
        return _esc(v)


def build_audit_report(
    profile,
    prefs,
    candidates,
    leaderboard_entries,
    tuning_results=None,
    evaluation_result=None,
    interpretation_result=None,
    leakage_suspects=None,
    warnings=None,
    sensitive_inputs=None,
) -> str:
    out: list[str] = []
    task = getattr(prefs.task_type, "value", prefs.task_type)
    out.append("# DataPilot Audit Report")
    out.append(f"_Generated {datetime.now():%Y-%m-%d %H:%M}_\n")

    # 1. Dataset
    out.append("## 1. Dataset Profile")
    out.append(f"- Rows x columns: **{profile.n_rows:,} x {profile.n_cols}**")
    out.append(f"- Duplicate rows: {profile.n_duplicate_rows:,}")
    out.append(f"- Target: **{prefs.target_column}** ({task})")
    if profile.target_distribution:
        dist = ", ".join(f"{_esc(k)}: {_fmt(v, 3)}" for k, v in list(profile.target_distribution.items())[:10])
        out.append(f"- Target distribution: {dist}")
    flagged = [c for c in profile.columns.values()
               if c.is_likely_identifier or c.is_constant or c.is_free_text or c.is_datetime or c.missing_ratio >= 0.2]
    if flagged:
        out.append("\n| Column | Type | Missing | Unique | Flags |\n|---|---|---|---|---|")
        for c in flagged[:40]:
            flags = [n for n, on in [("identifier?", c.is_likely_identifier), ("constant", c.is_constant),
                                      ("free text", c.is_free_text), ("datetime", c.is_datetime)] if on]
            out.append(f"| {_esc(c.name)} | {_esc(c.dtype)} | {c.missing_ratio:.1%} | {c.n_unique} | {', '.join(flags) or '-'} |")
        if len(flagged) > 40:
            out.append(f"\n_{len(flagged) - 40} more flagged columns not shown._")
    out.append("")

    # 2. Preprocessing decisions
    out.append("## 2. Preprocessing & Feature Engineering Decisions")
    for cand in candidates:
        out.append(f"### Pipeline: {cand.name}")
        if cand.description:
            out.append(cand.description)
        if cand.decisions:
            out.append("\n| What | Why | Evidence | Alternatives | Risk | Effect | Reversible |\n|---|---|---|---|---|---|---|")
            for d in cand.decisions:
                out.append(d.as_markdown_row().replace("\n", " "))
        out.append("")

    # 3. Leaderboard
    out.append("## 3. Model Leaderboard (cross-validated, training data only)")
    ok = sorted([e for e in leaderboard_entries if e.ok], key=lambda e: e.primary_mean, reverse=True)
    failed = [e for e in leaderboard_entries if not e.ok]
    if ok:
        out.append(f"Primary metric: {ok[0].primary_metric_name}\n")
        out.append("| Rank | Pipeline | Model | Mean | Std | Runtime (s) | Notes |\n|---|---|---|---|---|---|---|")
        for i, e in enumerate(ok, 1):
            notes = ", ".join(n for n, on in [("baseline", e.is_baseline), ("SMOTE", e.uses_smote)] if on) or "-"
            out.append(f"| {i} | {_esc(e.pipeline_name)} | {_esc(e.model_name)} | {_fmt(e.primary_mean)} | "
                       f"{_fmt(e.primary_std)} | {e.runtime_seconds:.1f} | {notes} |")
    else:
        out.append("No successful leaderboard entries.")
    for e in failed:
        out.append(f"- FAILED: {_esc(e.pipeline_name)} / {_esc(e.model_name)} — {_esc(e.error)}")
    out.append("")

    # 4. Tuning
    out.append("## 4. Model Improvement (hyperparameter tuning)")
    if tuning_results:
        out.append("| Pipeline | Model | Before | After | Change | Best params |\n|---|---|---|---|---|---|")
        for r in tuning_results:
            if not r.ok:
                out.append(f"| {_esc(r.pipeline_name)} | {_esc(r.model_name)} | - | - | - | FAILED: {_esc(r.error)} |")
            elif not r.tunable:
                out.append(f"| {_esc(r.pipeline_name)} | {_esc(r.model_name)} | {_fmt(r.base_mean)} | - | - | not tunable |")
            elif r.verified and not r.accepted:
                out.append(f"| {_esc(r.pipeline_name)} | {_esc(r.model_name)} | {_fmt(r.base_mean)} | "
                           f"{_fmt(r.tuned_mean)} | +0.0000 | defaults kept (tuned settings did not win on fresh folds) |")
            else:
                out.append(f"| {_esc(r.pipeline_name)} | {_esc(r.model_name)} | {_fmt(r.base_mean)} | "
                           f"{_fmt(r.tuned_mean)} | {r.improvement:+.4f} | {_esc(r.best_params)} |")
    else:
        out.append("Tuning was not run.")
    out.append("")

    # 5. Held-out evaluation
    out.append("## 5. Held-Out Test Evaluation")
    if evaluation_result is None:
        out.append("Final evaluation was not run.")
    elif evaluation_result.error:
        out.append(f"Evaluation failed: {_esc(evaluation_result.error)}")
    else:
        ev = evaluation_result
        out.append(f"Model: **{_esc(ev.model_name)}** on pipeline **{_esc(ev.pipeline_name)}**, {ev.n_test:,} test rows.\n")
        if ev.regression:
            r = ev.regression
            out.append("| R² | RMSE | MAE | MAPE | Residual mean | Residual std |\n|---|---|---|---|---|---|")
            out.append(f"| {_fmt(r.r2)} | {_fmt(r.rmse)} | {_fmt(r.mae)} | {_fmt(r.mape)} | "
                       f"{_fmt(r.residual_mean)} | {_fmt(r.residual_std)} |")
            if r.worst_errors:
                out.append("\nLargest errors:\n\n| Index | Actual | Predicted | Abs error |\n|---|---|---|---|")
                for w in r.worst_errors:
                    out.append(f"| {_esc(w.get('index'))} | {_fmt(w.get('actual'))} | {_fmt(w.get('predicted'))} | {_fmt(w.get('abs_error'))} |")
        if ev.classification:
            c = ev.classification
            out.append("| Accuracy | Precision | Recall | F1 | F2 | ROC-AUC |\n|---|---|---|---|---|---|")
            out.append(f"| {_fmt(c.accuracy)} | {_fmt(c.precision)} | {_fmt(c.recall)} | {_fmt(c.f1)} | {_fmt(c.f2)} | {_fmt(c.roc_auc)} |")
            if c.confusion and c.labels:
                labs = [_esc(l) for l in c.labels]
                out.append("\nConfusion matrix (rows = actual, columns = predicted):\n")
                out.append("| | " + " | ".join(labs) + " |\n|---|" + "---|" * len(labs))
                for lab, row in zip(labs, c.confusion):
                    out.append(f"| {lab} | " + " | ".join(str(v) for v in row) + " |")
    out.append("")

    # 6. Interpretation
    out.append("## 6. Interpretation")
    if interpretation_result is None or interpretation_result.kind == "unavailable":
        note = getattr(interpretation_result, "note", "") if interpretation_result else ""
        out.append(f"Interpretation unavailable. {_esc(note)}".strip())
    else:
        out.append(f"Method: {interpretation_result.kind}. {_esc(interpretation_result.note)}\n")
        out.append("| Feature | Importance | Direction |\n|---|---|---|")
        for f in interpretation_result.features:
            out.append(f"| {_esc(f.feature)} | {_fmt(f.importance)} | {f.direction or '-'} |")
    out.append("")

    out.append("## 7. Checks & Warnings")
    if not leakage_suspects and not warnings and not sensitive_inputs:
        out.append("No leakage suspects and no reliability warnings were raised.")
    for sus in leakage_suspects or []:
        out.append(f"- Possible leakage: **{_esc(sus.column)}** — {_esc(sus.evidence)}. "
                   "Check this column is available at prediction time.")
    if sensitive_inputs:
        out.append("- Sensitive attributes used as inputs (by column name): "
                   + ", ".join(f"**{_esc(c)}**" for c in sensitive_inputs)
                   + ". Review whether using them is appropriate and lawful for this decision.")
    for w in warnings or []:
        out.append(f"- {_esc(w)}")
    out.append("")
    return "\n".join(out)
