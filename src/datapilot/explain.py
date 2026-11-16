"""Plain-language reading of a result, and a one-example predictor.

Everything here uses numbers the run already measured on the held-out rows; nothing is
estimated or invented. Pure functions, so they are unit-tested without the interface."""
from __future__ import annotations

import numpy as np
import pandas as pd

# How the three data-preparation options are named for people (the code keeps the short names).
PREP_LABEL = {"conservative": "Simple cleaning", "balanced": "Standard cleaning", "aggressive": "Extra feature building"}


def prep_label(name: str) -> str:
    return PREP_LABEL.get(name, name)


def _pct(x: float) -> str:
    return f"{100 * x:.0f}%"


def unseen_phrase(result, n: int | None = None) -> str:
    """'N rows it had never seen', or, when no rows were hidden, how the N rows were checked."""
    n = result.n_test if n is None else n
    if getattr(result, "scored_by", "holdout") == "cv":
        return f"{n:,} rows cross-checked (each one predicted by a model that had not seen it)"
    return f"{n:,} rows it had never seen"


def plain_summary(result, target: str) -> list[str]:
    """A few sentences saying what the held-out score means in everyday terms."""
    out: list[str] = []
    n = result.n_test
    if result.regression is not None:
        r = result.regression
        out.append(f"On {unseen_phrase(result)}, its prediction of <b>{target}</b> was off by "
                   f"<b>{r.mae:,.2f}</b> on average (about {_pct(r.mape)} of the real value).")
        out.append(f"It explains about {_pct(max(r.r2, 0.0))} of the ups and downs in {target}; "
                   f"the rest comes from things the columns do not capture.")
        return out
    c = result.classification
    if c is None:
        return out
    conf = np.asarray(c.confusion, dtype=float)
    labels = [str(x) for x in c.labels]
    total = conf.sum() or 1.0
    guess = conf.sum(axis=1).max() / total
    out.append(f"It got <b>{_pct(c.accuracy)}</b> of {unseen_phrase(result)} right. Always guessing the most common "
               f"answer would get {_pct(guess)}.")
    if len(labels) <= 3:
        for i, lab in enumerate(labels):
            actual, said = conf[i].sum(), conf[:, i].sum()
            hit = conf[i, i]
            if actual:
                line = f"It found {int(hit):,} of the {int(actual):,} real <b>{lab}</b> cases ({_pct(hit / actual)})"
                if said:
                    line += f"; when it said {lab}, it was right {_pct(hit / said)} of the time"
                out.append(line + ".")
    else:
        recalls = [(conf[i, i] / conf[i].sum(), lab) for i, lab in enumerate(labels) if conf[i].sum()]
        if recalls:
            worst = min(recalls)
            out.append(f"Hardest answer to catch: <b>{worst[1]}</b> ({_pct(worst[0])} found).")
    return out


def readable_feature(name: str) -> str:
    """'num__age' -> 'age'; 'cat_low__sex_M' -> 'sex = M'. Falls back to the name as given."""
    if "__" in name:
        kind, rest = name.split("__", 1)
        if kind.startswith("cat") and "_" in rest:
            col, val = rest.rsplit("_", 1)
            return f"{col} = {val}"
        return rest
    return name


def accuracy_percent(result) -> float | None:
    """One 'accuracy' number for any task, from the held-out rows.
    Classification: share of rows answered correctly. Number targets: 100% minus the average percentage error (never below 0)."""
    if result is None:
        return None
    if result.classification is not None:
        return float(result.classification.accuracy)
    if result.regression is not None:
        return float(max(0.0, 1.0 - result.regression.mape))
    return None


def fmt_duration(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    if seconds < 1:
        return "under 1 s"
    if seconds < 60:
        return f"{seconds:.0f} s"
    m, s = divmod(int(round(seconds)), 60)
    return f"{m} min {s:02d} s"


def total_seconds(state=None, lb=None, tun=None, fin=None) -> float:
    return sum(float(d.get("seconds", 0.0)) for d in (state, lb, tun, fin) if d)


def run_summary(state: dict, lb: dict, tun: dict | None, fin: dict, target: str, display_score) -> list[str]:
    """Bullets telling the run from start to end, using only what was measured in this run."""
    res = fin["result"]
    prof = state["profile"]
    out: list[str] = []
    n_all = state["n_train"] + state["n_test"]
    what = "a number" if res.regression is not None else f"one of {len(res.classification.labels)} answers"
    out.append(f"<b>Read the data.</b> {prof.n_rows:,} rows and {prof.n_cols} columns. Goal: predict <b>{target}</b> ({what}).")
    left = [c.name for c in prof.columns.values() if c.name != target and (c.is_likely_identifier or c.is_constant)]
    cleaned = []
    if left:
        shown = ", ".join(left[:4]) + (f" and {len(left) - 4} more" if len(left) > 4 else "")
        cleaned.append(f"left out {len(left)} column(s) that cannot help ({shown})")
    if state.get("n_dropped_duplicates"):
        cleaned.append(f"removed {state['n_dropped_duplicates']:,} repeated rows")
    if state.get("n_dropped_missing_target"):
        cleaned.append(f"dropped {state['n_dropped_missing_target']:,} rows with no answer")
    out.append("<b>Checked the columns.</b> " + ("; ".join(cleaned).capitalize() + "." if cleaned else "All columns looked usable."))
    pct = state["n_test"] / max(n_all, 1)
    mode = state.get("score_mode", "holdout")
    if mode == "external":
        out.append(f"<b>Used your test file.</b> Its {state['n_test']:,} rows were kept for the final test and never used to train or to choose a model. "
                   f"All {state['n_train']:,} training rows were used to learn.")
    elif mode == "cv":
        out.append(f"<b>Kept every row for learning.</b> Nothing was hidden. The final score comes from cross-checking {state['n_test']:,} training rows: "
                   f"each one is predicted by a model that did not see it."
                   + (f" Those models learned from fewer rows than the final one (which used all {state['n_train']:,}), so the final model can do a bit better."
                      if state["n_train"] > 1.2 * state["n_test"] else ""))
    else:
        out.append(f"<b>Set rows aside.</b> {state['n_test']:,} rows ({pct:.0%}) were kept hidden. They were not used to train or to choose a model.")
    prep = prep_label(res.pipeline_name)
    out.append(f"<b>Prepared the data.</b> Filled gaps, turned text into numbers and evened out skewed columns, in 3 ways "
               f"(Simple, Standard, Extra feature building). <b>{prep}</b> worked best.")
    ok = [e for e in lb["entries"] if e.ok]
    names = {e.model_name for e in ok if not e.is_baseline}
    early = sum(1 for e in ok if e.stopped_early)
    from .scale import DEFAULT_SEARCH_ROWS
    samp = (f" on a {DEFAULT_SEARCH_ROWS:,}-row sample" if state["n_train"] > DEFAULT_SEARCH_ROWS else "")
    out.append(f"<b>Compared models.</b> {len(names)} kinds of model were scored on repeated practice splits of the training rows{samp}, "
               f"all CPU cores at once" + (f"; {early} slow ones clearly behind were stopped early" if early else "") + ".")
    tuned = [t for t in (tun or {}).get("results", []) if t.ok and getattr(t, "tunable", True)]
    if fin.get("used_tuned_params"):
        gains = [t for t in tuned if t.accepted]
        g = max(gains, key=lambda t: t.improvement) if gains else None
        out.append("<b>Fine-tuned the best models.</b> Their settings were searched, then re-checked on fresh splits; "
                   + (f"the winner improved from {display_score(lb['primary'], g.base_mean):.3f} to {display_score(lb['primary'], g.tuned_mean):.3f}."
                      if g else "the improved settings were kept."))
    else:
        out.append("<b>Fine-tuned the best models.</b> Nothing clearly beat the default settings on fresh splits, so the defaults were kept.")
    out.append(f"<b>Picked the winner.</b> <b>{res.model_name}</b>, refit on all training rows and "
               + ("cross-checked once." if mode == "cv" else "tested once on the hidden rows."))
    acc = accuracy_percent(res)
    if res.classification is not None:
        lines = plain_summary(res, target)
        out.append(f"<b>Result.</b> {lines[0]}")
        out.extend(lines[1:])
    else:
        r = res.regression
        out.append(f"<b>Result.</b> Accuracy <b>{acc:.1%}</b> on {unseen_phrase(res)} (100% minus the average error of {r.mape:.1%}). "
                   f"Predictions were off by {r.mae:,.2f} on average.")
        out.extend(plain_summary(res, target)[1:])
    t = {"read data and prepared": state.get("seconds"), "compared models": lb.get("seconds"),
         "tuned": (tun or {}).get("seconds"), "final fit and test": fin.get("seconds")}
    parts = ", ".join(f"{k} {fmt_duration(v)}" for k, v in t.items() if v is not None)
    out.append(f"<b>Time.</b> {fmt_duration(total_seconds(state, lb, tun, fin))} in total ({parts}).")
    return out


def time_parts(state, lb, tun, fin) -> list[tuple[str, float]]:
    """(stage label, seconds) for the stages that were timed in this run."""
    rows = [("Read and prepare", (state or {}).get("seconds")), ("Compare models", (lb or {}).get("seconds")),
            ("Fine-tune", (tun or {}).get("seconds")), ("Final test", (fin or {}).get("seconds"))]
    return [(k, float(v)) for k, v in rows if v is not None]


def summary_cards(bullets: list[str]) -> list[tuple[str, str]]:
    """Split run_summary bullets ('<b>Title.</b> text') into (title, text) cards; an untitled line joins the card above."""
    import re
    cards: list[tuple[str, str]] = []
    for b in bullets:
        m = re.match(r"<b>([^<]+?)\.?</b>\s*(.*)$", b, flags=re.S)
        if m and (m.group(1).strip() != "") and not b.startswith("It "):
            cards.append((m.group(1).strip(), m.group(2)))
        elif cards:
            cards[-1] = (cards[-1][0], cards[-1][1] + " " + b)
    return cards


def summary_html(cards: list[tuple[str, str]], parts: list[tuple[str, float]]) -> str:
    """Numbered step cards plus a bar of where the time went. Text is already HTML-safe markup from run_summary."""
    out = ["<div class='dp-sum'>"]
    n = 0
    for title, body in cards:
        if title == "Time":
            continue
        n += 1
        out.append(f"<div class='dp-sum-card' style='--i:{n}'><span class='num'>{n}</span>"
                   f"<div><div class='t'>{title}</div><div class='b'>{body}</div></div></div>")
    total = sum(v for _, v in parts) or 1.0
    if parts:
        segs = "".join(f"<i style='width:{max(2.0, 100 * v / total):.1f}%;--i:{k}' title='{k_} {fmt_duration(v)}'></i>"
                       for k, (k_, v) in enumerate(parts))
        legend = "".join(f"<span><b>{k_}</b> {fmt_duration(v)}</span>" for k_, v in parts)
        out.append(f"<div class='dp-sum-time'><div class='t'>Where the {fmt_duration(total)} went</div>"
                   f"<div class='bar'>{segs}</div><div class='legend'>{legend}</div></div>")
    out.append("</div>")
    return "".join(out)


def column_weights(feature_importances: list, columns: list[str]) -> dict[str, float]:
    """Share of the model's attention per original column (transformed names contain the column name)."""
    weights = {c: 0.0 for c in columns}
    for f in feature_importances:
        for c in sorted(columns, key=len, reverse=True):   # longest name first: "Whole weight.1" before "Whole weight"
            if c in f.feature:
                weights[c] += f.importance
                break
    return weights


def example_fields(X_train: pd.DataFrame, feature_importances: list, max_fields: int = 8) -> list[str]:
    """The columns worth asking about (the most relied-on ones); the rest are filled with typical values."""
    cols = list(X_train.columns)
    w = column_weights(feature_importances, cols)
    ranked = [c for c in sorted(cols, key=lambda c: -w[c]) if w[c] > 0]
    return (ranked or cols)[:max_fields]


def typical_row(X_train: pd.DataFrame) -> pd.Series:
    """Median for numbers, most common value for everything else."""
    vals = {}
    for c in X_train.columns:
        s = X_train[c]
        if pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s):
            vals[c] = float(s.median()) if s.notna().any() else 0.0
        else:
            m = s.mode(dropna=True)
            vals[c] = m.iloc[0] if len(m) else ""
    return pd.Series(vals)


def predict_example(pipeline, X_train: pd.DataFrame, overrides: dict, label_encoder=None) -> dict:
    """Predict one made-up row: typical values, with `overrides` on top.

    Returns {"prediction": value, "chances": {label: probability} or None}."""
    base = typical_row(X_train)
    for k, v in overrides.items():
        if k in base.index:
            base[k] = v
    row = pd.DataFrame([base])[list(X_train.columns)]
    for c in X_train.columns:   # keep each column's own type so the pipeline sees what it was trained on
        try:
            row[c] = row[c].astype(X_train[c].dtype)
        except (ValueError, TypeError):
            pass
    pred = pipeline.predict(row)[0]
    chances = None
    if hasattr(pipeline, "predict_proba"):
        try:
            proba = pipeline.predict_proba(row)[0]
            classes = list(getattr(pipeline, "classes_", range(len(proba))))
            if label_encoder is not None:
                classes = list(label_encoder.inverse_transform(np.asarray(classes).astype(int)))
            chances = {str(k): float(p) for k, p in zip(classes, proba)}
        except Exception:  # noqa: BLE001 - some models have no probabilities
            chances = None
    if label_encoder is not None:
        pred = label_encoder.inverse_transform(np.asarray([pred]).astype(int))[0]
    if isinstance(pred, (np.floating, float)):
        pred = float(pred)
    elif isinstance(pred, np.generic):
        pred = pred.item()
    return {"prediction": pred, "chances": chances}
