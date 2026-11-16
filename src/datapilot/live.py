"""HTML builders for the live-status UI: step list, progress bar, chips, the floating
status pill and the animated hero backdrop. Pure functions (no Streamlit import), so they
are unit-testable; every dynamic string is HTML-escaped."""
from __future__ import annotations

import html
import random

_CHECK = '<svg viewBox="0 0 12 12"><path d="M2.5 6.3 5 8.8 9.6 3.4"/></svg>'


def _e(x) -> str:
    return html.escape(str(x), quote=True)


def steps_html(title: str, steps: list[tuple[str, str, str]], finished: bool = False) -> str:
    """steps: (label, detail, state) with state in {'done','active','pending'}."""
    rows = []
    for label, detail, state in steps:
        if state not in ("done", "active", "pending"):
            raise ValueError(f"bad step state {state!r}")
        mark = f'<span class="mark">{_CHECK}</span>' if state == "done" else '<span class="mark"></span>'
        rows.append(f'<div class="dp-step {state}">{mark}<span>{_e(label)}</span>'
                    f'<span class="detail">{_e(detail)}</span></div>')
    cls = "dp-live finished" if finished else "dp-live"
    return f'<div class="{cls}"><div class="title"><span class="dot"></span>{_e(title)}</div>{"".join(rows)}</div>'


def progress_html(title: str, done: int, total: int, current: str = "", elapsed: float | None = None,
                  finished: bool = False) -> str:
    total = max(int(total), 1)
    pct = 100 if finished else max(0, min(100, round(100 * done / total)))
    left = _e(current) if current else ("Finished" if finished else "Starting")
    right = f"{done}/{total}" + (f" · {elapsed:.0f}s" if elapsed is not None else "")
    cls = "dp-live finished" if finished else "dp-live"
    return (f'<div class="{cls}"><div class="title"><span class="dot"></span>{_e(title)}</div>'
            f'<div class="dp-bar"><i style="width:{pct}%"></i></div>'
            f'<div class="dp-bar-meta"><span>{left}</span><span>{_e(right)}</span></div></div>')


def chips_html(title: str, chips: list[tuple[str, str]]) -> str:
    """chips: (label, value) pairs, shown with a staggered fade-in."""
    items = "".join(f'<span class="dp-chip" style="--i:{i}">{_e(k)} <b>{_e(v)}</b></span>'
                    for i, (k, v) in enumerate(chips))
    return (f'<div class="dp-live finished"><div class="title"><span class="dot"></span>{_e(title)}</div>'
            f'<div class="dp-chips">{items}</div></div>')


def pill_html(text: str, busy: bool = False) -> str:
    return (f'<div class="dp-live-pill{" busy" if busy else ""}"><span class="dot"></span>'
            f'<span class="txt">{_e(text)}</span></div>')


def hero_bars_svg(n: int = 44, seed: int = 7) -> str:
    """Box-plot glyphs (whisker + box) that gently breathe: the data-science answer to a
    candlestick backdrop. Deterministic for a given seed."""
    rng = random.Random(seed)
    w, h = 1000, 600
    step = w / n
    parts = []
    level = 0.45
    for i in range(n):
        level = min(0.85, max(0.2, level + rng.uniform(-.09, .09) + (0.004 if i < n / 2 else -0.004)))
        cx = step * (i + .5)
        box_h = rng.uniform(30, 120)
        top = h * (1 - level) - box_h / 2
        wick_t, wick_b = top - rng.uniform(20, 70), top + box_h + rng.uniform(20, 70)
        parts.append(
            f'<g class="bx" style="--i:{i}"><line x1="{cx:.1f}" x2="{cx:.1f}" y1="{wick_t:.1f}" y2="{wick_b:.1f}"/>'
            f'<rect x="{cx - step * .28:.1f}" y="{top:.1f}" width="{step * .56:.1f}" height="{box_h:.1f}" rx="1.5"/></g>')
    return f'<svg viewBox="0 0 {w} {h}" preserveAspectRatio="xMaxYMax slice" aria-hidden="true">{"".join(parts)}</svg>'


def rail_html(stage: int, labels=("Upload", "Configure", "Train", "Results")) -> str:
    """Progress rail for the hero: steps before `stage` are done, `stage` is active."""
    out = []
    for i, label in enumerate(labels):
        state = "done" if i < stage else ("active" if i == stage else "pending")
        out.append(f'<span class="{state}">{_e(label)}</span>')
    return f'<div class="dp-glass-nav" role="status" aria-label="Progress">{"".join(out)}</div>'



# ------------------------------------------------------------- live run ----
RUN_NODES = ["Read the data", "Build features", "Compare models", "Fine-tune", "Final test"]
_RUN_WHY = [
    "Looking at every column: its type, its gaps, and whether it can help predict.",
    "Cleaning the data and turning every column into numbers a model can use. New features are built here.",
    "Each kind of model plays the same practice rounds on the training rows. The best ones move on.",
    "Trying different settings on the best models and checking each one on fresh rows.",
    "Training the winner on all training rows, then testing it once on rows it has never seen.",
]


def clock_html(elapsed: float) -> str:
    """A running mm:ss that keeps ticking in the browser between updates (CSS counters; no script)."""
    return f'<span class="dp-clock" style="--e:{max(0, int(elapsed))}"><i class="m"></i>:<i class="s"></i></span>'


def _columns_visual(columns: list[tuple[str, str]]) -> str:
    shown = columns[:28]
    chips = "".join(f'<span class="dp-colchip {_e(role)}" style="--i:{i}">{_e(name)}</span>'
                    for i, (name, role) in enumerate(shown))
    more = f'<span class="dp-colchip more">+{len(columns) - len(shown)} more</span>' if len(columns) > len(shown) else ""
    legend = ('<div class="dp-legend"><span class="number">number</span><span class="category">category</span>'
              '<span class="text">text</span><span class="skip">left out</span><span class="target">to predict</span></div>')
    return f'<div class="dp-cols">{chips}{more}</div>{legend}'


def _features_visual(preps: list[tuple[str, int | None]], n_cols: int) -> str:
    biggest = max([n for _, n in preps if n] or [1])
    rows = []
    for name, n in preps:
        if n is None:
            rows.append(f'<div class="dp-prep active"><span class="k">{_e(name)}</span>'
                        f'<span class="bar"><i class="busy"></i></span><span class="v">working</span></div>')
        else:
            rows.append(f'<div class="dp-prep done"><span class="k">{_e(name)}</span>'
                        f'<span class="bar"><i style="width:{max(6, round(100 * n / biggest))}%"></i></span>'
                        f'<span class="v">{n:,} features</span></div>')
    note = ""
    last = [n for _, n in preps if n]
    if last:
        note = f'<div class="dp-note-line">{n_cols} columns became up to {max(last):,} features the models can read.</div>'
    return f'<div class="dp-preps">{"".join(rows)}</div>{note}'


def _models_visual(done_models: list[tuple[str, int]], done: int, total: int, n_prep: int) -> str:
    chips = []
    for i, (name, count) in enumerate(done_models):
        new = " new" if i == len(done_models) - 1 else ""
        full = " full" if count >= n_prep else ""
        chips.append(f'<span class="dp-model{full}{new}">{_e(name)}<b>{count}/{n_prep}</b></span>')
    pct = max(2, min(100, round(100 * done / max(total, 1))))
    return (f'<div class="dp-models">{"".join(chips)}</div>'
            f'<div class="dp-bar"><i style="width:{pct}%"></i></div>'
            f'<div class="dp-bar-meta"><span>{(f"{done} of {total} practice runs finished" if total > 1 else "setting up the practice rounds")}</span>'
            f'<span>all CPU cores in use</span></div>')


def _tuning_visual() -> str:
    tracks = "".join(f'<div class="dp-track"><span>{_e(n)}</span><div><i style="--d:{d}s"></i></div></div>'
                     for n, d in (("setting A", 2.6), ("setting B", 3.4), ("setting C", 2.1)))
    return (f'<div class="dp-tracks">{tracks}</div>'
            '<div class="dp-note-line">The sliders are only a picture. The search itself runs on your data; '
            'each try is scored on rows it was not trained on.</div>')


def _final_visual(model_name: str) -> str:
    return (f'<div class="dp-final"><span class="ring"></span><div><div class="n">{_e(model_name or "the winner")}</div>'
            f'<div class="s">fit on all training rows, then scored once on the hidden rows</div></div></div>')


def run_live_html(stage: int, details: list[str], elapsed: float, visual: str, title: str = "DataPilot is working") -> str:
    """Timeline of the five stages on the left, what is happening right now on the right.
    `stage` 0-4 is the active node; earlier ones are done. `details` is one short line per node."""
    nodes = []
    for i, name in enumerate(RUN_NODES):
        state = "done" if i < stage else "active" if i == stage else "pending"
        mark = (f'<span class="mark">{_CHECK}</span>' if state == "done" else '<span class="mark"></span>')
        det = f'<span class="d">{_e(details[i])}</span>' if i < len(details) and details[i] else ""
        nodes.append(f'<div class="dp-node {state}">{mark}<div><span class="n">{_e(name)}</span>{det}</div></div>')
    return (f'<div class="dp-run"><div class="dp-run-head"><span class="dot"></span><span class="ttl">{_e(title)}</span>'
            f'{clock_html(elapsed)}</div>'
            f'<div class="dp-run-body"><div class="dp-nodes">{"".join(nodes)}</div>'
            f'<div class="dp-stage-card"><div class="why">{_e(_RUN_WHY[min(max(stage, 0), 4)])}</div>{visual}</div></div></div>')


live_columns_visual = _columns_visual
live_features_visual = _features_visual
live_models_visual = _models_visual
live_tuning_visual = _tuning_visual
live_final_visual = _final_visual
