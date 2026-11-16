"""
DataPilot - the app.

    streamlit run src/datapilot/app.py

One click runs everything (see runner.py): build pipelines, race models, tune, final test.
An optional test file can be added at the start.
After a run: Results, Summary, Models, Program (read / edit / re-run the generated project)
and Competition (submission.csv for Kaggle-style tasks).
"""
from __future__ import annotations

import base64
import re
import time
from datetime import datetime
from html import escape as esc

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

from datapilot.audit_report import build_audit_report
from datapilot.clustering import default_cluster_inputs, run_clustering
from datapilot.diagnostics import (
    assess_results,
    detect_leakage_suspects,
    sensitive_input_columns,
)
from datapilot.experiment import SELECTABLE_PRIMARY, display_score
from datapilot.export import (
    build_project_zip,
    generate_prediction_script,
    serialize_pipeline,
)
from datapilot.explain import (accuracy_percent, example_fields, fmt_duration, run_summary, summary_cards,
                               summary_html, time_parts, total_seconds, predict_example, prep_label, readable_feature, typical_row)
from datapilot.interpretation import interpret
from datapilot.io import excel_sheets, is_excel, read_any, read_table
from datapilot.live import (
    RUN_NODES,
    chips_html,
    hero_bars_svg,
    live_columns_visual,
    live_features_visual,
    live_final_visual,
    live_models_visual,
    live_tuning_visual,
    pill_html,
    rail_html,
    run_live_html,
    steps_html,
)
from datapilot.model_zoo import unavailable_libraries
from datapilot.planner import PipelineOptions
from datapilot.profiler import DatasetProfiler
from datapilot.reasoner import Priority, TaskType
from datapilot.runner import (
    EFFORTS,
    RunConfig,
    build_state,
    finalize,
    race,
    run_all,
    search_data,
    tune,
)
from datapilot.setup import infer_task_type, suggest_input_exclusions, suggest_targets
from datapilot.stability import accuracy_interval, stability_check
from datapilot.submission import check_submission, make_submission, refit_on_all_rows
from datapilot.theme import THEME_CSS
from datapilot.workbench import (
    changed_files,
    dump_project,
    join_sections,
    load_project,
    ordered_names,
    run_script,
    run_script_files,
    split_sections,
)

SECTION_HELP = {
    "1": "The tools the program uses.",
    "2": "The switches: output column, test share, model and its values, tuning on/off.",
    "3": "Reads your CSV.",
    "4": "Removes duplicates and rows with no output, then sets the test rows aside.",
    "5": "Everything done to the inputs before the model sees them: missing values, encoding, scaling, new columns.",
    "6": "The model and its settings.",
    "7": "Optional search for better settings. Set RUN_TUNING = True in step 2.",
    "8": "Fits the model on the training rows.",
    "9": "Tests it once on rows it never saw.",
    "10": "Saves the trained model for predict.py.",
}


def _show_html(html: str, height: int) -> None:
    """Embed an HTML snippet; uses st.iframe where available, else the older component."""
    if hasattr(st, "iframe"):
        uri = "data:text/html;base64," + base64.b64encode(html.encode()).decode()
        st.iframe(uri, height=height)
    else:
        components.html(html, height=height, scrolling=True)

st.set_page_config(page_title="DataPilot", page_icon="🧭", layout="wide")
st.markdown(THEME_CSS, unsafe_allow_html=True)
ss = st.session_state
ss.setdefault("runs", [])

STAGES = ["build", "race", "tune", "final"]
NONE_CHOICE = "None - find groups instead"


# ------------------------------------------------------------------ state ---
def current_stage() -> int:
    if ss.get("final_eval_state"):
        return 3
    if ss.get("pipeline_state"):
        return 2
    if ss.get("analysed_key"):
        return 1
    return 0


def reset_downstream(keep: str = "") -> None:
    order = ["pipeline_state", "leaderboard_state", "tuning_state", "final_eval_state"]
    for k in order[order.index(keep) + 1:] if keep else order:
        ss.pop(k, None)
    ss.pop("derived", None)
    ss.pop("submission", None)


def derived() -> dict:
    """Things computed once per finished run (not on every click)."""
    tok = ss.get("final_token")
    d = ss.setdefault("derived", {})
    if d.get("token") != tok:
        d.clear()
        d["token"] = tok
    return d


def log_run(name: str, rows: int, **fields) -> None:
    for r in ss.runs:
        if r["name"] == name and r["rows"] == rows:
            r.update(fields)


# --------------------------------------------------------------- live UI ----
def live_columns(df, profile, target) -> list[tuple[str, str]]:
    """(column, role) for the live picture; roles match the chip colours."""
    out = []
    for c in df.columns:
        col = profile.columns.get(c)
        if c == target:
            role = "target"
        elif col is None or col.is_likely_identifier or col.is_constant:
            role = "skip"
        elif col.is_free_text:
            role = "text"
        elif col.is_numeric or col.is_boolean:
            role = "number"
        else:
            role = "category"
        out.append((str(c), role))
    out.sort(key=lambda x: (x[1] != "target", x[1] == "skip"))
    return out


def make_progress(slot, pill, columns=None, n_rows=0):
    """Progress callback that draws the live run picture from what the run reports."""
    t0 = time.time()
    columns = columns or []
    info = {"preps": [], "models": {}, "best": "", "done": 0, "total": 1}
    n_cols = sum(1 for _, r in columns if r not in ("target", "skip")) or len(columns)

    def cb(stage, done, total, label=""):
        idx = STAGES.index(stage)
        node = idx + 1 if idx >= 1 else 0      # build covers two nodes: read, then features
        if stage == "build":
            m = re.match(r"^Built (\d+) features: (.*)$", label)
            b = re.match(r"^Building features: (.*)$", label)
            if m:
                n, name = int(m.group(1)), m.group(2)
                info["preps"] = [(k, v) for k, v in info["preps"] if k != name] + [(name, n)]
            elif b and not any(k == b.group(1) for k, _ in info["preps"]):
                info["preps"].append((b.group(1), None))
            node = 0 if label.startswith(("Reading", "Setting")) else 1
            if label == "Ready":
                node = 2   # features are built: the model comparison is next, show it straight away
        elif stage == "race":
            info["done"], info["total"] = int(done), int(total)
            model = label.split(" on ")[0] if " on " in label else ""
            if model:
                info["models"][model] = info["models"].get(model, 0) + 1
        elif stage == "final" and done >= total and label:
            info["best"] = label
        n_prep = max(len(info["preps"]), 1)
        built = [n for _, n in info["preps"] if n]
        details = [f"{n_rows:,} rows, {len(columns)} columns" if columns else "",
                   (f"{len(built)} of {n_prep} ways done" if node == 1 else (f"up to {max(built):,} features" if built else "")),
                   (f"{info['done']} of {info['total']} runs" if node >= 2 and info["total"] > 1 else ""),
                   "searching settings" if node == 3 else "", info["best"] if node == 4 else ""]
        if node == 0:
            visual = live_columns_visual(columns)
        elif node == 1:
            visual = live_features_visual(info["preps"], n_cols)
        elif node == 2:
            visual = live_models_visual(list(info["models"].items()), info["done"], info["total"], n_prep)
        elif node == 3:
            visual = live_tuning_visual()
        else:
            visual = live_final_visual(info["best"])
        slot.markdown(run_live_html(node, details, time.time() - t0, visual), unsafe_allow_html=True)
        pill.markdown(pill_html(f"{RUN_NODES[node]} - {label}" if label else RUN_NODES[node], busy=True),
                      unsafe_allow_html=True)
    return cb


def store_run(state, lb=None, tun=None, fin=None):
    ss.pipeline_state = state
    if lb is not None:
        ss.leaderboard_state = lb
    if tun is not None:
        ss.tuning_state = tun
    if fin is not None:
        ss.final_eval_state = fin
        ss.final_token = time.time()
    ss.pop("derived", None)


def finish_run_log(name, rows, state, lb, tun, fin):
    res = fin["result"]
    acc = accuracy_percent(res) if res.ok else None
    log_run(name, rows, took=fmt_duration(total_seconds(state, lb, tun, fin)),
            accuracy=f"{acc:.1%}" if acc is not None else "-")


def update_runs_table(name, rows, state=None, lb=None):
    if state:
        log_run(name, rows, target=state["target"], task=state["task"], pipelines=len(state["candidates"]))
    if lb:
        ok = [e for e in lb["entries"] if e.ok]
        if ok:
            log_run(name, rows, best_model=f"{ok[0].model_name} ({ok[0].pipeline_name})",
                    best_score=f"{ok[0].primary_metric_name}={display_score(ok[0].primary_metric_name, ok[0].primary_mean):.3f}")


# ------------------------------------------------------------------ hero ----
pill_slot = st.empty()
_rendered_stage = current_stage()
_latest = ss.runs[-1] if ss.runs else None
if _latest:
    _panel = (
        f"<div class='dp-panel'><div class='dp-panel-head'><span class='fname mono'>{esc(_latest['name'])}</span>"
        f"<span class='tag'>{_latest['rows']:,} x {_latest['cols']}</span></div>"
        f"<div class='dp-row'><span class='k'>Missing data</span><span class='v'>{_latest['missing_pct']:.1%}</span></div>"
        f"<div class='dp-row'><span class='k'>Duplicate rows</span><span class='v'>{_latest['dup_rows']}</span></div>"
        f"<div class='dp-row'><span class='k'>Target</span><span class='v'>{esc(str(_latest.get('target', '-')))}</span></div>"
        f"<div class='dp-row'><span class='k'>Best</span><span class='v hi'>{esc(str(_latest.get('best_score', '-')))}</span></div>"
        f"<div class='dp-foot'>Latest dataset</div></div>")
else:
    _panel = (
        "<div class='dp-panel'><div class='dp-panel-head'><span class='fname mono'>How it works</span></div>"
        "<div class='dp-row'><span class='k'>1</span><span class='v'>Upload a CSV</span></div>"
        "<div class='dp-row'><span class='k'>2</span><span class='v'>Pick what to predict</span></div>"
        "<div class='dp-row'><span class='k'>3</span><span class='v hi'>Get a model and its code</span></div>"
        "<div class='dp-foot'>Everything is computed from your data</div></div>")

st.markdown(
    f"""
    <div class="dp-hero">
      <div class="dp-hero-bars">{hero_bars_svg()}</div>
      <div class="dp-nav-bar">
        <div class="dp-wordmark"><span class="grid"><i></i><i></i><i></i><i></i><i></i><i></i><i></i><i></i><i></i><i></i><i></i><i></i></span>DataPilot</div>
        {rail_html(_rendered_stage)}
      </div>
      <div class="dp-hero-grid">
        <div>
          <h1 class="dp-h1">AutoML for <em>tabular data.</em></h1>
          <div class="dp-lede">Upload a CSV or Excel file and pick what to predict, such as a price or who will leave. Get a tested, explained prediction model plus editable Python code.</div>
        </div>
        <div>{_panel}</div>
      </div>
      <div class="dp-strip">
        <span>CSV and Excel. Every step shown. Code you can edit.</span>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------- upload ----
st.markdown('<div class="dp-section-head"><h2 class="display">Start</h2></div>', unsafe_allow_html=True)
up1, up2 = st.columns([1, 1])
uploaded = up1.file_uploader("Training file: drop a CSV or Excel file here, or browse", type=["csv", "xlsx", "xls", "xlsm"])
test_uploaded = up2.file_uploader("Test file (CSV or Excel), if you have one", type=["csv", "xlsx", "xls", "xlsm"], key="test_file_start",
                                  help="A separate test file, like Kaggle's test.csv. With answers in it, it becomes the final test, "
                                       "so no training rows are set aside. Without answers, every training row is used to learn "
                                       "and you get predictions for the test file.")
sheet = None
if uploaded is not None and is_excel(uploaded):
    try:
        _sheets = excel_sheets(uploaded)
    except Exception as e:  # noqa: BLE001 - shown to the user
        st.error(f"Could not open this Excel file: {e}")
        st.stop()
    if len(_sheets) > 1:
        sheet = st.selectbox("Which sheet?", _sheets, key=f"sheet_{uploaded.name}")
    else:
        sheet = _sheets[0]

if uploaded is None:
    st.caption("Your data stays on this machine.")
else:
    key = f"{uploaded.name}:{getattr(uploaded, 'size', 0)}:{sheet}"
    first = ss.get("analysed_key") != key
    live_slot = st.empty()

    def beat(states, details, pause=0.0):
        if first:
            labels = ["Read the file", "Profile columns", "Find the output", "Ready"]
            live_slot.markdown(
                steps_html("Reading your data", list(zip(labels, details, states)),
                           finished=all(x == "done" for x in states)), unsafe_allow_html=True)
            pill_slot.markdown(pill_html("Reading your data", busy=True), unsafe_allow_html=True)
            if pause:
                time.sleep(pause)

    cache = ss.get("df_cache")
    if cache and cache[0] == key:
        _, df, headerless, profile = cache
    else:
        beat(["active", "pending", "pending", "pending"], [""] * 4, 0.2)
        try:
            df, headerless = read_table(uploaded, sheet)
        except Exception as e:  # noqa: BLE001 - shown to the user
            live_slot.empty()
            st.error(f"Could not read this file: {e}")
            st.stop()
        beat(["done", "active", "pending", "pending"], [f"{len(df):,} rows x {df.shape[1]} columns", "", "", ""], 0.25)
        profile = DatasetProfiler().profile(df)
        ss.df_cache = (key, df, headerless, profile)
        ss.pop("cluster_result", None)

    test_df = None
    if test_uploaded is not None:
        tkey = f"{test_uploaded.name}:{getattr(test_uploaded, 'size', 0)}"
        tc = ss.get("test_cache")
        if tc and tc[0] == tkey:
            test_df = tc[1]
        else:
            try:
                test_df, _ = read_table(test_uploaded, None)
            except Exception as e:  # noqa: BLE001 - shown to the user
                st.error(f"Could not read the test file: {e}")
                st.stop()
            ss.test_cache = (tkey, test_df)
    if ss.get("test_key") != (test_uploaded.name if test_uploaded is not None else None, getattr(test_uploaded, "size", 0)):
        ss.test_key = (test_uploaded.name if test_uploaded is not None else None, getattr(test_uploaded, "size", 0))
        reset_downstream()

    flagged = sum(1 for c in profile.columns.values()
                  if c.is_likely_identifier or c.is_constant or c.is_datetime or c.is_high_cardinality)
    suggestions = suggest_targets(df, profile)
    sug_txt = suggestions[0].column if suggestions else "none obvious"
    missing_pct = sum(c.n_missing for c in profile.columns.values()) / max(profile.n_rows * profile.n_cols, 1)

    if first:
        beat(["done", "done", "done", "done"],
             [f"{len(df):,} rows x {df.shape[1]} columns", f"{flagged} flagged", f"suggested: {sug_txt}", "ready"], 0.5)
        if not any(r["name"] == uploaded.name and r["rows"] == profile.n_rows for r in ss.runs):
            ss.runs.append({"name": uploaded.name, "rows": profile.n_rows, "cols": profile.n_cols,
                            "dup_rows": profile.n_duplicate_rows, "missing_pct": missing_pct,
                            "n_identifiers": sum(c.is_likely_identifier for c in profile.columns.values()),
                            "time": datetime.now().strftime("%H:%M:%S")})
        ss.analysed_key = key
        reset_downstream()
        pill_slot.empty()
        st.rerun()

    live_slot.markdown(chips_html("Data ready", [
        ("rows", f"{profile.n_rows:,}"), ("columns", str(profile.n_cols)), ("missing", f"{missing_pct:.1%}"),
        ("duplicates", str(profile.n_duplicate_rows)), ("flagged", str(flagged)), ("suggested output", sug_txt)]),
        unsafe_allow_html=True)
    if headerless:
        st.info("No header row found, so columns are named col_0, col_1, ...")

    with st.expander("Column profile"):
        st.dataframe(pd.DataFrame([{
            "column": c.name, "type": c.dtype, "missing": f"{c.missing_ratio:.1%}", "unique": c.n_unique,
            "skew": f"{c.skew:.2f}" if c.skew is not None else "-",
            "flag": ("identifier?" if c.is_likely_identifier else "datetime" if c.is_datetime else
                     "high-cardinality" if c.is_high_cardinality else "constant" if c.is_constant else ""),
        } for c in profile.columns.values()]), width="stretch", hide_index=True)
        if profile.top_correlations:
            st.caption("Strongly related inputs: " + " | ".join(
                f"{p.col_a} / {p.col_b} ({p.correlation:+.2f})" for p in profile.top_correlations[:5]))

    # ------------------------------------------------------------ configure ---
    st.markdown('<div class="dp-section-head"><h2 class="display">Predict</h2></div>', unsafe_allow_html=True)
    all_cols = list(df.columns)
    options = [NONE_CHOICE] + all_cols
    default_idx = options.index(suggestions[0].column) if suggestions else 0
    target_choice = st.selectbox("What should DataPilot predict?", options, index=default_idx,
                                 key=f"target_{key}")
    target = None if target_choice == NONE_CHOICE else target_choice

    if target is None:
        # ------------------------------------------------------- clustering ---
        cl_inputs = st.multiselect("Group by", all_cols, default=default_cluster_inputs(df, profile),
                                   key=f"cl_inputs_{key}")
        k1, k2 = st.columns([1, 1])
        k_fixed = k1.number_input("Groups (0 = automatic)", 0, 20, 0, key=f"cl_k_{key}")
        k2.markdown("<div style='height:28px'></div>", unsafe_allow_html=True)
        if k2.button("Find groups", type="primary", disabled=not cl_inputs):
            with st.spinner("Comparing clustering methods..."):
                try:
                    ss.cluster_result = (key, run_clustering(df, profile, cl_inputs, fixed_k=int(k_fixed) or None))
                except ValueError as e:
                    st.error(str(e))
        cr = ss.get("cluster_result")
        if cr and cr[0] == key:
            res = cr[1]
            st.success(f"{res.k} groups ({res.algorithm}, silhouette {res.silhouette:.2f})")
            for n in res.notes:
                st.caption(n)
            st.dataframe(res.profile, width="stretch")
            st.scatter_chart(res.pca.assign(cluster=res.pca["cluster"].astype(str)), x="pc1", y="pc2", color="cluster")
            with st.expander("All settings tried"):
                st.dataframe(res.search, width="stretch")
            st.download_button("Download data with group labels", df.assign(cluster=res.labels).to_csv(index=False).encode(),
                               "clustered.csv", "text/csv")
        st.stop()

    auto_excluded = suggest_input_exclusions(profile, target)
    input_candidates = [c for c in all_cols if c != target]
    with st.expander("Options"):
        selected_inputs = st.multiselect("Inputs the model may use", input_candidates,
                                         default=[c for c in input_candidates if c not in auto_excluded],
                                         key=f"inputs_{key}_{target}")
        if auto_excluded:
            st.caption("Left out: " + ", ".join(f"{c} ({why})" for c, why in auto_excluded.items()))
        o1, o2, o3 = st.columns(3)
        suggested_task = infer_task_type(df[target])
        task_vals = [TaskType.CLASSIFICATION.value, TaskType.REGRESSION.value]
        task_choice = o1.selectbox("Task", task_vals, index=task_vals.index(suggested_task.value), key=f"task_{key}_{target}")
        task = TaskType(task_choice)
        depth = o2.select_slider("Depth", options=list(EFFORTS), value="Balanced", key="depth",
                                 help="Quick: fewer models and less tuning. Thorough: the most.")
        prim_opts = SELECTABLE_PRIMARY[task]
        prim_default = "f1" if task == TaskType.CLASSIFICATION else "r2"
        primary = o3.selectbox("Rank models by", prim_opts, index=prim_opts.index(prim_default),
                               key=f"primary_{task_choice}") if len(prim_opts) > 1 else None
        priority = Priority(st.selectbox("Priority", [p.value for p in Priority], index=[p.value for p in Priority].index(
            Priority.BALANCED.value), key="priority"))

    def options_from_widgets() -> PipelineOptions | None:
        try:
            return PipelineOptions(
                excluded_columns=[c for c in all_cols if c != target and c not in selected_inputs],
                pipelines=ss.get("opt_pipelines") or None, numeric_impute=ss.get("opt_impute", "median"),
                skew_correction=ss.get("opt_skew", True), interactions=ss.get("opt_interact", True), string_shape=ss.get("opt_shape", True),
                clip_outliers=ss.get("opt_clip", "auto"), pca_variance=ss.get("opt_pca", 0.95),
                tfidf_max_features=(ss.get("opt_tfidf", 0) or None),
                missing_indicators={"follow preference": None, "yes": True, "no": False}[ss.get("opt_missing", "follow preference")])
        except ValueError as e:
            st.error(str(e))
            return None

    def cfg_now(opts, primary_override="__keep__") -> RunConfig:
        return RunConfig(target, task, priority, opts, depth, primary if primary_override == "__keep__" else primary_override,
                         test_df=test_df)

    def cfg_from_state(ps, primary_metric) -> RunConfig:
        return RunConfig(ps["target"], TaskType(ps["task"]), Priority(ps["priority"]), ps["options"], depth, primary_metric,
                         test_df=test_df)

    ps = ss.get("pipeline_state")
    if ps and ps.get("dataset") != uploaded.name:
        reset_downstream()
        ps = None
    lb, tun, fin = ss.get("leaderboard_state"), ss.get("tuning_state"), ss.get("final_eval_state")

    work_slot = st.empty()

    def fast_run(cfg: RunConfig):
        cb = make_progress(work_slot, pill_slot, live_columns(df, profile, cfg.target), len(df))
        try:
            state, lb_, tun_, fin_ = run_all(df, cfg, uploaded.name, cb)
        except Exception as e:  # noqa: BLE001 - shown, not hidden
            work_slot.empty(); pill_slot.empty()
            st.error(f"The run stopped: {e}")
            return
        reset_downstream()
        store_run(state, lb_, tun_, fin_)
        update_runs_table(uploaded.name, profile.n_rows, state, lb_)
        finish_run_log(uploaded.name, profile.n_rows, state, lb_, tun_, fin_)
        pill_slot.empty()
        st.rerun()

    # -------------------------------------------------------------- run bar ---
    test_problem = None
    if test_df is not None:
        miss = [c for c in selected_inputs if c not in test_df.columns]
        if miss:
            test_problem = f"The test file is missing columns the training file has: {', '.join(miss[:6])}."
            st.error(test_problem)
        elif target in test_df.columns and test_df[target].notna().mean() > 0.5:
            st.caption(f"Test file: {len(test_df):,} rows with answers. It will be the final test, so every training row is used to learn.")
        else:
            st.caption(f"Test file: {len(test_df):,} rows without a '{target}' column (like Kaggle's test.csv). Every training row is used to learn, "
                       "accuracy comes from cross-checking, and you get predictions for the test file.")
    test_csv_bytes = test_df.to_csv(index=False).encode() if (test_df is not None and ps and ps.get("score_mode") != "holdout") else None
    if not selected_inputs:
        st.warning("Pick at least one input column under Options.")
    elif not test_problem:
        label = "Run again" if fin else "Run DataPilot"
        if st.button(label, type="primary", key="run_fast"):
            for k in ("opt_pipelines", "opt_impute", "opt_missing", "opt_skew", "opt_interact", "opt_shape", "opt_clip", "opt_pca", "opt_tfidf"):
                ss.pop(k, None)
            o = options_from_widgets()
            if o is not None:
                fast_run(cfg_now(o))

    # ------------------------------------------------------------- results ----
    if ps:
        st.markdown('<div class="dp-section-head"><h2 class="display">Your run</h2></div>', unsafe_allow_html=True)
        tabs = st.tabs(["Results", "Summary", "Models", "Program", "Competition"])
        entries = lb["entries"] if lb else []
        ok_entries = [e for e in entries if e.ok]
        task_enum = TaskType(ps["task"])

        # ---------------------------------------------------------- Results ---
        with tabs[0]:
            if not fin:
                st.caption("Finish a run to see results here.")
            elif not fin["result"].ok:
                st.error(f"Final test failed: {fin['result'].error}")
            else:
                d = derived()
                result = fin["result"]
                for fn in fin.get("fallback_notes", []):
                    st.warning(fn)
                head_metric = ""
                if result.classification:
                    clf = result.classification
                    head_metric = f"F1 {clf.f1:.3f}" + (f" | AUC {clf.roc_auc:.3f}" if clf.roc_auc is not None else "")
                elif result.regression:
                    head_metric = f"R2 {result.regression.r2:.3f}"
                st.markdown(
                    f"<div class='dp-best'><span class='name'>{esc(result.model_name)}</span>"
                    f"<span class='meta'>{head_metric} | {'cross-checked on' if result.scored_by == 'cv' else 'tested on'} {result.n_test:,} rows{'' if result.scored_by == 'cv' else ' the model never saw'}</span></div>",
                    unsafe_allow_html=True)

                def kpis(items):
                    for col, item in zip(st.columns(len(items)), items):
                        lbl, val = item[0], item[1]
                        tip = f" title='{esc(item[2])}'" if len(item) > 2 else ""
                        col.markdown(f"<div class='dp-kpi'{tip}><div class='l'>{lbl}</div><div class='v'>{val}</div></div>",
                                     unsafe_allow_html=True)
                if result.regression:
                    reg = result.regression
                    kpis([("Accuracy", f"{accuracy_percent(result):.1%}", "100% minus the average percentage error"),
                          ("R2", f"{reg.r2:.3f}"), ("RMSE", f"{reg.rmse:,.2f}"), ("MAE", f"{reg.mae:,.2f}"),
                          ("MAPE", f"{reg.mape:.1%}")] + ([("RMSLE", f"{reg.rmsle:.4f}")] if reg.rmsle is not None else []))
                    with st.expander("Largest errors"):
                        st.dataframe(pd.DataFrame(reg.worst_errors)[["actual", "predicted", "abs_error"]],
                                     width="stretch", hide_index=True)
                if result.classification:
                    clf = result.classification
                    kpis([("Accuracy", f"{clf.accuracy:.1%}"), ("Precision", f"{clf.precision:.3f}"),
                          ("Recall", f"{clf.recall:.3f}"), ("F1", f"{clf.f1:.3f}"),
                          ("ROC-AUC", f"{clf.roc_auc:.3f}" if clf.roc_auc is not None else "-")])
                    if result.scored_by == "cv" and ps["n_train"] > 1.2 * result.n_test:
                        st.caption(f"The cross-check models learned from fewer rows than the final model (which used all {ps['n_train']:,}), "
                                   "so the final model can do a bit better than this score.")
                    lo, hi = accuracy_interval(clf.accuracy, result.n_test)
                    cv_best = max((e.primary_mean for e in ok_entries if not e.is_baseline), default=None)
                    st.markdown(
                        f"<div class='dp-sub' style='margin:2px 0 6px'>On {result.n_test:,} {'cross-checked' if result.scored_by == 'cv' else 'test'} rows, accuracy of "
                        f"{clf.accuracy:.3f} could honestly be anywhere from {lo:.2f} to {hi:.2f} on other rows."
                        + (f" Repeated tests on the training rows scored {cv_best:.3f} ({esc(str(lb.get('primary') or 'score'))})."
                           if cv_best is not None else "") + "</div>", unsafe_allow_html=True)
                    with st.expander("Confusion matrix"):
                        st.dataframe(pd.DataFrame(clf.confusion, index=[f"actual {x}" for x in clf.labels],
                                                  columns=[f"pred {x}" for x in clf.labels]), width="stretch")

                sb1, sb2 = st.columns([1, 3])
                if ps.get("score_mode") != "cv" and sb1.button("Re-test on other splits", key="stab_btn",
                              help="Train the winner again on 5 other random splits of your data and see how much the score moves."):
                    with st.spinner("Refitting on 5 other splits..."):
                        try:
                            d["stab"] = stability_check(fin["final_pipeline"], ps["X_train"], ps["y_train"],
                                                        ps["X_test"], ps["y_test"], task_enum, ps.get("label_encoder"))
                        except Exception as e:  # noqa: BLE001
                            d["stab"] = f"Could not run the check: {e}"
                stab = d.get("stab")
                if isinstance(stab, str):
                    sb2.warning(stab)
                elif stab is not None:
                    sb2.markdown(
                        f"<div class='dp-sub' style='padding-top:8px'>On {len(stab.scores)} other random splits, {stab.metric} was "
                        f"<b>{stab.mean:.3f}</b> on average, give or take {stab.std:.3f} "
                        f"(range {min(stab.scores):.3f} to {max(stab.scores):.3f}).</div>", unsafe_allow_html=True)

                if "interp" not in d:
                    d["interp"] = interpret(fin["final_pipeline"], X_train=ps["X_train"], top_n=12)
                interp = d["interp"]
                st.markdown("**What the prediction depends on most**")
                if interp.kind == "unavailable" or not interp.features:
                    st.caption(interp.note or "Not available for this model.")
                else:
                    top = max((f.importance for f in interp.features), default=1.0) or 1.0
                    for f in interp.features:
                        pct = max(2, round(100 * f.importance / top))
                        sign = f" ({f.direction})" if f.direction else ""
                        st.markdown(
                            f"<div style='display:flex;align-items:center;gap:10px;margin:4px 0;'>"
                            f"<div style='width:240px;font-size:12.5px;white-space:nowrap;overflow:hidden;"
                            f"text-overflow:ellipsis;' title='{esc(f.feature)}'>{esc(readable_feature(f.feature))}{sign}</div>"
                            f"<div style='flex:1;background:var(--line);height:8px;border-radius:4px;'>"
                            f"<div class='dp-grow' style='width:{pct}%;background:var(--accent);height:8px;border-radius:4px;'></div></div>"
                            f"<div class='mono' style='width:70px;text-align:right;font-size:12px;color:var(--muted);'>"
                            f"{f.importance:.4f}</div></div>", unsafe_allow_html=True)

                with st.expander("Try it on one example"):
                    Xt_ = ps["X_train"]
                    st.caption("Change a few values and see what the model predicts. Every other column is set to a typical value.")
                    fields = example_fields(Xt_, interp.features if interp.kind != "unavailable" else [])
                    typ = typical_row(Xt_)
                    ex_cols, vals = st.columns(2), {}
                    for i_, c_ in enumerate(fields):
                        host, s_ = ex_cols[i_ % 2], Xt_[c_]
                        if pd.api.types.is_numeric_dtype(s_) and not pd.api.types.is_bool_dtype(s_):
                            vals[c_] = host.number_input(c_, value=float(typ[c_]), key=f"ex_{c_}")
                        else:
                            opts_ = s_.dropna().astype(str).value_counts().index[:50].tolist() or [str(typ[c_])]
                            at_ = opts_.index(str(typ[c_])) if str(typ[c_]) in opts_ else 0
                            vals[c_] = host.selectbox(c_, opts_, index=at_, key=f"ex_{c_}")
                    if st.button("Predict", key="ex_go"):
                        try:
                            d["example"] = predict_example(fin["final_pipeline"], Xt_, vals, ps.get("label_encoder"))
                        except Exception as e:  # noqa: BLE001 - shown to the user
                            d["example"] = f"Could not predict this example: {e}"
                    ex = d.get("example")
                    if isinstance(ex, str):
                        st.warning(ex)
                    elif ex:
                        if ex["chances"]:
                            top3 = sorted(ex["chances"].items(), key=lambda kv: -kv[1])[:3]
                            st.success(f"Prediction: **{ex['prediction']}**  |  " + ", ".join(f"{k}: {p:.0%}" for k, p in top3))
                        else:
                            val = ex["prediction"]
                            st.success(f"Predicted {ps['target']}: **{val:,.2f}**" if isinstance(val, float) else f"Prediction: **{val}**")

                if "checks" not in d:
                    d["checks"] = (detect_leakage_suspects(ps["X_train"], ps["y_train"], task_enum),
                                   assess_results(lb["entries"], result), sensitive_input_columns(ps["X_train"].columns))
                leak, warns, sens = d["checks"]
                if leak or warns or sens:
                    st.markdown("**Checks**")
                    for s_ in leak:
                        st.warning(f"**{s_.column}** may give away the answer: {s_.evidence}. Remove it if you will not know it when you need the prediction.")
                    if sens:
                        st.warning("Sensitive attributes used as inputs: " + ", ".join(sens))
                    for w_ in warns:
                        st.info(w_)
                else:
                    st.caption("Nothing suspicious found.")

                if "audit" not in d:
                    d["audit"] = build_audit_report(ps["profile"], ps["prefs"], ps["candidates"], lb["entries"],
                                                    (tun or {}).get("results", []), result, interp,
                                                    leakage_suspects=leak, warnings=warns, sensitive_inputs=sens)
                st.markdown("**Download**")
                dl = st.columns(4)
                dl[0].download_button("Model (.joblib)", serialize_pipeline(fin["final_pipeline"]),
                                      "datapilot_model.joblib", "application/octet-stream")
                dl[1].download_button("Predict script", generate_prediction_script(
                    pipeline_name=result.pipeline_name, model_name=result.model_name, task_type_value=task_enum.value,
                    target_column=ps["target"], feature_columns=fin["feature_columns"]), "predict.py", "text/x-python")
                if "zip" not in d:
                    d["zip"] = _project_zip = build_project_zip(
                        target=ps["target"], task_type_value=task_enum.value, priority_value=ps["priority"],
                        preserve_missingness=ps["prefs"].preserve_meaningful_missingness, options=ps["options"],
                        pipeline_name=result.pipeline_name, model_name=fin["base_model_name"],
                        best_params=fin["best_params"], uses_smote=fin["uses_smote"],
                        feature_columns=fin["feature_columns"], fitted_pipeline=fin["final_pipeline"],
                        label_encoder=ps.get("label_encoder"), log_target=bool(ps.get("log_target")),
                        test_csv=test_csv_bytes)
                dl[2].download_button("Project (.zip)", d["zip"], "datapilot_project.zip", "application/zip")
                dl[3].download_button("Audit report (.md)", d["audit"], "datapilot_audit_report.md", "text/markdown")
                if "all_rows_model" not in d:
                    d["all_rows_model"] = None
                if ps.get("score_mode") != "cv" and st.toggle("Also prepare a model trained on all rows (for real use)", key="all_rows_toggle",
                             help="The scores above come from the 80% model. Once they look good, a model that has also learned from the test rows is slightly better in practice."):
                    if d["all_rows_model"] is None:
                        with st.spinner("Training on all rows..."):
                            d["all_rows_model"] = serialize_pipeline(refit_on_all_rows(
                                fin["final_pipeline"], ps["X_train"], ps["y_train"], ps["X_test"], ps["y_test"]))
                    st.download_button("Model trained on all rows (.joblib)", d["all_rows_model"],
                                       "datapilot_model_all_rows.joblib", "application/octet-stream")

        # ---------------------------------------------------------- Summary ---
        with tabs[1]:
            if not fin or not fin["result"].ok:
                st.caption("Finish a run to see what DataPilot did.")
            else:
                d = derived()
                if "summary" not in d:
                    d["summary"] = summary_cards(run_summary(ps, lb, tun, fin, ps["target"], display_score))
                st.markdown(summary_html(d["summary"], time_parts(ps, lb, tun, fin)), unsafe_allow_html=True)

        # ----------------------------------------------------------- Models ---
        with tabs[2]:
            if not lb:
                st.caption("Race the models to see the leaderboard.")
            else:
                if ok_entries:
                    keys = list(ok_entries[0].metrics.keys())
                    best_per_model: dict[str, object] = {}
                    for e in ok_entries:
                        if e.model_name not in best_per_model:
                            best_per_model[e.model_name] = e
                    shown = list(best_per_model.values())

                    def board(rows, with_prep):
                        return pd.DataFrame([{
                            "#": i, "model": e.model_name + (" (baseline)" if e.is_baseline else "") + (" (stopped early)" if e.stopped_early else ""),
                            **({"data preparation": prep_label(e.pipeline_name)} if with_prep else {}),
                            **{k: round(e.metrics[k], 4) for k in keys},
                            "stability": round(e.primary_std, 4), "seconds": round(e.runtime_seconds, 1),
                        } for i, e in enumerate(rows, 1)])
                    st.dataframe(board(shown, False), width="stretch", hide_index=True)
                    best = ok_entries[0]
                    base = [e.primary_mean for e in ok_entries if e.is_baseline]
                    shown_best = display_score(best.primary_metric_name, best.primary_mean)
                    gain = (f" | {abs(best.primary_mean - max(base)):.3f} better than guessing" if base and not best.is_baseline else "")
                    st.markdown(f"<div class='dp-sub'>Ranked by <b>{best.primary_metric_name}</b> (score {shown_best:.3f}). Best: "
                                f"<b>{esc(best.model_name)}</b>{gain}</div>", unsafe_allow_html=True)
                    with st.expander("Every model with every data preparation option"):
                        st.dataframe(board(ok_entries, True), width="stretch", hide_index=True)
                for n in (lb.get("smote_note"), lb.get("cv_fold_note"), lb.get("rescue_note"), (ps or {}).get("log_note")):
                    if n:
                        st.markdown(f"<div class='dp-note'>{esc(n)}</div>", unsafe_allow_html=True)
                miss = unavailable_libraries()
                if miss:
                    st.markdown(f"<div class='dp-note'>Not installed, so skipped: {', '.join(miss)}</div>", unsafe_allow_html=True)
                failed = [e for e in entries if not e.ok]
                if failed:
                    with st.expander(f"{len(failed)} combination(s) failed"):
                        for e in failed:
                            st.markdown(f"**{e.model_name} ({prep_label(e.pipeline_name)})**: `{e.error}`")
                _, _, snote = search_data(ps)
                if snote:
                    st.caption(snote)
                if tun and tun["results"]:
                    st.markdown("**Tuning**")
                    pm = (lb or {}).get("primary") or "r2"
                    for r in tun["results"]:
                        if not r.tunable:
                            st.markdown(f"<div class='dp-sub'>{esc(r.model_name)}: nothing to tune</div>", unsafe_allow_html=True)
                        elif not r.ok:
                            st.markdown(f"<div class='dp-sub' style='color:var(--danger)'>{esc(r.model_name)}: tuning failed - {esc(str(r.error))}</div>",
                                        unsafe_allow_html=True)
                        elif r.verified and not r.accepted:
                            st.markdown(f"<div class='dp-sub'><b>{esc(r.model_name)}</b> {display_score(pm, r.base_mean):.4f}: defaults kept. "
                                        "The tuned settings did not beat them on rows it had not tuned on.</div>", unsafe_allow_html=True)
                        else:
                            col = "var(--accent)" if r.improvement >= 0 else "var(--danger)"
                            params = ", ".join(f"{k.replace('model__', '')}={v}" for k, v in r.best_params.items())
                            st.markdown(f"<div class='dp-sub'><b>{esc(r.model_name)}</b> {display_score(pm, r.base_mean):.4f} -> <b>{display_score(pm, r.tuned_mean):.4f}</b> "
                                        f"<span style='color:{col}'>({r.improvement:+.4f})</span> "
                                        f"<span class='mono' style='font-size:12px'>{esc(params)}</span></div>", unsafe_allow_html=True)

        # ---------------------------------------------- data preparation (advanced) ---
        with st.expander("Advanced: how the data was prepared, and settings to change"):
            groups = ps["groups"]
            fit_by = {r["name"]: r for r in ps["fit_results"]}
            st.markdown(
                f"<div class='dp-sub'>{ps['n_train']:,} training rows | {ps['n_test']:,} test rows | "
                f"{len(groups.numeric)} numeric, {len(groups.categorical_low) + len(groups.categorical_high)} categorical, "
                f"{len(groups.datetime)} date, {len(groups.free_text)} text columns</div>", unsafe_allow_html=True)
            for n in (ps.get("target_note"), ps.get("stratify_note"),
                      f"{ps['n_dropped_duplicates']:,} duplicate rows dropped before splitting." if ps.get("n_dropped_duplicates") else "",
                      f"{ps['n_dropped_missing_target']} rows dropped for a missing target." if ps.get("n_dropped_missing_target") else ""):
                if n:
                    st.markdown(f"<div class='dp-note'>{esc(n)}</div>", unsafe_allow_html=True)
            labels = {"preprocess": f"preprocess: {len(groups.numeric)} numeric, {len(groups.categorical_low)} low-card, "
                                    f"{len(groups.categorical_high)} high-card, {len(groups.datetime)} date",
                      "select": "select: SelectKBest", "pca": f"pca: keeps {ps['options'].pca_variance:.0%} variance"}
            for pcol, cand in zip(st.columns(len(ps["candidates"])), ps["candidates"]):
                fit = fit_by.get(cand.name, {})
                steps = "".join(f"<li>{esc(labels.get(n, n))}</li>" for n, _ in cand.pipeline.steps)
                decisions = "".join(
                    f"<div class='dp-decision'><div class='row'><b>What</b> {esc(dd.what)}</div>"
                    f"<div class='row'><b>Why</b> {esc(dd.why)}</div></div>" for dd in cand.decisions)
                ok = fit.get("ok")
                fit_html = (f"<div class='dp-decision'><div class='row'><b>Check</b> <span style='color:var(--ok)'>"
                            f"fits, {fit['n_features']} features</span></div></div>" if ok else
                            f"<div class='dp-decision'><div class='row'><b>Check</b> <span style='color:var(--danger)'>"
                            f"failed: {esc(str(fit.get('error', '')))}</span></div></div>")
                pcol.markdown(f"<div class='dp-pipe-card'><div class='pipe-name'>{esc(prep_label(cand.name))}</div>"
                              f"<div class='pipe-desc'>{esc(cand.description)}</div><ul class='pipe-steps'>{steps}</ul>"
                              f"{decisions}{fit_html}</div>", unsafe_allow_html=True)
            with st.expander("Change settings"):
                o = ps["options"]
                ec = st.columns(3)
                ec[0].multiselect("Data preparation options to try", ["conservative", "balanced", "aggressive"],
                                  default=[c.name for c in ps["candidates"]], key="opt_pipelines", format_func=prep_label)
                ec[0].selectbox("Numeric imputation", ["median", "mean", "most_frequent"],
                                index=["median", "mean", "most_frequent"].index(o.numeric_impute), key="opt_impute")
                ec[0].selectbox("Missingness indicators", ["follow preference", "yes", "no"], key="opt_missing")
                ec[1].checkbox("Skew correction", value=o.skew_correction, key="opt_skew")
                ec[1].checkbox("Interaction features", value=o.interactions, key="opt_interact")
                ec[1].checkbox("Mine names, tickets, codes", value=o.string_shape, key="opt_shape",
                               help="Pulls titles, prefixes and lengths out of code-like text columns.")
                ec[1].selectbox("Clip outliers", ["auto", "yes", "no"], index=["auto", "yes", "no"].index(o.clip_outliers), key="opt_clip")
                ec[2].slider("PCA variance kept", 0.80, 0.99, float(o.pca_variance), 0.01, key="opt_pca")
                ec[2].number_input("Text vocabulary (0 = default)", 0, 2000, int(o.tfidf_max_features or 0), 50, key="opt_tfidf")
                if st.button("Apply and rebuild + run", key="rebuild_btn"):
                    o2 = options_from_widgets()
                    if o2 is not None:
                        fast_run(cfg_now(o2))

        # ---------------------------------------------------------- Program ---
        with tabs[3]:
            if not fin or not fin["result"].ok:
                st.caption("Finish a run to open its program.")
            else:
                d = derived()
                if "files" not in d:
                    if "zip" not in d:
                        r_ = fin["result"]
                        d["zip"] = build_project_zip(
                            target=ps["target"], task_type_value=task_enum.value, priority_value=ps["priority"],
                            preserve_missingness=ps["prefs"].preserve_meaningful_missingness, options=ps["options"],
                            pipeline_name=r_.pipeline_name, model_name=fin["base_model_name"],
                            best_params=fin["best_params"], uses_smote=fin["uses_smote"],
                            feature_columns=fin["feature_columns"], fitted_pipeline=fin["final_pipeline"],
                            label_encoder=ps.get("label_encoder"), log_target=bool(ps.get("log_target")),
                        test_csv=test_csv_bytes)
                    d["files"] = load_project(d["zip"])
                    d["orig"] = dict(d["files"])
                files, orig = d["files"], d["orig"]
                st.markdown("<div class='dp-sub'>Your whole program is one file, <b>train.py</b>. "
                            "Pick a step, change it, run it.</div>", unsafe_allow_html=True)
                head, blocks = split_sections(files["train.py"])
                labels = {t: t.split(" (")[0] for t, _ in blocks}
                titles = [t for t, _ in blocks]
                pick = st.radio("Step", ["Whole file", *titles], horizontal=True, key="wb_sec",
                                format_func=lambda t: labels.get(t, t), label_visibility="collapsed")
                if pick != "Whole file":
                    st.caption(SECTION_HELP.get(pick.split(".")[0], ""))
                c_edit, c_reset = st.columns([1, 1])
                edit_on = c_edit.toggle("Edit", key="wb_edit_on")
                if c_reset.button("Undo all edits", key="wb_reset"):
                    d["files"] = dict(orig)
                    for k_ in [k_ for k_ in ss if str(k_).startswith("wb_edit_")]:
                        ss.pop(k_, None)
                    st.rerun()
                if pick == "Whole file":
                    shown = files["train.py"]
                else:
                    shown = dict(blocks)[pick]
                if edit_on:
                    new = st.text_area("Edit " + pick, shown, height=420, key=f"wb_edit_{pick}", label_visibility="collapsed")
                    if new != shown:
                        if pick == "Whole file":
                            files["train.py"] = new
                        else:
                            files["train.py"] = join_sections(head, [(t, new if t == pick else b_) for t, b_ in blocks])
                else:
                    st.code(shown, language="python", line_numbers=True, height=420)
                a1, a2, a3, a4 = st.columns([1.2, 1.2, 1.4, 2])
                data_bytes = df.to_csv(index=False).encode()
                run_which = None
                if a1.button("Run program", type="primary", key="wb_run_train"):
                    run_which = "train.py"
                if a2.button("Run predict.py", key="wb_run_pred"):
                    run_which = "predict.py"
                a3.download_button("Download project", dump_project(files), "datapilot_project.zip", "application/zip")
                chg = changed_files(orig, files)
                a4.markdown(f"<div class='dp-sub' style='padding-top:8px'>{len(chg)} file(s) edited" +
                            (": " + ", ".join(chg) if chg else "") + "</div>", unsafe_allow_html=True)
                if run_which:
                    with st.spinner(f"Running {run_which} on your data (a retrain can take a while)..."):
                        d["run"] = (run_which, *run_script(files, run_which, data_bytes))
                if d.get("run"):
                    which, code, out, err = d["run"]
                    (st.success if code == 0 else st.error)(f"{which} {'finished' if code == 0 else f'exited with code {code}'}")
                    if out:
                        st.code(out[-6000:], language="text")
                    if err and code != 0:
                        st.code(err[-6000:], language="text")
                with st.expander("Other project files (advanced)"):
                    others = [n for n in ordered_names(files) if n != "train.py"]
                    pick2 = st.selectbox("File", others, key="wb_file",
                                         format_func=lambda n: n + ("  (binary)" if isinstance(files[n], bytes) else ""))
                    c2 = files[pick2]
                    if isinstance(c2, bytes):
                        st.caption(f"{len(c2):,} bytes. Binary file; not shown.")
                    else:
                        st.code(c2, language="python" if pick2.endswith(".py") else "markdown", line_numbers=True, height=300)
                        st.caption("To change a file here, edit it after downloading the project. "
                                   "Library files (datapilot/) are the exact code the run used.")
                with st.expander("Model structure"):
                    try:
                        from sklearn.utils import estimator_html_repr
                        _show_html(estimator_html_repr(fin["final_pipeline"]), 380)
                    except Exception as e:  # noqa: BLE001
                        st.caption(f"Diagram not available: {e}")
                    st.code(repr(fin["final_pipeline"]), language="text")

        # ------------------------------------------------------ Competition ---
        with tabs[4]:
            if not fin or not fin["result"].ok:
                st.caption("Finish a run first. This part is optional.")
            else:
                st.markdown("<div class='dp-sub'>Kaggle-style files: add the competition's test file, get a submission, "
                            "and a script you can change.</div>", unsafe_allow_html=True)
                opts_ = SELECTABLE_PRIMARY[task_enum]
                cur = (lb or {}).get("primary") or ("f1" if task_enum == TaskType.CLASSIFICATION else "r2")
                k1, k2 = st.columns([1.2, 2])
                metric = k1.selectbox("Competition metric", opts_, index=opts_.index(cur) if cur in opts_ else 0, key="kg_metric")
                k2.markdown("<div style='height:28px'></div>", unsafe_allow_html=True)
                if metric != cur and k2.button(f"Re-run, ranking by {metric}", key="kg_rerun"):
                    fast_run(cfg_from_state(ps, metric))
                elif metric == cur:
                    k2.markdown(f"<div class='dp-sub' style='padding-top:8px'>This run already ranks by {cur}.</div>", unsafe_allow_html=True)
                u1, u2 = st.columns(2)
                if test_uploaded is not None:
                    test_up = test_uploaded
                    u1.markdown(f"<div class='dp-sub' style='padding-top:8px'>Using the test file you added at the start: <b>{esc(test_uploaded.name)}</b></div>",
                                unsafe_allow_html=True)
                else:
                    test_up = u1.file_uploader("test file (CSV or Excel)", type=["csv", "xlsx", "xls"], key="kg_test")
                samp_up = u2.file_uploader("sample submission (optional)", type=["csv", "xlsx", "xls"], key="kg_sample")
                all_rows_already = ps.get("score_mode") == "cv"
                use_all = False if all_rows_already else st.checkbox(
                    "Train on all rows first (recommended for a submission)", value=True, key="kg_all",
                    help="The score is already measured, so the final model can learn from every labelled row.")
                if all_rows_already:
                    st.caption("The model already learned from every training row, so predictions come straight from it.")
                if test_up is not None:
                    try:
                        sub_test_df = test_df if (test_uploaded is not None and test_df is not None) else read_any(test_up)
                        samp_df = read_any(samp_up) if samp_up is not None else None
                        d = derived()
                        key_ = (use_all, test_up.name, getattr(test_up, "size", 0))
                        if d.get("kg_key") != key_:
                            with st.spinner("Training on all rows and predicting..." if use_all else "Predicting..."):
                                model_ = (refit_on_all_rows(fin["final_pipeline"], ps["X_train"], ps["y_train"],
                                                            ps["X_test"], ps["y_test"]) if use_all else fin["final_pipeline"])
                                d["kg_sub"] = make_submission(model_, sub_test_df, samp_df, fin["feature_columns"],
                                                              ps.get("label_encoder"))
                                d["kg_key"] = key_
                        sub_df, sub_note = d["kg_sub"]
                        st.success(f"{len(sub_df):,} predictions. {sub_note}")
                        for level, msg in check_submission(sub_df, samp_df, sub_test_df):
                            {"ok": st.caption, "warn": st.warning, "error": st.error}[level](msg)
                        st.dataframe(sub_df.head(8), width="stretch", hide_index=True)
                        st.download_button("Download submission.csv", sub_df.to_csv(index=False).encode(),
                                           "submission.csv", "text/csv", type="primary")
                        st.markdown("<div class='dp-sub' style='padding-top:10px'>Want to change how it is made? "
                                    "<b>make_submission.py</b> is in your project. Edit it in the box below and run it "
                                    "on these same files.</div>", unsafe_allow_html=True)
                        pf = derived().get("files")
                        if pf and "make_submission.py" in pf:
                            with st.expander("make_submission.py"):
                                pf["make_submission.py"] = st.text_area(
                                    "make_submission.py", pf["make_submission.py"], height=380,
                                    key="kg_script", label_visibility="collapsed")
                                if st.button("Run this script", key="kg_run_script"):
                                    extra = {"test.csv": sub_test_df.to_csv(index=False).encode()}
                                    if samp_up is not None:
                                        extra["sample_submission.csv"] = samp_df.to_csv(index=False).encode()
                                    with st.spinner("Running..."):
                                        d["kg_run"] = run_script_files(pf, "make_submission.py", df.to_csv(index=False).encode(),
                                                                       extra=extra, collect=("submission.csv",))
                                if d.get("kg_run"):
                                    code, out, err, got = d["kg_run"]
                                    (st.success if code == 0 else st.error)("Finished" if code == 0 else f"Exited with code {code}")
                                    st.code((out or "")[-3000:] + ("" if code == 0 else "\n" + (err or "")[-3000:]), language="text")
                                    if "submission.csv" in got:
                                        st.download_button("Download this submission.csv", got["submission.csv"],
                                                           "submission_edited.csv", "text/csv", key="kg_dl2")
                        else:
                            st.caption("Open the Program tab once to prepare the script, then come back.")
                    except Exception as e:  # noqa: BLE001
                        st.error(f"Could not build the submission: {e}")
                st.caption("DataPilot cannot submit or see leaderboards; scores here are local only.")

# ---------------------------------------------------------------- footer ----
if ss.runs:
    with st.expander(f"This session ({len(ss.runs)} dataset{'s' if len(ss.runs) != 1 else ''})"):
        rdf = pd.DataFrame(ss.runs)
        for c_ in ("target", "task", "best_model", "best_score", "took", "accuracy"):
            if c_ not in rdf:
                rdf[c_] = "-"
        st.dataframe(rdf[["name", "rows", "cols", "target", "task", "best_model", "best_score", "took", "accuracy"]].fillna("-"),
                     width="stretch", hide_index=True)
st.markdown('<div class="dp-footer"><span>DataPilot</span><span>Runs on your machine</span></div>', unsafe_allow_html=True)

if current_stage() != _rendered_stage:
    st.rerun()  # redraw the hero progress rail
