"""Single-file program export, section editing, and the competition submission flow."""
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

from datapilot import build_project_zip
from datapilot.reasoner import TaskType
from datapilot.runner import RunConfig, run_all
from datapilot.submission import check_submission, make_submission, refit_on_all_rows
from datapilot.transformers import StringShape
from datapilot.workbench import join_sections, load_project, split_sections


def _data(n=260, seed=3):
    rng = np.random.default_rng(seed)
    title = rng.choice(["Mr.", "Mrs.", "Miss.", "Dr."], n, p=[.5, .2, .25, .05])
    df = pd.DataFrame({
        "id": range(n),
        "name": [f"Surname{i}, {t} First{i}" for i, t in enumerate(title)],
        "ticket": [rng.choice(["PC ", "A/5 ", ""]) + str(rng.integers(1000, 99999)) for _ in range(n)],
        "age": rng.normal(35, 10, n),
        "city": rng.choice(list("abc"), n),
    })
    df["y"] = np.where((np.isin(title, ["Mrs.", "Miss."])) ^ (rng.random(n) < .1), "yes", "no")
    return df


@pytest.fixture(scope="module")
def run():
    df = _data()
    cfg = RunConfig("y", TaskType.CLASSIFICATION, effort="Quick")
    state, lb, tun, fin = run_all(df, cfg, "t")
    zbytes = build_project_zip(
        target="y", task_type_value="classification", priority_value=cfg.priority.value,
        preserve_missingness=True, options=state["options"], pipeline_name=fin["result"].pipeline_name,
        model_name=fin["base_model_name"], best_params=fin["best_params"], uses_smote=fin["uses_smote"],
        feature_columns=fin["feature_columns"], fitted_pipeline=fin["final_pipeline"],
        label_encoder=state.get("label_encoder"))
    return df, state, fin, load_project(zbytes)


def test_string_shape_finds_titles_and_prefixes():
    df = _data()
    t = StringShape().fit(df[["name", "ticket"]])
    names = list(t.get_feature_names_out())
    assert any("title=mr" in n for n in names) and any("prefix=PC" in n for n in names)
    assert t.transform(df[["name", "ticket"]]).shape == (len(df), len(names))
    # unseen values and missing values do not crash
    odd = pd.DataFrame({"name": [None, "x"], "ticket": ["??", None]})
    assert t.transform(odd).shape[1] == len(names)


def test_train_py_is_one_sectioned_file_and_runs(run, tmp_path):
    df, state, fin, files = run
    src = files["train.py"]
    head, blocks = split_sections(src)
    assert [t.split(".")[0] for t, _ in blocks] == [str(i) for i in range(1, 11)]
    assert join_sections(head, blocks) == src
    assert "preprocess = Pipeline(" in src and "..." not in src.split("# %% 5")[1].split("# %% 6")[0]
    csv = tmp_path / "d.csv"
    df.to_csv(csv, index=False)
    for name, content in files.items():
        p = tmp_path / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(content if isinstance(content, bytes) else content.encode())
    r = subprocess.run([sys.executable, "train.py", str(csv)], cwd=tmp_path, capture_output=True, text=True, timeout=300)
    assert r.returncode == 0, r.stderr[-800:]
    assert "accuracy" in r.stdout


def test_section_edit_changes_program(run, tmp_path):
    df, state, fin, files = run
    head, blocks = split_sections(files["train.py"])
    edited = join_sections(head, [(t, b.replace("TEST_SIZE = 0.2", "TEST_SIZE = 0.3") if t.startswith("2.") else b)
                                  for t, b in blocks])
    assert "TEST_SIZE = 0.3" in edited and edited != files["train.py"]


def test_make_submission_script_and_refit(run, tmp_path):
    df, state, fin, files = run
    assert "make_submission.py" in files
    test = _data(60, seed=9).drop(columns=["y"])
    sample = pd.DataFrame({"id": test["id"], "y": "no"})
    for name, content in files.items():
        p = tmp_path / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(content if isinstance(content, bytes) else content.encode())
    df.to_csv(tmp_path / "data.csv", index=False)
    test.to_csv(tmp_path / "test.csv", index=False)
    sample.to_csv(tmp_path / "sample_submission.csv", index=False)
    r = subprocess.run([sys.executable, "make_submission.py", "data.csv"], cwd=tmp_path, capture_output=True, text=True, timeout=300)
    assert r.returncode == 0, r.stderr[-800:]
    sub = pd.read_csv(tmp_path / "submission.csv")
    assert list(sub.columns) == ["id", "y"] and len(sub) == 60 and set(sub["y"]) <= {"yes", "no"}
    # same thing through the app's helpers
    model = refit_on_all_rows(fin["final_pipeline"], state["X_train"], state["y_train"], state["X_test"], state["y_test"])
    sub2, _ = make_submission(model, test, sample, fin["feature_columns"], state.get("label_encoder"))
    assert len(sub2) == 60 and all(lvl != "error" for lvl, _ in check_submission(sub2, sample, test))
