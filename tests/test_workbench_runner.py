"""Program workbench, one-shot runner, submission checks and the progress rail."""
import numpy as np
import pandas as pd
import pytest

from datapilot.live import rail_html
from datapilot.submission import check_submission
from datapilot.workbench import (
    changed_files,
    dump_project,
    load_project,
    ordered_names,
    run_script,
)


def _files():
    return {"train.py": "print('hi')\n", "predict.py": "import sys\nprint(sys.argv[1:])\n",
            "README.md": "# x\n", "datapilot/a.py": "A = 1\n", "model.joblib": b"\x00\x01"}


def test_dump_load_roundtrip_and_order():
    f = _files()
    back = load_project(dump_project(f))
    assert back == f
    assert ordered_names(back)[:2] == ["predict.py", "train.py"] or ordered_names(back)[:2] == ["train.py", "predict.py"]
    assert ordered_names(back)[-1] == "model.joblib"


def test_changed_files():
    a = _files(); b = dict(a); b["train.py"] = "print('x')\n"
    assert changed_files(a, b) == ["train.py"]


def test_run_script_runs_edit_and_passes_data():
    f = _files()
    code, out, _ = run_script(f, "train.py", None, timeout=60)
    assert code == 0 and "hi" in out
    f["train.py"] = "print('edited')\n"
    assert "edited" in run_script(f, "train.py", None, timeout=60)[1]
    code, out, _ = run_script(f, "predict.py", b"a,b\n1,2\n", timeout=60)
    assert code == 0 and "data.csv" in out


def test_run_script_reports_errors_timeout_and_blocks_escape():
    f = {"train.py": "raise SystemExit(3)\n"}
    assert run_script(f, "train.py", None, timeout=60)[0] == 3
    f = {"train.py": "import time; time.sleep(30)\n"}
    assert run_script(f, "train.py", None, timeout=1)[0] == -1
    with pytest.raises(ValueError):
        run_script({"train.py": "", "../evil.py": "x"}, "train.py", None)
    with pytest.raises(ValueError):
        run_script(_files(), "README.md", None)


def test_check_submission():
    test = pd.DataFrame({"id": [1, 2, 3], "x": [1, 2, 3]})
    sample = pd.DataFrame({"id": [1, 2, 3], "y": [0, 0, 0]})
    good = pd.DataFrame({"id": [1, 2, 3], "y": [1, 0, 1]})
    assert all(lvl == "ok" for lvl, _ in check_submission(good, sample, test))
    bad = pd.DataFrame({"id": [1, 2], "target": [1, np.nan]})
    levels = [lvl for lvl, _ in check_submission(bad, sample, test)]
    assert levels.count("error") >= 2


def test_rail_states():
    h = rail_html(2)
    assert h.count('class="done"') == 2 and h.count('class="active"') == 1 and h.count('class="pending"') == 1


def test_run_all_quick_end_to_end():
    from datapilot.reasoner import TaskType
    from datapilot.runner import RunConfig, run_all
    rng = np.random.default_rng(1)
    n = 240
    df = pd.DataFrame({"a": rng.normal(size=n), "b": rng.choice(list("xyz"), n)})
    df["y"] = np.where(df.a + rng.normal(0, .5, n) > 0, "p", "q")
    seen = []
    state, lb, tun, fin = run_all(df, RunConfig("y", TaskType.CLASSIFICATION, effort="Quick"),
                                  "t", lambda *a: seen.append(a[0]))
    assert fin["result"].error is None and {"build", "race", "tune", "final"} <= set(seen)
