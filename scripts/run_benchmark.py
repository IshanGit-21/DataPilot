"""Run: python scripts/run_benchmark.py [--tune] [--online]
Writes reports/benchmark.md. Offline cases use scikit-learn's bundled data."""
import argparse
from pathlib import Path

import pandas as pd
from sklearn import datasets as skd

from datapilot.benchmark import run_case, benchmark_to_markdown

ONLINE = {
    "titanic": ("https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv", "Survived"),
}


def sk_case(loader, target):
    b = loader(as_frame=True)
    df = b.frame.copy()
    if "target" in df.columns and target != "target":
        df = df.rename(columns={"target": target})
    return df, target


def imbalanced():
    X, y = skd.make_classification(n_samples=3000, n_features=12, n_informative=5, weights=[0.96, 0.04],
                                   flip_y=0.01, random_state=0)
    df = pd.DataFrame(X, columns=[f"f{i}" for i in range(12)])
    df["fraud"] = y
    return df, "fraud"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tune", action="store_true")
    ap.add_argument("--online", action="store_true")
    a = ap.parse_args()
    cases = {
        "breast_cancer": sk_case(skd.load_breast_cancer, "malignant_flag"),
        "wine": sk_case(skd.load_wine, "cultivar"),
        "iris": sk_case(skd.load_iris, "species"),
        "diabetes": sk_case(skd.load_diabetes, "progression"),
        "synthetic_imbalanced_3000": imbalanced(),
    }
    if a.online:
        for name, (url, tgt) in ONLINE.items():
            cases[name] = (pd.read_csv(url), tgt)
    rows = []
    for name, (df, tgt) in cases.items():
        print("running", name, flush=True)
        rows.append(run_case(name, df, tgt, tune=a.tune))
    md = benchmark_to_markdown(rows)
    out = Path(__file__).resolve().parents[1] / "reports" / "benchmark.md"
    out.parent.mkdir(exist_ok=True)
    out.write_text(md)
    print(md)


if __name__ == "__main__":
    main()
