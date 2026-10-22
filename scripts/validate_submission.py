"""Structural check for a C1 submission (agent.md section 20.3). Does NOT measure quality.

Usage:
    python scripts/validate_submission.py --test ../test.csv --sub submission.csv
"""
import argparse

import pandas as pd

ALLOWED = {"BENIGN", "JAILBREAK", "PROMPT_INJECTION"}

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", default="../test.csv")
    ap.add_argument("--sub", default="submission.csv")
    a = ap.parse_args()
    test = pd.read_csv(a.test, dtype={"id": str}, encoding="utf-8")
    sub = pd.read_csv(a.sub, dtype={"id": str}, encoding="utf-8")
    assert sub.columns.tolist() == ["id", "label"], sub.columns.tolist()
    assert len(sub) == len(test) == 6000, (len(sub), len(test))
    assert not sub[["id", "label"]].isna().any().any()
    assert sub["id"].is_unique
    assert set(sub["id"]) == set(test["id"])
    assert sub["label"].isin(ALLOWED).all(), sub.loc[~sub["label"].isin(ALLOWED), "label"].unique()
    print("Submission structure passed; prediction quality is not checked.")
    print(sub["label"].value_counts(normalize=True).round(3).to_string())
