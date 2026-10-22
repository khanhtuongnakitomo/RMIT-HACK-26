"""Create the shared, fixed 5-fold split used by every C1 experiment.

Usage:
    python scripts/make_folds.py --train ../train.csv --out data/folds.csv

Split: StratifiedKFold(n_splits=5, shuffle=True, random_state=42) on language + "__" + label,
applied to train.csv in its original row order. Fold 0 is the 20% validation set for Sub 1 / Sub 2.
Prints a SHA-256 fingerprint of the sorted fold-0 ids so any re-generation can be verified.
"""
import argparse
import hashlib

import pandas as pd
from sklearn.model_selection import StratifiedKFold

N_SPLITS, SEED = 5, 42


def make_folds(train: pd.DataFrame) -> pd.DataFrame:
    strata = train["language"] + "__" + train["label"]
    fold = pd.Series(-1, index=train.index, dtype=int)
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    for k, (_, valid_idx) in enumerate(skf.split(train, strata)):
        fold.iloc[valid_idx] = k
    return pd.DataFrame({"id": train["id"], "fold": fold})


def fold_fingerprint(folds: pd.DataFrame, k: int = 0) -> str:
    ids = sorted(folds.loc[folds["fold"] == k, "id"])
    return hashlib.sha256(",".join(ids).encode("utf-8")).hexdigest()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", default="../train.csv")
    ap.add_argument("--out", default="data/folds.csv")
    a = ap.parse_args()
    train = pd.read_csv(a.train, dtype=str, keep_default_na=False, encoding="utf-8")
    folds = make_folds(train)
    assert (folds["fold"] >= 0).all()
    folds.to_csv(a.out, index=False)
    merged = train.merge(folds, on="id")
    print(pd.crosstab(merged["fold"], merged["label"]))
    print("fold-0 valid rows:", int((folds["fold"] == 0).sum()))
    print("fold-0 sha256:", fold_fingerprint(folds, 0))
