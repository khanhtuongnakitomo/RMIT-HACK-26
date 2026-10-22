# SELFVAL — offline scoring of Sub 1/2/3 without spending submissions

> **Audience:** whoever is free while the three fine-tunes run (CPU only, laptop is fine).
> **Team:** UTS · Saigon South (SGS) · InnoLab. **Deadline:** Kaggle closes **22 Oct 2026, 18:00 UTC+7**.
> **Inputs:** the `valid_probs.csv` + `test_probs.csv` that [Sub 1](SUB1_wolf_defender.md), [Sub 2](SUB2_xlmr_large.md) and [Sub 3](SUB3_granite_lora.md) already save (§6 of each plan). No retraining, no GPU, no Kaggle submission.
> **Runtime target:** < 5 min per invocation; re-run each time a new run lands.

---

## 0. Intention — why this test, and what it decides

We have **6 submissions** and three candidate models. The public leaderboard scores only ≈ 30% of test
(≈ 1,800 rows, ≈ 126 `PROMPT_INJECTION` rows), so one flipped injection prediction moves macro F1 by
≈ 0.003 and two runs within ≈ 0.01 of each other cannot be told apart there. Fold 0 has **2,800 labelled
rows (196 injection)** — larger than the public LB — and every run already predicts it under the same contract.
Scoring those predictions properly answers the questions the leaderboard answers badly:

| Question | Why it matters | Test in this plan |
|---|---|---|
| Q1. Is each run **really** above the floor, not just by luck? | The 0.93 gate in the Sub plans is a point estimate; a run at 0.935 ± 0.009 is not clearly better than TF-IDF. | §3.2 bootstrap 95% CI vs a TF-IDF baseline scored on **the same fold 0** |
| Q2. Is run A **really** better than run B? | Picks which run becomes `UTS_SGS_InnoLab_C1` and which runs deserve Sub 4 (5-fold) GPU time. | §3.3 paired bootstrap of the macro-F1 difference |
| Q3. Will Sub 5 threshold / class-bias tuning help **on unseen rows**? | Tuning on fold 0 and scoring on fold 0 always "improves" — that number is meaningless. | §3.4 cross-fitted bias tuning: tune on one half of fold 0, score on the other |
| Q4. What public-LB score should we **expect**? | When a real LB score arrives, we can tell "noise" from "broken test pipeline". | §3.5 simulate 1,800-row LB subsets of fold 0 |
| Q5. Is the **test** prediction file sane? | Fold-0 F1 cannot catch a test-only bug (row order, truncation, tokenizer, wrong checkpoint). | §3.6 test-vs-valid distribution and cross-model agreement checks |

**Outcome:** a short report with a ranked shortlist and explicit go / no-go rules (§4), so submissions are spent
on runs that self-validation already supports, and the leaderboard is used as confirmation rather than as the
validation set.

## 1. Inputs and layout

Put downloaded run folders under `outputs/` (git-ignored):

```
outputs/
  sub1_wolfdefender_f0/   valid_probs.csv  test_probs.csv  metrics.json
  sub2_xlmrlarge_f0/      valid_probs.csv  test_probs.csv  metrics.json
  sub3_granite_f0/        valid_probs.csv  test_probs.csv  metrics.json
data/
  train.csv  test.csv  folds.csv        # competition CSVs are git-ignored; folds.csv is in the repo
```

File formats (from the Sub plans, §6): `valid_probs.csv` = `id, label, p_BENIGN, p_JAILBREAK, p_PROMPT_INJECTION`
(2,800 rows, fold 0); `test_probs.csv` = `id, p_BENIGN, p_JAILBREAK, p_PROMPT_INJECTION` (6,000 rows, test.csv order).
The harness must work with **any subset** of the three runs present.

## 2. Implementation

One script, `scripts/self_eval.py`, run with the same tooling as the README:

```powershell
uv run --no-project --python 3.11 --with pandas --with scikit-learn --with scipy python scripts/self_eval.py `
  --data data --runs outputs/sub1_wolfdefender_f0 outputs/sub2_xlmrlarge_f0 outputs/sub3_granite_f0 `
  --out outputs/selfval
```

Fixed settings: `seed=42`, `B=2000` bootstrap resamples, `LB_ROWS=1800`, `LB_DRAWS=2000`, bias grid
`-2.0 … 2.0` step `0.1`, cross-fit `5 repeats × 2 halves`.

## 3. Tests

### 3.1 Load and integrity checks (abort on any failure)

```python
import hashlib, numpy as np, pandas as pd
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold

LABELS = ["BENIGN", "JAILBREAK", "PROMPT_INJECTION"]; PCOLS = [f"p_{l}" for l in LABELS]
FOLD0_SHA = "0d043c7efa1612b707d4a52e4b73a0aa311ee37fb3f9cadf213ec47269a57fc4"
read = lambda p: pd.read_csv(p, dtype=str, keep_default_na=False, encoding="utf-8")
rng = np.random.default_rng(42)

train, test, folds = read(f"{DATA}/train.csv"), read(f"{DATA}/test.csv"), read(f"{DATA}/folds.csv")
folds["fold"] = folds["fold"].astype(int)
va = (train.merge(folds, on="id", validate="one_to_one")
           .query("fold == 0").sort_values("id").reset_index(drop=True))
assert len(va) == 2800 and hashlib.sha256(",".join(va.id).encode()).hexdigest() == FOLD0_SHA
y = va.label.map({l: i for i, l in enumerate(LABELS)}).values

def load_run(d):
    v, t = read(f"{d}/valid_probs.csv"), read(f"{d}/test_probs.csv")
    assert len(v) == 2800 and set(v.id) == set(va.id), "valid ids differ from fold 0"
    v = v.set_index("id").loc[va.id]
    assert (v.label.values == va.label.values).all(), "valid labels differ from train.csv"
    assert t.id.tolist() == test.id.tolist(), "test_probs not in test.csv order"
    pv, pt = v[PCOLS].astype(float).values, t[PCOLS].astype(float).values
    for p in (pv, pt):
        assert np.isfinite(p).all() and np.allclose(p.sum(1), 1, atol=1e-3), "probs do not sum to 1"
    return pv, pt
```

Also add a **TF-IDF reference run** scored on the same fold 0: train the `inspect2.py` setup (char_wb 2–5 +
word 1–2 TF-IDF, `LogisticRegression(C=10, max_iter=3000, class_weight="balanced")`) on folds 1–4, predict fold 0
and test, and treat it as run `tfidf`. The published 0.928 is a 5-fold average, not a fold-0 number, so it is not
a fair floor on its own.

### 3.2 Point score + bootstrap CI per run (Q1)

```python
def mf1(y, yhat):
    cm = np.bincount(3 * y + yhat, minlength=9).reshape(3, 3)
    tp, denom = np.diag(cm), cm.sum(0) + cm.sum(1)          # F1_c = 2TP / (2TP + FP + FN)
    return float(np.mean(np.where(denom > 0, 2 * tp / np.maximum(denom, 1), 0.0)))

assert abs(mf1(y, y[::-1].copy()) - f1_score(y, y[::-1], average="macro")) < 1e-12   # self-check

BOOT = rng.integers(0, len(y), size=(B, len(y)))          # shared by all runs -> paired
boot = lambda yhat: np.array([mf1(y[i], yhat[i]) for i in BOOT])
```

Report per run: macro F1, 95% CI (2.5/97.5 percentiles of `boot`), per-class P/R/F1, confusion matrix,
per-language macro F1 with `n`, and the gap to the `tfidf` run.

### 3.3 Paired comparison between runs (Q2)

For every pair (A, B), using the **same** `BOOT` indices: `d = boot(yhat_A) - boot(yhat_B)`.
Report mean `d`, 95% CI, and `P(d > 0)`. Also report a per-class version (injection F1 difference), since
injection decides most of the macro-F1 gap.

### 3.4 Cross-fitted class-bias tuning (Q3)

Sub 5 adds a bias vector `b = [0, b_JB, b_INJ]` to the log-probabilities before `argmax`. Estimate its real
gain without scoring on the rows it was tuned on:

```python
GRID = np.round(np.arange(-2.0, 2.01, 0.1), 2)

def fit_bias(lp, yy):
    best_s, best_b = mf1(yy, lp.argmax(1)), np.zeros(3)       # start at "no change"
    for bj in GRID:
        for bi in GRID:
            b = np.array([0.0, bj, bi]); s = mf1(yy, (lp + b).argmax(1))
            if s > best_s + 1e-6: best_s, best_b = s, b
    return best_b

def crossfit_gain(p, repeats=5):
    lp, strata, gains = np.log(np.clip(p, 1e-9, 1)), (va.language + "__" + va.label).values, []
    for r in range(repeats):
        for a, b in StratifiedKFold(2, shuffle=True, random_state=r).split(lp, strata):
            bias = fit_bias(lp[a], y[a])
            gains.append(mf1(y[b], (lp[b] + bias).argmax(1)) - mf1(y[b], lp[b].argmax(1)))
    return np.array(gains)                                    # 10 held-out gains
```

Report mean gain, min/max, number of positive halves, and the biases chosen in each half (stable biases are a
good sign; biases that jump across the grid mean the tuning is fitting noise). If tuning is adopted (§4), fit
the final bias on all of fold 0, apply it to `test_probs`, and write `outputs/selfval/submission_<run>_tuned.csv`
(validate with `scripts/validate_submission.py`).

### 3.5 Expected public-LB band (Q4)

Draw 2,000 random 1,800-row subsets of fold 0 without replacement and score each run's argmax predictions.
Report the 5th / 50th / 95th percentiles. This is the range of public scores the run would produce from
**sampling noise alone**, assuming test looks like train (verified in
[`dataset_findings_and_approach.md`](../docs/dataset_findings_and_approach.md) §1, findings 2–3).

### 3.6 Test-side sanity checks (Q5)

| Check | Flag if |
|---|---|
| Predicted class shares on test vs fold 0 (train prior ≈ 0.62 / 0.31 / 0.07) | any class differs by > 3 percentage points |
| Mean max-probability on test vs fold 0 | test lower by > 0.03 (a confidence drop suggests a test-only preprocessing difference) |
| Pairwise argmax agreement between runs on test vs on fold 0 | agreement on test lower by > 3 percentage points |
| Agreement of each run with `tfidf` on test vs on fold 0 | same rule; catches a single broken run even if only one run is present |

## 4. Decision rules (written into the report)

1. **Floor:** a run is submission-worthy only if the lower bound of its 95% CI is above the `tfidf` fold-0 point
   score, all §3.1 checks pass, and no §3.6 flag fires.
2. **Ranking:** A beats B only if the paired 95% CI of `d` excludes 0. Otherwise treat them as **tied** and
   prefer the run that can be reproduced inside a Kaggle notebook (Sub 3 depends on the open Colab rules question),
   then the one with higher injection F1.
3. **Sub 5 tuning:** adopt only if the mean cross-fit gain is > 0 **and** at least 8 of the 10 halves are positive.
   After adoption, the tuned model's fold-0 score is no longer an honest estimate — quote the cross-fit gain instead.
4. **LB check:** when a public score arrives, it should fall inside the §3.5 band. Below the band → inspect the test
   pipeline (row order, truncation length, tokenizer, loaded checkpoint) before changing the model. Above the band →
   accept it, but do not rank runs by it.
5. **Sub 4 GPU time** goes to the top-ranked run, or to the reproducible one if the top two are tied.

## 5. Outputs

| File | Content |
|---|---|
| `outputs/selfval/report.md` | One table per question (§3.2–3.6), flags, and the §4 decisions in plain sentences |
| `outputs/selfval/summary.json` | All numbers above, plus runs found, seed, `B`, grid, and input file hashes |
| `outputs/selfval/submission_<run>_tuned.csv` | Only if rule 3 adopts tuning for that run |

Paste the headline numbers (macro F1 + CI, cross-fit gain, LB band) into
[`experiments/EXPERIMENT_LOG.md`](../experiments/EXPERIMENT_LOG.md) next to each run.

## 6. Known limits

- Fold 0 was also used to **pick the best checkpoint** (≈ 4–6 evaluations per run), so every fold-0 score is
  slightly optimistic. The bias is similar for all three runs, so the ranking is fairer than the absolute numbers.
- 196 injection rows make injection F1 noisy (one row ≈ 0.005 injection F1); the CIs show this — do not read
  past them.
- The harness detects test-pipeline bugs only through proxies (§3.6); it cannot detect a genuine train/test shift
  that the EDA did not see. The private-LB split is unknown.
- Averaging probabilities **across different backbones** is not tested here: the rules allow one backbone per
  notebook (`docs/agent.md`).

## 7. Do-not list

- Do not tune anything on the public LB or on `test_probs`.
- Do not report a bias-tuned fold-0 score as a validation result.
- Do not change `data/folds.csv` or the fold-0 hash check.
- Do not commit competition CSVs, run outputs, or generated submissions.
