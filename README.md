# RMIT-HACK-26 — Team InnoLab (UTS · SGS)

RMIT Hackathon 2026, Challenge 1 (Kaggle `rmit-hackathon-2026`): multilingual 3-class prompt-attack
classification (`BENIGN` / `JAILBREAK` / `PROMPT_INJECTION`), metric **macro F1**.
Kaggle deadline: **22 Oct 2026, 18:00 UTC+7**. Final notebook name: **`UTS_SGS_InnoLab_C1`**.

## Layout

| Path | What |
|---|---|
| [`docs/agent.md`](docs/agent.md) | Competition briefing, rules, constraints, open questions |
| [`docs/dataset_findings_and_approach.md`](docs/dataset_findings_and_approach.md) | Data audit, TF-IDF baseline (0.928 CV macro F1), model choice, overall plan |
| [`plans/SUB1_wolf_defender.md`](plans/SUB1_wolf_defender.md) | Hand-off spec: Sub 1 — `wolf-defender-prompt-injection`, fold 0 |
| [`plans/SUB2_xlmr_large.md`](plans/SUB2_xlmr_large.md) | Hand-off spec: Sub 2 — `xlm-roberta-large`, fold 0 |
| [`experiments/EXPERIMENT_LOG.md`](experiments/EXPERIMENT_LOG.md) | Shared results table + submission ledger |
| [`data/folds.csv`](data/folds.csv) | Fixed 5-fold split (`id,fold`), stratified on language × label, seed 42 |
| [`scripts/make_folds.py`](scripts/make_folds.py) | Regenerates `folds.csv` and prints the fold-0 fingerprint |
| [`scripts/validate_submission.py`](scripts/validate_submission.py) | Structural check of a submission CSV |
| [`scripts/eda/`](scripts/eda/) | Dataset inspection + TF-IDF baseline scripts |

## Data

Competition CSVs are **not** in this repo (licence CC BY-NC-SA 4.0 and competition rules restrict
redistribution). Put `train.csv`, `test.csv`, `sample_submission.csv` in the parent folder (`../`) locally,
or attach the competition dataset on Kaggle.

Fold-0 fingerprint: `sha256(",".join(sorted(fold0_ids)))` =
`0d043c7efa1612b707d4a52e4b73a0aa311ee37fb3f9cadf213ec47269a57fc4` (2,800 rows).

## Local quick commands

```powershell
uv run --no-project --python 3.11 --with pandas --with scikit-learn python scripts/make_folds.py --train ../train.csv --out data/folds.csv
uv run --no-project --python 3.11 --with pandas python scripts/validate_submission.py --test ../test.csv --sub submission.csv
```
