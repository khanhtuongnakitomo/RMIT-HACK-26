# Experiment log — C1

All runs use the shared contract in the plan files (fold 0 of `data/folds.csv`, raw text, head+tail truncation,
sqrt-inverse-frequency class weights, best checkpoint by fold-0 macro F1, argmax predictions).

## Results

| Run | Model | Max len | LR | Best step | Val macro F1 | F1 BEN | F1 JB | F1 INJ | pt / fr / it / en | Train min | Public LB | Decision |
|---|---|---:|---|---:|---:|---:|---:|---:|---|---:|---:|---|
| tfidf | char+word TF-IDF + LR (balanced), 5-fold CV — reference only, not submitted | — | — | — | 0.928 | 0.951 | 0.919 | 0.914 | 0.910 / 0.952 / 0.941 / 0.922 | — | — | floor to beat |
| sub1 | wolf-defender-prompt-injection (UTS_SGS_InnoLab_C1 v4) | | 3e-5 | | | | | | | | 0.96557 | Kaggle attempt 1 |
| sub2 | xlm-roberta-large | 512 | 1e-5 | | | | | | | | | Kaggle attempt 2 (score unlogged) |
| sub3 | granite-3.3-2b-instruct + LoRA r=16 (Colab A100) | 1024 | 1e-4 | 700 | 0.9589 | 0.9765 | 0.9600 | 0.9403 | not recomputed | 23.0 | 0.96808 | Kaggle attempt 3; gate pass |
| sub4 | wolf-defender-prompt-injection (adaptive-K + 100% full-data single model, Kaggle T4 ×2) | 1024 | 3e-5 | | | | | | | | | Ready (`plans/SUB4_wolf_adaptive_full.md`) |

## Submission ledger (6 attempts — reset period unconfirmed; NO ENSEMBLES allowed)

| # | Time (UTC+7) | Run | File | Val macro F1 | Public LB | Selected for final? |
|---|---|---|---|---:|---:|---|
| 1 | ~13:38 | sub1 wolf-defender-prompt-injection | `UTS_SGS_InnoLab_C1` Version 4 | | 0.96557 | |
| 2 | | sub2 | | | | |
| 3 | ~14:12 | sub3 Granite-3.3-2B LoRA fold 0 | `submission.csv` | 0.9589 | 0.96808 | |
| 4 | | | | | | |
| 5 | | | | | | |
| 6 | | | | | | |

Note: the public LB uses ≈30% of test (≈1,800 rows, ≈126 injection rows) — one flipped injection prediction moves
macro F1 by ≈0.003. Prefer validation evidence for small differences.
