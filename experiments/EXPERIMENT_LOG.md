# Experiment log — C1

All runs use the shared contract in the plan files (fold 0 of `data/folds.csv`, raw text, head+tail truncation,
sqrt-inverse-frequency class weights, best checkpoint by fold-0 macro F1, argmax predictions).

## Results

| Run | Model | Max len | LR | Best step | Val macro F1 | F1 BEN | F1 JB | F1 INJ | pt / fr / it / en | Train min | Public LB | Decision |
|---|---|---:|---|---:|---:|---:|---:|---:|---|---:|---:|---|
| tfidf | char+word TF-IDF + LR (balanced), 5-fold CV — reference only, not submitted | — | — | — | 0.928 | 0.951 | 0.919 | 0.914 | 0.910 / 0.952 / 0.941 / 0.922 | — | — | floor to beat |
| sub1 | wolf-defender-prompt-injection (or mmBERT-base fallback) | | 3e-5 | | | | | | | | | |
| sub2 | xlm-roberta-large | 512 | 1e-5 | | | | | | | | | |

## Submission ledger (6 attempts — reset period unconfirmed)

| # | Time (UTC+7) | Run | File | Val macro F1 | Public LB | Selected for final? |
|---|---|---|---|---:|---:|---|
| 1 | | | | | | |
| 2 | | | | | | |
| 3 | | | | | | |
| 4 | | | | | | |
| 5 | | | | | | |
| 6 | | | | | | |

Note: the public LB uses ≈30% of test (≈1,800 rows, ≈126 injection rows) — one flipped injection prediction moves
macro F1 by ≈0.003. Prefer validation evidence for small differences.
