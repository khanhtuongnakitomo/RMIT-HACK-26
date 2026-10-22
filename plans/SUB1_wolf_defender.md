# SUB 1 — `wolf-defender-prompt-injection`, fold 0 (Kaggle)

> **Audience:** the co-worker agent running this experiment on Kaggle.
> **Team:** UTS · Saigon South (SGS) · InnoLab. **Final C1 notebook name:** `UTS_SGS_InnoLab_C1` (only the winning pipeline gets this name).
> **Deadline:** Kaggle closes **22 Oct 2026, 18:00 UTC+7 (11:00 UTC)**. Target: this run finished and reported by **~11:30 UTC+7**.
> **Sister experiment:** [SUB2_xlmr_large.md](SUB2_xlmr_large.md) runs in parallel on another account with the *identical contract* below — do not change the contract, otherwise the two results are not comparable.

---

## 0. Goal in one paragraph

Fine-tune **one** approved backbone, `patronus-studio/wolf-defender-prompt-injection`, as a 3-class classifier (`BENIGN`, `JAILBREAK`, `PROMPT_INJECTION`) on folds 1–4 of the shared split, select the best checkpoint by **macro F1 on fold 0**, predict all 6,000 test rows, and save probabilities + metrics + a valid `submission.csv`. This is an *exploration* submission: it answers "does a multilingual encoder that was already tuned for prompt-injection detection, with long context, beat a bigger generic encoder (Sub 2)?"

## 1. Background you need (short)

- Task: multilingual (pt 45%, fr 25%, it 20%, en 10%) prompt-attack classification. Metric: **macro F1** — the 7% `PROMPT_INJECTION` class weighs as much as the 62% `BENIGN` class.
- Train 14,000 rows (`id,text,language,label`), test 6,000 (`id,text,language`). Files are clean UTF-8; newlines were stripped from all texts.
- Class counts: BENIGN 8,680 · JAILBREAK 4,340 · PROMPT_INJECTION 980. Every language has the same class ratio.
- Hardest confusion (from our TF-IDF analysis, 5-fold macro F1 **0.928**): BENIGN ↔ JAILBREAK (same role-play style, differ only by harmful intent), then INJECTION → BENIGN. **Your result must beat 0.93 to be worth submitting.**
- Rules: exactly **one** approved pretrained backbone per notebook; name it at the top of the notebook; no other pretrained models, no external data.
- Full context: [`docs/agent.md`](../docs/agent.md), [`docs/dataset_findings_and_approach.md`](../docs/dataset_findings_and_approach.md).

## 2. Kaggle setup

1. New notebook, working title **`InnoLab_sub1_wolfdefender_f0`**.
2. **Accelerator: GPU T4 ×2** (use one GPU — see §5). Internet: optional (only needed if `transformers` must be upgraded).
3. **Add inputs:**
   - Competition data: `rmit-hackathon-2026`.
   - Model: Kaggle Model `aisuko/wolf-defender-prompt-injection` → Transformers → default ([link](https://www.kaggle.com/models/aisuko/wolf-defender-prompt-injection/Transformers/default)).
   - Fold file: upload [`data/folds.csv`](../data/folds.csv) from this repo as a private Kaggle Dataset (e.g. `innolab-folds`) and attach it. If that is not possible, use the regeneration fallback in §4.
4. **Discover real paths** — do not hard-code version folders:
   ```python
   import glob
   print(glob.glob("/kaggle/input/**/train.csv", recursive=True))
   print(glob.glob("/kaggle/input/**/folds.csv", recursive=True))
   print([p for p in glob.glob("/kaggle/input/**/config.json", recursive=True) if "wolf" in p.lower()])
   ```
5. First cell (markdown) must state: **`Model: patronus-studio/wolf-defender-prompt-injection (single approved backbone)`**, seed, split, and hyperparameters.

## 3. Shared experiment contract (identical in Sub 1 and Sub 2 — do not change)

| Item | Value |
|---|---|
| Label map | `LABELS = ["BENIGN", "JAILBREAK", "PROMPT_INJECTION"]`, `label2id = {l: i ...}`, `id2label` inverse. Export **strings**. |
| Split | `data/folds.csv` (`id,fold`). **Train = folds 1–4 (11,200 rows), Valid = fold 0 (2,800 rows).** |
| Split check | Fold-0 count **2,800**; `sha256(",".join(sorted(fold0_ids)))` == `0d043c7efa1612b707d4a52e4b73a0aa311ee37fb3f9cadf213ec47269a57fc4`. Abort if it differs. |
| Input | raw `text` only, unmodified (no lowercasing, no cleaning, no language prefix). Read CSVs with `dtype=str, keep_default_na=False, encoding="utf-8"`. |
| Long inputs | head + tail truncation: keep first 75% / last 25% of the token budget (code in §4). |
| Loss | cross-entropy with class weights `w_c = sqrt(n_max / n_c)` computed on the **training folds** → ≈ `[1.00, 1.41, 2.98]`. |
| Seed | `transformers.set_seed(42)` |
| Selection | evaluate **twice per epoch**; `metric_for_best_model="macro_f1"`, `load_best_model_at_end=True`, `save_total_limit=1`. |
| Test prediction | plain `argmax` of softmax probabilities. **No threshold tuning** (that is Sub 5). |
| Gate | submit to Kaggle only if fold-0 macro F1 ≥ **0.93** and the validator passes. |

## 4. Reference code for the contract-critical parts

```python
import hashlib, json, time, numpy as np, pandas as pd, torch
from sklearn.metrics import f1_score, classification_report, confusion_matrix

LABELS = ["BENIGN", "JAILBREAK", "PROMPT_INJECTION"]
label2id = {l: i for i, l in enumerate(LABELS)}; id2label = {i: l for l, i in label2id.items()}
FOLD0_SHA = "0d043c7efa1612b707d4a52e4b73a0aa311ee37fb3f9cadf213ec47269a57fc4"

read = lambda p: pd.read_csv(p, dtype=str, keep_default_na=False, encoding="utf-8")
train, test, folds = read(TRAIN_PATH), read(TEST_PATH), read(FOLDS_PATH)
folds["fold"] = folds["fold"].astype(int)
train = train.merge(folds, on="id", how="left", validate="one_to_one")
assert train["fold"].notna().all()
fold0 = sorted(train.loc[train.fold == 0, "id"])
assert len(fold0) == 2800 and hashlib.sha256(",".join(fold0).encode()).hexdigest() == FOLD0_SHA
tr_df, va_df = train[train.fold != 0].reset_index(drop=True), train[train.fold == 0].reset_index(drop=True)

# Fallback ONLY if folds.csv cannot be attached (must reproduce the same hash):
# from sklearn.model_selection import StratifiedKFold
# skf = StratifiedKFold(5, shuffle=True, random_state=42)  # on train.csv in original row order
# for k, (_, vi) in enumerate(skf.split(train, train.language + "__" + train.label)): train.loc[vi, "fold"] = k

def encode(texts, tok, max_len, head_frac=0.75):
    """Tokenize WITH special tokens, no truncation; if too long keep head + tail.
    Slicing the full sequence keeps the first ([CLS]/<bos>) and last ([SEP]/<eos>) special tokens,
    so this works for any tokenizer without relying on build_inputs_with_special_tokens."""
    enc = tok(list(texts), add_special_tokens=True, truncation=False)
    out, n_trunc = [], 0
    for ids in enc["input_ids"]:
        if len(ids) > max_len:
            h = int(max_len * head_frac); ids = ids[:h] + ids[-(max_len - h):]; n_trunc += 1
        out.append(ids)
    print(f"truncated {n_trunc}/{len(out)} ({n_trunc/len(out):.2%}) at max_len={max_len}")
    return out

class DS(torch.utils.data.Dataset):
    def __init__(self, ids, labels=None): self.ids, self.labels = ids, labels
    def __len__(self): return len(self.ids)
    def __getitem__(self, i):
        d = {"input_ids": self.ids[i], "attention_mask": [1] * len(self.ids[i])}
        if self.labels is not None: d["labels"] = int(self.labels[i])
        return d
# collator: transformers.DataCollatorWithPadding(tok)  (dynamic padding)

counts = tr_df.label.value_counts().reindex(LABELS).values
CLASS_W = torch.tensor(np.sqrt(counts.max() / counts), dtype=torch.float)
print("class weights", CLASS_W)

from transformers import Trainer
class WeightedTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        loss = torch.nn.functional.cross_entropy(outputs.logits.float(), labels,
                                                 weight=CLASS_W.to(outputs.logits.device))
        return (loss, outputs) if return_outputs else loss

def compute_metrics(p):
    preds = p.predictions.argmax(-1)
    return {"macro_f1": f1_score(p.label_ids, preds, average="macro")}
```

## 5. Sub-1-specific instructions

### 5.1 Verify the backbone before training (mandatory)

Our information about this model (ModernBERT / `mmBERT-base` backbone, ~300M params, ~2k-token context, original **2-class** head) comes from web-search summaries and is **unverified**. Print and record:

```python
from transformers import AutoConfig, AutoTokenizer, AutoModelForSequenceClassification
import transformers; print("transformers", transformers.__version__)   # ModernBERT needs >= 4.48
cfg = AutoConfig.from_pretrained(MODEL_DIR)
print(cfg.model_type, cfg.max_position_embeddings, getattr(cfg, "id2label", None),
      getattr(cfg, "classifier_pooling", None))
tok = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_DIR, num_labels=3, id2label=id2label, label2id=label2id,
    problem_type="single_label_classification", ignore_mismatched_sizes=True,
    attn_implementation="sdpa")   # T4 has no FlashAttention-2; fall back to "eager" if sdpa errors
print(tok("Olá, mundo")["input_ids"])   # confirm special tokens at both ends
```

- Expect a warning that the classifier weights were **newly initialised / resized** (2 → 3 classes). That is correct. Any *other* missing-weight warning (e.g. encoder layers) is a red flag — stop and report.
- If `transformers < 4.48` and the model type is `modernbert`: `pip install -U "transformers>=4.48"` (needs Internet ON), restart kernel.
- **Fallback:** if the model cannot be loaded, or is not a multilingual backbone, switch to the approved `jhu-clsp/mmBERT-base` (Kaggle model `aisuko/mmbert-base`) with the same recipe and rename the run `InnoLab_sub1_mmbertbase_f0`. Report which one was used.

### 5.2 Choose `max_len` by measurement

Tokenize all train texts without truncation and print the share of texts > 512 and > 1024 tokens and the 99th percentile.
- If ≤ 1% of texts exceed 1024 → `MAX_LEN = 1024`; otherwise still use 1024 (head+tail handles the rest).
- If 1024 runs out of memory even at batch 4 with gradient checkpointing → `MAX_LEN = 512`. Record the choice and truncation rate.

### 5.3 Hyperparameters

| Setting | Value |
|---|---|
| `learning_rate` | `3e-5` |
| `num_train_epochs` | `3` |
| `warmup_ratio` / scheduler | `0.1` / linear |
| `weight_decay` | `0.01` |
| `per_device_train_batch_size` × `gradient_accumulation_steps` | `8 × 4` (effective 32); drop to `4 × 8` on OOM |
| `per_device_eval_batch_size` | `32` (reduce if OOM) |
| `fp16` | `True` (T4 has no bf16) |
| `group_by_length` | `True` |
| `eval_strategy` / `save_strategy` | `"steps"` / `"steps"`, `eval_steps = save_steps = 175` (≈ ½ epoch: 11,200 / 32 = 350 steps/epoch) |
| `gradient_checkpointing` | off; turn on only if OOM |
| `report_to` | `"none"` |
| `dataloader_num_workers` | `2` |

- Use a **single GPU**: put `import os; os.environ["CUDA_VISIBLE_DEVICES"] = "0"` in the very first code cell, before importing torch. (Keeps results reproducible and avoids DataParallel issues with ModernBERT unpadding.)
- In recent `transformers` the Trainer argument is `processing_class=tok`; on older versions use `tokenizer=tok`. `evaluation_strategy` was renamed `eval_strategy`.
- **Run as a committed version** (Save Version → *Save & Run All*) once a 50-step smoke run has worked, so training survives browser disconnects and outputs persist.

### 5.4 Smoke run first (≈ 3 min)

Train for `max_steps=50` on the real data with one evaluation at step 50. Confirm: loss decreases, no OOM, eval returns `macro_f1`, prediction shape `(2800, 3)`. Record **seconds per step** and extrapolate total time. Then launch the full run.

## 6. After training — evaluation and outputs

Load best model (Trainer does this), then:

```python
OUT = "/kaggle/working/sub1_wolfdefender_f0"   # create it
from scipy.special import softmax as _sm
softmax = lambda z: _sm(z, axis=1)

va_pred = trainer.predict(va_ds); va_p = softmax(va_pred.predictions.astype(np.float64))
te_pred = trainer.predict(te_ds); te_p = softmax(te_pred.predictions.astype(np.float64))
y = va_df.label.map(label2id).values; yhat = va_p.argmax(1)

# per-language macro F1 with support, full report, confusion matrix
per_lang = {g: {"macro_f1": f1_score(y[m], yhat[m], average="macro"), "n": int(m.sum())}
            for g in ["pt", "fr", "it", "en"] for m in [(va_df.language == g).values]}
```

Save into `OUT/`:

| File | Content |
|---|---|
| `valid_probs.csv` | `id, label, p_BENIGN, p_JAILBREAK, p_PROMPT_INJECTION` (2,800 rows, fold 0) |
| `test_probs.csv` | `id, p_BENIGN, p_JAILBREAK, p_PROMPT_INJECTION` (6,000 rows, **test.csv order**) |
| `submission.csv` | `id,label` built as `pd.DataFrame({"id": test.id, "label": [LABELS[i] for i in te_p.argmax(1)]})`, `index=False` |
| `metrics.json` | `model_id` (actual one used), `macro_f1`, per-class P/R/F1/support, confusion matrix (rows = true, order = LABELS), `per_lang`, `best_step`, `max_len`, truncation rate, all hyperparameters, `class_weights`, transformers/torch versions, GPU name, train runtime, test-inference runtime, fold-0 sha check passed |
| `best_model/` | `trainer.save_model()` + tokenizer (keep for Sub 4/5) |

Then run the structural validator (same logic as [`scripts/validate_submission.py`](../scripts/validate_submission.py)):

```python
sub = pd.read_csv(f"{OUT}/submission.csv", dtype={"id": str})
assert sub.columns.tolist() == ["id", "label"] and len(sub) == 6000
assert sub.id.is_unique and set(sub.id) == set(test.id)
assert sub.label.isin(LABELS).all() and not sub.isna().any().any()
print(sub.label.value_counts(normalize=True))   # sanity: roughly 0.62 / 0.31 / 0.07
```

## 7. Submit / report

1. If **fold-0 macro F1 ≥ 0.93** and the validator passes → submit `submission.csv` to the competition with description `sub1 wolfdefender f0 valF1=<x.xxxx> len=<L>`.
2. If < 0.93 → **do not submit**; report back with metrics and the 10 most confident errors per confusion type.
3. Report to the lead (paste into [`experiments/EXPERIMENT_LOG.md`](../experiments/EXPERIMENT_LOG.md) row `sub1`): model actually used, val macro F1, per-class F1, per-language F1, best step, max_len + truncation %, train minutes, Kaggle public score (if submitted), notebook version link.
4. Download `OUT/` (or keep it as notebook output) — Sub 4/5 need `valid_probs.csv` and `test_probs.csv`.

## 8. Do-not list

- Do not change the split, label order, input text, or loss — that breaks comparison with Sub 2.
- Do not use any second pretrained model, external data, translation, or pseudo-labels.
- Do not tune thresholds or pick checkpoints using the public leaderboard.
- Do not use `id` or `language` as features.
- Do not hand-edit test predictions.
- Do not make the notebook public during the competition.

## 9. Failure playbook

| Symptom | Action |
|---|---|
| Fold-0 hash mismatch | Wrong `folds.csv` or row order — re-attach the repo file; never proceed with a different split |
| `KeyError: 'modernbert'` / unknown model type | Upgrade `transformers` (§5.1) |
| OOM | batch 4 × 8 → gradient checkpointing → `MAX_LEN=512` (in that order) |
| Macro F1 stuck ≈ 0.25–0.4 after first eval | Head not training / label bug: check `id2label`, that `labels` reach the loss, LR not 0 |
| Warning about missing **encoder** weights | Wrong checkpoint path (pointing to a sub-folder / ONNX export) — find the folder with `model.safetensors` + `config.json` |
| Very slow (> 1.5 s/step at batch 8) | Confirm GPU is used (`torch.cuda.is_available()`), `fp16=True`, `group_by_length=True` |
