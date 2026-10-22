# SUB 5 — `granite-3.3-2b-instruct` LoRA, full-data single model (Colab A100 / Kaggle Notebook Fast Path)

> **Audience:** the person or agent running this on Kaggle Notebook or Colab.
> **Team:** UTS · Saigon South (SGS) · InnoLab.
> **HARD CUTOFF:** CSV uploaded by **17:45 UTC+7**.
> **KAGGLE FAST PACKAGE (< 17:45 UTC+7):** Ready in [`../kaggle_sub5_package/`](../kaggle_sub5_package/):
> - Notebook to import: [`../kaggle_sub5_package/UTS_SGS_InnoLab_C1_Sub5_Kaggle.ipynb`](../kaggle_sub5_package/UTS_SGS_InnoLab_C1_Sub5_Kaggle.ipynb)
> - Dataset zip to attach: [`../kaggle_sub5_package/kaggle_sub5_dataset.zip`](../kaggle_sub5_package/kaggle_sub5_dataset.zip) (`train.csv`, `test.csv`, `folds.csv`, `sub3_submission.csv`, `sub5_kaggle_prepare.py`, `sub5_kaggle_train.py`)
> - Runs `MODE="full"` on all 14,000 rows for `2.0` epochs (875 steps, cosine schedule, `wd=0.01`), auto-adapts precision/batch for Kaggle T4 (`fp16` Tensor Cores + `fp32` master LoRA weights, `BS=8×4`, `MAX_LEN=768`) vs L4/A100 (`bf16`, `BS=16×2`, `MAX_LEN=1024`), writes a validated `/kaggle/working/submission.csv` at **every** snapshot (`[1.0, 1.5, 2.0]` epochs), and enforces a hard **17:25 UTC+7** (`10:25 UTC`) wall-clock guard.
> **Builds on:** [SUB3_granite_lora.md](SUB3_granite_lora.md) (same backbone, fold-0 **0.9589**, public **0.96808**, current best) and [SELFVAL_offline_scoring.md](SELFVAL_offline_scoring.md) (bias-tuning rule).
> **Team rule:** **no ensembles.** One trained model produces the CSV. The "shadow" run below only *chooses settings*; its predictions are never submitted or averaged.

---

## 0. Goal in one paragraph

Train **one** Granite-3.3-2B LoRA classifier on **all 14,000 labelled rows** (Sub 3 saw 11,200, including only
785 of the 980 injection rows) with a **longer 3-epoch cosine schedule**, take its predictions at the
**training length chosen by evidence**, and apply a **class-bias correction only if it is proven on held-out rows**.
Because a full-data model has no validation set, a second identical-recipe run on folds 1–4 (the **shadow**)
scores fold 0 every half epoch; it picks the epoch E\* and the bias. The full run saves its test logits at the same
half-epoch points, so no retraining is needed after selection.

## 1. What the Sub 3 results tell us (evidence for each change)

| Observation (Sub 3, `workplace.ipynb`) | Consequence for Sub 5 |
|---|---|
| Fold-0 macro F1 per half epoch: **0.824 → 0.869 → 0.945 → 0.960**. It was still rising when the 2-epoch schedule ended. | Train longer (**3 epochs**) and *select* the epoch rather than guess it. |
| Training loss at step 700 = **0.011**; the model already fits the training rows | Capacity is not the bottleneck → keep LoRA **r = 16**; more *data* is the lever, not a bigger adapter. |
| Injection: precision **0.9526**, recall **0.9282** (195 rows) | Injection is under-predicted → a held-out-verified bias toward `PROMPT_INJECTION` may add points; §6 tests it. |
| Injection is 7% of rows but 1/3 of macro F1; fold 1–4 training saw 785 injection rows | Full data adds **+195 injection rows (+25%)**, the most direct fix for the weakest class. |
| `trainer.predict` after training scored **0.3409**. Reloading the same checkpoint and predicting in a plain ordered loop scored **0.9589**. | The 0.3409 per-class F1s (BENIGN 0.621, JAILBREAK 0.302) **equal the class shares** (0.62, 0.31): that is the signature of correct predictions written to the **wrong rows**, not broken weights. Likely cause (hypothesis): length-grouped batching also reorders `predict`. **Rule: never use `trainer.predict`, never use `load_best_model_at_end`.** All predictions come from the ordered loop in §4. |
| 700 steps took **22.9 min** including 4 evals + 4 × 5 GB checkpoint saves → ≈ **1.85 s/step** at 16 × 2 | Full run 3 epochs ≈ 1,313 steps ≈ **41 min** + 6 test passes ≈ **48 min**. Shadow 1,050 steps ≈ **33 min** + 6 × (valid + test) ≈ **42 min**. |

## 2. Recipe (Sub 3 contract + changes in bold)

| Item | Sub 3 | **Sub 5** |
|---|---|---|
| Training rows | folds 1–4 (11,200) | **full run: all 14,000** · shadow: folds 1–4 |
| Backbone / head | Granite-3.3-2B, LoRA r 16, α 32, dropout 0.05, all 7 projections, zero-init last-token head | same |
| Input | raw text, head 75% + tail 25% of 1,024 tokens | same |
| Loss | CE, weights `sqrt(n_max / n_c)` ≈ `[1.00, 1.41, 2.98]` | same (recomputed on the rows trained on; identical values) |
| Epochs / schedule | 2, linear, warmup 5% | **3, cosine, warmup 5%** |
| LR / weight decay | 1e-4 / 0.0 | 1e-4 / **0.01** |
| Batch / precision | 16 × 2, bf16, `group_by_length` | same |
| Checkpointing | Trainer eval + save every 175 steps, `load_best_model_at_end` | **no Trainer eval/save**; a callback snapshots at epochs 0.5, 1.0 … 3.0 |
| Prediction | `trainer.predict` (row-order bug) | **ordered inference loop** (§4) |
| Decision | `argmax` | **`argmax(logits + b)`**, `b = [0, b_JB, b_INJ]`, `b = 0` unless §6 adopts it |

**Considered and deliberately left out** (no time to A/B them, and the evidence does not point to them):
bigger LoRA rank (train loss already 0.011); `max_len` 2048 (only 0.81% of rows exceed 1,024); an instruction
prompt or chat template (changes the input and is unvalidated); label smoothing / R-Drop / adversarial training
(unvalidated); multiple seeds or fold averaging (ensemble, not allowed); pseudo-labels or external data (rules).

## 3. Runtimes and order

Sub 5 shares A100 runtimes with [Sub 6 (wolf-defender plus)](SUB6_wolf_plus.md). **Sub 6 gets a runtime before the Sub 5 shadow
does**: the shadow only tunes Sub 5's schedule a little, while Sub 6 produces the second backbone for the final picks. Sub 6 can also run on a
Kaggle T4 × 2 account instead (see SUB6 §2), which frees the A100s for the Sub 5 full + shadow pair.

| A100 runtimes available | Sub 5 | Sub 6 |
|---|---|---|
| **3** | Runtime A = `MODE="full"` (3-epoch cosine), Runtime B = `MODE="shadow"`, both launched together, done ≈ 50 min later | Runtime C |
| **2** | Runtime A = **Lite** (below), ≈ 30 min | Runtime B |
| **1** | **Lite** first (≈ 15:05–15:40) | Same runtime after Sub 5 Lite finishes: Runtime → Restart session, then Sub 6 cells |
| Full run (3-epoch) fails once, or starts after 15:45 | Switch to **Lite** | — |

**Lite** = `MODE="full"`, `EPOCHS=2.0`, `SCHED="linear"`, `WD=0.0`: exactly the Sub 3 schedule (fold-0 **0.9589**) on 14,000 rows,
≈ 875 steps, ≈ 30 min. It needs no shadow, because that recipe was already validated on fold 0. E\* = 2.0. For §5–7, create
`sub5_granite_shadow/` from the **corrected** Sub 3 outputs: `valid_logits_e2.0.npy` = `va_logits` and `test_logits_e2.0.npy` = `te_logits`
from `workplace.ipynb` Cell 6 (`np.save` them while that runtime is alive), plus `snapshots.json` = `[{"epoch": 2.0, "fold0_macro_f1": 0.9589}]`.
**Do not use Sub 3's old `valid_probs.csv` / `test_probs.csv`.** They came from `trainer.predict` and are in the wrong row order.
If the corrected logits are gone, run Lite without a shadow folder (bias off, E\* = 2.0).

With 3 runtimes and the full + shadow pair: if you have to restart the session between runs on one VM, snapshots are on Drive.
Re-run Cell 2 and then Cell 3; Cell 1 is not needed again on the same VM.

If someone is already running the "Submission #4" Granite fold-0 job from `docs/agent.md` §26.8 (3 epochs, cosine, 1e-4,
wd 0.01), it can serve as the shadow **only if** it saves fold-0 logits from an ordered loop at every half epoch. Otherwise
use the shadow below.

## 4. Cells

**Cell 1** — fresh runtime, then restart (same as Sub 3):

```python
!pip uninstall -y -q torchao
```

**Cell 2** — upload `train.csv`, `test.csv`, `folds.csv` to `/content/`, then paste and run
[`scripts/sub3_prepare.py`](../scripts/sub3_prepare.py) **unchanged**. It sets the seed, checks the fold-0 hash, tokenizes
(`tr_ids`, `va_ids`, `te_ids`), defines `encode`, `DS`, `GraniteClassifier`, `build_model`, `WeightedTrainer`, and builds a
fresh `model`. It must print `PREPARE_OK`.

**Cell 3** — Sub 5 training (set `MODE` first):

```python
import json, os, time
import numpy as np, pandas as pd, torch
from sklearn.metrics import f1_score
from transformers import DataCollatorWithPadding, TrainerCallback, TrainingArguments

MODE = "full"                       # "full" = all 14,000 rows | "shadow" = folds 1-4, scores fold 0
EPOCHS, SCHED, WD = 3.0, "cosine", 0.01           # Lite: 2.0, "linear", 0.0
BS, GA = 16, 2                                    # effective batch 32; OOM -> 8, 4
GRID_E = [e / 2 for e in range(1, int(EPOCHS * 2) + 1)]
GUARD_E = 1.0                                     # shadow stops if fold-0 F1 < 0.50 at this epoch
SAVE_ADAPTERS = MODE == "full"
from google.colab import drive; drive.mount("/content/drive")
OUT5 = f"/content/drive/MyDrive/innolab/sub5_granite_{MODE}"      # on Drive: survives a runtime reset
os.makedirs(OUT5, exist_ok=True)
SUB3_CSV = "/content/sub3_submission.csv"         # upload the 0.96808 file here (agreement check only)

fit_df = train.reset_index(drop=True) if MODE == "full" else tr_df
fit_ids = encode(fit_df.text)[0] if MODE == "full" else tr_ids
counts = fit_df.label.value_counts().reindex(LABELS).values
CLASS_W = torch.tensor(np.sqrt(counts.max() / counts), dtype=torch.float)   # read by WeightedTrainer
fit_ds = DS(fit_ids, fit_df.label.map(label2id).tolist())
print(MODE, "rows", len(fit_df), "class weights", CLASS_W.tolist())

sub3_lab = None
if os.path.exists(SUB3_CSV):
    s3 = pd.read_csv(SUB3_CSV, dtype=str)
    assert s3.id.tolist() == test.id.tolist(), "Sub 3 CSV not in test.csv order"
    sub3_lab = s3.label.map(label2id).values

collate = DataCollatorWithPadding(tok)
def ordered_logits(m, ids, bs=32):
    """Length-sorted batches, results written back to the ORIGINAL row order."""
    was_training, order = m.training, np.argsort([len(x) for x in ids])
    out = np.zeros((len(ids), 3), dtype=np.float32)
    m.eval()
    with torch.no_grad():
        for s in range(0, len(ids), bs):
            idx = order[s:s + bs]
            b = collate([{"input_ids": ids[i], "attention_mask": [1] * len(ids[i])} for i in idx])
            out[idx] = m(input_ids=b["input_ids"].cuda(), attention_mask=b["attention_mask"].cuda()).logits.float().cpu().numpy()
    m.train(was_training)
    return out

class Snapshot(TrainerCallback):
    def on_train_begin(self, args, state, control, **kw):
        self.at = {round(state.max_steps * e / EPOCHS): e for e in GRID_E}
        self.recs, self.t0 = [], time.time()
        print("max_steps", state.max_steps, "snapshots at", self.at)
    def on_step_end(self, args, state, control, model=None, **kw):
        e = self.at.get(state.global_step)
        if e is None:
            return
        rec = {"epoch": e, "step": state.global_step, "minutes": round((time.time() - self.t0) / 60, 1)}
        te = ordered_logits(model, te_ids); np.save(f"{OUT5}/test_logits_e{e}.npy", te)
        p = te.argmax(1)
        rec["test_shares"] = (np.bincount(p, minlength=3) / len(p)).round(3).tolist()
        if sub3_lab is not None:
            rec["agree_sub3"] = round(float((p == sub3_lab).mean()), 4)
        if MODE == "shadow":
            va = ordered_logits(model, va_ids); np.save(f"{OUT5}/valid_logits_e{e}.npy", va)
            rec["fold0_macro_f1"] = round(float(f1_score(va_y, va.argmax(1), average="macro")), 4)
            if e == GUARD_E and rec["fold0_macro_f1"] < 0.50:
                print("COLLAPSE_GUARD: stopping"); control.should_training_stop = True
        if SAVE_ADAPTERS:
            d = f"{OUT5}/adapter_e{e}"
            model.backbone.save_pretrained(d); torch.save(model.score.state_dict(), f"{d}/score.pt")
        if MODE == "full" and e >= 1.5 and rec.get("agree_sub3", 1.0) < 0.90:
            print("WARNING: full run disagrees with Sub 3 on > 10% of test rows")
        self.recs.append(rec)
        json.dump(self.recs, open(f"{OUT5}/snapshots.json", "w"), indent=1)
        print("SNAPSHOT", rec)

args = TrainingArguments(
    output_dir=f"/content/sub5_{MODE}_tmp", num_train_epochs=EPOCHS, per_device_train_batch_size=BS,
    gradient_accumulation_steps=GA, learning_rate=1e-4, lr_scheduler_type=SCHED, warmup_ratio=0.05,
    weight_decay=WD, max_grad_norm=1.0, bf16=True, fp16=False, group_by_length=True,
    eval_strategy="no", save_strategy="no", report_to="none", logging_steps=25,
    label_names=["labels"], dataloader_num_workers=2, seed=42)
trainer = WeightedTrainer(model=model, args=args, train_dataset=fit_ds,
                          data_collator=collate, processing_class=tok, callbacks=[Snapshot()])
t0 = time.time(); trainer.train()
print(f"TRAIN_DONE {MODE} {(time.time() - t0) / 60:.1f} min")
```

Check within the first 2 minutes: initial loss ≈ 1.1 (zero-init head), `snapshots at` lists 6 steps
(full ≈ 219 … 1,313; shadow 175 … 1,050), and the logged s/step is ≈ 1.8–2.0. If it is above 3 s/step, switch to the Lite fallback.

**What to expect at each `SNAPSHOT`:** shadow `fold0_macro_f1` should climb past ≈ 0.94 by epoch 1.5. Test shares should
approach 0.62 / 0.31 / 0.07, and `agree_sub3` should rise towards ≈ 0.96–0.98 (two good models agree on most rows).

## 5. Selection — E\* (CPU, < 2 min)

Run in the full-run runtime after copying the shadow's `snapshots.json` + `valid_logits_e*.npy` + `test_logits_e*.npy`
(≈ 1 MB) into `/content/drive/MyDrive/innolab/sub5_granite_shadow/`. Cell 2 must have run in this runtime (it provides `va_df`, `va_y`, `test`).

```python
S, F = "/content/drive/MyDrive/innolab/sub5_granite_shadow", "/content/drive/MyDrive/innolab/sub5_granite_full"
shadow = json.load(open(f"{S}/snapshots.json")) if os.path.exists(f"{S}/snapshots.json") else []
f1_at = {r["epoch"]: r["fold0_macro_f1"] for r in shadow}
full_eps = [r["epoch"] for r in json.load(open(f"{F}/snapshots.json"))]
usable = {e: v for e, v in f1_at.items() if e in full_eps}
if usable:
    best = max(usable.values())
    E = max(e for e, v in usable.items() if v >= best - 0.002)     # ties within 0.002 -> later, more-annealed snapshot
else:
    E = max(full_eps)                                              # no shadow: end of schedule, no bias
print("fold-0 F1 per epoch", f1_at, "-> E* =", E)
```

## 6. Class bias — adopt only if it helps on held-out rows

Same rule as [SELFVAL §3.4](SELFVAL_offline_scoring.md): tune `b` on one half of fold 0 and score on the other half,
5 repeats × 2 halves = 10 held-out gains. **Adopt only if the mean gain > 0 and at least 8 of the 10 halves are positive.**
Then fit the final `b` on all of fold 0.

```python
from sklearn.model_selection import StratifiedKFold
def mf1(y, yhat):
    cm = np.bincount(3 * y + yhat, minlength=9).reshape(3, 3)
    tp, denom = np.diag(cm), cm.sum(0) + cm.sum(1)
    return float(np.mean(np.where(denom > 0, 2 * tp / np.maximum(denom, 1), 0.0)))
GRID = np.round(np.arange(-2.0, 2.01, 0.1), 2)
def fit_bias(lg, yy):
    best_s, best_b = mf1(yy, lg.argmax(1)), np.zeros(3)
    for bj in GRID:
        for bi in GRID:
            b = np.array([0.0, bj, bi]); s = mf1(yy, (lg + b).argmax(1))
            if s > best_s + 1e-6: best_s, best_b = s, b
    return best_b

bias, gains = np.zeros(3), []
if usable:
    lg, yv = np.load(f"{S}/valid_logits_e{E}.npy"), np.array(va_y)
    strata = (va_df.language + "__" + va_df.label).values
    for r in range(5):
        for a, b in StratifiedKFold(2, shuffle=True, random_state=r).split(lg, strata):
            gains.append(mf1(yv[b], (lg[b] + fit_bias(lg[a], yv[a])).argmax(1)) - mf1(yv[b], lg[b].argmax(1)))
    if np.mean(gains) > 0 and sum(g > 0 for g in gains) >= 8:
        bias = fit_bias(lg, yv)
print("held-out gains", np.round(gains, 4), "-> bias", bias)
```

The full-data model may be slightly more confident than the shadow, so the bias transfers approximately, not exactly.
That is why it must pass the held-out rule and why §7 caps how many labels it may change.

## 7. Finalize, checks, gate

```python
te = np.load(f"{F}/test_logits_e{E}.npy")
pred, raw = (te + bias).argmax(1), te.argmax(1)
sub = pd.DataFrame({"id": test.id, "label": [LABELS[i] for i in pred]})
shares = np.bincount(pred, minlength=3) / len(pred)
checks = {
    "E_star": E, "bias": bias.tolist(), "shadow_f1_at_E": f1_at.get(E),
    "labels_changed_by_bias": int((pred != raw).sum()),
    "test_shares": shares.round(3).tolist(),
    "shares_ok": bool(np.abs(shares - np.array([0.62, 0.31, 0.07])).max() <= 0.03),
    "agree_sub3": float((pred == sub3_lab).mean()) if sub3_lab is not None else None,
}
if usable and os.path.exists(f"{S}/test_logits_e{E}.npy"):
    checks["agree_shadow_at_E"] = float((pred == np.load(f"{S}/test_logits_e{E}.npy").argmax(1)).mean())
assert list(sub.columns) == ["id", "label"] and len(sub) == 6000 and sub.id.is_unique
assert sub.id.tolist() == test.id.tolist() and sub.label.isin(LABELS).all()
sub.to_csv(f"{F}/submission.csv", index=False)
json.dump(checks, open(f"{F}/final_report.json", "w"), indent=1)
print(json.dumps(checks, indent=1))
```

**Upload `submission.csv` only if all of these hold:**

1. Shadow fold-0 macro F1 at E\* is **≥ 0.955**. That is Sub 3's 0.9589 minus half its ≈ ±0.008 confidence range, so the new recipe is at least as good as Sub 3's. With no shadow (one runtime ran out of time), skip this check but require check 3.
2. `shares_ok` is true.
3. `agree_sub3` **≥ 0.95** and, if a shadow exists, `agree_shadow_at_E` **≥ 0.95**. Lower agreement means a pipeline problem, not a better model.
4. `labels_changed_by_bias` **≤ 120** (2% of test). More than that means the bias does not transfer; resubmit with `bias = 0`.
5. It is before **17:45 UTC+7**.

Description: `sub5 granite full14k E*=<E> shadowF1=<x.xxxx> bias=<adopted|off>`.

Keep `adapter_e{E}/` (adapter + `score.pt`, ≈ 115 MB) and `final_report.json`; they are needed if this model has to be
reproduced inside `UTS_SGS_InnoLab_C1`. Then fill a `sub5` row and the ledger in
[`experiments/EXPERIMENT_LOG.md`](../experiments/EXPERIMENT_LOG.md): E\*, shadow F1 per epoch, held-out bias gains,
bias, labels changed, agreement numbers, s/step, minutes, public score.

## 8. Time plan (UTC+7)

| Time | 3 runtimes (full + shadow) | 1–2 runtimes (Lite) |
|---|---|---|
| 15:00–15:15 | Cells 1–2; upload CSVs and the Sub 3 submission; save Sub 3's corrected logits | same |
| 15:15 | Launch full (A) and shadow (B) | Launch Lite |
| ≈ 15:45 | — | Lite done → §5–7 → **upload ≈ 15:55**; with 1 runtime, hand the GPU to Sub 6 |
| ≈ 16:05 | Both done → §5–7 | — |
| ≈ 16:15 | **Upload** | — |
| **17:45** | **Hard cutoff** | **Hard cutoff** |

At 17:20, if not ready: stop. Sub 3 (0.96808) stays our best.

## 9. Failure playbook

| Symptom | Action |
|---|---|
| `torchao` import error / `GraniteConfig` not recognised | See [SUB3 §9](SUB3_granite_lora.md); Cell 1 and the wrapper in `sub3_prepare.py` handle both |
| OOM | Batch `8 × 4` (`per_device_train_batch_size=8`, `gradient_accumulation_steps=4`); inference `bs=16` |
| Initial loss ≈ 10–15 | The head is not zero-initialised; re-run Cell 2 for a fresh model |
| `COLLAPSE_GUARD` in the shadow | Restart the shadow once with `learning_rate=5e-5`; if the full run shows low `agree_sub3`, it collapsed too, so restart it with the same change or switch to the Lite fallback |
| Runtime disconnects | Snapshots and `snapshots.json` are on Drive. Use the latest full snapshot as E\* (no bias) if the shadow cannot finish |
| Selection finds no shadow epoch that the full run reached | The full run was cut short. Use its latest snapshot only if it is ≥ 2.0 epochs; otherwise do not upload |

## 10. Do-not list

- Do not use `trainer.predict`, `load_best_model_at_end`, or Trainer checkpoints for predictions.
- Do not submit, average, or blend the shadow's predictions. They only choose E\* and `b`.
- Do not pick E\* or `b` from the public leaderboard or from test predictions.
- Do not change the fold split, label order, input text, or loss.
- Do not use a second pretrained model, external data, pseudo-labels, or a prompt template. Do not hand-edit predictions.
