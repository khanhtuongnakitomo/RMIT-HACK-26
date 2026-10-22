# SUB 4 — `wolf-defender-prompt-injection`, adaptive K + full-data single model (Kaggle T4 ×2)

> **Audience:** the co-worker agent running this experiment on Kaggle.
> **Team:** UTS · Saigon South (SGS) · InnoLab. **Notebook:** a **new version of `UTS_SGS_InnoLab_C1`** (Version 4 = Sub 1 stays in the version history).
> **HARD CUTOFF:** the CSV must be **uploaded by 17:45 UTC+7 (10:45 UTC)**. Target upload **17:15 UTC+7**. Kaggle shows UTC (= UTC+7 − 7 h).
> **Sister experiments:** [SUB1_wolf_defender.md](SUB1_wolf_defender.md) (same backbone, fold 0 only → public **0.96557**), [SUB3_granite_lora.md](SUB3_granite_lora.md) (Granite, public **0.96808**, current best). Same contract (§3) except where this file says otherwise.
> **Script:** everything runs from [`scripts/sub4_wolf_adaptive.py`](../scripts/sub4_wolf_adaptive.py) (one file, four modes).

---

## 0. Goal in one paragraph

Submit **one single model**: `patronus-studio/wolf-defender-prompt-injection` fine-tuned on **all 14,000 labelled rows**. **No ensemble** — nothing is averaged. "**Adaptive K**" = K-fold runs are used **only to choose settings automatically**, and **K adapts to the time left** (as many of the 5 folds as can finish before 17:00 UTC+7; minimum useful K = 2). The folds choose (1) **E\*** = how many epochs to train (best mean fold macro F1 on a half-epoch grid) and (2) a **per-class decision bias** `b = [0, b_JAILBREAK, b_INJECTION]` added to the model's logits before `argmax`, kept **only if** it improves macro F1 on folds it was not tuned on. The full-data model runs **in parallel** with the folds and saves its test logits every half epoch, so after selection we just take its predictions at E\* — no retraining. This is our strongest Kaggle-native attempt: same backbone as Sub 1 (0.96557) + 25% more data (all 980 injection rows instead of 785) + evidence-based training length + held-out-verified injection calibration + 1024-token head+tail input.

## 1. Background you need (short)

- Task: multilingual (pt 45%, fr 25%, it 20%, en 10%) prompt-attack classification. Metric: **macro F1** — the 7% `PROMPT_INJECTION` class weighs as much as the 62% `BENIGN` class.
- Train 14,000 rows (`id,text,language,label`), test 6,000 (`id,text,language`). Clean UTF-8; newlines stripped.
- Class counts: BENIGN 8,680 · JAILBREAK 4,340 · PROMPT_INJECTION 980. Every language has the same class ratio.
- Scores so far: Sub 1 wolf-defender fold 0 → public **0.96557**; Sub 3 Granite LoRA fold 0 → fold-0 **0.9589**, public **0.96808**. TF-IDF floor 0.928.
- **Team rule (confirmed): ensembles are NOT permitted.** Every submission = one trained model.
- Rules: exactly **one** approved pretrained backbone per notebook; name it at the top; no other pretrained models, no external data.
- Full context: [`docs/agent.md`](../docs/agent.md) §26.

## 2. Kaggle setup

1. Open `UTS_SGS_InnoLab_C1` → **Edit**. Add the Sub 4 cells (§5) as a new section; this run becomes the next notebook version.
2. **Accelerator: GPU T4 ×2** — this plan uses **both GPUs**. Internet: only needed for option B below.
3. **Inputs** (probably already attached from Sub 1): competition data `rmit-hackathon-2026`; Kaggle Model `aisuko/wolf-defender-prompt-injection` → Transformers → default; the private dataset with `folds.csv`.
4. **Script onto disk** (pick one):
   - **A (preferred, keeps the official notebook self-contained):** a cell beginning with `%%writefile /kaggle/working/sub4.py`, followed by the full contents of [`scripts/sub4_wolf_adaptive.py`](../scripts/sub4_wolf_adaptive.py).
   - **B (Internet ON):** `!wget -q https://raw.githubusercontent.com/khanhtuongnakitomo/RMIT-HACK-26/main/scripts/sub4_wolf_adaptive.py -O /kaggle/working/sub4.py`
5. First markdown cell of the section must state: **`Sub 4 — Model: patronus-studio/wolf-defender-prompt-injection (single approved backbone). One model trained on all 14,000 rows; K-fold runs only choose epochs and class bias (adaptive K). No ensemble.`**

## 3. Experiment contract (Sub 1 contract + the Sub 4 changes in bold)

| Item | Value |
|---|---|
| Label map | `LABELS = ["BENIGN", "JAILBREAK", "PROMPT_INJECTION"]`. Export **strings**. |
| Split | `data/folds.csv` (`id,fold`). Fold-0 check: 2,800 rows, sha256 `0d043c7efa1612b707d4a52e4b73a0aa311ee37fb3f9cadf213ec47269a57fc4` (the script aborts on mismatch). |
| **Training data** | **Fold runs:** train on folds ≠ k, score fold k. **Final model: all 14,000 rows** (no holdout). |
| Input | raw `text` only, unmodified. |
| Long inputs | head + tail truncation, first 75% / last 25% of **1024** tokens. |
| Loss | cross-entropy, class weights `sqrt(n_max / n_c)` from the training rows ≈ `[1.00, 1.41, 2.98]`. |
| Seed | 42 |
| **Schedule** | **identical in epoch units for folds and final model: 4 epochs, linear warmup 10% → decay to 0 at epoch 4.** |
| **Selection** | **E\* = half-epoch (0.5 … 4.0) with the highest mean fold macro F1** across the K finished folds. |
| **Test prediction** | **`argmax(logits + b)`**, `b = [0, b_JB, b_INJ]` — `b = 0` unless the leave-one-fold-out rule (§5.5) adopts it. |
| Gate | upload only if `finalize` prints `SUBMISSION_READY` **and** `cv_macro_f1_at_E ≥ 0.950` (§7). |

## 4. How the script works (read once; no need to edit it)

[`scripts/sub4_wolf_adaptive.py`](../scripts/sub4_wolf_adaptive.py), `--mode`:

| Mode | Device | What it does | Writes |
|---|---|---|---|
| `fold --fold k` | 1 GPU | trains on folds ≠ k with the 4-epoch schedule; scores fold k at every half epoch; collapse guard stops the fold if macro F1 < 0.50 at epoch 1.0 | `sub4/fold{k}/record.npz` (valid logits per half epoch), `meta.json`, `test_logits_final.npy` (sanity check only), `DONE` |
| `full` | 1 GPU | trains on all 14,000 rows with the same schedule; at every half epoch saves **test logits**, a train-sample macro F1, and fp16 weights | `sub4/full/record.npz`, `weights/e{E}.pt`, `tokenizer/`, `meta.json`, `DONE` |
| `select` | CPU | uses only folds with `DONE`; picks E\*; fits bias with leave-one-fold-out check | `sub4/selection.json` |
| `finalize` | CPU | takes the full model's test logits at E\*, applies bias if adopted, writes and validates the CSV, runs sanity checks, keeps only the selected weights | `sub4_final/submission.csv`, `test_probs.csv`, `final_report.json`, `best_model/` |

Implementation notes:
- **Plain PyTorch loop** (AdamW, fp16 autocast + `GradScaler`, grad-clip 1.0, weight decay 0.01 on weight matrices only, length-grouped shuffled batches). No HF `Trainer` — so the Sub 3 "Trainer reload broke the weights" bug cannot happen, and it does not depend on `transformers` argument names.
- Predictions are made in **length-sorted batches and written back to the original row order**; `finalize` asserts the test ids match `test.csv` order.
- `torch.compile` is disabled for ModernBERT (`reference_compile=False`) to avoid compile stalls on T4.
- The checkpoint's 2-class head is replaced by a fresh 3-class head (`ignore_mismatched_sizes=True`) — the load report shows `classifier.weight | MISMATCH | Reinit`, which is expected.

## 5. Sub-4-specific instructions

### 5.1 Paths (cell 1)

```python
import glob, os
MODEL_DIR = os.path.dirname([p for p in glob.glob("/kaggle/input/**/config.json", recursive=True) if "wolf" in p.lower()][0])
FOLDS = glob.glob("/kaggle/input/**/folds.csv", recursive=True)[0]
OUT, FINAL = "/kaggle/working/sub4", "/kaggle/working/sub4_final"
os.makedirs(OUT, exist_ok=True)
print(MODEL_DIR, FOLDS, sep="\n")
!nvidia-smi --query-gpu=index,name,memory.total --format=csv
```

`MODEL_DIR` must contain `config.json` + `model.safetensors` (not an ONNX export). Two GPUs must be listed.

### 5.2 Hyperparameters (script defaults — do not change)

| Setting | Value | Flag |
|---|---|---|
| max length | `1024` (head + tail) | `--max_len` |
| learning rate | `3e-5` | `--lr` |
| schedule length | `4` epochs (grid 0.5 … 4.0) | `--epochs` |
| warmup | `0.1`, linear decay to 0 | `--warmup` |
| weight decay | `0.01` | `--wd` |
| batch | `8` × accumulation `4` = effective 32; **out of memory → `--bs 4`** (accumulation becomes 8, effective stays 32) | `--bs` |
| precision | fp16 autocast | — |
| seed | `42` | `--seed` |

Steps: fold = 11,200 / 32 = 350 per epoch → 1,400 total; full = 14,000 / 32 = 437.5 per epoch → 1,750 total.

### 5.3 Smoke run first (≈ 3 min, cell 2)

Same purpose as the smoke run in Sub 1/2/3: prove model load + training + output on the real GPU and **measure seconds per step** before committing the GPUs.

```python
!CUDA_VISIBLE_DEVICES=0 python /kaggle/working/sub4.py --mode fold --fold 0 --model_dir "$MODEL_DIR" --folds "$FOLDS" \
   --out /kaggle/working/smoke --limit 640 --epochs 1.0 2>&1 | grep -v "Token indices" | tail -n 25
!rm -rf /kaggle/working/smoke
```

Confirm: `classifier … MISMATCH | Reinit` only (no missing encoder weights), `step 20/20` reached, loss falls, no OOM, `DONE` printed. **Write down `s/step`.**

### 5.4 Launch: two GPUs, adaptive K (cell 3 — non-blocking)

GPU 0 takes the **full** run first; GPU 1 takes fold 0; each GPU then pulls the next fold **only if it can finish before 17:00 UTC+7** (estimate updated with real fold times).

```python
import subprocess, threading, queue, time, datetime as dt, json
SEC_PER_STEP = 0.8            # <- s/step measured in 5.3
EXTRA = []                    # ["--bs", "4"] if 5.3 needed it
LAST_FOLD_END = dt.datetime(2026, 10, 22, 10, 0, tzinfo=dt.timezone.utc)   # 17:00 UTC+7
est_fold_min = (1400 * SEC_PER_STEP + 8 * 25 + 90) / 60
print(f"estimated minutes: fold {est_fold_min:.0f}, full {(1750 * SEC_PER_STEP + 8 * 60) / 60:.0f}")

jobs = queue.Queue()
for j in ["full", "fold0", "fold1", "fold2", "fold3", "fold4"]:
    jobs.put(j)
fold_minutes, status = [], {}

def run(job, gpu):
    args = ["python", "/kaggle/working/sub4.py", "--model_dir", MODEL_DIR, "--folds", FOLDS, "--out", OUT] + EXTRA
    args += ["--mode", "full"] if job == "full" else ["--mode", "fold", "--fold", job[-1]]
    t = time.time(); status[job] = f"running on GPU {gpu}"
    with open(f"{OUT}/{job}.log", "w") as f:
        rc = subprocess.run(args, env=dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu)), stdout=f, stderr=subprocess.STDOUT).returncode
    mins = (time.time() - t) / 60
    status[job] = f"{'done' if rc == 0 else 'FAILED rc=' + str(rc)} in {mins:.1f} min (GPU {gpu})"
    if job.startswith("fold") and rc == 0:
        fold_minutes.append(mins)

def worker(gpu):
    while True:
        try:
            job = jobs.get_nowait()
        except queue.Empty:
            return
        est = sum(fold_minutes) / len(fold_minutes) if fold_minutes else est_fold_min
        if job.startswith("fold") and dt.datetime.now(dt.timezone.utc) + dt.timedelta(minutes=est) > LAST_FOLD_END:
            status[job] = f"skipped (would end after 17:00 UTC+7, est {est:.0f} min)"; continue
        run(job, gpu)

threads = [threading.Thread(target=worker, args=(0,), daemon=True)]
threads[0].start(); time.sleep(3)
threads.append(threading.Thread(target=worker, args=(1,), daemon=True)); threads[1].start()
print("launched", dt.datetime.now(dt.timezone.utc).strftime("%H:%M UTC"))
```

**Monitor (cell 4, re-run any time; does not interrupt training):**

```python
print(dt.datetime.now(dt.timezone.utc).strftime("%H:%M UTC"), json.dumps(status, indent=1))
!for f in /kaggle/working/sub4/*.log; do echo "== $f"; grep -E "valid macro|train-sample|step [0-9]+/|DONE|Error|error" $f | tail -n 3; done
```

Expect `[fold k] … valid macro F1` ≈ 0.90+ by epoch 1.0 and ≈ 0.95+ later; `[full] … train-sample F1` rising towards 0.99 with `test shares` ≈ 0.62 / 0.31 / 0.07.

*(Only for a "Save & Run All" commit, add a cell `for t in threads: t.join()` so the kernel waits.)*

### 5.5 Select + finalize (cell 5, CPU, < 2 min)

Run when `full` is `done` and no fold is running — or at **17:05 UTC+7 (10:05 UTC)** at the latest (unfinished folds are ignored automatically; only folds with `DONE` count).

```python
!python /kaggle/working/sub4.py --mode select   --out "$OUT" --folds "$FOLDS"
!python /kaggle/working/sub4.py --mode finalize --out "$OUT" --folds "$FOLDS" --final_dir "$FINAL"
```

**Bias rule (leave-one-fold-out):** for each finished fold k, tune `b` on the other folds' logits at E\* (grid −3 … 3, step 0.1, ties → smallest |b|) and measure the macro-F1 change on fold k. Adopt only if the mean change > 0 **and** it is positive on all folds (K = 2) or on at least K − 1 folds (K ≥ 3). Then the final `b` is fitted on all K folds pooled. Otherwise `b = [0, 0, 0]` (plain argmax). The `pooled_oof_macro_f1_biased_optimistic` number in `selection.json` is optimistic — quote the leave-one-fold-out gains instead.

**Fallback:** no fold finished → `E* = 3.0` (Sub 1 recipe), no bias; `finalize` still works because the full run saved epoch 3.0.

### 5.6 Collapse guard

Built in: a fold whose macro F1 is < 0.50 at epoch 1.0 stops itself and writes `FAILED`; the queue carries on and `select` skips it. If the **full** run's train-sample F1 is still < 0.90 at epoch 2.0 in the monitor, the full run is broken → see §9.

## 6. After training — outputs

| File | Content |
|---|---|
| `sub4/selection.json` | `K`, folds used, mean ± std fold macro F1 per half epoch, **`E_star`**, **`cv_macro_f1_at_E`**, `lofo_gains`, `bias_adopted`, `bias` |
| `sub4_final/submission.csv` | `id,label`, 6,000 rows, validated (columns, ids, labels) |
| `sub4_final/test_probs.csv` | `id, p_BENIGN, p_JAILBREAK, p_PROMPT_INJECTION` (softmax of raw logits at E\*, test.csv order) |
| `sub4_final/final_report.json` | E\*, bias, test class shares, agreement with each fold model's test predictions, train-sample F1 at E\*, `labels_changed_by_bias`, `flags`, `upload_ok` |
| `sub4_final/best_model/` | `state_dict_fp16.pt` of the selected full-data model + tokenizer/config. Rebuild with `AutoModelForSequenceClassification.from_pretrained(MODEL_DIR, num_labels=3, ignore_mismatched_sizes=True)` then `load_state_dict(torch.load(...))` |
| `sub4/*/meta.json`, `sub4/*.log` | per-run timings, s/step, versions |

`finalize` deletes the unselected weight files (≈ 0.6 GB each) so `/kaggle/working` stays under 20 GB.

## 7. Submit / report

1. Upload **`/kaggle/working/sub4_final/submission.csv`** (download from the output panel → competition **Submit Prediction**) only if **all** hold:
   - `finalize` printed **`SUBMISSION_READY`** (test class shares within 3 points of the prior, agreement with every fold model ≥ 0.95, train-sample F1 at E\* ≥ 0.97);
   - `cv_macro_f1_at_E` **≥ 0.950** (Granite fold 0 was 0.9589 for reference);
   - it is **before 17:45 UTC+7**.
   Description: `sub4 wolf full-data E*=<E> K=<K> cv=<x.xxxx> bias=<adopted|off>`.
2. Otherwise **do not upload** — report `final_report.json` and `selection.json`; Granite (0.96808) stays our best.
3. Fill row `sub4` and ledger row 4 in [`experiments/EXPERIMENT_LOG.md`](../experiments/EXPERIMENT_LOG.md): K + folds used, E\*, `cv_macro_f1_at_E` ± std, per-epoch mean macro F1, LOFO gains, bias adopted / values, `labels_changed_by_bias`, s/step, full-run and per-fold minutes, public score, notebook version.
4. After uploading, **save the notebook version** (Quick Save is fine) so judges can see the code. Final-submission choice (Sub 4 vs Sub 3) is made by the lead.

**Time plan (UTC+7):**

| Time | Step |
|---|---|
| 14:50–15:05 | setup, script on disk, smoke run (§2, §5.1, §5.3) |
| 15:05 | launch (§5.4) |
| 15:05–17:00 | monitor; folds start only if they can end by 17:00 |
| 17:00–17:05 | select + finalize (§5.5) |
| **17:15** | **upload target** |
| **17:45** | **hard cutoff — nothing after this** |

Rough durations (unmeasured — the smoke run gives the real s/step): at ≈ 0.8 s/step full ≈ 31 min, fold ≈ 23 min → all 5 folds by ≈ 16:20; at ≈ 1.5 s/step full ≈ 52 min, fold ≈ 40 min → K ≈ 3–4.

## 8. Do-not list

- Do not average folds, seeds or models; do not submit a fold model's predictions (no ensemble).
- Do not change the split, label order, input text, or the 4-epoch schedule (E\* is only meaningful if folds and full run share it).
- Do not choose E\* or the bias from the public leaderboard.
- Do not use any second pretrained model, external data, translation, or pseudo-labels; do not hand-edit test predictions.
- Do not make the notebook public. Do not upload after **17:45 UTC+7**.

## 9. Failure playbook

| Symptom | Action |
|---|---|
| OOM in smoke run or launch | `EXTRA = ["--bs", "4"]` (and `--bs 4` in the smoke command) |
| `fold-0 hash mismatch` | wrong `folds.csv` — attach the repo file; never change the split |
| Missing **encoder** weights in the load report | `MODEL_DIR` is the wrong folder — use the one with `model.safetensors` + `config.json` |
| `collapse guard` in a fold log | that fold is skipped automatically; nothing to do unless all folds fail (then LR/model problem — report) |
| `full` FAILED | relaunch alone: `!CUDA_VISIBLE_DEVICES=0 python /kaggle/working/sub4.py --mode full --model_dir "$MODEL_DIR" --folds "$FOLDS" --out "$OUT"`; if it cannot reach epoch 3–4 by ≈ 17:10, skip Sub 4 |
| `E*=… not in full-run epochs` | full run has not reached E\* yet — wait if time allows; otherwise stop and report |
| `CHECK_FLAGS_BEFORE_UPLOAD` | read `flags` in `final_report.json`; low agreement or odd class shares = pipeline problem → do not upload |
| 17:20 and not `SUBMISSION_READY` | stop; do not start anything new |
