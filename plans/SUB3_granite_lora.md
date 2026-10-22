# SUB 3 — `granite-3.3-2b-instruct` + LoRA classifier, fold 0 (Colab A100)

> **Audience:** the person or agent running this experiment.
> **Team:** UTS · Saigon South (SGS) · InnoLab. **Final C1 notebook name:** `UTS_SGS_InnoLab_C1` (only the winning pipeline gets this name).
> **Deadline:** Kaggle closes **22 Oct 2026, 18:00 UTC+7 (11:00 UTC)**. Target: this run finished and reported by **~14:30 UTC+7**.
> **Sister experiments:** [SUB1_wolf_defender.md](SUB1_wolf_defender.md) and [SUB2_xlmr_large.md](SUB2_xlmr_large.md). Same contract (§3) — do not change it, otherwise the three results are not comparable.
> **Open rules question:** whether training on Colab (external compute) is allowed for C1 is unconfirmed (`docs/agent.md` §23.1). Ask the organizers before submitting a Colab-trained result.

---

## 0. Goal in one paragraph

Fine-tune **one** approved backbone, `ibm-granite/granite-3.3-2b-instruct` (decoder LLM, 2.56B params, 40 layers, natively supports pt/fr/it/en), as a 3-class classifier with **LoRA adapters + a linear head on the last token**, train on folds 1–4, select the best checkpoint by **macro F1 on fold 0**, predict all 6,000 test rows, and save probabilities + metrics + a valid `submission.csv`. This is the third *exploration* submission: it answers "does a 5× larger, instruction/safety-aligned LLM read the subtle BENIGN ↔ JAILBREAK intent better than the encoders in Sub 1 and Sub 2?"

## 1. Background you need (short)

- Task: multilingual (pt 45%, fr 25%, it 20%, en 10%) prompt-attack classification. Metric: **macro F1** — the 7% `PROMPT_INJECTION` class weighs as much as the 62% `BENIGN` class.
- Train 14,000 rows (`id,text,language,label`), test 6,000 (`id,text,language`). Clean UTF-8; newlines were stripped from all texts.
- Class counts: BENIGN 8,680 · JAILBREAK 4,340 · PROMPT_INJECTION 980. Every language has the same class ratio.
- Hardest confusion (TF-IDF 5-fold macro F1 **0.928**): BENIGN ↔ JAILBREAK (same role-play style, differ only by harmful intent), then INJECTION → BENIGN. **Your result must beat 0.93 to be worth submitting.**
- Rules: exactly **one** approved pretrained backbone per notebook; name it at the top; no other pretrained models, no external data. LoRA on an approved model is the organizer-listed route for the LLMs.

## 2. Verified environment facts (checked on Colab A100, 22 Oct ~13:00)

| Fact | Consequence |
|---|---|
| `transformers` 5.20.1 has **no** `GraniteForSequenceClassification`; `AutoModelForSequenceClassification` cannot load Granite | Use the small custom `GraniteClassifier` wrapper in §5.1 (backbone `AutoModel` + linear head on the last real token). `docs/agent.md` §26.4 suggests `AutoModelForSequenceClassification` — that does not work here. |
| Colab ships `torchao` 0.10.0; `peft` 0.21.2 refuses it (`ImportError: ... only versions above 0.16.0`) when creating LoRA layers | First code cell: `!pip uninstall -y -q torchao`, then restart the kernel. LoRA does not need torchao. |
| Tokenizer: `pad = eos = bos = <|end_of_text|>` (id 0), `padding_side="left"`, **no special tokens added** to plain text | Do not rely on `pad_token_id` to find the last token (it equals EOS). The wrapper locates the last real token from `attention_mask`, which works with left or right padding. |
| Granite token lengths on `train.csv`: median 90, p95 481, p99 918; **> 512: 4.21%**, **> 1024: 0.81%**, > 2048: 0.13% | `MAX_LEN = 1024` truncates < 1% of rows (head + tail rule still applies). |
| `flash-attn` not installed; bf16 supported on A100 | `attn_implementation="sdpa"`, `bf16=True`. |
| Smoke test, worst case (16 longest train texts × 1024 tokens, LoRA r=16, gradient checkpointing, bf16): **2.94 s fwd+bwd, peak 11.1 GiB**; 8 × 1024: 1.88 s, 8.0 GiB | Batch **16 × 2** (effective 32) fits with large headroom on 40 GB. |
| Smoke test, `WeightedTrainer` 8 steps at batch 16 on random rows + eval + checkpoint save + best-model reload + `predict`: **all passed**; ≈ 1.4 s/step; a full-model checkpoint save/reload costs ≈ 20 s | Estimated full run (700 steps + 4 evals + 4 saves + test inference): **≈ 40–45 min** — an extrapolation, confirm with §5.3. |
| A head initialised with `normal(std=0.02)` starts at loss ≈ 14.6 (Granite hidden states have a large norm); a **zero-initialised** head starts at 1.10 (≈ ln 3) | Use `nn.init.zeros_` for the head (§5.1). |
| `TrainingArguments(save_safetensors=...)` raises `TypeError` in transformers 5.20 | Do not pass it. |

Hardware: **use an A100 (or L4) Colab runtime.** A T4 has no bf16; fp16 training of Granite is prone to overflow and is several times slower. Do not run this on Kaggle T4 unless you accept that risk.

## 3. Shared experiment contract (identical in Sub 1, 2, 3 — do not change)

| Item | Value |
|---|---|
| Label map | `LABELS = ["BENIGN", "JAILBREAK", "PROMPT_INJECTION"]`, `label2id = {l: i ...}`, `id2label` inverse. Export **strings**. |
| Split | `data/folds.csv` (`id,fold`). **Train = folds 1–4 (11,200 rows), Valid = fold 0 (2,800 rows).** |
| Split check | Fold-0 count **2,800**; `sha256(",".join(sorted(fold0_ids)))` == `0d043c7efa1612b707d4a52e4b73a0aa311ee37fb3f9cadf213ec47269a57fc4`. Abort if it differs. (Verified on the Colab copy at `/content/folds.csv`.) |
| Input | raw `text` only, unmodified (no lowercasing, no cleaning, no prompt template, no language prefix). Read CSVs with `dtype=str, keep_default_na=False, encoding="utf-8"`. |
| Long inputs | head + tail truncation: keep first 75% / last 25% of the token budget (`encode` in Sub 1/2 §4 — reuse it unchanged; Granite adds no special tokens, so the slice is plain text tokens). |
| Loss | cross-entropy with class weights `w_c = sqrt(n_max / n_c)` computed on the **training folds** → ≈ `[1.00, 1.41, 2.98]` (`WeightedTrainer` from Sub 1/2 §4). |
| Seed | `transformers.set_seed(42)` |
| Selection | evaluate **twice per epoch**; `metric_for_best_model="macro_f1"`, `load_best_model_at_end=True`, `save_total_limit=1`. |
| Test prediction | plain `argmax` of softmax probabilities. **No threshold tuning** (that is Sub 5). |
| Gate | submit to Kaggle only if fold-0 macro F1 ≥ **0.93** and the validator passes. |

Paths on Colab: `TRAIN_PATH = "/content/train.csv"`, `TEST_PATH = "/content/test.csv"`, `FOLDS_PATH = "/content/folds.csv"`. Model: `MODEL_DIR = "ibm-granite/granite-3.3-2b-instruct"` (Hugging Face download, ~5 GB, ≈ 1 min on Colab). On Kaggle use the attached `aisuko/granite-3.3-2b-instruct` model folder instead.

## 4. Setup cells (run in this order)

```python
# Cell 1 — once per fresh runtime, then Runtime/Kernel → Restart
!pip uninstall -y -q torchao
```

```python
# Cell 2 — after the restart
import os; os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import hashlib, json, time, numpy as np, pandas as pd, torch, torch.nn as nn, transformers, peft
from sklearn.metrics import f1_score, classification_report, confusion_matrix
print("transformers", transformers.__version__, "| peft", peft.__version__, "| GPU", torch.cuda.get_device_name(0))
assert torch.cuda.is_bf16_supported(), "Use an A100/L4 runtime for Sub 3"
transformers.set_seed(42)
# then: LABELS / folds / hash check / tr_df / va_df / encode / DS / CLASS_W / WeightedTrainer / compute_metrics
# exactly as in SUB1_wolf_defender.md §4
```

First markdown cell must state: **`Model: ibm-granite/granite-3.3-2b-instruct (single approved backbone, LoRA r=16 + linear head)`**, seed, split, and hyperparameters.

## 5. Sub-3-specific instructions

### 5.1 Model wrapper (verified to load, train, checkpoint and predict with transformers 5.20.1 / peft 0.21.2)

```python
from transformers import AutoModel, AutoTokenizer
from transformers.modeling_outputs import SequenceClassifierOutput
from peft import LoraConfig, get_peft_model

class GraniteClassifier(nn.Module):
    """Granite backbone + linear head on the last real (non-pad) token."""
    def __init__(self, path, num_labels=3):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(path, dtype=torch.bfloat16, attn_implementation="sdpa")
        self.score = nn.Linear(self.backbone.config.hidden_size, num_labels, bias=False)  # fp32, trainable
        nn.init.zeros_(self.score.weight)  # random init gives initial loss ≈ 14.6 because of the large hidden-state norm
    def forward(self, input_ids, attention_mask, labels=None):
        h = self.backbone(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        last = attention_mask.shape[1] - 1 - attention_mask.flip(1).argmax(1)  # works for left or right padding
        logits = self.score(h[torch.arange(h.size(0), device=h.device), last].float())
        return SequenceClassifierOutput(logits=logits)

tok = AutoTokenizer.from_pretrained(MODEL_DIR)
model = GraniteClassifier(MODEL_DIR)
model.backbone.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
model.backbone = get_peft_model(model.backbone, LoraConfig(
    r=16, lora_alpha=32, lora_dropout=0.05,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]))
model.cuda()
print(sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6, "M trainable")  # expect ≈ 28.2M
```

- The head is created fresh (there is no pretrained classification head); that is expected.
- Gradient checkpointing only works in training mode. `Trainer` calls `model.train()` for you; if you write any manual training step, call `model.train()` first or a batch of 8 × 1024 tokens will run out of 40 GB.
- Do **not** pass `gradient_checkpointing=True` to `TrainingArguments` (the wrapper is a plain `nn.Module`); it is enabled on the backbone above.
- `WeightedTrainer` is required: the wrapper returns only logits, so the stock `Trainer` loss path would fail.

### 5.2 Hyperparameters

| Setting | Value |
|---|---|
| `MAX_LEN` | `1024` (0.81% of train rows truncated; record the real rate printed by `encode`) |
| `learning_rate` | `1e-4` (LoRA + head) |
| `num_train_epochs` | `2` |
| `warmup_ratio` / scheduler | `0.05` / linear |
| `weight_decay` | `0.0` |
| `max_grad_norm` | `1.0` |
| `per_device_train_batch_size` × `gradient_accumulation_steps` | `16 × 2` (effective 32; measured peak 11.1 GiB); drop to `8 × 4` on OOM |
| `per_device_eval_batch_size` | `16` |
| `bf16` / `fp16` | `True` / `False` |
| `group_by_length` | `True` |
| `eval_strategy` / `save_strategy` | `"steps"` / `"steps"`, `eval_steps = save_steps = 175` (≈ ½ epoch: 11,200 / 32 = 350 steps/epoch → 700 steps total, 4 evaluations) |
| `save_only_model` | `True` (checkpoints hold the full 5 GB state; optimizer state is not needed) |
| `label_names` | `["labels"]` |
| `report_to` | `"none"` |
| `dataloader_num_workers` | `2` |
| `processing_class` | `tok` (`tokenizer=` was removed in transformers 5) |

### 5.3 Smoke run first (≈ 3 min)

Train for `max_steps=20` with one evaluation at step 20 on 256 fold-0 rows. Confirm: loss decreases, no OOM, eval returns `macro_f1`, prediction shape `(n, 3)`. Record **seconds per step** and extrapolate: 700 steps + 4 fold-0 evaluations + 6,000-row test inference. If the extrapolated total is beyond **75 minutes**, set `num_train_epochs=1` (350 steps) and record that.

### 5.4 Collapse guard

If fold-0 macro F1 at the first evaluation (step 175) is **< 0.50**, stop and restart once with `learning_rate=5e-5` (keep everything else). If it collapses again, stop and report — do not keep burning GPU time.

## 6. After training — evaluation and outputs

Identical to Sub 1/2 §6, with `OUT = "/content/outputs/sub3_granite_f0"`:

| File | Content |
|---|---|
| `valid_probs.csv` | `id, label, p_BENIGN, p_JAILBREAK, p_PROMPT_INJECTION` (2,800 rows, fold 0) |
| `test_probs.csv` | `id, p_BENIGN, p_JAILBREAK, p_PROMPT_INJECTION` (6,000 rows, **test.csv order**) |
| `submission.csv` | `id,label` from `argmax`, `index=False` |
| `metrics.json` | as Sub 1/2, plus `lora_r`, `lora_alpha`, `target_modules`, epochs actually used, seconds/step |
| `best_model/` | **adapter + head only**: `model.backbone.save_pretrained(f"{OUT}/best_model")` and `torch.save(model.score.state_dict(), f"{OUT}/best_model/score.pt")` (≈ 115 MB instead of 5 GB) |

Run the structural validator (`scripts/validate_submission.py` logic) on `submission.csv`.

**Download `OUT/` before the Colab runtime ends** (server files view → right-click → Download), or mount Drive and set `OUT` under `/content/drive/MyDrive/...`. Sub 4/5 need `valid_probs.csv` and `test_probs.csv`.

## 7. Submit / report

1. If **fold-0 macro F1 ≥ 0.93** and the validator passes → submit with description `sub3 granite-lora f0 valF1=<x.xxxx> len=1024`.
2. If < 0.93 → **do not submit**; report metrics and the 10 most confident errors per confusion type.
3. Fill row `sub3` in [`experiments/EXPERIMENT_LOG.md`](../experiments/EXPERIMENT_LOG.md): val macro F1, per-class F1, per-language F1, best step, truncation %, epochs used, train minutes, seconds/step, Kaggle public score (if submitted).

## 8. Do-not list

- Do not change the split, label order, input text, or loss — that breaks comparison with Sub 1/2.
- Do not wrap texts in a chat template or instruction prompt (changes the input contract).
- Do not use any second pretrained model, external data, translation, or pseudo-labels.
- Do not tune thresholds or pick checkpoints using the public leaderboard.
- Do not use `id` or `language` as features. Do not hand-edit test predictions.
- Do not make the notebook public during the competition.

## 9. Failure playbook

| Symptom | Action |
|---|---|
| `ImportError: Found an incompatible version of torchao` | §4 cell 1 (uninstall torchao) and restart the kernel |
| `ValueError: Unrecognized configuration class ... GraniteConfig ... AutoModelForSequenceClassification` | Use the §5.1 wrapper, not `AutoModelForSequenceClassification` |
| `CUDA out of memory` | Confirm `model.train()` / Trainer is used (checkpointing only active in training mode) → batch `8 × 4` → `MAX_LEN = 768` (1.66% truncated). After an OOM in a notebook, the traceback keeps the tensors alive: run `import sys, gc, torch; sys.last_exc = sys.last_traceback = sys.last_value = None; gc.collect(); torch.cuda.empty_cache()` or restart the runtime |
| Initial loss ≈ 10–15 instead of ≈ 1.1 | Head was randomly initialised — use `nn.init.zeros_` (§5.1) |
| Trainer error about missing `loss` in outputs | Use `WeightedTrainer` (§3) |
| Macro F1 stuck ≈ 0.25–0.4 | Head not training: check `model.score.weight.requires_grad` is `True`, labels reach the loss, LR not 0 |
| Loss `nan` | Confirm `bf16=True` (not fp16) and loss computed on `logits.float()`; lower LR to 5e-5 |
| Very slow (> 4 s per optimizer step at 16 × 2) | Confirm A100/L4, `bf16=True`, `group_by_length=True`; otherwise drop to 1 epoch |
