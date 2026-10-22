"""Cell 3+5+6+7 for Kaggle Notebook — Sub 5 Granite-3.3-2B LoRA (Fast Lite + Time-Guarded).

Designed to finish before 17:45 UTC+7 on Kaggle:
1. Uses Sub 5 Lite schedule by default (MODE="full", all 14,000 rows, 2.0 epochs = 875 steps,
   cosine schedule, wd=0.01) so no second shadow GPU run is needed.
2. Evaluates test logits using the ordered inference loop (never trainer.predict) at
   epochs [1.0, 1.5, 2.0] and IMMEDIATELY writes a validated /kaggle/working/submission.csv
   at each snapshot so a submission-ready CSV is always on disk.
3. Enforces a wall-clock safety guard (17:25 UTC+7 / 10:25 UTC, or MAX_TRAIN_MINUTES=38):
   if training approaches the cutoff after >= 1.0 epoch, it stops cleanly, runs final ordered
   inference on test.csv, and writes /kaggle/working/submission.csv immediately.
"""
import datetime as dt
import json
import os
import time

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score
from transformers import DataCollatorWithPadding, TrainerCallback, TrainingArguments

MODE = "full"  # Train single model on all 14,000 labelled rows (no ensemble)
EPOCHS = 2.0   # 875 optimizer steps at effective batch 32 (~25-32 min)
SCHED = "cosine"
WD = 0.01

# Auto-size micro-batch for 16 GB T4 (8 x 4 = 32) vs >=24 GB L4/A100 (16 x 2 = 32)
vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
BS, GA = (16, 2) if vram_gb >= 22.0 else (8, 4)
INF_BS = 32 if vram_gb >= 22.0 else 16

# Snapshot only at 1.0, 1.5, 2.0 epochs (saves ~7 min vs 6 snapshots while keeping safety checkpoints)
GRID_E = [1.0, 1.5, 2.0]

# Hard wall-clock cutoff: 2026-10-22 10:25 UTC (17:25 UTC+7) or 38 minutes max runtime
HARD_CUTOFF_UTC = dt.datetime(2026, 10, 22, 10, 25, tzinfo=dt.timezone.utc)
MAX_TRAIN_MINUTES = 38.0

WORK_ROOT = "/kaggle/working" if os.path.exists("/kaggle/working") else "/content"
OUT5 = os.path.join(WORK_ROOT, f"sub5_granite_{MODE}")
os.makedirs(OUT5, exist_ok=True)
TOP_SUBMISSION_CSV = os.path.join(WORK_ROOT, "submission.csv")

fit_df = train.reset_index(drop=True) if MODE == "full" else tr_df
fit_ids = all_train_ids if MODE == "full" else tr_ids

counts = fit_df.label.value_counts().reindex(LABELS).values
CLASS_W = torch.tensor(np.sqrt(counts.max() / counts), dtype=torch.float)
fit_ds = DS(fit_ids, fit_df.label.map(label2id).tolist())
print(f"MODE={MODE} | rows={len(fit_df)} | BS={BS}x{GA} | class_weights={CLASS_W.tolist()}")

sub3_lab = None
if SUB3_CSV and os.path.exists(SUB3_CSV):
    s3 = pd.read_csv(SUB3_CSV, dtype=str)
    if s3.id.tolist() == test.id.tolist():
        sub3_lab = s3.label.map(label2id).values
        print(f"Loaded Sub 3 reference ({SUB3_CSV}) for agreement check.")

collate = DataCollatorWithPadding(tok)


def ordered_logits(m, ids, bs=INF_BS):
    """Length-sorted batches, results written back to the ORIGINAL row order."""
    was_training, order = m.training, np.argsort([len(x) for x in ids])
    out = np.zeros((len(ids), 3), dtype=np.float32)
    m.eval()
    with torch.no_grad():
        for s in range(0, len(ids), bs):
            idx = order[s : s + bs]
            b = collate([{"input_ids": ids[i], "attention_mask": [1] * len(ids[i])} for i in idx])
            logits = m(
                input_ids=b["input_ids"].cuda(),
                attention_mask=b["attention_mask"].cuda(),
            ).logits.float()
            out[idx] = torch.nan_to_num(logits, nan=0.0).cpu().numpy()
    m.train(was_training)
    return out


def write_validated_submission(te_logits, tag_epoch):
    pred = te_logits.argmax(1)
    sub = pd.DataFrame({"id": test.id, "label": [LABELS[i] for i in pred]})
    shares = np.bincount(pred, minlength=3) / len(pred)
    shares_ok = bool(np.abs(shares - np.array([0.62, 0.31, 0.07])).max() <= 0.03)
    agree_s3 = float((pred == sub3_lab).mean()) if sub3_lab is not None else None

    assert list(sub.columns) == ["id", "label"] and len(sub) == 6000 and sub.id.is_unique
    assert sub.id.tolist() == test.id.tolist() and sub.label.isin(LABELS).all()

    snap_csv = os.path.join(OUT5, f"submission_e{tag_epoch}.csv")
    sub.to_csv(snap_csv, index=False)
    # Always keep the latest valid submission at /kaggle/working/submission.csv
    sub.to_csv(TOP_SUBMISSION_CSV, index=False)
    sub.to_csv(os.path.join(OUT5, "submission.csv"), index=False)

    report = {
        "E_star": tag_epoch,
        "bias": [0.0, 0.0, 0.0],
        "labels_changed_by_bias": 0,
        "test_shares": shares.round(3).tolist(),
        "shares_ok": shares_ok,
        "agree_sub3": round(agree_s3, 4) if agree_s3 is not None else None,
        "submission_path": TOP_SUBMISSION_CSV,
    }
    json.dump(report, open(os.path.join(OUT5, "final_report.json"), "w"), indent=2)
    return report


class FastKaggleSnapshot(TrainerCallback):
    def on_train_begin(self, args, state, control, **kw):
        self.at = {round(state.max_steps * e / EPOCHS): e for e in GRID_E}
        self.recs = []
        self.t0 = time.time()
        self.emergency_saved = False
        print("max_steps:", state.max_steps, "| scheduled snapshots:", self.at)

    def _run_snapshot(self, e, step, model):
        mins = round((time.time() - self.t0) / 60, 1)
        te = ordered_logits(model, te_ids)
        np.save(os.path.join(OUT5, f"test_logits_e{e}.npy"), te)
        rep = write_validated_submission(te, e)
        rec = {
            "epoch": e,
            "step": step,
            "minutes": mins,
            "test_shares": rep["test_shares"],
            "shares_ok": rep["shares_ok"],
            "agree_sub3": rep["agree_sub3"],
        }
        self.recs.append(rec)
        json.dump(self.recs, open(os.path.join(OUT5, "snapshots.json"), "w"), indent=2)
        print(f"SNAPSHOT_SAVED -> {TOP_SUBMISSION_CSV} :", rec)

    def on_step_end(self, args, state, control, model=None, **kw):
        e = self.at.get(state.global_step)
        if e is not None:
            self._run_snapshot(e, state.global_step, model)
            if e == EPOCHS:
                d = os.path.join(OUT5, f"adapter_e{e}")
                model.backbone.save_pretrained(d)
                torch.save(model.score.state_dict(), os.path.join(d, "score.pt"))
            return

        # Time-cutoff guard: if past 1.0 epoch and approaching 17:25 UTC+7 or MAX_TRAIN_MINUTES
        cur_ep = round(EPOCHS * state.global_step / max(state.max_steps, 1), 2)
        elapsed_min = (time.time() - self.t0) / 60.0
        now_utc = dt.datetime.now(dt.timezone.utc)
        if (
            not self.emergency_saved
            and cur_ep >= 1.0
            and (elapsed_min >= MAX_TRAIN_MINUTES or now_utc >= HARD_CUTOFF_UTC)
        ):
            self.emergency_saved = True
            print(
                f"TIME_CUTOFF_GUARD triggered at step {state.global_step} "
                f"(epoch {cur_ep}, {elapsed_min:.1f} min). Saving final test predictions now..."
            )
            self._run_snapshot(cur_ep, state.global_step, model)
            d = os.path.join(OUT5, f"adapter_e{cur_ep}")
            model.backbone.save_pretrained(d)
            torch.save(model.score.state_dict(), os.path.join(d, "score.pt"))
            control.should_training_stop = True


args = TrainingArguments(
    output_dir=os.path.join(WORK_ROOT, f"sub5_{MODE}_tmp"),
    num_train_epochs=EPOCHS,
    per_device_train_batch_size=BS,
    gradient_accumulation_steps=GA,
    learning_rate=1e-4,
    lr_scheduler_type=SCHED,
    warmup_ratio=0.05,
    weight_decay=WD,
    max_grad_norm=1.0,
    bf16=USE_NATIVE_BF16,
    fp16=not USE_NATIVE_BF16,
    group_by_length=True,
    eval_strategy="no",
    save_strategy="no",
    report_to="none",
    logging_steps=25,
    label_names=["labels"],
    dataloader_num_workers=2,
    seed=42,
)

trainer = WeightedTrainer(
    model=model,
    args=args,
    train_dataset=fit_ds,
    data_collator=collate,
    processing_class=tok,
    callbacks=[FastKaggleSnapshot()],
)

t0 = time.time()
trainer.train()
print(f"TRAIN_DONE {MODE} in {(time.time() - t0) / 60:.1f} min")

final_rep_path = os.path.join(OUT5, "final_report.json")
if os.path.exists(final_rep_path):
    final_rep = json.load(open(final_rep_path))
    print("\nFINAL_REPORT:")
    print(json.dumps(final_rep, indent=2))
    if final_rep["shares_ok"] and (final_rep["agree_sub3"] is None or final_rep["agree_sub3"] >= 0.95):
        print(f"\nSUBMISSION_READY -> {TOP_SUBMISSION_CSV}")
