import gc, json, time
import numpy as np, torch
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from transformers import TrainingArguments, DataCollatorWithPadding

# 20-step smoke on the real data, evaluated on 256 fold-0 rows.
smoke = WeightedTrainer(
    model=model, args=TrainingArguments(
        output_dir=f"{OUT}/smoke", max_steps=20, per_device_train_batch_size=16,
        gradient_accumulation_steps=2, per_device_eval_batch_size=16, learning_rate=1e-4,
        bf16=True, eval_strategy="steps", eval_steps=20, save_strategy="no",
        report_to="none", logging_steps=10, label_names=["labels"], group_by_length=True,
        dataloader_num_workers=2, seed=42, warmup_ratio=0.05, weight_decay=0.0, max_grad_norm=1.0),
    train_dataset=tr_ds, eval_dataset=DS(va_ids[:256], va_y[:256]),
    data_collator=DataCollatorWithPadding(tok), compute_metrics=compute_metrics, processing_class=tok)
t0 = time.time()
smoke.train()
smoke_s = time.time() - t0
sec_per_step = smoke_s / 20
projected_min = sec_per_step * 700 / 60
N_EPOCHS = 1 if projected_min > 75 else 2
print(f"SMOKE {smoke_s:.0f}s for 20 steps ({sec_per_step:.2f}s/step); 700-step projection {projected_min:.0f} min -> epochs {N_EPOCHS}")
del smoke
gc.collect(); torch.cuda.empty_cache()

# Fresh model so the smoke steps do not count as training.
del model
gc.collect(); torch.cuda.empty_cache()
transformers.set_seed(42)
model = build_model()

args = TrainingArguments(
    output_dir=f"{OUT}/checkpoints", num_train_epochs=N_EPOCHS, per_device_train_batch_size=16,
    gradient_accumulation_steps=2, per_device_eval_batch_size=16, learning_rate=1e-4,
    warmup_ratio=0.05, weight_decay=0.0, max_grad_norm=1.0, bf16=True, fp16=False,
    group_by_length=True, eval_strategy="steps", save_strategy="steps",
    eval_steps=175, save_steps=175, save_total_limit=1, save_only_model=True,
    load_best_model_at_end=True, metric_for_best_model="macro_f1", greater_is_better=True,
    report_to="none", logging_steps=25, label_names=["labels"], dataloader_num_workers=2, seed=42)
trainer = WeightedTrainer(
    model=model, args=args, train_dataset=tr_ds, eval_dataset=va_ds,
    data_collator=DataCollatorWithPadding(tok), compute_metrics=compute_metrics, processing_class=tok)
t0 = time.time()
trainer.train()
train_s = time.time() - t0
print(f"TRAIN_DONE {train_s / 60:.1f} min, best step {trainer.state.best_model_checkpoint}")

from scipy.special import softmax as _sm
def probs(ds):
    pred = trainer.predict(ds)
    logits = pred.predictions[0] if isinstance(pred.predictions, tuple) else pred.predictions
    return _sm(logits.astype(np.float64), axis=1)

t0 = time.time(); va_p = probs(va_ds); va_s = time.time() - t0
t0 = time.time(); te_p = probs(te_ds); te_s = time.time() - t0
y = np.array(va_y); yhat = va_p.argmax(1)
macro = f1_score(y, yhat, average="macro")
print("VAL_MACRO_F1", round(float(macro), 4))
print(classification_report(y, yhat, target_names=LABELS, digits=4))
print("confusion rows=true", confusion_matrix(y, yhat).tolist())

va_out = va_df[["id", "label"]].copy()
for i, name in enumerate(LABELS):
    va_out[f"p_{name}"] = va_p[:, i]
te_out = pd.DataFrame({"id": test.id})
for i, name in enumerate(LABELS):
    te_out[f"p_{name}"] = te_p[:, i]
sub = pd.DataFrame({"id": test.id, "label": [LABELS[i] for i in te_p.argmax(1)]})
va_out.to_csv(f"{OUT}/valid_probs.csv", index=False)
te_out.to_csv(f"{OUT}/test_probs.csv", index=False)
sub.to_csv(f"{OUT}/submission.csv", index=False)

assert list(sub.columns) == ["id", "label"] and len(sub) == 6000
assert sub.id.is_unique and set(sub.id) == set(test.id)
assert sub.label.isin(LABELS).all() and not sub.isna().any().any()
print(sub.label.value_counts(normalize=True).round(3).to_dict())
print("VALIDATOR_OK")

per_lang = {}
for g in ["pt", "fr", "it", "en"]:
    m = (va_df.language == g).values
    per_lang[g] = {"macro_f1": float(f1_score(y[m], yhat[m], average="macro")), "n": int(m.sum())}
report = classification_report(y, yhat, target_names=LABELS, output_dict=True, digits=4)
metrics = {
    "model_id": MODEL_DIR, "macro_f1": float(macro),
    "per_class": {k: {m: float(report[k][m]) for m in ("precision", "recall", "f1-score", "support")} for k in LABELS},
    "confusion_matrix": confusion_matrix(y, yhat).tolist(), "per_lang": per_lang,
    "best_checkpoint": str(trainer.state.best_model_checkpoint), "max_len": MAX_LEN,
    "truncation_rate_train_folds": TRUNC_RATE, "epochs": N_EPOCHS, "sec_per_smoke_step": sec_per_step,
    "class_weights": [float(x) for x in CLASS_W], "lora_r": 16, "lora_alpha": 32,
    "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    "learning_rate": 1e-4, "transformers": transformers.__version__, "torch": torch.__version__,
    "gpu": torch.cuda.get_device_name(0), "train_minutes": train_s / 60,
    "valid_predict_s": va_s, "test_predict_s": te_s, "fold0_sha_ok": True,
    "gate_pass": bool(macro >= 0.93),
}
(json_path := f"{OUT}/metrics.json")
open(json_path, "w", encoding="utf-8").write(json.dumps(metrics, indent=2))
model.backbone.save_pretrained(f"{OUT}/best_model")
torch.save(model.score.state_dict(), f"{OUT}/best_model/score.pt")
tok.save_pretrained(f"{OUT}/best_model")
print("SAVED", OUT, "gate_pass", metrics["gate_pass"])
