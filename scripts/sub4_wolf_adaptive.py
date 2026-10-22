"""Sub 4: wolf-defender with adaptive-K selection, then ONE model trained on all 14,000 rows (Kaggle T4).

Single model, no ensemble. The K-fold runs are used only to *choose settings*; they never vote on test.

Modes
  fold      --fold k   Train on folds != k with the full 4-epoch schedule; score fold k every half epoch.
  full                 Train on all 14,000 rows with the same schedule; save test logits (+ fp16 weights)
                       every half epoch. Runs at the same time as the folds (it does not need their results).
  select               CPU. Use the folds that finished (K adapts to the time left) to choose
                       E* = number of epochs, and a per-class logit bias that is kept only if it
                       helps on held-out folds (leave-one-fold-out).
  finalize             CPU. Take the full-data model's test logits at E*, apply the bias if kept,
                       write submission.csv, run checks, keep only the selected weights.

Everything uses a plain PyTorch loop (no HF Trainer), so it works across transformers versions and the
"Trainer reload" bug from Sub 3 cannot happen.
"""
import argparse
import glob
import hashlib
import json
import math
import os
import random
import shutil
import sys
import time

import numpy as np
import pandas as pd

LABELS = ["BENIGN", "JAILBREAK", "PROMPT_INJECTION"]
L2I = {l: i for i, l in enumerate(LABELS)}
FOLD0_SHA = "0d043c7efa1612b707d4a52e4b73a0aa311ee37fb3f9cadf213ec47269a57fc4"
EFFECTIVE_BATCH = 32
PRIOR = np.array([0.62, 0.31, 0.07])


# ----------------------------------------------------------------------------- utilities
def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def read(p):
    return pd.read_csv(p, dtype=str, keep_default_na=False, encoding="utf-8")


def resolve(path, name):
    if path:
        return path
    for pattern in (f"/kaggle/input/**/{name}", f"data/{name}", f"../{name}", name):
        hits = sorted(glob.glob(pattern, recursive=True))
        if hits:
            return hits[0]
    sys.exit(f"cannot find {name}; pass it explicitly")


def mf1(y, yhat):
    """Macro F1 for 3 classes (same definition as sklearn average='macro')."""
    cm = np.bincount(3 * y + yhat, minlength=9).reshape(3, 3)
    tp, denom = np.diag(cm), cm.sum(0) + cm.sum(1)
    return float(np.mean(np.where(denom > 0, 2 * tp / np.maximum(denom, 1), 0.0)))


def softmax(z):
    z = z - z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


def grid_epochs(total, step=0.5):
    return [round(step * i, 4) for i in range(1, int(round(total / step)) + 1)]


def ekey(e):
    return f"{e:.1f}"


def load_data(a):
    train, test, folds = read(resolve(a.train, "train.csv")), read(resolve(a.test, "test.csv")), read(resolve(a.folds, "folds.csv"))
    folds["fold"] = folds["fold"].astype(int)
    train = train.merge(folds, on="id", how="left", validate="one_to_one")
    assert train["fold"].notna().all(), "some train ids missing from folds.csv"
    f0 = sorted(train.loc[train.fold == 0, "id"])
    assert len(f0) == 2800 and hashlib.sha256(",".join(f0).encode()).hexdigest() == FOLD0_SHA, "fold-0 hash mismatch"
    assert len(train) == 14000 and len(test) == 6000
    return train, test


# ----------------------------------------------------------------------------- GPU part
def gpu_main(a):
    import torch
    import torch.nn.functional as F
    from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed); torch.cuda.manual_seed_all(a.seed)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp = dev.type == "cuda"
    import transformers
    log("mode", a.mode, "fold", a.fold, "| torch", torch.__version__, "| transformers", transformers.__version__,
        "| device", torch.cuda.get_device_name(0) if amp else "cpu")

    train, test = load_data(a)
    if a.mode == "fold":
        tr, va = train[train.fold != a.fold].reset_index(drop=True), train[train.fold == a.fold].reset_index(drop=True)
    else:
        tr, va = train.reset_index(drop=True), None
    te = test.reset_index(drop=True)
    if a.limit:  # smoke test only (test set is kept whole so finalize can be exercised)
        tr = tr.sample(n=min(a.limit, len(tr)), random_state=0).reset_index(drop=True)
        va = va.sample(n=min(a.limit // 4, len(va)), random_state=0).reset_index(drop=True) if va is not None else None

    out = os.path.join(a.out, f"fold{a.fold}" if a.mode == "fold" else "full")
    os.makedirs(out, exist_ok=True)
    for stale in ("DONE", "FAILED"):
        if os.path.exists(os.path.join(out, stale)):
            os.remove(os.path.join(out, stale))

    tok = AutoTokenizer.from_pretrained(a.model_dir)
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else (tok.eos_token_id or 0)

    def encode(texts):
        enc = tok(list(texts), add_special_tokens=True, truncation=False)["input_ids"]
        res, cut = [], 0
        for ids in enc:
            if len(ids) > a.max_len:  # head + tail; keeps the first and last special tokens
                h = int(a.max_len * 0.75)
                ids = ids[:h] + ids[-(a.max_len - h):]
                cut += 1
            res.append(ids)
        return res, cut / max(1, len(res))

    tr_x, tr_cut = encode(tr.text)
    te_x, _ = encode(te.text)
    va_x = encode(va.text)[0] if va is not None else None
    tr_y = tr.label.map(L2I).values
    va_y = va.label.map(L2I).values if va is not None else None
    log(f"rows train {len(tr_x)} valid {0 if va is None else len(va_x)} test {len(te_x)} | truncated {tr_cut:.2%} at {a.max_len}")
    # fixed train sample to check the full-data model actually learned (no holdout exists in full mode)
    fit_idx = np.random.default_rng(0).choice(len(tr_x), size=min(2000, len(tr_x)), replace=False)

    cfg = AutoConfig.from_pretrained(a.model_dir)
    log("backbone", cfg.model_type, "| original labels", getattr(cfg, "id2label", None))
    if hasattr(cfg, "reference_compile"):
        cfg.reference_compile = False  # avoid torch.compile on Kaggle T4
    cfg.num_labels, cfg.id2label, cfg.label2id = 3, dict(enumerate(LABELS)), dict(L2I)
    cfg.problem_type = "single_label_classification"
    kw = dict(config=cfg, ignore_mismatched_sizes=True)
    try:
        model = AutoModelForSequenceClassification.from_pretrained(a.model_dir, attn_implementation="sdpa", **kw)
    except (TypeError, ValueError) as e:
        log("sdpa unavailable, default attention:", e)
        model = AutoModelForSequenceClassification.from_pretrained(a.model_dir, **kw)
    model.to(dev).train()

    def collate(seqs):
        m = max(len(s) for s in seqs)
        ids = torch.full((len(seqs), m), pad_id, dtype=torch.long)
        att = torch.zeros((len(seqs), m), dtype=torch.long)
        for i, s in enumerate(seqs):
            ids[i, : len(s)] = torch.tensor(s, dtype=torch.long)
            att[i, : len(s)] = 1
        return ids.to(dev), att.to(dev)

    @torch.no_grad()
    def predict(seqs, bs=32):
        was = model.training
        model.eval()
        order = np.argsort([-len(s) for s in seqs], kind="stable")
        res = np.zeros((len(seqs), 3), dtype=np.float32)
        for s in range(0, len(order), bs):
            b = order[s : s + bs]
            ids, att = collate([seqs[i] for i in b])
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=amp):
                lg = model(input_ids=ids, attention_mask=att).logits
            res[b] = lg.float().cpu().numpy()  # written back to original row order
        if was:
            model.train()
        return res

    # schedule: identical in epoch units for folds and the full run
    accum = max(1, EFFECTIVE_BATCH // a.bs)
    spe = len(tr_x) / (a.bs * accum)  # optimizer steps per epoch
    max_steps = max(1, int(round(a.epochs * spe)))
    grid = {max(1, int(round(e * spe))): e for e in grid_epochs(a.epochs)}
    counts = np.bincount(tr_y, minlength=3)
    w = torch.tensor(np.sqrt(counts.max() / np.maximum(counts, 1)), dtype=torch.float, device=dev)
    decay = [p for p in model.parameters() if p.requires_grad and p.ndim >= 2]
    no_decay = [p for p in model.parameters() if p.requires_grad and p.ndim < 2]
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": a.wd}, {"params": no_decay, "weight_decay": 0.0}], lr=a.lr)
    sched = get_linear_schedule_with_warmup(opt, int(round(a.warmup * max_steps)), max_steps)
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    log(f"bs {a.bs} x accum {accum} | steps/epoch {spe:.1f} | max_steps {max_steps} | grid {sorted(grid.values())} | class w {w.tolist()}")

    rec = {"epochs": [], "valid_logits": [], "test_logits": [], "macro_f1": [], "fit_f1": []}
    if a.mode == "fold":
        rec["valid_ids"], rec["valid_y"] = va.id.values, va_y

    def at_grid(step, e):
        t = time.time()
        if a.mode == "fold":
            lv = predict(va_x)
            f = mf1(va_y, lv.argmax(1))
            rec["valid_logits"].append(lv); rec["macro_f1"].append(f)
            log(f"[fold {a.fold}] epoch {e} step {step}: valid macro F1 {f:.4f} ({time.time() - t:.0f}s)")
            if e == 1.0 and f < 0.5 and not a.limit:
                raise RuntimeError(f"collapse guard: macro F1 {f:.3f} at epoch {e}")
        else:
            lt = predict(te_x)
            lf = predict([tr_x[i] for i in fit_idx])
            f = mf1(tr_y[fit_idx], lf.argmax(1))
            rec["test_logits"].append(lt); rec["fit_f1"].append(f)
            shares = np.bincount(lt.argmax(1), minlength=3) / len(lt)
            log(f"[full] epoch {e} step {step}: train-sample F1 {f:.4f} | test shares {np.round(shares, 3).tolist()} ({time.time() - t:.0f}s)")
            if a.save_weights:
                wdir = os.path.join(out, "weights"); os.makedirs(wdir, exist_ok=True)
                torch.save({k: v.detach().half().cpu() for k, v in model.state_dict().items()}, os.path.join(wdir, f"e{ekey(e)}.pt"))
        rec["epochs"].append(e)
        save_rec()

    def save_rec():
        arrs = {"epochs": np.array(rec["epochs"])}
        if a.mode == "fold":
            arrs.update(valid_logits=np.stack(rec["valid_logits"]), macro_f1=np.array(rec["macro_f1"]),
                        valid_ids=rec["valid_ids"], valid_y=rec["valid_y"])
        else:
            arrs.update(test_logits=np.stack(rec["test_logits"]), fit_f1=np.array(rec["fit_f1"]), test_ids=te.id.values)
        np.savez_compressed(os.path.join(out, "record.npz"), **arrs)

    t0, step, micro, run_loss = time.time(), 0, 0, 0.0
    rng = np.random.default_rng(a.seed)
    lengths = np.array([len(s) for s in tr_x])
    while step < max_steps:
        idx = rng.permutation(len(tr_x))  # length-grouped shuffling: sort within chunks, shuffle batches
        chunks = [idx[s : s + a.bs * 64] for s in range(0, len(idx), a.bs * 64)]
        batches = [c[np.argsort(-lengths[c], kind="stable")][i : i + a.bs] for c in chunks for i in range(0, len(c), a.bs)]
        for bi in rng.permutation(len(batches)):
            b = batches[bi]
            ids, att = collate([tr_x[i] for i in b])
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=amp):
                logits = model(input_ids=ids, attention_mask=att).logits
            loss = F.cross_entropy(logits.float(), torch.as_tensor(tr_y[b], device=dev), weight=w) / accum
            scaler.scale(loss).backward()
            run_loss += loss.item()
            micro += 1
            if micro % accum:
                continue
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True); sched.step()
            step += 1
            if step % 25 == 0 or step == 1:
                el = time.time() - t0
                log(f"step {step}/{max_steps} loss {run_loss / (25 if step > 1 else 1):.4f} | {el / step:.2f}s/step | eta {(max_steps - step) * el / step / 60:.1f} min")
                run_loss = 0.0
            if step in grid:
                at_grid(step, grid[step])
            if step >= max_steps:
                break

    meta = {"mode": a.mode, "fold": a.fold, "model_dir": a.model_dir, "model_type": cfg.model_type, "max_len": a.max_len,
            "lr": a.lr, "epochs": a.epochs, "bs": a.bs, "accum": accum, "seed": a.seed, "steps": step,
            "sec_per_step": (time.time() - t0) / max(1, step), "train_minutes": (time.time() - t0) / 60,
            "truncated_train": tr_cut, "class_weights": w.tolist(), "limit": a.limit,
            "gpu": torch.cuda.get_device_name(0) if amp else "cpu", "torch": torch.__version__, "transformers": transformers.__version__}
    if a.mode == "fold":
        meta["macro_f1_by_epoch"] = {ekey(e): f for e, f in zip(rec["epochs"], rec["macro_f1"])}
        lt = predict(te_x)  # final-epoch test logits: used only for an agreement sanity check, never submitted
        np.save(os.path.join(out, "test_logits_final.npy"), lt)
    else:
        meta["fit_f1_by_epoch"] = {ekey(e): f for e, f in zip(rec["epochs"], rec["fit_f1"])}
        tok.save_pretrained(os.path.join(out, "tokenizer"))
        cfg.save_pretrained(os.path.join(out, "tokenizer"))
    json.dump(meta, open(os.path.join(out, "meta.json"), "w"), indent=2)
    open(os.path.join(out, "DONE"), "w").write("ok")
    log("DONE", out, f"{meta['train_minutes']:.1f} min")


# ----------------------------------------------------------------------------- CPU part
def fit_bias(L, y, lim=3.0, step=0.1):
    """Best b = [0, b_JB, b_INJ] added to logits for macro F1; ties go to the smallest |b|."""
    g = np.round(np.arange(-lim, lim + 1e-9, step), 2)
    combos = sorted(((bj, bi) for bj in g for bi in g), key=lambda t: (abs(t[0]) + abs(t[1]), t))
    best_b, best = np.zeros(3), mf1(y, L.argmax(1))
    for bj, bi in combos:
        b = np.array([0.0, bj, bi])
        s = mf1(y, (L + b).argmax(1))
        if s > best + 1e-9:
            best, best_b = s, b
    return best_b, best


def select_main(a):
    done = sorted(d for d in glob.glob(os.path.join(a.out, "fold*")) if os.path.exists(os.path.join(d, "DONE")))
    recs = [np.load(os.path.join(d, "record.npz"), allow_pickle=True) for d in done]
    K = len(recs)
    sel = {"K": K, "folds_used": [os.path.basename(d) for d in done]}
    if K == 0:
        sel.update(E_star=3.0, bias=[0.0, 0.0, 0.0], bias_adopted=False,
                   note="no fold finished: fallback to the Sub 1 recipe (3 epochs, argmax)")
    else:
        common = sorted(set.intersection(*[set(map(float, r["epochs"])) for r in recs]))
        F = np.array([[float(r["macro_f1"][list(map(float, r["epochs"])).index(e)]) for e in common] for r in recs])
        mean, std = F.mean(0), F.std(0)
        E = common[int(np.argmax(mean))]
        sel.update(epochs=common, mean_macro_f1={ekey(e): float(m) for e, m in zip(common, mean)},
                   std_macro_f1={ekey(e): float(s) for e, s in zip(common, std)},
                   per_fold={sel["folds_used"][k]: {ekey(e): float(F[k, j]) for j, e in enumerate(common)} for k in range(K)},
                   E_star=float(E), cv_macro_f1_at_E=float(mean[common.index(E)]))
        Ls = [r["valid_logits"][list(map(float, r["epochs"])).index(E)] for r in recs]
        ys = [r["valid_y"].astype(int) for r in recs]
        gains, biases = [], []
        if K >= 2:
            for k in range(K):  # leave-one-fold-out: tune on the others, score on fold k
                b, _ = fit_bias(np.concatenate([Ls[j] for j in range(K) if j != k]), np.concatenate([ys[j] for j in range(K) if j != k]))
                gains.append(mf1(ys[k], (Ls[k] + b).argmax(1)) - mf1(ys[k], Ls[k].argmax(1)))
                biases.append(b.tolist())
        need = K if K <= 2 else K - 1
        adopt = K >= 2 and float(np.mean(gains)) > 0 and sum(g > 0 for g in gains) >= need
        Lall, yall = np.concatenate(Ls), np.concatenate(ys)
        b_all, f_b = fit_bias(Lall, yall)
        sel.update(lofo_gains=[float(g) for g in gains], lofo_biases=biases, lofo_required_positive=need,
                   bias_adopted=bool(adopt), bias=(b_all if adopt else np.zeros(3)).tolist(),
                   pooled_oof_macro_f1_raw=mf1(yall, Lall.argmax(1)),
                   pooled_oof_macro_f1_biased_optimistic=float(f_b))
    os.makedirs(a.out, exist_ok=True)
    json.dump(sel, open(os.path.join(a.out, "selection.json"), "w"), indent=2)
    print(json.dumps(sel, indent=2))


def finalize_main(a):
    sel = json.load(open(os.path.join(a.out, "selection.json")))
    full = os.path.join(a.out, "full")
    r = np.load(os.path.join(full, "record.npz"), allow_pickle=True)
    eps = [round(float(e), 4) for e in r["epochs"]]
    E = float(sel["E_star"])
    if E not in eps:
        sys.exit(f"E*={E} not in full-run epochs {eps} (full run not finished that far?)")
    L = r["test_logits"][eps.index(E)].astype(np.float64)
    bias = np.array(sel["bias"], dtype=np.float64)
    pred = (L + bias).argmax(1)
    test = read(resolve(a.test, "test.csv"))
    assert list(r["test_ids"]) == test.id.tolist(), "test logits not in test.csv order"
    probs = softmax(L)

    sub = pd.DataFrame({"id": test.id, "label": [LABELS[i] for i in pred]})
    os.makedirs(a.final_dir, exist_ok=True)
    sub.to_csv(os.path.join(a.final_dir, "submission.csv"), index=False)
    tp = pd.DataFrame({"id": test.id, **{f"p_{l}": probs[:, i] for i, l in enumerate(LABELS)}})
    tp.to_csv(os.path.join(a.final_dir, "test_probs.csv"), index=False)

    chk = pd.read_csv(os.path.join(a.final_dir, "submission.csv"), dtype={"id": str})
    assert chk.columns.tolist() == ["id", "label"] and len(chk) == 6000
    assert chk.id.is_unique and set(chk.id) == set(test.id) and chk.label.isin(LABELS).all() and not chk.isna().any().any()

    shares = np.bincount(pred, minlength=3) / len(pred)
    flags = []
    if np.abs(shares - PRIOR).max() > 0.03:
        flags.append(f"test class shares {np.round(shares, 3).tolist()} differ from prior {PRIOR.tolist()} by > 3 points")
    agree = {}
    for d in sorted(glob.glob(os.path.join(a.out, "fold*"))):
        f = os.path.join(d, "test_logits_final.npy")
        if os.path.exists(f):
            agree[os.path.basename(d)] = float((np.load(f).argmax(1) == pred).mean())
    if agree and min(agree.values()) < 0.95:
        flags.append(f"agreement with a fold model below 0.95: {agree}")
    fit = json.load(open(os.path.join(full, "meta.json"))).get("fit_f1_by_epoch", {}).get(ekey(E))
    if fit is not None and fit < 0.97:
        flags.append(f"full model train-sample macro F1 {fit:.4f} < 0.97 at E*")

    wsrc = os.path.join(full, "weights", f"e{ekey(E)}.pt")
    if os.path.exists(wsrc):
        bdir = os.path.join(a.final_dir, "best_model"); os.makedirs(bdir, exist_ok=True)
        shutil.copy(wsrc, os.path.join(bdir, "state_dict_fp16.pt"))
        for f in glob.glob(os.path.join(full, "tokenizer", "*")):
            shutil.copy(f, bdir)
        if a.cleanup:
            shutil.rmtree(os.path.join(full, "weights"), ignore_errors=True)

    report = {"E_star": E, "bias": bias.tolist(), "bias_adopted": sel.get("bias_adopted"), "K": sel.get("K"),
              "cv_macro_f1_at_E": sel.get("cv_macro_f1_at_E"), "test_shares": shares.tolist(),
              "agreement_with_fold_models": agree, "train_sample_f1_at_E": fit,
              "labels_changed_by_bias": int(((L + bias).argmax(1) != L.argmax(1)).sum()),
              "flags": flags, "upload_ok": not flags}
    json.dump(report, open(os.path.join(a.final_dir, "final_report.json"), "w"), indent=2)
    print(json.dumps(report, indent=2))
    print("SUBMISSION_READY" if not flags else "CHECK_FLAGS_BEFORE_UPLOAD", os.path.join(a.final_dir, "submission.csv"))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--mode", required=True, choices=["fold", "full", "select", "finalize"])
    p.add_argument("--fold", type=int, default=-1)
    p.add_argument("--model_dir", default=None)
    p.add_argument("--train"); p.add_argument("--test"); p.add_argument("--folds")
    p.add_argument("--out", default="/kaggle/working/sub4")
    p.add_argument("--final_dir", default="/kaggle/working/sub4_final")
    p.add_argument("--max_len", type=int, default=1024)
    p.add_argument("--lr", type=float, default=3e-5)
    p.add_argument("--epochs", type=float, default=4.0)
    p.add_argument("--warmup", type=float, default=0.1)
    p.add_argument("--wd", type=float, default=0.01)
    p.add_argument("--bs", type=int, default=8)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--limit", type=int, default=0, help="smoke test only: subsample rows")
    p.add_argument("--save_weights", type=int, default=1)
    p.add_argument("--cleanup", type=int, default=1)
    a = p.parse_args()
    if a.mode in ("fold", "full"):
        if not a.model_dir:
            sys.exit("--model_dir is required for fold/full")
        if a.mode == "fold" and a.fold not in range(5):
            sys.exit("--fold must be 0..4")
        try:
            gpu_main(a)
        except Exception:
            out = os.path.join(a.out, f"fold{a.fold}" if a.mode == "fold" else "full")
            os.makedirs(out, exist_ok=True)
            open(os.path.join(out, "FAILED"), "w").write("failed")
            raise
    elif a.mode == "select":
        select_main(a)
    else:
        finalize_main(a)
