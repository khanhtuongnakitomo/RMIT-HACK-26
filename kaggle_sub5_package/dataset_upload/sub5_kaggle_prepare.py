"""Cell 2 for Kaggle Notebook — Sub 5 Granite-3.3-2B LoRA Full-14k Preparation.

Auto-discovers train.csv, test.csv, folds.csv, sub3_submission.csv, and
ibm-granite/granite-3.3-2b-instruct from /kaggle/input/**, /content/**, or HuggingFace.
Automatically selects bf16 (on L4/A100 sm_80+) or safe fp16 with fp32 master LoRA weights
(on Kaggle T4 sm_75) and sets MAX_LEN / batch size so training finishes before 17:45 UTC+7.
"""
import gc
import glob
import hashlib
import os
import time

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
# Pin to single GPU so HF Trainer does not wrap PEFT + gradient_checkpointing in DataParallel
if "CUDA_VISIBLE_DEVICES" not in os.environ:
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import numpy as np
import pandas as pd
import peft
import torch
import torch.nn as nn
import transformers
from peft import LoraConfig, get_peft_model
from sklearn.metrics import f1_score
from transformers import (
    AutoModel,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)
from transformers.modeling_outputs import SequenceClassifierOutput

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
transformers.set_seed(42)

assert torch.cuda.is_available(), "No CUDA GPU detected! Enable GPU accelerator in Kaggle Notebook settings."
gpu_name = torch.cuda.get_device_name(0)
cc_major, cc_minor = torch.cuda.get_device_capability(0)
vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
USE_NATIVE_BF16 = cc_major >= 8  # Ampere/Ada (A100, L4) -> native bf16; Turing (T4) -> fp16 Tensor Cores
AMP_DTYPE = torch.bfloat16 if USE_NATIVE_BF16 else torch.float16

print(
    f"transformers {transformers.__version__} | peft {peft.__version__} | "
    f"GPU: {gpu_name} (sm_{cc_major}{cc_minor}, {vram_gb:.1f} GB) | "
    f"precision: {'bf16' if USE_NATIVE_BF16 else 'fp16'}"
)

LABELS = ["BENIGN", "JAILBREAK", "PROMPT_INJECTION"]
label2id = {l: i for i, l in enumerate(LABELS)}
id2label = {i: l for l, i in label2id.items()}
FOLD0_SHA = "0d043c7efa1612b707d4a52e4b73a0aa311ee37fb3f9cadf213ec47269a57fc4"


def find_file(filename, required=True):
    search_patterns = [
        f"/kaggle/input/**/{filename}",
        f"/kaggle/working/**/{filename}",
        f"/content/**/{filename}",
        f"data/{filename}",
        filename,
    ]
    for pattern in search_patterns:
        hits = sorted(glob.glob(pattern, recursive=True))
        if hits:
            return hits[0]
    if required:
        raise FileNotFoundError(
            f"Could not find {filename}. Attach the dataset containing {filename} under /kaggle/input/."
        )
    return None


def find_granite_model():
    for pattern in ["/kaggle/input/**/config.json", "/content/**/config.json"]:
        for cfg in sorted(glob.glob(pattern, recursive=True)):
            if "granite" in cfg.lower():
                return os.path.dirname(cfg)
    return "ibm-granite/granite-3.3-2b-instruct"


TRAIN_PATH = find_file("train.csv", required=True)
TEST_PATH = find_file("test.csv", required=True)
FOLDS_PATH = find_file("folds.csv", required=True)
SUB3_CSV = find_file("sub3_submission.csv", required=False)
MODEL_DIR = find_granite_model()

# On 16 GB T4, MAX_LEN=768 truncates only 1.66% of rows (head 75% + tail 25%) and runs ~35% faster.
# On >=24 GB L4/A100, MAX_LEN=1024 truncates 0.81% of rows.
MAX_LEN = 1024 if vram_gb >= 22.0 else 768
WORK_ROOT = "/kaggle/working" if os.path.exists("/kaggle/working") else "/content"
OUT = os.path.join(WORK_ROOT, "sub5_granite_full")
os.makedirs(OUT, exist_ok=True)

print("TRAIN_PATH :", TRAIN_PATH)
print("TEST_PATH  :", TEST_PATH)
print("FOLDS_PATH :", FOLDS_PATH)
print("SUB3_CSV   :", SUB3_CSV)
print("MODEL_DIR  :", MODEL_DIR)
print("MAX_LEN    :", MAX_LEN)
print("OUT        :", OUT)

read = lambda p: pd.read_csv(p, dtype=str, keep_default_na=False, encoding="utf-8")
train, test, folds = read(TRAIN_PATH), read(TEST_PATH), read(FOLDS_PATH)
folds["fold"] = folds["fold"].astype(int)
train = train.merge(folds, on="id", how="left", validate="one_to_one").reset_index(drop=True)
assert train["fold"].notna().all()

fold0 = sorted(train.loc[train.fold == 0, "id"])
assert (
    len(fold0) == 2800
    and hashlib.sha256(",".join(fold0).encode()).hexdigest() == FOLD0_SHA
), "fold-0 hash mismatch"

tr_mask = (train.fold != 0).values
va_mask = (train.fold == 0).values
tr_df = train[tr_mask].reset_index(drop=True)
va_df = train[va_mask].reset_index(drop=True)
print("rows (folds 1-4 / fold 0 / test / total train):", len(tr_df), len(va_df), len(test), len(train))

tok = AutoTokenizer.from_pretrained(MODEL_DIR)


def encode(texts, max_len=MAX_LEN, head_frac=0.75):
    enc = tok(list(texts), add_special_tokens=True, truncation=False)
    out, n_trunc = [], 0
    for ids in enc["input_ids"]:
        if len(ids) > max_len:
            h = int(max_len * head_frac)
            ids = ids[:h] + ids[-(max_len - h) :]
            n_trunc += 1
        out.append(ids)
    print(f"truncated {n_trunc}/{len(out)} ({n_trunc / len(out):.2%}) at max_len={max_len}")
    return out, n_trunc / len(out)


all_train_ids, TRUNC_RATE = encode(train.text)
tr_ids = [all_train_ids[i] for i in range(len(train)) if tr_mask[i]]
va_ids = [all_train_ids[i] for i in range(len(train)) if va_mask[i]]
te_ids, te_trunc = encode(test.text)


class DS(torch.utils.data.Dataset):
    def __init__(self, ids, labels=None):
        self.ids, self.labels = ids, labels

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        d = {"input_ids": self.ids[i], "attention_mask": [1] * len(self.ids[i])}
        if self.labels is not None:
            d["labels"] = int(self.labels[i])
        return d


tr_y = tr_df.label.map(label2id).tolist()
va_y = va_df.label.map(label2id).tolist()
tr_ds, va_ds, te_ds = DS(tr_ids, tr_y), DS(va_ids, va_y), DS(te_ids)
counts = tr_df.label.value_counts().reindex(LABELS).values
CLASS_W = torch.tensor(np.sqrt(counts.max() / counts), dtype=torch.float)
print("class weights", CLASS_W.tolist())


class GraniteClassifier(nn.Module):
    def __init__(self, path, num_labels=3):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(
            path, dtype=AMP_DTYPE, attn_implementation="sdpa"
        )
        self.score = nn.Linear(self.backbone.config.hidden_size, num_labels, bias=False)
        nn.init.zeros_(self.score.weight)

    def forward(self, input_ids, attention_mask, labels=None):
        with torch.amp.autocast("cuda", dtype=AMP_DTYPE):
            h = self.backbone(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        last = attention_mask.shape[1] - 1 - attention_mask.flip(1).argmax(1)
        pooled = h[torch.arange(h.size(0), device=h.device), last].float()
        pooled = torch.nan_to_num(pooled, nan=0.0, posinf=1e4, neginf=-1e4)
        logits = self.score(pooled)
        return SequenceClassifierOutput(logits=logits)


def build_model():
    model = GraniteClassifier(MODEL_DIR)
    model.backbone.config.use_cache = False
    model.backbone.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )
    model.backbone = get_peft_model(
        model.backbone,
        LoraConfig(
            r=16,
            lora_alpha=32,
            lora_dropout=0.05,
            target_modules=[
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ],
        ),
    )
    # Keep trainable LoRA + head weights in fp32 so GradScaler (fp16 mode on T4) never fails
    for p in model.parameters():
        if p.requires_grad:
            p.data = p.data.float()
    model.cuda()
    return model


class WeightedTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        loss = nn.functional.cross_entropy(
            outputs.logits.float(), labels, weight=CLASS_W.to(outputs.logits.device)
        )
        return (loss, outputs) if return_outputs else loss


def compute_metrics(p):
    logits = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
    return {"macro_f1": f1_score(p.label_ids, np.argmax(logits, axis=-1), average="macro")}


t0 = time.time()
model = build_model()
print(
    f"model ready in {time.time() - t0:.0f}s | trainable "
    f"{sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6:.1f}M"
)
print("PREPARE_OK")
