import gc, hashlib, os, time
import numpy as np, pandas as pd, torch, torch.nn as nn, transformers, peft
from sklearn.metrics import f1_score
from transformers import AutoModel, AutoTokenizer
from transformers.modeling_outputs import SequenceClassifierOutput
from peft import LoraConfig, get_peft_model

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
os.environ["TOKENIZERS_PARALLELISM"] = "false"
print("transformers", transformers.__version__, "| peft", peft.__version__, "| GPU", torch.cuda.get_device_name(0))
assert torch.cuda.is_bf16_supported(), "Use an A100 or L4 runtime for Sub 3"
transformers.set_seed(42)

LABELS = ["BENIGN", "JAILBREAK", "PROMPT_INJECTION"]
label2id = {l: i for i, l in enumerate(LABELS)}
id2label = {i: l for l, i in label2id.items()}
FOLD0_SHA = "0d043c7efa1612b707d4a52e4b73a0aa311ee37fb3f9cadf213ec47269a57fc4"
TRAIN_PATH, TEST_PATH, FOLDS_PATH = "/content/train.csv", "/content/test.csv", "/content/folds.csv"
MODEL_DIR = "ibm-granite/granite-3.3-2b-instruct"
MAX_LEN = 1024
OUT = "/content/outputs/sub3_granite_f0"
os.makedirs(OUT, exist_ok=True)

read = lambda p: pd.read_csv(p, dtype=str, keep_default_na=False, encoding="utf-8")
train, test, folds = read(TRAIN_PATH), read(TEST_PATH), read(FOLDS_PATH)
folds["fold"] = folds["fold"].astype(int)
train = train.merge(folds, on="id", how="left", validate="one_to_one")
assert train["fold"].notna().all()
fold0 = sorted(train.loc[train.fold == 0, "id"])
assert len(fold0) == 2800 and hashlib.sha256(",".join(fold0).encode()).hexdigest() == FOLD0_SHA, "fold-0 hash mismatch"
tr_df = train[train.fold != 0].reset_index(drop=True)
va_df = train[train.fold == 0].reset_index(drop=True)
print("rows", len(tr_df), len(va_df), len(test))

tok = AutoTokenizer.from_pretrained(MODEL_DIR)

def encode(texts, max_len=MAX_LEN, head_frac=0.75):
    enc = tok(list(texts), add_special_tokens=True, truncation=False)
    out, n_trunc = [], 0
    for ids in enc["input_ids"]:
        if len(ids) > max_len:
            h = int(max_len * head_frac)
            ids = ids[:h] + ids[-(max_len - h):]
            n_trunc += 1
        out.append(ids)
    print(f"truncated {n_trunc}/{len(out)} ({n_trunc / len(out):.2%}) at max_len={max_len}")
    return out, n_trunc / len(out)

tr_ids, tr_trunc = encode(tr_df.text)
va_ids, va_trunc = encode(va_df.text)
te_ids, te_trunc = encode(test.text)
TRUNC_RATE = tr_trunc

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
        self.backbone = AutoModel.from_pretrained(path, dtype=torch.bfloat16, attn_implementation="sdpa")
        self.score = nn.Linear(self.backbone.config.hidden_size, num_labels, bias=False)
        nn.init.zeros_(self.score.weight)
    def forward(self, input_ids, attention_mask, labels=None):
        h = self.backbone(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        last = attention_mask.shape[1] - 1 - attention_mask.flip(1).argmax(1)
        logits = self.score(h[torch.arange(h.size(0), device=h.device), last].float())
        return SequenceClassifierOutput(logits=logits)

def build_model():
    model = GraniteClassifier(MODEL_DIR)
    model.backbone.config.use_cache = False
    model.backbone.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.backbone = get_peft_model(model.backbone, LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]))
    model.cuda()
    return model

from transformers import Trainer, TrainingArguments, DataCollatorWithPadding

class WeightedTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        loss = nn.functional.cross_entropy(outputs.logits.float(), labels, weight=CLASS_W.to(outputs.logits.device))
        return (loss, outputs) if return_outputs else loss

def compute_metrics(p):
    logits = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
    return {"macro_f1": f1_score(p.label_ids, np.argmax(logits, axis=-1), average="macro")}

t0 = time.time()
model = build_model()
print(f"model ready in {time.time() - t0:.0f}s | trainable {sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6:.1f}M")
print("PREPARE_OK")
