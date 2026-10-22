# Sub 5 (`granite-3.3-2b-instruct` LoRA Full-14k) — Kaggle Fast Package (< 17:45 UTC+7)

This folder (`d:\Coding\Prj\RMITHACK\RMIT-HACK-26\kaggle_sub5_package`) contains everything prepared and pre-configured to run **Sub 5** on a **Kaggle Notebook** and finish well before the **17:45 UTC+7** deadline.

---

## 1. Local File Tree Prepared for You

```text
RMIT-HACK-26/
└── kaggle_sub5_package/
    ├── UTS_SGS_InnoLab_C1_Sub5_Kaggle.ipynb   <-- Import directly into Kaggle Notebook
    ├── kaggle_sub5_dataset.zip                <-- Upload once as a Kaggle Input Dataset (~4.2 MB)
    ├── README_KAGGLE_SETUP.md                 <-- This quickstart guide
    └── dataset_upload/                        <-- Uncompressed contents of kaggle_sub5_dataset.zip
        ├── train.csv                          <-- 14,000 labelled rows
        ├── test.csv                           <-- 6,000 test rows
        ├── folds.csv                          <-- 5-fold split (fold-0 SHA verified)
        ├── sub3_submission.csv                <-- Sub 3 (0.96808 LB) reference for agreement check
        ├── sub5_kaggle_prepare.py             <-- Cell 2 script (auto-detects Kaggle paths + GPU)
        └── sub5_kaggle_train.py               <-- Cell 3 script (fast 2.0-epoch + 17:25 UTC+7 guard)
```

---

## 2. How It Maps Inside the Kaggle Notebook Runtime

Once you attach `kaggle_sub5_dataset.zip` (and optionally the Kaggle Model `aisuko/granite-3.3-2b-instruct`), the notebook automatically resolves and creates this runtime tree on Kaggle:

```text
/kaggle/
├── input/
│   ├── rmit-sub5-inputs/                      <-- From uploading kaggle_sub5_dataset.zip
│   │   ├── train.csv
│   │   ├── test.csv
│   │   ├── folds.csv
│   │   ├── sub3_submission.csv
│   │   ├── sub5_kaggle_prepare.py
│   │   └── sub5_kaggle_train.py
│   └── granite-3-3-2b-instruct/               <-- (Optional) Attached Kaggle Model or downloaded from HF
└── working/
    ├── submission.csv                         <-- AUTO-UPDATED at Epoch 1.0, 1.5, and 2.0 (ready to submit!)
    └── sub5_granite_full/
        ├── snapshots.json                     <-- Step timings, class shares, and Sub 3 agreement
        ├── final_report.json                  <-- Gate check summary
        ├── submission.csv                     <-- Copy of latest validated submission
        ├── submission_e1.0.csv                <-- Safety checkpoint at Epoch 1.0 (~14 min)
        ├── submission_e1.5.csv                <-- Safety checkpoint at Epoch 1.5 (~21 min)
        ├── submission_e2.0.csv                <-- Final 2.0-epoch checkpoint (~28 min)
        ├── test_logits_e1.0.npy
        ├── test_logits_e1.5.npy
        ├── test_logits_e2.0.npy
        └── adapter_e2.0/                      <-- Saved LoRA adapter + score.pt (~115 MB)
```

---

## 3. What Was Modified from `SUB5_granite_full.md` to Finish Before 17:45 on Kaggle

1. **Removed Colab-only dependencies (`google.colab.drive` & `/content` hardcoding):**
   - `sub5_kaggle_prepare.py` and `sub5_kaggle_train.py` automatically search `/kaggle/input/**` for `train.csv`, `test.csv`, `folds.csv`, `sub3_submission.csv`, and `granite-3.3-2b-instruct`, and write outputs directly to `/kaggle/working/`.
2. **Fixed Kaggle T4 (`sm_75`) vs L4/A100 (`sm_80+`) Hardware Precision & Memory:**
   - Original `sub3_prepare.py` had `assert torch.cuda.is_bf16_supported()` and forced `bf16=True` (which either fails or emulates slowly without Tensor Cores on Kaggle T4).
   - Now auto-detects compute capability: uses native `bf16` (`BS=16, GA=2, MAX_LEN=1024`) on L4/A100, and uses **native FP16 Tensor Cores** (`BS=8, GA=4, MAX_LEN=768`, which truncates only 1.66% of rows per `SUB3 §9`) with **FP32 master LoRA/head weights** and `nan_to_num` pooling clamp on Kaggle T4 so `GradScaler` never errors and activations never overflow.
   - Pins `CUDA_VISIBLE_DEVICES="0"` so HuggingFace `Trainer` does not wrap `PEFT + gradient_checkpointing` in `nn.DataParallel` on 2×T4.
3. **Switched to Fast Sub 5 Lite Schedule (`EPOCHS = 2.0`, 875 steps, single `full` run):**
   - Trains on **all 14,000 rows** (including all 980 `PROMPT_INJECTION` rows, +25% over Sub 3) for **2.0 epochs** (875 steps at effective batch 32) with `cosine` schedule and `weight_decay=0.01`.
   - Snapshots only at **`[1.0, 1.5, 2.0]` epochs** (saving 4 unnecessary inference passes = ~12 minutes saved).
4. **Progressive `/kaggle/working/submission.csv` Writing + Hard 17:25 UTC+7 Wall-Clock Guard:**
   - At **Epoch 1.0** (~14 min), **Epoch 1.5** (~21 min), and **Epoch 2.0** (~28 min), `FastKaggleSnapshot` runs the ordered inference loop (`ordered_logits`) and immediately writes `/kaggle/working/submission.csv`.
   - If the clock reaches **17:25 UTC+7** (`10:25 UTC`) or `38 minutes` elapsed at any point after Epoch 1.0, `TIME_CUTOFF_GUARD` cleanly stops training, runs ordered test inference, saves the adapter, and writes `/kaggle/working/submission.csv` so you are guaranteed to have 20 minutes before 17:45 UTC+7 to upload.

---

## 4. 3-Minute Launch Steps on Kaggle

1. Open Kaggle → **New Notebook** (or open `UTS_SGS_InnoLab_C1`).
2. Click **File → Import Notebook** and select:
   `d:\Coding\Prj\RMITHACK\RMIT-HACK-26\kaggle_sub5_package\UTS_SGS_InnoLab_C1_Sub5_Kaggle.ipynb`
3. In the right sidebar (**Settings**):
   - **Accelerator:** Select **GPU T4 × 2** (or **GPU L4 / P100**).
   - **Internet:** Turn **ON** (so `ibm-granite/granite-3.3-2b-instruct` can download from HuggingFace if not attached as a Kaggle Model).
4. In the right sidebar (**Input → + Add Input → Upload**):
   - Upload `d:\Coding\Prj\RMITHACK\RMIT-HACK-26\kaggle_sub5_package\kaggle_sub5_dataset.zip` and name it e.g. `rmit-sub5-inputs`.
5. Run **Cell 1** (`!pip uninstall -y -q torchao`), **Cell 2** (Prepare — prints `PREPARE_OK`), **Cell 3** (Train + Snapshot), and **Cell 4** (Verify).
6. Download `/kaggle/working/submission.csv` from the Output panel and upload to Kaggle Competition before **17:45 UTC+7**!
