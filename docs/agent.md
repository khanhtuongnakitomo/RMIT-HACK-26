# RMIT Hackathon 2026 — Competition Briefing and Team Handoff

> Prepared for a four-person student team from the research, local checks, and discussions in this chat.
>
> **As of:** 22 October 2026. **Reference timezone:** Vietnam / Asia–Bangkok, UTC+7.
>
> **Workspace:** `D:\Coding\Prj\RMITHACK`.
>
> This is a requested reference document, not an automatically loaded `AGENTS.md`, a new global rule, or authorization to execute every suggested action. It records competition facts, observed state, proposed approaches, and unresolved decisions separately.

## 1. Read this first

The published Kaggle task is **multilingual three-class prompt-attack classification**. Fine-tune **one approved pretrained model**, predict labels for 6,000 test prompts, and submit an `id,label` CSV. The metric is **macro F1**.

The broader event also has challenges on **RMIT RACE Cloud**. Their briefs and overall scoring were not available in the material inspected. A good Kaggle result does not establish completion of the whole hackathon.

The recommended approach is:

1. Confirm official registration, team membership, scoring weights, allowed resources, and deadlines.
2. Establish a working GPU environment and load an approved model.
3. Build a complete training → validation → prediction → CSV pipeline.
4. Secure an early valid submission during the authorized competition window.
5. Use a fixed validation split and error analysis to choose a small number of improvements.
6. Allocate the four people across challenges according to marks, difficulty, remaining time, and execution risk.
7. Preserve a known-working model, notebook, and submission before further experimentation.

**Time-sensitive:** Kaggle displayed its closing time as **22 October 2026 at 18:00 UTC+7**, equivalent to **11:00 UTC**. The wider event continues on 23 October. Recheck the live page and event briefing for changes; do not assume the second event day extends the Kaggle deadline. [S1]

**Scoring ambiguity:** the overview says a first valid submission earns full Challenge 1 marks, but also encourages leaderboard improvement. The public material does not explain how that interacts with overall awards. Clarify this before dedicating the whole team to leaderboard optimization. [S1]

## 2. Evidence labels and authority

This document uses these distinctions:

| Label | Meaning |
|---|---|
| **Official** | Published on the event website, current Kaggle competition pages, or official product documentation inspected in this chat |
| **Verified locally** | Directly checked in this workspace or the selected local installation |
| **Recommendation** | Proposed strategy or implementation choice; not a competition requirement or a user-approved architecture |
| **Unconfirmed** | Not established by the accessible evidence |
| **Not performed** | Discussed or proposed, but not installed, executed, or behaviorally verified |

For competition decisions, use the current competition rules, task description, organizer clarifications, and event announcements. Kaggle's foundational rules explicitly take precedence over conflicting competition-specific rules. Apparent inconsistencies need clarification, not an invented resolution. [S4]

For local setup, use the actual environment rather than treating an earlier inspection as permanently current. For learning resources, distinguish official requirements from participant notes, mock data, prior-year tasks, and speculative preparation material.

## 3. User intent and conversation context

### 3.1 What the user asked for

The user is preparing to compete as a team of four and requested:

- A comprehensive explanation of the event, rules, instructions, and needed preparation.
- Use of the referenced ChatGPT conversation **AI Learning** together with current web research.
- Checks of this machine and workspace before the competition.
- Whether VS Code can connect to Kaggle using SSH or another supported method.
- Whether this machine is ready to open and execute notebooks.
- A practical competition strategy and technical direction.
- This detailed Markdown handoff, explicitly named **`agent.md`**.

The discussion is preparation and planning. It does not establish that the proposed model, split, hyperparameters, or task ownership have been accepted as final team decisions.

### 3.2 Referenced AI Learning conversation

- **Title:** AI Learning.
- **Conversation ID:** `6acc9fd6-6670-83ec-82dd-2e07adcd5bab`.
- **Reference:** `chatgpt-conversation://6acc9fd6-6670-83ec-82dd-2e07adcd5bab`.
- The conversation was retrieved through the chat-reading capability.
- The user's questions were readable, but the assistant replies appeared only as unavailable content-reference placeholders.

Readable user interests included:

1. Security, GenAI, and low-resource-language hackathon preparation.
2. A list of concepts to teach teammates.
3. An emphasis on training and fine-tuning in Kaggle.
4. Explanations of AI concepts, mechanisms, benefits, and use cases.
5. The steps and flow of a hybrid-search RAG system.

**Limitation:** earlier assistant explanations could not be recovered. Do not attribute detailed technical decisions to that conversation or claim its full contents were read.

### 3.3 Relevant working preferences

- Windows 11 / PowerShell is the working environment.
- Lead with the practical outcome and explain why important choices work.
- Keep confirmed facts separate from proposals and unknowns.
- Preserve unrelated files and existing work.
- Do not install persistent settings, rules, hooks, skills, or agent definitions without explicit direction.
- Do not commit, push, create or modify GitHub issues/PRs, publish, deploy, or message others without applicable explicit authorization.
- This document's creation was explicitly requested. It does not authorize external publication or unrelated setup changes.

## 4. Official event overview

| Item | Published information |
|---|---|
| Event | RMIT Hackathon 2026 |
| Themes | Security × Generative AI × Low-Resource Languages |
| Dates | 22–23 October 2026 |
| Published hours | 22 October: 08:00–18:00; 23 October: 08:00–12:00; confirm venue-local scheduling |
| Team size | Exactly four currently enrolled undergraduate or postgraduate students |
| Attendance | All members present at the same venue; no online or cross-venue participation |
| Event language | English |
| Registration | One registration per team, submitted by the leader |
| Registration status | Closed early on 13 October, replacing the original 15 October deadline |
| Cost | Free |
| Advertised total prize value | Up to VND 100 million; breakdown not verified |
| Official contact | `GenAICC@rmit.edu.au` |

Venue choices:

- **Melbourne:** B091.01.001, 110 Victoria Street, Carlton VIC 3053.
- **Saigon South:** 702 Nguyen Van Linh Boulevard, Tan Hung Ward, Ho Chi Minh City.
- **Hanoi:** Handi Resco Building, 521 Kim Ma, Giang Vo Ward, Hanoi.
- **Malaysia:** Taylor's University Lakeside Campus, No. 1 Jalan Taylor's, 47500 Subang Jaya, Selangor.

Source: official event website. [S6]

**Registration limitation:** neither event acceptance nor the team's chosen venue was verified in this session. Access to Kaggle does not establish a physical event place.

## 5. Event structure and unresolved mapping

The event website advertises **one warm-up and two challenges**. The Kaggle overview describes **Challenge 1** and states that the other challenges are handled through **RMIT RACE Cloud**. [S1, S6]

The inspected public material does not establish:

- Exactly how the warm-up maps to the numbered challenges.
- The RACE task statements or scoring functions.
- Overall weights or whether different award categories use different criteria.
- A mandatory final presentation, report, video, or pitch format.

Therefore:

- Treat the published classifier as a confirmed task.
- Obtain the other briefs and rubrics before selecting their architectures.
- Do not import a previous year's attack-generation, RAG, binary-classification, or submission format into 2026 without evidence.
- Do not assume a generic hackathon product demo or polished UI earns marks unless the rubric says so.

## 6. Challenge 1: exact task definition

### 6.1 Input and output

Given a prompt in Portuguese, French, Italian, or English, predict exactly one label:

| Label | Organizer's intended category |
|---|---|
| `BENIGN` | Normal, safe user request |
| `JAILBREAK` | Attempt to overcome the model's safety restrictions |
| `PROMPT_INJECTION` | Attempt to hijack the model's instructions |

The conceptual boundaries can overlap. Use the supplied labelled training examples to understand the competition's operational distinction. Do not silently relabel examples according to a different taxonomy. [S1, S3]

### 6.2 Why these languages count as low-resource here

Portuguese, French, and Italian have substantial general NLP resources. The competition describes them as comparatively low-resource **for AI safety**, where prompt-attack and guardrail data are often English-centric. [S1]

The task is not specifically Vietnamese/Malay classification. An earlier broad theme should not override the published four-language dataset.

### 6.3 Required workflow

1. Choose one approved model.
2. Fine-tune on `train.csv`.
3. Predict every row of `test.csv`.
4. Export a valid `id,label` CSV.
5. Submit to Kaggle and make the notebook accessible as instructed by the organizers. [S1, S3]

No competition model training or submission has been performed in this chat.

## 7. Dataset contract and verified local audit

### 7.1 Files

| File | Rows | Columns | Role |
|---|---:|---|---|
| `train.csv` | 14,000 | `id,text,language,label` | Labelled training data; make validation splits from this file |
| `test.csv` | 6,000 | `id,text,language` | Unlabelled evaluation inputs |
| `sample_submission.csv` | 6,000 | `id,label` | Submission format and ID template |

The Data page lists **CC BY-NC-SA 4.0** for the competition dataset. Read the actual licence and organizer restrictions before reuse or redistribution. Public availability does not remove licensing obligations. [S2]

### 7.2 Local files already present

The workspace contained these files at the latest inspection:

```text
D:\Coding\Prj\RMITHACK\
├── rmit-hackathon-2026.zip
├── train.csv
├── test.csv
└── sample_submission.csv
```

`agent.md` is added by this request. No training notebook, project environment, dependency manifest, model checkpoint, or generated prediction file was present when the preparatory checks were made.

### 7.3 Checks actually performed

Python's standard CSV reader was used for structural inspection:

- Correct row counts and headers in all three CSV files.
- Zero duplicate IDs within each file.
- Zero blank or missing cells under the check used.
- Sample-submission IDs match the test IDs as a set.
- No duplicate training texts under exact matching.
- No duplicate training texts after collapsing whitespace and case-folding.

**Boundaries:** this is not a comprehensive data-quality audit. It did not establish absence of semantic duplicates, translated templates, label noise, source contamination, train/test overlap, or hidden evaluation artifacts. The checks did not identify the private test labels.

### 7.4 Actual training label counts

| Label | Count | Share |
|---|---:|---:|
| `BENIGN` | 8,680 | 62% |
| `JAILBREAK` | 4,340 | 31% |
| `PROMPT_INJECTION` | 980 | 7% |
| **Total** | **14,000** | **100%** |

### 7.5 Actual language and class counts

| Language | Code | Benign | Jailbreak | Injection | Total | Share |
|---|---|---:|---:|---:|---:|---:|
| Portuguese | `pt` | 3,906 | 1,953 | 441 | 6,300 | 45% |
| French | `fr` | 2,170 | 1,085 | 245 | 3,500 | 25% |
| Italian | `it` | 1,736 | 868 | 196 | 2,800 | 20% |
| English | `en` | 868 | 434 | 98 | 1,400 | 10% |
| **Total** | | **8,680** | **4,340** | **980** | **14,000** | **100%** |

Every language has the same class proportions. Language alone therefore offers no class-prior advantage in this training file. It remains useful for stratification and diagnosing model weaknesses; the organizers explicitly allow using it as an input. [S3]

### 7.6 Training-text lengths measured locally

These are **characters, not model tokens**. The 95th-percentile values use the sorted element at index `int(0.95 * (n - 1))`.

| Label | Median characters | 95th-percentile characters | Maximum characters |
|---|---:|---:|---:|
| `BENIGN` | 115 | 1,582 | 18,550 |
| `JAILBREAK` | 565 | 1,517 | 3,752 |
| `PROMPT_INJECTION` | 99 | 1,429 | 5,128 |

The official description gives an overall prompt-length range of 10–18,550 characters and median of 294. [S2]

**Interpretation:** some inputs are long enough that token truncation deserves measurement. These statistics do not prove that longer context improves performance, or justify classifying by length. Test length-related hypotheses on validation data.

## 8. Evaluation and leaderboard behavior

### 8.1 Macro F1

For each class, measure precision and recall, combine them into F1, then average the three class F1 values equally:

```text
Precision = true positives / predicted positives
Recall    = true positives / actual positives
F1        = 2 × precision × recall / (precision + recall)

Macro F1 = (F1_benign + F1_jailbreak + F1_injection) / 3
```

Use the metric library's defined zero-division behavior for empty cases. The central consequence is that the 7% injection class has the same final metric weight as the 62% benign class. [S1]

Illustrative scores, not measured competition results:

| Candidate | Benign F1 | Jailbreak F1 | Injection F1 | Macro F1 |
|---|---:|---:|---:|---:|
| A | 0.95 | 0.90 | 0.40 | 0.750 |
| B | 0.92 | 0.88 | 0.70 | 0.833 |

Candidate B wins despite lower scores on the two larger classes. Conversely, increasing injection recall by marking many benign prompts as attacks can reduce precision and hurt macro F1. Measure both.

The organizers report an all-benign score around 0.26. Perfect predictions score 1.0. [S1]

### 8.2 Public and private evaluation

The leaderboard inspected in this chat states:

- Public scores use approximately **30%** of test data.
- Final scores use the other **70%**.
- Final rankings can change. [S5]

This is a reason to use trustworthy local validation. A small public-score gain is not proof of a better generalizing model. Do not infer hidden labels from repeated leaderboard probing.

## 9. Approved models and model-selection strategy

### 9.1 Official approved list

The overview requires **one** approved pretrained backbone and prohibits other pretrained backbones. It also requires identifying the selected model at the top of the notebook. [S1]

| Approved model | Published type / guidance | Kaggle model |
|---|---|---|
| `distilbert/distilbert-base-multilingual-cased` | Encoder; fastest baseline | [Provided model](https://www.kaggle.com/models/aisuko/distilbert-base-multilingual-cased/Transformers/default) |
| `jhu-clsp/mmBERT-small` | Encoder; fast multilingual option | [Provided model](https://www.kaggle.com/models/aisuko/mmbert-small/Transformers/default) |
| `jhu-clsp/mmBERT-base` | Encoder; strong multilingual option | [Provided model](https://www.kaggle.com/models/aisuko/mmbert-base/Transformers/default) |
| `FacebookAI/xlm-roberta-base` | Encoder; strong multilingual option | [Provided model](https://www.kaggle.com/models/aisuko/xlm-roberta-base/Transformers/default) |
| `FacebookAI/xlm-roberta-large` | Larger encoder; more GPU memory required | [Provided model](https://www.kaggle.com/models/aisuko/xlm-roberta-large/Transformers/default) |
| `google/rembert` | Multilingual encoder; organizers specify batch size 8 | [Provided model](https://www.kaggle.com/models/aisuko/rembert/Transformers/default) |
| `protectai/deberta-v3-base-prompt-injection-v2` | Safety-tuned encoder | [Provided model](https://www.kaggle.com/models/aisuko/deberta-v3-base-prompt-injection-v2/Transformers/default) |
| `patronus-studio/wolf-defender-prompt-injection` | Safety-tuned encoder | [Provided model](https://www.kaggle.com/models/aisuko/wolf-defender-prompt-injection/Transformers/default) |
| `ibm-granite/granite-3.3-2b-instruct` | LLM; fine-tuning / LoRA route | [Provided model](https://www.kaggle.com/models/aisuko/granite-3.3-2b-instruct/Transformers/default) |
| `mistralai/Mistral-7B-Instruct-v0.3` | LLM; fine-tuning / LoRA route | [Provided model](https://www.kaggle.com/models/aisuko/mistral-7b-instruct-v0.3/Transformers/default) |

Do not interpret the table's published descriptions as measured performance on this team's split. No model comparison was run here.

### 9.2 Recommended starting decision

- Start with **XLM-RoBERTa-base** if it loads, trains, and predicts within the available memory and time. The organizer's example uses it.
- Use **multilingual DistilBERT** if the priority is the fastest working end-to-end baseline or setup is blocking progress.
- Consider a larger approved encoder only after the baseline is reliable and timing measurements justify it.
- Choose an approved LLM with LoRA/QLoRA when the team already has relevant implementation experience and a concrete reason to expect a useful tradeoff.

An encoder with a classification head directly solves the requested output problem. A generative route adds label-format handling and training complexity. That does not prove an encoder wins; it makes it the proposed lower-complexity initial route.

### 9.3 Important loading checks

- Use the tokenizer associated with the selected backbone.
- Configure exactly three output labels.
- Keep `label2id` and `id2label` consistent throughout training and inference.
- Inspect warnings about newly initialized heads or incompatible checkpoint shapes.
- A safety-tuned checkpoint may have a different original label scheme; do not reuse its old class semantics blindly.
- Verify attached model paths in the actual Kaggle session; do not assume version directories remain identical.
- Multi-backbone ensembles, external teacher models, and externally trained adapters are not established as permitted.

## 10. Rules and submission contract

### 10.1 Notebook name

```text
<university>_<campus>_<team>_C1
```

- Campus code: `SGS`, `HNI`, `MEL`, or `MYS`.
- University and campus fields use letters.
- Team field allows letters, digits, underscores, and hyphens.
- Use the team's real institution and chosen campus.

Published examples include `RMIT_SGS_a_C1` and `Taylor_MYS_sunny_C1`. [S1]

### 10.2 Prediction CSV

```csv
id,label
t000006,BENIGN
t000007,PROMPT_INJECTION
t000008,JAILBREAK
```

Requirements:

- Exactly 6,000 prediction rows plus the header.
- Exactly the required `id,label` columns.
- One prediction per test ID, with no duplicates or omissions.
- Labels spelled exactly as specified.
- No dataframe index column.
- Predictions explicitly matched to IDs, rather than assuming a shuffled dataloader preserves order. [S1, S2]

Use the sample file as a template. Its contents are a format example, not evidence of correct test answers and not a legitimate model result.

### 10.3 Attempts and final selection

The overview specifies **six submission attempts**. The FAQ's submission-count answer was incomplete when inspected. The reset period and number of final selections were not confirmed. Plan conservatively around six total attempts until clarified. [S1, S3]

Keep an internal record of candidate name, checkpoint, validation score, upload time, Kaggle score, and whether it was selected for final evaluation. Do not assume a particular automatic final-selection rule without checking the interface and rules.

### 10.4 Foundational conduct

The foundational rules include:

- Each person uses one unique Kaggle account; no proxy accounts or submission-limit evasion.
- Each member has a separate account, enters individually, and confirms official team membership.
- A participant may join or form only one team.
- No private competition code/data sharing between competing teams, subject to the rules' stated exceptions.
- Public sharing must follow Kaggle competition channels and licence conditions.
- Do not incorporate hand-labeling or human prediction of held-out organizer validation/test records into submissions.
- Meet all deadlines and required formats.
- Respect ownership, licences, eligibility, and verification requirements. [S4]

Analyzing errors in a validation split drawn from the supplied labelled training data is ordinary development. It is distinct from manually assigning the hidden answers to organizer test records.

Kaggle's standard eligibility language includes a registered account, an age/majority requirement with specified exceptions, and sanctions/export restrictions. Read the full rules if any eligibility condition could affect a member. The event's student eligibility requirements also apply. [S4, S6]

### 10.5 Notebook sharing

The FAQ directs sharing with specified people or through Kaggle Groups. Its illustration shows a private notebook and a public-sharing lock until the competition ends. The judge identities/group and required permissions were not established. [S3]

Kaggle also discourages late public release of high-scoring notebooks because it can disrupt a competition. [S7]

Do not confuse notebook collaboration permissions with permission to share privately with another competing team.

## 11. Machine and workspace readiness

### 11.1 Verified local installation snapshot

| Component | Observed state |
|---|---|
| Operating environment | Windows / PowerShell; Windows 11 is the user's stated default |
| Python | 3.11.9 |
| Python executable | `C:\Users\ADMIN\AppData\Local\Programs\Python\Python311\python.exe` |
| Git | 2.55.0.windows.5 |
| `uv` | Installed at `C:\Users\ADMIN\.local\bin\uv.exe` |
| VS Code launcher | `D:\Coding\Application\Microsoft VS Code\bin\code.cmd` |
| GPU | NVIDIA GeForce RTX 5060 Laptop GPU |
| GPU memory | 8,151 MiB reported by `nvidia-smi`, approximately 8 GB |
| NVIDIA driver | 581.83 |
| System RAM | Approximately 31.2 GiB visible, commonly described as 32 GB installed |
| Free storage at inspection | C: approximately 55.9 GB; D: approximately 338.2 GB |
| Workspace environment | No project `.venv` found |

Free memory and disk capacity are point-in-time observations, not guaranteed current values.

### 11.2 Python packages checked

In the selected Python installation:

- `numpy` was installed at version **2.4.6**.
- The following were not installed: `torch`, `transformers`, `datasets`, `accelerate`, `scikit-learn`, `pandas`, `safetensors`, `sentencepiece`, `peft`, `bitsandbytes`, `kaggle`, `ipykernel`, `jupyter`, `jupyterlab`, `notebook`, `nbformat`, and `ipython`.

These checks do not prove the packages are absent from every other Python or Conda environment on the machine.

### 11.3 GPU verification boundary

`nvidia-smi` successfully identified the GPU and driver. A Python check failed to import PyTorch because it was not installed.

Therefore:

- **Hardware/driver detected:** yes.
- **PyTorch CUDA working:** not verified.
- **Notebook GPU training working:** not verified.
- **Competition model fits in local VRAM:** not measured.

Do not report local training readiness based only on the GPU's presence.

### 11.4 VS Code extension check

The inspected VS Code CLI profile listed Markdown, C/C++, browser-debugging, and OpenAI-related extensions. It did **not** list:

- Microsoft Python: `ms-python.python`.
- Microsoft Jupyter: `ms-toolsai.jupyter`.

Other VS Code profiles or separate installations were not exhaustively inspected.

### 11.5 Actual preparation status

| Activity | Status |
|---|---|
| Read official public competition materials | Performed |
| Retrieve AI Learning reference | Performed, with unavailable assistant replies |
| Inspect local data structure and counts | Performed |
| Inspect selected Python and VS Code extension inventory | Performed |
| Install notebook/training packages | Not performed |
| Install Python/Jupyter VS Code extensions | Not performed |
| Create a virtual environment | Not performed |
| Create or run a training notebook | Not performed |
| Download/load an approved model | Not performed |
| Connect VS Code to Kaggle | Not performed |
| Verify Kaggle account, GPU quota, or team | Not performed |
| Train or benchmark a model | Not performed |
| Generate/upload predictions | Not performed |
| Create this requested handoff | Performed by this task |

## 12. Three notebook workflows

| Workflow | Editor | Execution location | Local requirements | Current status |
|---|---|---|---|---|
| Kaggle browser | Browser | Kaggle | Working browser and Kaggle access | Local packages unnecessary; account/session unverified |
| VS Code with Kaggle kernel | Local VS Code | Kaggle | Notebook extensions and supported remote connection | Extensions missing in inspected profile; connection untested |
| Local notebooks | VS Code or local JupyterLab | Laptop | Notebook kernel, dependencies, optional GPU-compatible PyTorch | Not configured in this workspace |

**Recommendation:** prioritize Kaggle browser or VS Code connected to Kaggle for competition work. Treat the browser as a fallback if the experimental connection causes delays. Local GPU training is an optional backup, subject to organizer permission for external compute.

## 13. VS Code → Kaggle: supported remote connection

### 13.1 Answer to the SSH question

Kaggle documents an **experimental Jupyter Server connection** for external notebook editors, including VS Code. This is the supported route identified in the research. It is not the same as VS Code Remote SSH, and it does not automatically provide a full remote folder/terminal session. [S8, S9]

No SSH tunnel, external tunnelling service, or custom remote-access workaround was configured or recommended as the primary route.

### 13.2 Connection procedure

These instructions were checked against documentation, but the connection was not exercised on the user's account:

1. Install Microsoft's Python and Jupyter extensions in the VS Code profile used for this project.
2. Open the competition notebook on Kaggle.
3. Attach the competition input and selected approved model.
4. Select an available GPU and start the session.
5. Open **Run → Kaggle Jupyter Server**.
6. Under **Manually Connect**, copy the **VS Code Compatible URL**.
7. In VS Code, open a local `.ipynb`.
8. Open the notebook kernel picker.
9. Choose **Existing Jupyter Server**; some versions put this under **Select Another Kernel**.
10. Enter the URL and select the remote Python kernel.
11. Run a small environment/GPU diagnostic. [S8, S10]

### 13.3 Execution and saving model

```text
Local .ipynb in VS Code
        │ sends cells
        ▼
Kaggle Jupyter kernel
        │ uses Kaggle packages, files, GPU
        ▼
Results displayed in VS Code
```

- Local PyTorch and a local `ipykernel` are not required for remote cell execution.
- A normal local VS Code terminal still runs on Windows.
- Notebook cells see Kaggle's filesystem, not the local D: drive automatically.
- Local notebook edits should be saved explicitly; do not assume automatic synchronization with the Kaggle website's notebook copy.
- Kaggle session limits and quota still apply; a stopped session requires reconnection.
- The server URL contains authentication information. Keep it out of shared documents, repositories, and public screenshots.

The connection is a development convenience. It does not itself publish a Kaggle notebook version, submit a CSV, or meet notebook-sharing requirements.

## 14. Setup checklists and diagnostic cells

### 14.1 Accounts, venue, and collaboration

- [ ] Event registration/acceptance confirmed for this team.
- [ ] All four students can attend the same chosen venue.
- [ ] Every member can access their own Kaggle account.
- [ ] Each member has entered the competition and accepted the rules.
- [ ] Official Kaggle team membership is confirmed.
- [ ] Any verification required for accelerator access is completed.
- [ ] GPU allocation and remaining quota are visible and sufficient.
- [ ] RACE Cloud access and challenge briefs obtained.
- [ ] Shared working location and communication channel agreed.
- [ ] Notebook collaborators/judge group identified.
- [ ] Primary submission owner and backup assigned.

### 14.2 Kaggle environment

- [ ] GPU selected and kernel started.
- [ ] Competition CSV input attached.
- [ ] One approved model and tokenizer attached.
- [ ] Paths verified in the running session.
- [ ] Required imports succeed.
- [ ] A GPU tensor calculation succeeds.
- [ ] Tokenizer loads.
- [ ] Three-class model loads with understood warnings.
- [ ] A small dummy batch can run through the model.
- [ ] Notebook naming convention satisfied.
- [ ] Output/checkpoint saving behavior understood.

Kaggle's documentation describes GPU selection through notebook settings. Availability varies; inspect the actual account rather than assuming a particular GPU type or a fixed weekly allowance. [S8]

### 14.3 GPU diagnostic

Run in the environment whose readiness you want to establish. For the recommended remote workflow, run this **as a Kaggle-backed notebook cell**, not in a local Windows terminal.

```python
import sys
import torch
import pandas
import transformers
import datasets
import accelerate
import sklearn

print("Python:", sys.version)
print("Executable:", sys.executable)
print("PyTorch:", torch.__version__)
print("Transformers:", transformers.__version__)
print("CUDA build:", torch.version.cuda)
print("GPU available:", torch.cuda.is_available())

assert torch.cuda.is_available(), "No usable CUDA GPU in this kernel."
print("GPU:", torch.cuda.get_device_name(0))
x = torch.ones((32, 32), device="cuda")
assert (x @ x)[0, 0].item() == 32
print("GPU calculation passed.")
```

**Expected observation:** imports succeed, CUDA is available, the expected remote GPU is named, and the matrix calculation passes. This does not prove model loading, training, memory capacity for your batch, or submission correctness.

**Execution status:** this cell was proposed, not successfully run in Kaggle during the chat. The local PyTorch import failed because the package was absent.

### 14.4 Optional local environment procedure

These are **unexecuted setup instructions** for PowerShell on this Windows laptop, not a record of completed installation:

```powershell
Set-Location 'D:\Coding\Prj\RMITHACK'
uv venv --python 3.11 .venv
uv pip install --python .venv\Scripts\python.exe ipykernel pandas scikit-learn transformers datasets accelerate safetensors sentencepiece
```

Then:

1. Select a Windows GPU-compatible PyTorch build using the current official installer. [S11]
2. Install it into this same `.venv`, using the installer-selected package index/build. Do not blindly select an old CUDA wheel for the RTX 5060.
3. In VS Code's notebook kernel picker, select this environment.
4. Run the diagnostic cell and an actual model smoke check.
5. Record compatible package versions after the environment works.

`ipykernel` supplies local execution for VS Code notebooks. A separate JupyterLab browser UI is optional for that workflow. If a browser UI is wanted, install `jupyterlab` into the environment and run it from that environment. No global kernel registration or persistent editor setting is necessary merely to describe this option.

Avoid upgrading a working Kaggle environment indiscriminately on event day. Confirm installed versions and change only a dependency that blocks the chosen approach.

## 15. Proposed implementation structure

No notebook has been implemented yet. The proposed notebook sections are:

1. **Run identity:** selected model, seed, split definition, hyperparameters, package versions.
2. **Environment:** GPU calculation and input-path checks.
3. **Data:** schema, counts, missing cells, label map, language breakdown.
4. **Validation split:** fixed and reusable across comparable experiments.
5. **Tokenization:** paired tokenizer, max length, truncation, dynamic padding if suitable.
6. **Model:** approved backbone and three-class head.
7. **Training:** baseline loss and checkpoint evaluation.
8. **Evaluation:** macro F1, per-class precision/recall/F1, confusion matrix, language slices.
9. **Error analysis:** validation mistakes and confidence outputs.
10. **Test inference:** stable mapping from rows to predictions.
11. **Export:** sample-template merge, exact labels, structural checks.
12. **Artifacts:** model/tokenizer/config, notebook, scores, valid CSV.

Recommended label map:

```python
LABELS = ["BENIGN", "JAILBREAK", "PROMPT_INJECTION"]
label2id = {label: index for index, label in enumerate(LABELS)}
id2label = {index: label for label, index in label2id.items()}
```

The actual order is a design choice; consistency is mandatory. The exported labels must be the strings, not numeric class IDs.

Do not use `id` as a predictive shortcut. Use it to preserve row identity and audit outputs.

## 16. Validation design and learning discipline

### 16.1 Recommended first split

- **80% training / 20% validation** from the labelled training file.
- Fixed random seed, for example 42.
- Stratify jointly by language and label so the small groups remain represented.
- Use the exact same split for baseline and initial comparisons.

Illustrative split construction, not an executed training step:

```python
from sklearn.model_selection import train_test_split

strata = train["language"] + "__" + train["label"]
train_part, valid_part = train_test_split(
    train,
    test_size=0.20,
    random_state=42,
    stratify=strata,
)
```

The organizer's example uses a 10% split stratified by label. The 20% joint-stratified split is our recommendation, not an official requirement. [S3]

With only 98 English injection examples in the full training file, even a 20% split leaves approximately 20 such validation cases. Treat small-slice F1 changes cautiously and include support counts.

### 16.2 Template leakage

If prompts share an obvious template, source, paraphrase family, or translated counterpart, a random split may put very similar items on both sides. That can inflate validation scores. Group such families when there is sufficient evidence to identify them reliably.

The exact-duplicate checks performed do not settle this question. Do not invent grouping labels or spend the whole event building a complex clustering pipeline without evidence that this is the dominant risk.

### 16.3 Choosing a candidate

- Optimize validation macro F1, not training accuracy or training loss alone.
- Preserve per-class precision and recall to explain changes.
- If a result improves only one tiny slice, inspect uncertainty before declaring victory.
- If time permits, confirm the best meaningful change with another seed or a second sensible split.
- Repeatedly tuning on one holdout can overfit it; keep the number of experiments deliberate.

## 17. Recommended experiment sequence

### 17.1 Baseline

Proposed starting configuration:

| Setting | Starting value | Status |
|---|---|---|
| Backbone | XLM-RoBERTa-base | Recommendation |
| Output labels | 3 | Required by the task |
| Maximum length | 256 tokens | Starting point from organizer example |
| Learning rate | `2e-5` | Starting point from organizer example |
| Epoch limit | 3 | Starting point, evaluate along the way |
| Batch size | Fit to actual GPU memory | Must be measured |
| Loss | Ordinary three-class cross-entropy | Establish reference before changes |
| Checkpoint choice | Best validation macro F1 | Recommendation |
| Validation | Fixed 80/20 joint-stratified split | Recommendation |

Run a small smoke check before the full run. Measure training speed, GPU memory, evaluation time, and full-test inference duration. No runtime estimate in this document is a measured guarantee.

### 17.2 Diagnosis-driven comparisons

| Observed problem | Candidate change | Evidence needed |
|---|---|---|
| Injection recall poor | Compare moderate class weighting with baseline loss | Better macro F1 without unacceptable false positives |
| Excess benign false positives | Inspect examples; reduce excessive weighting if implicated | Improved benign precision/recall and overall metric |
| Truncated examples fail | Compare 256 with 512 tokens, or test head/tail retention as a later option | Token truncation measurements and validation improvement |
| One language weak | Examine its token lengths, errors, support, and content patterns | Repeatable failure pattern, not a handful of cases |
| Training/validation gap widens | Earlier checkpoint, fewer epochs, or smaller update rate | Better held-out performance |
| Baseline remains limited | Compare another approved backbone if allowed under the workflow and time | Meaningful score gain with viable runtime |

Change one major factor per comparison. Do not simultaneously change model, split, sequence length, class weights, and learning rate and then claim a specific cause.

### 17.3 Class-imbalance caution

Class weighting, focal loss, and resampling are suggested by the organizers. They are options, not guaranteed improvements. [S1]

- Compute any weights from the **training portion**, not hidden labels.
- Try one imbalance method at a time initially.
- Extreme weights may increase minority recall while hurting precision and other classes.
- Avoid automatically combining inverse-frequency weights with heavy oversampling.
- Keep the standard-loss baseline for comparison.

### 17.4 Text handling

Initially preserve punctuation, capitalization, Unicode, code fragments, and instruction structure. These may distinguish attacks from benign requests.

Measure token lengths with the actual chosen tokenizer. Character counts do not determine token counts uniformly across languages.

Translation, synthetic augmentation, external corpora, teacher-model labeling, and pseudo-labeling remain subject to unconfirmed permissions. Do not treat them as approved because they might be technically useful.

### 17.5 Final training decision

Once hyperparameters and epoch count are selected, retraining on all labelled training rows can use more data if time permits. Keep the previous validated checkpoint as a fallback.

If refitting on all rows, do not report scores on those same rows as new validation evidence. Preserve the earlier split-based score and identify the final model as a full-data refit whose hidden-test performance is unknown.

If a refit threatens the submission deadline, keep the known-working validated model.

## 18. Error analysis procedure

After each serious run, save:

- Macro F1.
- Per-class precision, recall, F1, and support.
- Confusion matrix.
- Per-language metrics with support counts.
- Validation IDs, true labels, predicted labels, and class probabilities.
- Input token length and truncation indicator if available.
- Runtime and relevant training settings.

Review approximately 20–30 mistakes from the training-derived validation split. Ask:

1. Are jailbreaks confused with injections?
2. Are benign discussions of attacks misclassified as actual attacks?
3. Are malicious instructions missed in long inputs?
4. Are failures concentrated in a language, template, or formatting pattern?
5. Is the apparent improvement a recall gain paid for by many false positives?
6. Do warnings or class-map mistakes explain the results better than model capacity?

Do not manually correct organizer test predictions based on human guesses. The error-analysis workflow is for improving a general model using permitted labelled data.

## 19. Team strategy and execution ownership

### 19.1 Proposed initial roles

These are suggestions, not confirmed assignments or assessments of any teammate's ability.

| Member | Initial ownership | Expected output |
|---|---|---|
| User / integration lead | Baseline architecture, integration, experiment decisions | Reproducible working pipeline |
| Data/evaluation lead | Split, reports, label handling, error analysis | Comparable scores and concrete next hypotheses |
| Training experiment lead | Bounded controlled comparisons | Candidate checkpoints, settings, timing, and results |
| Event/RACE lead | Other briefs, rubric, access, deadlines, final checks | Confirmed requirements and parallel challenge progress |

One person should own actual submissions and one should be the backup. Name the people before work starts; roles alone do not establish accountability.

### 19.2 Coordinating experiments

Use a small shared experiment table. Do not assume an experiment tracker service is required.

| Run | Model | Split | Length | Loss | LR | Epoch/checkpoint | Macro F1 | Injection F1 | Runtime | Decision |
|---|---|---|---:|---|---|---|---:|---:|---|---|
| baseline | TBD | fixed | 256 | standard | `2e-5` | TBD | — | — | — | pending |

Do not have all four people independently rebuild data loading, label maps, or incompatible baseline notebooks. Independent experiments should share the evaluation contract while keeping their outputs distinct.

### 19.3 Overall event allocation

At the briefing, prioritize tasks by:

- Marks/award relevance.
- Completion probability in remaining time.
- Dependencies and access bottlenecks.
- Measured time to train, infer, and package results.

If the first valid C1 submission secures all available C1 marks, move more people to other challenges. If leaderboard placement provides additional rewards, retain a focused modelling effort without abandoning the remaining deliverables.

### 19.4 Conditional directions for other briefs

These are planning branches, not confirmed event tasks:

| If a brief actually asks for… | First direction |
|---|---|
| Prompt-attack detection | Approved supervised classifier with held-out evaluation |
| Grounded question answering | Baseline retrieval → relevant context → grounded response; add hybrid retrieval/reranking only when measured failures justify it |
| Guardrail defense | Define permitted/forbidden behavior and attack/benign evaluation; measure missed attacks and overblocking |
| Attack generation | Work only within the authorized challenge targets and scoring protocol; maintain reproducible permitted outputs |
| An application/demo | Implement the smallest complete flow that directly satisfies the published rubric |

Hybrid RAG was a learning interest in AI Learning, but is not a published requirement of the inspected Kaggle task.

## 20. Submission safety and finishing plan

### 20.1 Use attempts deliberately

Submit an actual trained baseline early in the authorized window. Use local validation for most comparisons. Further uploads should correspond to credible candidates or a clearly defined diagnostic purpose.

Keep a fallback submission and reserve remaining attempts for meaningful improvements and recovery. The six-attempt reset period is still unconfirmed.

### 20.2 Time buffer

Measure the full inference/export pipeline. Reserve at least **twice that measured duration plus an upload/checking buffer**. A fixed arbitrary estimate is less reliable than a measurement.

Before the buffer begins:

- Freeze model and label mapping.
- Confirm all input/model files are available.
- Save the checkpoint/tokenizer and known-working notebook.
- Check enough storage remains for outputs.

At finish:

- Run final inference using the selected candidate.
- Validate CSV structure and ID alignment.
- Save the file and relevant outputs.
- Upload under the actual competition instructions.
- Confirm the platform reports a successful scored submission.
- Confirm final-submission selection if required.
- Confirm notebook accessibility and any other challenge deliverables.

### 20.3 Example CSV validator

This checks structure, not prediction quality or rule compliance. Run against a real candidate after inference. It has not been run against a generated model submission because no model submission exists yet.

```python
from pathlib import Path
import pandas as pd

# Set these paths for the current local or Kaggle environment.
test_path = Path("test.csv")
submission_path = Path("submission.csv")

test = pd.read_csv(test_path, dtype={"id": str})
submission = pd.read_csv(submission_path, dtype={"id": str})
allowed = {"BENIGN", "JAILBREAK", "PROMPT_INJECTION"}

assert submission.columns.tolist() == ["id", "label"]
assert len(submission) == len(test) == 6000
assert not submission[["id", "label"]].isna().any().any()
assert submission["id"].is_unique
assert set(submission["id"]) == set(test["id"])
assert submission["label"].isin(allowed).all()
print("Submission structure passed; prediction quality is not checked.")
```

Do not interpret passing this validator as proof the notebook is reproducible, the model is approved, or Kaggle accepted the upload.

## 21. Common failure modes and response

| Failure | First check / response |
|---|---|
| GPU unavailable in Kaggle | Account verification prompts, remaining quota, selected accelerator, session state; contact support/organizers for account blocks |
| VS Code cannot connect | Active Kaggle session, fresh VS Code-compatible URL, correct remote kernel; use Kaggle browser if connection troubleshooting consumes competition time |
| Imports fail | Confirm which kernel is executing and its installed versions before installing packages |
| Model not found | Attached approved model input and actual directory/version path |
| GPU out of memory | Reduce batch size; measure effects of sequence length; consider gradient accumulation or a smaller approved backbone |
| Wrong number of classes | Model head configuration and label map; understand checkpoint head warnings |
| Validation suspiciously high | Leakage, near-identical templates, label/map mistakes, accidental training on validation rows |
| Injection score poor | Confusion matrix, precision/recall, train imbalance, and label interpretation |
| Score changes after export | Prediction ordering, label decoding, checkpoint identity, and IDs |
| Notebook works interactively but not from a fresh run | Hidden state, missing saved artifacts, download dependencies, relative paths |
| Public leaderboard gain conflicts with validation | Preserve the validation-based candidate and investigate; do not assume either signal is definitive |
| Approaching deadline | Stop ambitious changes and submit the known-working candidate with a measured buffer |

Do not assume two GPUs automatically combine their VRAM or that training automatically uses both. Verify the actual training setup before depending on multi-GPU speed or capacity.

## 22. Learning priorities for teammate preparation

| Priority | Topic | Explain-back or practical check |
|---|---|---|
| Essential | Train/validation/test boundaries | Explain which labels are available and what each split is for |
| Essential | Tokenization and truncation | Show how text becomes tokens and what a max length discards |
| Essential | Encoder and classification head | Explain how pretrained text features become three class scores |
| Essential | Fine-tuning | Explain why model weights change and what the training labels teach |
| Essential | Learning rate, batch size, epochs | Explain one practical consequence of changing each |
| Essential | Precision, recall, macro F1 | Read a confusion matrix and identify the harmful error type |
| Essential | Imbalance | Explain why 62% accuracy from a majority-class predictor is inadequate |
| Essential | Kaggle runtime and files | Identify where cells execute and where inputs/outputs live |
| Essential | Reproducibility | Rerun the notebook without relying on invisible interactive state |
| Useful next | Class weights/focal loss/resampling | Compare one method to the same baseline fairly |
| Useful next | Per-language evaluation | Report performance with support counts and avoid overreading tiny slices |
| Optional | LoRA, QLoRA, quantization | Explain trainable adapters versus memory reduction and their implementation costs |
| Brief-dependent | Hybrid RAG and reranking | Explain retrieval → ranking → context → grounded answer only if a relevant task exists |

The team should be able to explain the selected baseline before adding sophisticated components. Training a model from scratch is not the published requirement.

## 23. Organizer questions and unresolved decisions

### 23.1 Blocking or consequential questions

| Question | Why it changes the plan |
|---|---|
| Is the team officially accepted, and at which venue? | Physical participation eligibility |
| What are the complete rubrics and weights? | Where the team should spend its time |
| Is C1 fully completion-based or does rank affect awards? | Value of continued classifier optimization |
| What are the RACE briefs, access details, and deadlines? | Architecture, ownership, and integration |
| May training begin before the opening session? | Permitted preparation versus competition work |
| Are external datasets, synthetic data, translation, or pseudo-labels allowed? | Data and augmentation strategy |
| Are AI coding assistants or external model APIs allowed, and must they be disclosed? | Development workflow and data handling |
| Must training/inference run on Kaggle, or is external compute allowed? | Local GPU/cloud backup plan |
| Are six attempts total or daily? | Submission budget |
| How many final submissions may be selected? | Final candidate selection |
| Are multi-model ensembles or multiple backbones permitted? | Potential late-stage approach |
| Who must receive notebook access, and with what permissions? | Judge accessibility |
| Is a saved/runnable notebook version required in addition to CSV? | Reproducibility and submission completion |
| Are reports, slides, demos, or source archives required? | Packaging effort |

An unanswered question is not permission and is not proof of a ban. Seek the current official clarification. No organizer was messaged during this chat.

### 23.2 Team decisions still open

- Exact team name and members' responsibilities.
- Selected approved model.
- Primary editor: Kaggle browser versus VS Code remote connection.
- Whether to configure a local backup environment.
- Accepted split definition and experiment budget.
- Final submission owner and backup.
- How effort is divided after the full briefs are known.

## 24. Completion criteria for the next implementation task

A future implementation should be considered ready only when the relevant items are demonstrated:

### Environment ready

- Target execution location explicitly identified.
- Required imports work in that environment.
- GPU calculation passes if GPU training is intended.
- Approved model/tokenizer load.
- A dummy batch runs successfully.

### Baseline ready

- Split and label mapping are reproducible.
- Training completes and a selected checkpoint is saved.
- Validation macro F1 and class reports are recorded.
- Inference produces exactly one prediction per test ID.
- CSV validator passes.
- Notebook reruns with declared inputs and dependencies.

### Competition submission complete

- Submission is actually uploaded during the permitted window.
- Kaggle confirms success and scoring.
- Final selection is confirmed when applicable.
- Notebook naming and access meet organizer instructions.
- Other event deliverables are separately completed.

Local preparation, a passing script, a valid CSV, a successful upload, and overall hackathon completion are different states. Report only the states actually verified.

## 25. Sources and verification notes

The competition pages were inspected through the browser when the web search reader could not load them. These links identify the public evidence; they are not a guarantee that the pages will remain unchanged.

| ID | Source | What it supports |
|---|---|---|
| S1 | [Competition overview](https://www.kaggle.com/competitions/rmit-hackathon-2026/overview) | Task, model list, naming, metric, attempts, marks statement, deadline displayed by Kaggle |
| S2 | [Competition data](https://www.kaggle.com/competitions/rmit-hackathon-2026/data) | Schema, row counts, prompt descriptions, licence |
| S3 | [Pinned organizer FAQ](https://www.kaggle.com/competitions/rmit-hackathon-2026/discussion/748423) | Language distribution, approved-model guidance, starter example, sharing |
| S4 | [Competition rules](https://www.kaggle.com/competitions/rmit-hackathon-2026/rules) | Foundational rules, eligibility, teams, sharing, entry and winner requirements |
| S5 | [Leaderboard](https://www.kaggle.com/competitions/rmit-hackathon-2026/leaderboard) | Approximately 30% public / 70% private split |
| S6 | [Official event website](https://rmit-hackathon.com/) | Event dates, venues, team size, attendance, registration, contact |
| S7 | [Kaggle Spoiler Alert](https://www.kaggle.com/spoiler-alert/) | Late public notebook-sharing guidance |
| S8 | [Kaggle notebook documentation](https://www.kaggle.com/docs/notebooks) | Notebook operation, GPU settings, experimental external Jupyter connection |
| S9 | [Kaggle's external Jupyter connection announcement](https://www.kaggle.com/discussions/product-announcements/567740) | Official external-editor connection feature and experimental status |
| S10 | [VS Code Jupyter documentation](https://code.visualstudio.com/docs/datascience/jupyter-notebooks) | Notebook support and remote-kernel selection |
| S11 | [PyTorch local installation selector](https://pytorch.org/get-started/locally/) | Current supported local-install choices |

Additional directly inspected organizer resource: [notebook-sharing illustration](https://storage.googleapis.com/kaggle-forum-message-attachments/3537136/52679/share.png).

Local numerical evidence comes from the actual CSV files and tool checks described in sections 7 and 11. The local files were not compared to organizer checksums, and no claim is made about their independent provenance beyond their presence and agreement with the published schema/counts.

### Final handoff status

The competition briefing, local readiness inspection, notebook-connection explanation, and proposed strategy are documented. Training setup, model implementation, live Kaggle connection, model results, and submissions remain outstanding. This Markdown file is a detailed context handoff, not evidence that those later stages have been completed.
