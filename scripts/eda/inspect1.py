import pandas as pd, numpy as np, re, sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
D = sys.argv[1] if len(sys.argv) > 1 else ".."  # folder containing the competition CSVs
tr = pd.read_csv(f"{D}/train.csv", dtype=str, keep_default_na=False)
te = pd.read_csv(f"{D}/test.csv", dtype=str, keep_default_na=False)
ss = pd.read_csv(f"{D}/sample_submission.csv", dtype=str, keep_default_na=False)
print(tr.shape, te.shape, ss.shape)
print("ss ids == test ids (ordered):", (ss.id.values == te.id.values).all())
print(pd.crosstab(tr.language, tr.label, margins=True))
print("test lang:\n", te.language.value_counts())
for df, n in [(tr, "train"), (te, "test")]:
    df["nch"] = df.text.str.len()
    df["nw"] = df.text.str.split().str.len()
print(tr.groupby("label").nch.describe(percentiles=[.5, .9, .95, .99]))
print("test nch:", te.nch.describe(percentiles=[.5, .9, .95, .99]))
print("train nch:", tr.nch.describe(percentiles=[.5, .9, .95, .99]))

# ID numbering: interleaved?
tr["idn"] = tr.id.str[1:].astype(int); te["idn"] = te.id.str[1:].astype(int)
print("id ranges train", tr.idn.min(), tr.idn.max(), "test", te.idn.min(), te.idn.max())
tr["idbin"] = pd.cut(tr.idn, 10)
print(pd.crosstab(tr.idbin, tr.label, normalize="index").round(3))

# exact / normalized overlap train-test
norm = lambda s: re.sub(r"\s+", " ", s.lower()).strip()
trn = set(tr.text.map(norm)); ten = te.text.map(norm)
print("test texts exactly in train (normalized):", ten.isin(trn).sum())
print("dup within test:", te.text.map(norm).duplicated().sum())

# features per class
pats = {
 "placeholder [CAPS]": r"\[[A-Z_ ]{3,}\]",
 "ignore/previous instr (multi)": r"(ignore|ignor|oubli|esque|dimentic|instru|previous|précédent|anterior|precedent)",
 "DAN / dev mode": r"\b(DAN|developer mode|modo desenvolvedor|mode développeur|modalità sviluppatore)\b",
 "url": r"https?://",
 "code fence/brace": r"```|\{|\}",
 "newline": r"\n",
 "quotes \"": r"\"",
 "colon-ended": r":\s*$",
 "starts lower": r"^[a-zà-ú]",
 "contains 'system'": r"(?i)system|sistema|système",
 "contains 'prompt'": r"(?i)prompt",
 "contains 'role/act as'": r"(?i)act as|aja como|agisci come|agis comme|agissez|finja|fai finta|imagine",
}
rows = []
for k, p in pats.items():
    r = {"feat": k}
    for lab in ["BENIGN", "JAILBREAK", "PROMPT_INJECTION"]:
        r[lab] = round(tr[tr.label == lab].text.str.contains(p, regex=True).mean(), 3)
    r["TEST"] = round(te.text.str.contains(p, regex=True).mean(), 3)
    rows.append(r)
print(pd.DataFrame(rows).to_string())

pd.set_option("display.max_colwidth", 300)
rng = np.random.RandomState(0)
for lab in ["BENIGN", "JAILBREAK", "PROMPT_INJECTION"]:
    print("\n=====", lab)
    for lang in ["en", "pt", "fr", "it"]:
        s = tr[(tr.label == lab) & (tr.language == lang)].sample(3, random_state=1)
        for t in s.text: print(f"[{lang}] {t[:300]!r}")
