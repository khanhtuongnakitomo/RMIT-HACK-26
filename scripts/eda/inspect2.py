import pandas as pd, numpy as np, sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import f1_score, classification_report, confusion_matrix
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import hstack

D = sys.argv[1] if len(sys.argv) > 1 else ".."  # folder containing the competition CSVs
tr = pd.read_csv(f"{D}/train.csv", dtype=str, keep_default_na=False)
te = pd.read_csv(f"{D}/test.csv", dtype=str, keep_default_na=False)
L = ["BENIGN", "JAILBREAK", "PROMPT_INJECTION"]
y = tr.label.map({l: i for i, l in enumerate(L)}).values

vc = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=3, sublinear_tf=True, max_features=300000)
vw = TfidfVectorizer(analyzer="word", ngram_range=(1, 2), min_df=2, sublinear_tf=True, max_features=200000)
allt = pd.concat([tr.text, te.text])
vc.fit(allt); vw.fit(allt)
Xtr = hstack([vc.transform(tr.text), vw.transform(tr.text)]).tocsr()
Xte = hstack([vc.transform(te.text), vw.transform(te.text)]).tocsr()

skf = StratifiedKFold(5, shuffle=True, random_state=42)
strata = tr.language + "_" + tr.label
for cw in [None, "balanced"]:
    clf = LogisticRegression(C=10, max_iter=3000, class_weight=cw)
    p = cross_val_predict(clf, Xtr, y, cv=skf.split(Xtr, strata), method="predict_proba")
    pred = p.argmax(1)
    print(f"\n### TF-IDF LR class_weight={cw}: macroF1={f1_score(y, pred, average='macro'):.4f}")
    print(classification_report(y, pred, target_names=L, digits=3))
    print(confusion_matrix(y, pred))
    for lang in ["en", "pt", "fr", "it"]:
        m = (tr.language == lang).values
        print(lang, round(f1_score(y[m], pred[m], average="macro"), 4))
    if cw == "balanced":
        tr["p_inj"] = p[:, 2]; tr["pred"] = [L[i] for i in pred]

# misclassified injection examples
pd.set_option("display.max_colwidth", 250)
print("\n--- injection missed (sample):")
for _, r in tr[(tr.label == "PROMPT_INJECTION") & (tr.pred != "PROMPT_INJECTION")].sample(12, random_state=0).iterrows():
    print(f"[{r.language}] ->{r.pred}: {r.text[:250]!r}")
print("\n--- false injection (sample):")
for _, r in tr[(tr.label != "PROMPT_INJECTION") & (tr.pred == "PROMPT_INJECTION")].sample(8, random_state=0).iterrows():
    print(f"[{r.language}] {r.label}->: {r.text[:250]!r}")
print("\n--- JB<->BENIGN confusions (sample):")
for _, r in tr[((tr.label == "JAILBREAK") & (tr.pred == "BENIGN")) | ((tr.label == "BENIGN") & (tr.pred == "JAILBREAK"))].sample(10, random_state=0).iterrows():
    print(f"[{r.language}] {r.label}->{r.pred}: {r.text[:250]!r}")

# Near-duplicates: train-train (cross-language translations?) and train-test
Xc = vc.transform(tr.text); Xct = vc.transform(te.text)
nn = NearestNeighbors(n_neighbors=2, metric="cosine").fit(Xc)
d, i = nn.kneighbors(Xc)
sim = 1 - d[:, 1]; j = i[:, 1]
print("\ntrain-train NN char-sim quantiles:", np.quantile(sim, [.5, .9, .95, .99]).round(3))
hi = sim > 0.9
print("pairs sim>0.9:", hi.sum(), " label-conflict among them:", (tr.label.values[hi] != tr.label.values[j[hi]]).sum())
d2, i2 = nn.kneighbors(Xct, n_neighbors=1)
s2 = 1 - d2[:, 0]
print("test->train NN char-sim quantiles:", np.quantile(s2, [.5, .9, .95, .99]).round(3), " >0.9:", (s2 > 0.9).sum(), " >0.8:", (s2 > 0.8).sum())
for k in np.where(hi & (tr.label.values != tr.label.values[j]))[0][:6]:
    print(f"  CONFLICT {sim[k]:.2f} {tr.label[k]} vs {tr.label[j[k]]}: {tr.text[k][:150]!r} || {tr.text[j[k]][:150]!r}")
