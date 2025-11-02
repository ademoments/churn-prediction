from pathlib import Path
import numpy as np, pandas as pd, joblib
from sklearn.metrics import accuracy_score, roc_auc_score, precision_recall_fscore_support, confusion_matrix
from sklearn.model_selection import train_test_split

DATA_CANDS = ["data/WA_Fn-UseC_-Telco-Customer-Churn.csv","data/telco.csv"]
csv = next((Path(p) for p in DATA_CANDS if Path(p).exists()), None)
assert csv, "Place le CSV dans ./data/"
df = pd.read_csv(csv)


s = pd.to_numeric(df["TotalCharges"].astype(str).str.strip().replace({"": np.nan}), errors="coerce")
df["TotalCharges"] = s.fillna(s.median())

y = df["Churn"].astype(str).str.lower().isin(["yes","1","true"]).astype(int)
X = df.drop(columns=[c for c in ["Churn","customerID","CustomerID","customerId"] if c in df])


Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)


models = []
for name in ["logreg","rf","xgb","lgbm"]:
    p = Path(f"models/{name}.joblib")
    if not p.exists():
        continue
    if name == "lgbm":
        try:
            import lightgbm  # noqa
        except Exception:
            print("[WARN] LightGBM indisponible, skip.")
            continue
    models.append((name, joblib.load(p)))

rows, cms = [], {}
for name, pipe in models:
    proba = pipe.predict_proba(Xte)[:,1]
    pred = (proba >= 0.5).astype(int)
    acc = accuracy_score(yte, pred)
    auc = roc_auc_score(yte, proba)
    prec, rec, f1, _ = precision_recall_fscore_support(yte, pred, average="binary", pos_label=1, zero_division=0)
    rows.append(dict(model=name, accuracy=acc, roc_auc=auc, precision=prec, recall=rec, f1=f1))
    cms[name] = confusion_matrix(yte, pred)

res = pd.DataFrame(rows).sort_values("roc_auc", ascending=False)
Path("reports").mkdir(exist_ok=True)
md = ["| Model | Accuracy | ROC-AUC | Precision | Recall | F1 |",
      "|------:|---------:|--------:|----------:|-------:|---:|"]
for r in res.itertuples(index=False):
    md.append(f"| {r.model} | {r.accuracy:.4f} | {r.roc_auc:.4f} | {r.precision:.4f} | {r.recall:.4f} | {r.f1:.4f} |")
Path("reports/metrics.md").write_text("\n".join(md), encoding="utf-8")

for name, cm in cms.items():
    np.savetxt(f"reports/confusion_{name}.csv", cm, fmt="%d", delimiter=",")
print("[OK] reports/metrics.md (test split) + confusion_*.csv")