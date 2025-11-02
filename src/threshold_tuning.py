from pathlib import Path
import numpy as np, pandas as pd, joblib
from sklearn.metrics import precision_recall_curve, roc_auc_score

DATA = next((Path(p) for p in ["data/WA_Fn-UseC_-Telco-Customer-Churn.csv","data/telco.csv"] if Path(p).exists()), None)
assert DATA, "CSV manquant dans ./data/"
df = pd.read_csv(DATA)

# Nettoyage minimal
s = pd.to_numeric(df["TotalCharges"].astype(str).str.strip().replace({"": np.nan}), errors="coerce")
df["TotalCharges"] = s.fillna(s.median())

y = df["Churn"].astype(str).str.lower().isin(["yes","1","true"]).astype(int)
X = df.drop(columns=[c for c in ["Churn","customerID","CustomerID","customerId"] if c in df.columns])

pipe = joblib.load("models/logreg.joblib")  # modèle gagnant
proba = pipe.predict_proba(X)[:,1]

precision, recall, thresholds = precision_recall_curve(y, proba)
f1 = 2*(precision*recall)/(precision+recall+1e-9)
ix = np.nanargmax(f1)
best_t = thresholds[ix] if ix < len(thresholds) else 0.5

print(f"AUC={roc_auc_score(y, proba):.4f} | seuil F1={best_t:.3f} (P={precision[ix]:.3f}, R={recall[ix]:.3f})")
Path("models").mkdir(exist_ok=True)
Path("models/threshold.txt").write_text(f"{best_t:.4f}\n", encoding="utf-8")
print("[OK] models/threshold.txt écrit")
