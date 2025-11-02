import sys, pandas as pd, numpy as np, joblib
from pathlib import Path

if len(sys.argv) < 2:
    print("Usage: python src/predict_csv.py <chemin_csv>")
    sys.exit(1)

csv_in = Path(sys.argv[1]); assert csv_in.exists(), f"Fichier introuvable: {csv_in}"
pipe = joblib.load("models/logreg.joblib")
thresh = 0.5
thf = Path("models/threshold.txt")
if thf.exists():
    try: thresh = float(thf.read_text().strip())
    except: pass

df = pd.read_csv(csv_in)
if "TotalCharges" in df:
    s = pd.to_numeric(df["TotalCharges"].astype(str).str.strip().replace({"": np.nan}), errors="coerce")
    df["TotalCharges"] = s.fillna(s.median())

drop_cols = [c for c in ["Churn","customerID","CustomerID","customerId"] if c in df.columns]
X = df.drop(columns=drop_cols, errors="ignore")

proba = pipe.predict_proba(X)[:,1]
pred = (proba >= thresh).astype(int)

out = df.copy()
out["churn_proba"] = proba
out["churn_pred"]  = pred
out_path = csv_in.with_suffix(".pred.csv")
out.to_csv(out_path, index=False)
print(f"[OK] {len(out)} lignes prédites → {out_path} (seuil={thresh:.3f})")
