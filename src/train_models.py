# src/train_models.py
"""
Churn Prediction - Comparaison de modèles
- Charge le CSV Telco depuis ./data/
- Nettoyage robuste de TotalCharges
- Création de X (features) et y (cible binaire)
- Pipeline: OneHotEncoder + StandardScaler (numériques) + modèle
- Modèles testés: LogisticRegression, RandomForest (+ XGBoost/LightGBM si dispo)
- Affiche ACC/AUC sur le test set
- Sauvegarde chaque pipeline entraîné dans ./models/<name>.joblib
"""

from pathlib import Path
import sys
import numpy as np
import pandas as pd
from joblib import dump

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

# ---------------------------
# 0) Localiser le dataset
# ---------------------------
DATA = Path("data/WA_Fn-UseC_-Telco-Customer-Churn.csv")
if not DATA.exists():
    alt = Path("data/telco.csv")
    if alt.exists():
        DATA = alt
    else:
        print("[ERREUR] Place le CSV dans ./data/ (ex: WA_Fn-UseC_-Telco-Customer-Churn.csv)")
        sys.exit(1)

# ---------------------------
# 1) Charger + nettoyer
# ---------------------------
df = pd.read_csv(DATA)

# Nettoyage robuste de TotalCharges (souvent 'object' avec espaces)
if "TotalCharges" in df.columns:
    s = pd.to_numeric(
        df["TotalCharges"].astype(str).str.strip().replace({"": np.nan}),
        errors="coerce"
    )
    df["TotalCharges"] = s.fillna(s.median())

# Cible binaire y
if "Churn" not in df.columns:
    print("[ERREUR] Colonne 'Churn' introuvable dans le CSV.")
    sys.exit(1)

y = df["Churn"].astype(str).str.lower().isin(["yes", "1", "true"]).astype(int)

# Features X (on retire target + identifiants)
drop_cols = [c for c in ["Churn", "customerID", "CustomerID", "customerId"] if c in df.columns]
X = df.drop(columns=drop_cols)

# Colonnes numériques / catégorielles
num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
cat_cols = X.select_dtypes(exclude=[np.number]).columns.tolist()

# ---------------------------
# 2) Split train/test
# ---------------------------
Xtr, Xte, ytr, yte = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

# ---------------------------
# 3) Préprocesseur commun
# ---------------------------
# OneHotEncoder -> on force la sortie dense pour compatibilité simple
pre = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), num_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols),
    ],
    remainder="drop",
)

# ---------------------------
# 4) Définir les modèles
# ---------------------------
models = {
    "logreg": LogisticRegression(max_iter=1000),
    "rf": RandomForestClassifier(n_estimators=400, random_state=42),
}

# XGBoost (si installé)
try:
    from xgboost import XGBClassifier
    models["xgb"] = XGBClassifier(
        n_estimators=500,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1,
    )
except Exception:
    pass  # ignoré si non installé

# LightGBM (si installé)
try:
    from lightgbm import LGBMClassifier
    models["lgbm"] = LGBMClassifier(
        n_estimators=600,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=42,
        n_jobs=-1,
    )
except Exception:
    pass  # ignoré si non installé

# ---------------------------
# 5) Entraîner / Évaluer / Sauver
# ---------------------------
Path("models").mkdir(exist_ok=True)
results = []

for name, model in models.items():
    pipe = Pipeline([("pre", pre), ("clf", model)])
    pipe.fit(Xtr, ytr)

    proba = pipe.predict_proba(Xte)[:, 1]
    pred = (proba >= 0.5).astype(int)

    acc = accuracy_score(yte, pred)
    auc = roc_auc_score(yte, proba)
    results.append((name, acc, auc))

    out_path = Path(f"models/{name}.joblib")
    dump(pipe, out_path)
    print(f"[OK] Modèle sauvegardé → {out_path}")

# ---------------------------
# 6) Afficher le tableau des scores
# ---------------------------
print("\n[RESULTATS - triés par AUC]")
for name, acc, auc in sorted(results, key=lambda t: t[2], reverse=True):
    print(f"{name:6s}  ACC={acc:.4f}  AUC={auc:.4f}")
