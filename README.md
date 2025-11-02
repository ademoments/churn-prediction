# **Churn Prediction – Telco Customer Dataset**

## **Présentation du projet**

Ce projet a été réalisé dans le cadre de mon parcours en **Master Machine Learning for Data Science** à l’Université **Paris Descartes**.  
L’objectif est de développer une **pipeline complète de machine learning** capable de prédire la **probabilité qu’un client quitte son opérateur télécom (churn)**, à partir du jeu de données **Telco Customer Churn**.

Le projet a été conçu de manière modulaire et industrialisable :
- **Exploration et préparation des données**
- **Entraînement, évaluation et interprétation des modèles**
- **Optimisation automatique du seuil de décision**
- **Conteneurisation via Docker et orchestration avec Docker Compose**

Les données utilisées proviennent du dataset public **Telco Customer Churn** disponible sur Kaggle.

---

## **Objectifs**

- **Nettoyer et préparer les données** :
  - Gestion des valeurs manquantes  
  - Conversion des types et imputation de `TotalCharges`  
  - Encodage des variables catégorielles et normalisation  

- **Entraîner plusieurs modèles supervisés de classification** :
  - Logistic Regression  
  - Random Forest  
  - XGBoost  
  - LightGBM  

- **Comparer leurs performances selon plusieurs métriques** :
  - Accuracy, AUC, Précision, Rappel, F1-score  

- **Optimiser le seuil de décision du churn**  
- **Automatiser le processus d’entraînement et de prédiction avec Docker Compose**

---

## **Structure du projet**
```plaintext
.
├── README.md                     # Documentation du projet
├── requirements.txt              # Dépendances Python
├── Dockerfile                    # Image Docker principale (pipeline complet)
├── docker-compose.yml            # Orchestration multi-services
├── .dockerignore                 # Exclusions pour le build Docker
│
├── data/
│   ├── WA_Fn-UseC_-Telco-Customer-Churn.csv        # Jeu de données brut
│   ├── WA_Fn-UseC_-Telco-Customer-Churn.pred.csv   # Fichier prédit
│   └── telco-customer-churn.zip
│
├── models/                       # Modèles entraînés (.joblib)
│   ├── logreg.joblib
│   ├── rf.joblib
│   ├── xgb.joblib
│   ├── lgbm.joblib
│   └── threshold.txt
│
├── notebooks/
│   └── eda_churn.ipynb           # Analyse exploratoire (EDA)
│
├── reports/                      # Résultats et visualisations
│   ├── metrics.md
│   ├── confusion_logreg.csv
│   ├── confusion_rf.csv
│   ├── confusion_xgb.csv
│   ├── confusion_lgbm.csv
│   ├── logreg_top_positive_features.csv
│   └── logreg_top_negative_features.csv
│
└── src/                          # Code source principal
    ├── train_models.py           # Entraînement des modèles
    ├── evaluate_models.py        # Évaluation et génération de rapports
    ├── threshold_tuning.py       # Optimisation du seuil de churn
    ├── explain_logreg.py         # Analyse des coefficients du modèle logistique
    └── predict_csv.py            # Prédiction sur de nouveaux fichiers CSV

```
---

## **Prérequis**

- Python 3.12  
- Docker et Docker Compose  
- Environnement Linux ou WSL2 (Windows Subsystem for Linux)

### Installation des dépendances
```bash
python3 -m venv env  
source env/bin/activate  
python -m pip install --upgrade pip  
python -m pip install -r requirements.txt  
```
---

## **Étapes du projet**

### 1. **Entraînement des modèles**
```bash
python src/train_models.py  
```
Les modèles suivants sont entraînés :
- Logistic Regression  
- Random Forest  
- XGBoost  
- LightGBM  

Les résultats (Accuracy, AUC) sont affichés dans la console et sauvegardés dans `models/`.

---

### 2. **Évaluation et rapports**
```bash
python src/evaluate_models.py  
```
Cette étape génère automatiquement :
- `reports/metrics.md` : tableau comparatif des performances  
- `reports/confusion_*.csv` : matrices de confusion par modèle  

---

### 3. **Optimisation du seuil de décision**
```bash
python src/threshold_tuning.py  
```
Le seuil optimal maximisant le F1-score est calculé et enregistré dans :
```bash
models/threshold.txt  
```
---

### 4. **Interprétation du modèle logistique**
```bash
python src/explain_logreg.py  
```
Les coefficients du modèle sont analysés pour comprendre les facteurs du churn.  
Résultats exportés dans :
```bash
- reports/logreg_top_positive_features.csv  
- reports/logreg_top_negative_features.csv  
```
---

### 5. **Prédiction sur de nouvelles données**
```bash
python src/predict_csv.py data/WA_Fn-UseC_-Telco-Customer-Churn.csv  
```
Sortie :
```bash
data/WA_Fn-UseC_-Telco-Customer-Churn.pred.csv  
```
Contient :
- churn_proba → probabilité de churn  
- churn_pred → prédiction binaire (0 = reste, 1 = quitte)  

---

## **Exécution complète via Docker**

### 1. Construire l’image
```bash
docker build -t churn-prediction:latest .  
```
### 2. Lancer un entraînement simple
```bash
docker run --rm \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/models:/app/models \
  -v $(pwd)/reports:/app/reports \
  churn-prediction:latest  
```
### 3. Orchestration complète (multi-services)
```bash
docker compose build  
docker compose run --rm train  
docker compose run --rm eval  
docker compose run --rm threshold  
docker compose run --rm predict  
```
Les sorties générées sont automatiquement montées dans :
- data/  
- models/  
- reports/  

---

## **Résultats clés**

| Modèle              | Accuracy | AUC  |
|---------------------|-----------|------|
| Logistic Regression | 0.808     | 0.846 |
| Random Forest       | 0.784     | 0.823 |
| XGBoost             | 0.789     | 0.829 |
| LightGBM            | 0.782     | 0.822 |

### Principaux enseignements :
- Les contrats “Month-to-month” sont fortement associés au churn.  
- L’absence de Tech Support ou Online Security augmente le risque.  
- Le mode de paiement Electronic Check est souvent corrélé au départ.  
- Une plus grande ancienneté (tenure) diminue le risque de churn.  

---

## **Améliorations futures**

- Optimisation des hyperparamètres (GridSearchCV, Optuna)  
- Déploiement d’une API FastAPI ou d’une interface Streamlit  
- Monitoring du churn en temps réel avec Docker Compose et Grafana  
- Intégration CI/CD via GitHub Actions  

---

## **Auteur**

**Adem Bounaidja-Rachedi**  
_Master 1 – Machine Learning for Data Science_  
_Université Paris Descartes, France_  
[GitHub – ademoments](https://github.com/ademoments)
