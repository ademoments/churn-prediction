Churn Prediction – Telco Customer Dataset
Présentation du projet

Dans le cadre de mon parcours en Master Machine Learning for Data Science à l’Université Paris Descartes, j’ai réalisé ce projet afin d’explorer l’application de l’apprentissage automatique à la prédiction de la fidélité client.
L’objectif est de concevoir un modèle capable de prédire la probabilité qu’un client quitte son opérateur télécom (“churn”) à partir de données réelles issues du jeu de données Telco Customer Churn.

Le projet vise à illustrer une démarche complète de data science :
préparation des données, modélisation, évaluation et interprétation des résultats dans une optique métier.

Les données utilisées proviennent du dataset public Telco Customer Churn disponible sur Kaggle
.

Objectifs

Préparer et nettoyer les données :

Gestion des valeurs manquantes.

Encodage des variables catégorielles.

Normalisation des variables numériques.

Concevoir et entraîner plusieurs modèles de classification supervisée :

Régression logistique.

Forêt aléatoire.

XGBoost et LightGBM.

Comparer leurs performances à l’aide de métriques standards :

Accuracy, AUC, Précision, Rappel, F1-score.

Interpréter le modèle le plus performant pour identifier les facteurs clés du churn.

Fournir une pipeline complète et reproductible pour l’entraînement et l’inférence.

Structure du projet
.
├── data/                      # Données brutes (Telco dataset)
│   ├── WA_Fn-UseC_-Telco-Customer-Churn.csv
│   └── telco-customer-churn.zip
├── notebooks/
│   └── eda_churn.ipynb        # Analyse exploratoire (EDA)
├── src/                       # Scripts sources du projet
│   ├── train_models.py        # Entraînement et sauvegarde des modèles
│   ├── evaluate_models.py     # Évaluation et génération des rapports
│   ├── threshold_tuning.py    # Optimisation du seuil de classification
│   ├── explain_logreg.py      # Interprétation du modèle logistique
│   └── predict_csv.py         # Prédiction sur de nouveaux jeux de données
├── models/                    # Modèles entraînés (.joblib)
├── reports/                   # Résultats et rapports (.csv / .md)
│   ├── metrics.md
│   ├── confusion_*.csv
│   ├── logreg_top_positive_features.csv
│   └── logreg_top_negative_features.csv
├── requirements.txt
└── README.md

Prérequis

Python 3.9 à 3.12.

Un environnement virtuel est recommandé (venv ou conda).

IDE compatible (VS Code, JupyterLab ou autre).

Installation des dépendances
python3 -m venv env
source env/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

Étapes du projet
1. Préparation et exploration des données

Chargement du jeu de données Telco Customer Churn.

Nettoyage des colonnes numériques et catégorielles.

Traitement de la colonne TotalCharges (conversion et imputation).

Étude des corrélations et visualisation des distributions.

Analyse de la variable cible Churn.

2. Entraînement des modèles
python src/train_models.py


Les modèles testés :

Logistic Regression

Random Forest

XGBoost

LightGBM

Les pipelines incluent la normalisation et l’encodage des variables.

Les modèles sont sauvegardés dans le dossier models/.

3. Évaluation et comparaison
python src/evaluate_models.py


Les métriques (Accuracy, AUC, F1, Recall) sont exportées dans reports/metrics.md
Les matrices de confusion pour chaque modèle sont disponibles dans reports/confusion_*.csv.

4. Ajustement du seuil de décision
python src/threshold_tuning.py


Recherche du seuil optimal de probabilité de churn maximisant le F1-score.
Le seuil choisi est enregistré dans models/threshold.txt.

5. Interprétation du modèle
python src/explain_logreg.py


Analyse des coefficients du modèle logistique pour identifier les facteurs influençant le churn.
Résultats exportés dans :

reports/logreg_top_positive_features.csv (facteurs pro-churn)

reports/logreg_top_negative_features.csv (facteurs anti-churn)

6. Prédiction sur de nouvelles données
python src/predict_csv.py data/WA_Fn-UseC_-Telco-Customer-Churn.csv


Le fichier de sortie (ex. telco.pred.csv) contient :

churn_proba : probabilité de départ.

churn_pred : prédiction binaire (0 = reste, 1 = quitte).

Résultats clés
Modèle	Accuracy	AUC
Logistic Regression	0.808	0.846
Random Forest	0.784	0.823
XGBoost	0.789	0.829
LightGBM	0.782	0.822

Meilleur modèle : Régression logistique, simple, stable et interprétable.

Principaux enseignements

Les clients sous contrat “Month-to-month” ont un fort risque de churn.

L’absence de services “Tech Support” ou “Online Security” est corrélée au départ.

Les clients utilisant le mode de paiement “Electronic Check” partent plus souvent.

Une ancienneté élevée (“tenure”) réduit significativement le risque de churn.

Les contrats sur 2 ans sont les plus stables.

Améliorations possibles

Optimiser les hyperparamètres des modèles avec GridSearchCV ou Optuna.

Ajouter des variables agrégées (revenu mensuel total, fidélité).

Tester des modèles de gradient boosting avancés (CatBoost).

Déployer une API de prédiction (FastAPI) ou une interface Streamlit.

Intégrer des alertes automatiques de churn sur de nouvelles données clients.

Auteur

Adem Bounaidja-Rachedi
Master 1 – Machine Learning for Data Science
Université Paris Descartes, France
GitHub – ademoments