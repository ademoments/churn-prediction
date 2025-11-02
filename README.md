<!doctype html>
<html lang="fr">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>Churn Prediction – Telco Customer Dataset</title>
<style>
  :root{
    --bg:#0b1220; --fg:#e7ecf3; --muted:#a7b1c2; --card:#121a2b;
    --accent:#4da3ff; --accent2:#7bd389; --border:#1b2740; --code:#0f172a;
  }
  *{box-sizing:border-box}
  body{
    margin:0; font:16px/1.6 system-ui,-apple-system,Segoe UI,Roboto,Ubuntu,Helvetica,Arial,sans-serif;
    color:var(--fg); background:linear-gradient(160deg,#0a1120 0%, #0c1324 70%, #0e1a33 100%);
  }
  .wrap{max-width:1000px; margin:0 auto; padding:48px 20px 72px}
  header{display:flex; align-items:flex-start; justify-content:space-between; gap:16px; margin-bottom:24px}
  .title{font-size:34px; line-height:1.2; margin:0}
  .subtitle{color:var(--muted); margin:6px 0 0}
  .badges{display:flex; flex-wrap:wrap; gap:8px; margin-top:12px}
  .badge{
    font-size:12px; border:1px solid var(--border); background:#0e1630; color:#cfe3ff;
    padding:6px 10px; border-radius:999px;
  }
  section{background:rgba(18,26,43,0.78); border:1px solid var(--border); border-radius:16px; padding:22px 20px; margin:18px 0}
  h2{font-size:22px; margin:0 0 12px; color:#dfe9ff}
  p{margin:10px 0}
  ul{margin:8px 0 0 20px}
  code, pre{font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace}
  pre{
    background:var(--code); color:#e6edf3; border:1px solid #1e293b; border-radius:12px;
    padding:14px; overflow:auto; margin:14px 0;
  }
  .tree{white-space:pre; background:#0e1528}
  .grid{display:grid; grid-template-columns:1fr 1fr; gap:14px}
  @media (max-width:800px){ .grid{grid-template-columns:1fr} header{flex-direction:column} }
  table{width:100%; border-collapse:separate; border-spacing:0; overflow:hidden; border-radius:12px; border:1px solid var(--border); background:#0f1730}
  th, td{padding:10px 12px; text-align:left; border-bottom:1px solid var(--border)}
  th{color:#cfe3ff; background:#0d1530}
  tr:last-child td{border-bottom:0}
  .pill{display:inline-block; padding:2px 8px; border-radius:999px; background:#0c1a33; border:1px solid var(--border); color:#cfe3ff}
  .note{color:var(--muted); font-size:14px}
  .kbd{background:#0f1730; border:1px solid var(--border); border-bottom-color:#0a0f1f; padding:1px 6px; border-radius:6px; font-family:inherit}
  .footer{color:#b8c3d6; font-size:14px}
  a{color:var(--accent); text-decoration:none}
  a:hover{text-decoration:underline}
  .ok{color:var(--accent2)}
</style>
</head>
<body>
  <div class="wrap">
    <header>
      <div>
        <h1 class="title">Churn Prediction – Telco Customer Dataset</h1>
        <p class="subtitle">Pipeline ML de bout en bout : préparation, entraînement, évaluation, interprétation, seuil optimal, Docker & Compose.</p>
        <div class="badges">
          <span class="badge">Python 3.12</span>
          <span class="badge">scikit-learn · XGBoost · LightGBM</span>
          <span class="badge">JupyterLab</span>
          <span class="badge">Docker · Compose</span>
          <span class="badge">WSL2</span>
        </div>
      </div>
    </header>

    <section>
      <h2>Présentation du projet</h2>
      <p>
        Projet académique (Master Machine Learning for Data Science – Université Paris Descartes) visant à prédire la probabilité
        de départ client (<em>churn</em>) pour un opérateur télécom à partir du dataset public Telco Customer Churn (Kaggle).
      </p>
      <p>Objectifs : pipeline modulaire et reproductible couvrant préparation des données, modélisation multi-algorithmes, évaluation, interprétation et exécution conteneurisée.</p>
    </section>

    <section>
      <h2>Objectifs</h2>
      <div class="grid">
        <div>
          <ul>
            <li>Nettoyage et préparation : imputation <code>TotalCharges</code>, encodage catégoriels, normalisation numériques.</li>
            <li>Modèles entraînés : Logistic Regression, Random Forest, XGBoost, LightGBM.</li>
            <li>Comparaison via Accuracy, AUC, Precision, Recall, F1.</li>
          </ul>
        </div>
        <div>
          <ul>
            <li>Optimisation du seuil de décision (F1 optimal).</li>
            <li>Interprétation du modèle logistique (coefficients, features clés).</li>
            <li>Conteneurisation Docker + orchestration avec Docker Compose.</li>
          </ul>
        </div>
      </div>
    </section>

    <section>
      <h2>Structure du projet</h2>
      <pre class="tree"><code>.
├── README.md
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
├── .dockerignore
├── data/
│   ├── WA_Fn-UseC_-Telco-Customer-Churn.csv
│   ├── WA_Fn-UseC_-Telco-Customer-Churn.pred.csv
│   └── telco-customer-churn.zip
├── models/
│   ├── logreg.joblib
│   ├── rf.joblib
│   ├── xgb.joblib
│   ├── lgbm.joblib
│   └── threshold.txt
├── notebooks/
│   └── eda_churn.ipynb
├── reports/
│   ├── metrics.md
│   ├── confusion_logreg.csv
│   ├── confusion_rf.csv
│   ├── confusion_xgb.csv
│   ├── confusion_lgbm.csv
│   ├── logreg_top_positive_features.csv
│   └── logreg_top_negative_features.csv
└── src/
    ├── train_models.py
    ├── evaluate_models.py
    ├── threshold_tuning.py
    ├── explain_logreg.py
    └── predict_csv.py
</code></pre>
      <p class="note">Remarque : les dossiers <span class="pill">models</span> et <span class="pill">reports</span> sont générés/actualisés par les scripts et via Docker/Compose.</p>
    </section>

    <section>
      <h2>Prérequis</h2>
      <ul>
        <li>Python 3.12, Docker et Docker Compose, Linux/WSL2.</li>
      </ul>
      <pre><code>python3 -m venv env
source env/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
</code></pre>
    </section>

    <section>
      <h2>Étapes (exécution locale)</h2>

      <h3>1) Entraînement</h3>
      <pre><code>python src/train_models.py
</code></pre>
      <p class="note">Sorties attendues : modèles <code>.joblib</code> dans <code>models/</code>, affichage Accuracy & AUC.</p>

      <h3>2) Évaluation & rapports</h3>
      <pre><code>python src/evaluate_models.py
</code></pre>
      <p class="note">Génère <code>reports/metrics.md</code> + <code>reports/confusion_*.csv</code>.</p>

      <h3>3) Seuil de décision (F1 optimal)</h3>
      <pre><code>python src/threshold_tuning.py
</code></pre>
      <p class="note">Écrit le seuil recommandé dans <code>models/threshold.txt</code>.</p>

      <h3>4) Interprétation (Logistic Regression)</h3>
      <pre><code>python src/explain_logreg.py
</code></pre>
      <p class="note">Exporte les features pro/anti-churn dans <code>reports/logreg_top_*.csv</code>.</p>

      <h3>5) Inférence sur CSV</h3>
      <pre><code>python src/predict_csv.py data/WA_Fn-UseC_-Telco-Customer-Churn.csv
</code></pre>
      <p class="note">Produit <code>data/WA_Fn-UseC_-Telco-Customer-Churn.pred.csv</code> avec <code>churn_proba</code> et <code>churn_pred</code>.</p>
    </section>

    <section>
      <h2>Exécution via Docker</h2>

      <h3>Image unique</h3>
      <pre><code>docker build -t churn-prediction:latest .
docker run --rm \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/models:/app/models \
  -v $(pwd)/reports:/app/reports \
  churn-prediction:latest
</code></pre>

      <h3>Orchestration avec Docker Compose</h3>
      <pre><code>docker compose build
docker compose run --rm train
docker compose run --rm eval
docker compose run --rm threshold
docker compose run --rm predict
</code></pre>
      <p class="note">Les volumes montés garantissent la persistance dans <code>data/</code>, <code>models/</code>, <code>reports/</code>.</p>
    </section>

    <section>
      <h2>Résultats (exemple)</h2>
      <table>
        <thead>
          <tr><th>Modèle</th><th>Accuracy</th><th>AUC</th></tr>
        </thead>
        <tbody>
          <tr><td>Logistic Regression</td><td>0.808</td><td>0.846</td></tr>
          <tr><td>Random Forest</td><td>0.784</td><td>0.823</td></tr>
          <tr><td>XGBoost</td><td>0.789</td><td>0.829</td></tr>
          <tr><td>LightGBM</td><td>0.782</td><td>0.822</td></tr>
        </tbody>
      </table>

      <h3>Insights principaux</h3>
      <ul>
        <li>Contrat <strong>Month-to-month</strong> fortement associé au churn.</li>
        <li>Absence de <strong>Tech Support</strong> ou <strong>Online Security</strong> corrélée au départ.</li>
        <li>Paiement <strong>Electronic Check</strong> plus à risque.</li>
        <li><strong>Tenure</strong> élevé diminue la probabilité de churn.</li>
      </ul>
    </section>

    <section>
      <h2>Améliorations possibles</h2>
      <ul>
        <li>Optimisation hyperparamètres (GridSearchCV / Optuna).</li>
        <li>API FastAPI ou interface Streamlit pour la démo.</li>
        <li>Monitoring/observabilité (Compose + Grafana/Prometheus).</li>
        <li>CI/CD (GitHub Actions) et packaging reproductible.</li>
      </ul>
    </section>

    <section class="footer">
      <h2>Auteur</h2>
      <p><strong>Adem Bounaidja-Rachedi</strong><br/>
         Master 1 – Machine Learning for Data Science, Université Paris Descartes<br/>
         GitHub&nbsp;: <a href="https://github.com/ademoments" target="_blank" rel="noopener">ademoments</a></p>
    </section>
  </div>
</body>
</html>
