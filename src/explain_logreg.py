import numpy as np, pandas as pd, joblib
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

pipe = joblib.load("models/logreg.joblib")
pre: ColumnTransformer = pipe.named_steps["pre"]
clf = pipe.named_steps["clf"]

num_names = pre.transformers_[0][2]
ohe: OneHotEncoder = pre.transformers_[1][1]
cat_raw = pre.transformers_[1][2]
cat_names = ohe.get_feature_names_out(cat_raw)
feat_names = np.concatenate([num_names, cat_names])

coefs = pd.Series(clf.coef_.ravel(), index=feat_names).sort_values()
coefs.tail(15).to_csv("reports/logreg_top_positive_features.csv")  # pro-churn
coefs.head(15).to_csv("reports/logreg_top_negative_features.csv")  # anti-churn
print("[OK] reports/logreg_top_* générés")
