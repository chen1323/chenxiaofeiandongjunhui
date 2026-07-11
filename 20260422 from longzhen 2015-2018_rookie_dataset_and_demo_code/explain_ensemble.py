import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, f1_score
from imblearn.over_sampling import SMOTE
import shap
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# 1. Load Data
data = pd.read_csv('2015-2018_rookie_dataset.csv', index_col=0)
train = data[data.year < 2018]
test = data[data.year == 2018]

# 2. Define Features for C, D, E
X_trainC = train.loc[:, 'gender':'multi_language']
X_testC = test.loc[:, 'gender':'multi_language']

X_trainD = train.loc[:, 'Bachelor_top':'second_language_euro']
X_testD = test.loc[:, 'Bachelor_top':'second_language_euro']

# Use 3072 embeddings as in the robust run
X_trainE = train.loc[:, '0_dt':'3071_dt']
X_testE = test.loc[:, '0_dt':'3071_dt']

target = 'pub_top_5pct'
y_train0 = train[target]
y_test = test[target]

# Oversampling
smote = SMOTE(sampling_strategy=1, random_state=0)
X_trainC_sm, y_trainC_sm = smote.fit_resample(X_trainC, y_train0)
X_trainD_sm, y_trainD_sm = smote.fit_resample(X_trainD, y_train0)
X_trainE_sm, y_trainE_sm = smote.fit_resample(X_trainE, y_train0)

# 3. Train Models
modelC = LogisticRegression(C=0.1, max_iter=1000)
modelC.fit(X_trainC_sm, y_trainC_sm)

modelD = LogisticRegression(C=0.1, max_iter=1000)
modelD.fit(X_trainD_sm, y_trainD_sm)

modelE = LogisticRegression(C=0.1, max_iter=1000)
modelE.fit(X_trainE_sm, y_trainE_sm)

# 4. Predict and Evaluate CDE Ensemble
probC = modelC.predict_proba(X_testC)[:, 1]
probD = modelD.predict_proba(X_testD)[:, 1]
probE = modelE.predict_proba(X_testE)[:, 1]

# Average probabilities (Late-Fusion)
prob_CDE = (probC + probD + probE) / 3.0
pred_CDE = (prob_CDE >= 0.5).astype(int)

auc = roc_auc_score(y_test, prob_CDE)
f1 = f1_score(y_test, pred_CDE)

print("=== Ensemble CDE Evaluation ===")
print(f"Target: {target}")
print(f"ROC-AUC: {auc:.4f}")
print(f"F1-Score: {f1:.4f}")
print("===============================\n")

# 5. SHAP Interpretation
X_train_combined = pd.concat([X_trainC, X_trainD, X_trainE], axis=1)
X_test_combined = pd.concat([X_testC, X_testD, X_testE], axis=1)

def predict_ensemble(X_combined):
    # X_combined is a numpy array here
    X_C = X_combined[:, :X_trainC.shape[1]]
    X_D = X_combined[:, X_trainC.shape[1] : X_trainC.shape[1] + X_trainD.shape[1]]
    X_E = X_combined[:, X_trainC.shape[1] + X_trainD.shape[1]:]
    
    pC = modelC.predict_proba(X_C)[:, 1]
    pD = modelD.predict_proba(X_D)[:, 1]
    pE = modelE.predict_proba(X_E)[:, 1]
    return (pC + pD + pE) / 3.0

# K-means summary of training data (background)
background = shap.kmeans(X_train_combined, 10)
explainer = shap.KernelExplainer(predict_ensemble, background)

# Compute SHAP values for a subset of the test data (e.g., first 30 samples) to save time
print("Computing SHAP values... This may take a moment.")
shap_values = explainer.shap_values(X_test_combined.iloc[:30], nsamples=100)

plt.figure(figsize=(10, 6))
shap.summary_plot(shap_values, X_test_combined.iloc[:30], show=False)
plt.savefig("shap_summary_CDE_ensemble.png", bbox_inches='tight')
print("Saved SHAP summary plot to shap_summary_CDE_ensemble.png")
