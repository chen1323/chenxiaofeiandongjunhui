import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression

# Load original data
data = pd.read_csv('20260509_2015-2018_rookie_dataset.csv')
train = data[data.year < 2018].copy()
test = data[data.year == 2018].copy()

# Load predictions
preds = pd.read_csv('output_prediction_main_2018.csv')
preds = preds.merge(test[['ID', 'Placerank', 'research_oriented', 'Bachelor_top', 'gender']], on='ID', how='inner')

# Identify top K for pub_top_5pct (CDE ensemble)
target = 'pub_top_5pct'
score_col = f'CDE_{target}_score'
true_col = f'CDE_{target}'

k = int(preds[true_col].sum())
top_k_preds = preds.sort_values(score_col, ascending=False).head(k)

print(f"\n--- Top {k} Candidates (CDE Ensemble) ---")
for _, row in top_k_preds.iterrows():
    is_fp = row[true_col] == 0
    status = "FALSE POSITIVE" if is_fp else "TRUE POSITIVE"
    print(f"ID: {row['ID']} | Score: {row[score_col]:.4f} | True: {row[true_col]} -> {status}")
    print(f"   Features: Placerank={row['Placerank']}, Research_Oriented={row['research_oriented']}, Bachelor_top={row['Bachelor_top']}")

# Let's see what features are heavily weighted by the IPW Logit model for 'CDE'
# CDE corresponds to features C, D, E
# But the IPW is calculated on C, D, F
print("\n--- IPW Weights Investigation ---")
X_trainC = train.loc[:, 'gender':'multi_language']
X_trainD = train.loc[:, 'Bachelor_top':'second_language_euro']
X_trainF = train[['Placerank']]

X_ps_train = pd.concat([X_trainC, X_trainD, X_trainF], axis=1)
D = train['research_oriented']
model = LogisticRegression(random_state=42, max_iter=1000)
model.fit(X_ps_train, D)

# Display largest coefficients in PS model
coefs = pd.Series(model.coef_[0], index=X_ps_train.columns)
print("Top Positive Coefs for Research_Oriented:")
print(coefs.sort_values(ascending=False).head(5))
print("\nTop Negative Coefs for Research_Oriented:")
print(coefs.sort_values(ascending=True).head(5))

# Calculate training IPW weights
prop_scores = model.predict_proba(X_ps_train)[:, 1]
p_marginal = D.mean()
weights = np.where(D == 1, p_marginal / prop_scores, (1 - p_marginal) / (1 - prop_scores))
weights = np.clip(weights, 0, np.percentile(weights, 99))
train['ipw_weight'] = weights

# See which candidates in training had the highest weights
top_weights = train.sort_values('ipw_weight', ascending=False).head(5)
print("\n--- Training Candidates with Highest IPW Weights ---")
for _, row in top_weights.iterrows():
    print(f"ID: {row['ID']} | Weight: {row['ipw_weight']:.2f} | Research_Oriented: {row['research_oriented']} | Placerank: {row['Placerank']}")

