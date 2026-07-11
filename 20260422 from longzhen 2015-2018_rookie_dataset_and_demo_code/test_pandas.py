import pandas as pd
m_param = {'Logit_c': {'C': 0.001}}
metric = pd.DataFrame({'AUC': [0.9]}, index=[str(m_param)])
metric['model'] = 'Logit_c'
best = metric.groupby('model')['AUC'].idxmax()
print("best:")
print(best)
print("best.index[0]:", best.index[0])
print("best[0]:", best.iloc[0])
