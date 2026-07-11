import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
import warnings

warnings.filterwarnings('ignore')

def calculate_ipw(X, D, bound, seed):
    ps_model = LogisticRegression(random_state=seed, max_iter=1000)
    ps_model.fit(X, D)
    prop_scores = ps_model.predict_proba(X)[:, 1]
    
    p_marginal = D.mean()
    weights = np.where(D == 1, p_marginal / prop_scores, (1 - p_marginal) / (1 - prop_scores))
    
    p_low = np.percentile(weights, bound[0])
    p_high = np.percentile(weights, bound[1])
    weights = np.clip(weights, p_low, p_high)
    return weights

def run_optimized_pipeline(target, seed, ps_name):
    data = pd.read_csv('20260509_2015-2018_rookie_dataset.csv', index_col=0)
    
    train = data[data.year < 2018].copy()
    test = data[data.year == 2018].copy()
    
    X_trainC = train.loc[:, 'gender':'multi_language']
    X_testC = test.loc[:, 'gender':'multi_language']
    
    X_trainD = train.loc[:, 'Bachelor_top':'second_language_euro']
    X_testD = test.loc[:, 'Bachelor_top':'second_language_euro']
    
    X_trainE = train.loc[:, '0_dt':'255_dt']
    X_testE = test.loc[:, '0_dt':'255_dt']
    
    X_trainF = train[['Placerank']]
    
    if ps_name == 'CDF':
        X_ps_train = pd.concat([X_trainC, X_trainD, X_trainF], axis=1)
    else:
        X_ps_train = pd.concat([X_trainD, X_trainF], axis=1)
        
    bound = (1, 99)
    C_params = np.logspace(-4, 4, 10)
    
    test_preds = pd.DataFrame(index=test.index)
    
    for set_name, (X_tr, X_te) in zip(['C', 'D', 'E'], [(X_trainC, X_testC), (X_trainD, X_testD), (X_trainE, X_testE)]):
        y_tr = train[target]
        D_tr = train['research_oriented']
        
        X_tr_80, X_val, y_tr_80, y_val = train_test_split(X_tr, y_tr, test_size=0.2, random_state=seed)
        
        X_ps_80 = X_ps_train.loc[X_tr_80.index]
        D_80 = D_tr.loc[X_tr_80.index]
        weights_80 = calculate_ipw(X_ps_80, D_80, bound, seed)
        
        best_c = None
        best_auc = -1
        from sklearn.metrics import roc_auc_score
        for c in C_params:
            model = LogisticRegression(C=c, random_state=seed, max_iter=1000)
            model.fit(X_tr_80, y_tr_80, sample_weight=weights_80)
            y_val_score = model.predict_proba(X_val)[:, 1]
            auc = roc_auc_score(y_val, y_val_score)
            if auc > best_auc:
                best_auc = auc
                best_c = c
                
        weights_100 = calculate_ipw(X_ps_train, D_tr, bound, seed)
        final_model = LogisticRegression(C=best_c, random_state=seed, max_iter=1000)
        final_model.fit(X_tr, y_tr, sample_weight=weights_100)
        
        test_preds[set_name] = final_model.predict_proba(X_te)[:, 1]
        
    cde_score = test_preds[['C', 'D', 'E']].mean(axis=1)
    
    y_test = test[target]
    pred_binary = (cde_score >= 0.5).astype(int)
    f1 = f1_score(y_test, pred_binary)
    print(f"Target: {target}, F1-Score: {f1:.4f}")

if __name__ == '__main__':
    run_optimized_pipeline('pub_top_5pct', 1, 'DF')
    run_optimized_pipeline('pub_w_top_5pct', 21, 'CDF')
