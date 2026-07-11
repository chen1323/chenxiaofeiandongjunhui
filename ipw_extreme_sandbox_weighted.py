import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import make_pipeline
import warnings

warnings.filterwarnings('ignore')

def ndcg_at_k(y_true, y_score, k):
    temp = pd.DataFrame({'y_true': y_true, 'y_score': y_score})
    temp_pred = temp.sort_values('y_score', ascending=False).reset_index(drop=True)
    temp_pred['gain'] = 2 ** temp_pred['y_true'] - 1
    temp_pred['discount'] = np.log2(np.arange(len(temp_pred)) + 2)
    dcg_k = (temp_pred.loc[:k-1, 'gain'] / temp_pred.loc[:k-1, 'discount']).sum()
    
    temp_ideal = temp.sort_values('y_true', ascending=False).reset_index(drop=True)
    temp_ideal['gain'] = 2 ** temp_ideal['y_true'] - 1
    temp_ideal['discount'] = np.log2(np.arange(len(temp_ideal)) + 2)
    idcg_k = (temp_ideal.loc[:k-1, 'gain'] / temp_ideal.loc[:k-1, 'discount']).sum()
    
    return dcg_k / idcg_k if idcg_k > 0 else 0.0

def precision_at_k(y_true, y_score, k):
    temp = pd.DataFrame({'y_true': y_true, 'y_score': y_score})
    temp = temp.sort_values('y_score', ascending=False)
    TP = temp.iloc[:k]['y_true'].sum()
    return TP / k

def calculate_ipw(X, D, bound, seed, poly=False):
    if poly:
        ps_model = make_pipeline(PolynomialFeatures(degree=2, include_bias=False), StandardScaler(), LogisticRegression(random_state=seed, max_iter=1000))
    else:
        ps_model = LogisticRegression(random_state=seed, max_iter=1000)
    
    ps_model.fit(X, D)
    prop_scores = ps_model.predict_proba(X)[:, 1]
    
    p_marginal = D.mean()
    weights = np.where(D == 1, p_marginal / prop_scores, (1 - p_marginal) / (1 - prop_scores))
    
    p_low = np.percentile(weights, bound[0])
    p_high = np.percentile(weights, bound[1])
    weights = np.clip(weights, p_low, p_high)
    return weights

def get_data():
    data = pd.read_csv('20260509_2015-2018_rookie_dataset.csv', index_col=0)
    target = 'pub_w_top_5pct'
    
    train = data[data.year < 2018].copy()
    test = data[data.year == 2018].copy()
    
    X_trainC = train.loc[:, 'gender':'multi_language']
    X_testC = test.loc[:, 'gender':'multi_language']
    X_trainD = train.loc[:, 'Bachelor_top':'second_language_euro']
    X_testD = test.loc[:, 'Bachelor_top':'second_language_euro']
    X_trainE = train.loc[:, '0_dt':'255_dt']
    X_testE = test.loc[:, '0_dt':'255_dt']
    X_trainF = train[['Placerank']]
    
    X_ps_train = pd.concat([X_trainC, X_trainD, X_trainF], axis=1) # CDF
    
    return train, test, target, X_trainC, X_testC, X_trainD, X_testD, X_trainE, X_testE, X_ps_train

def run_method_1_poly_ps():
    print("\n--- Method 1: Non-linear Propensity Scores (Poly degree 2) ---")
    train, test, target, X_trainC, X_testC, X_trainD, X_testD, X_trainE, X_testE, X_ps_train = get_data()
    seed = 21
    bound = (1, 99)
    C_params = np.logspace(-4, 4, 10)
    
    test_preds = pd.DataFrame(index=test.index)
    
    for set_name, (X_tr, X_te) in zip(['C', 'D', 'E'], [(X_trainC, X_testC), (X_trainD, X_testD), (X_trainE, X_testE)]):
        y_tr = train[target]
        D_tr = train['research_oriented']
        X_tr_80, X_val, y_tr_80, y_val = train_test_split(X_tr, y_tr, test_size=0.2, random_state=seed)
        
        X_ps_80 = X_ps_train.loc[X_tr_80.index]
        D_80 = D_tr.loc[X_tr_80.index]
        weights_80 = calculate_ipw(X_ps_80, D_80, bound, seed, poly=True)
        
        best_c = None
        best_auc = -1
        for c in C_params:
            model = LogisticRegression(C=c, random_state=seed, max_iter=1000)
            model.fit(X_tr_80, y_tr_80, sample_weight=weights_80)
            y_val_score = model.predict_proba(X_val)[:, 1]
            auc = roc_auc_score(y_val, y_val_score)
            if auc > best_auc:
                best_auc = auc; best_c = c
                
        weights_100 = calculate_ipw(X_ps_train, D_tr, bound, seed, poly=True)
        final_model = LogisticRegression(C=best_c, random_state=seed, max_iter=1000)
        final_model.fit(X_tr, y_tr, sample_weight=weights_100)
        test_preds[set_name] = final_model.predict_proba(X_te)[:, 1]
        
    cde_score = test_preds[['C', 'D', 'E']].mean(axis=1)
    y_test = test[target]
    k = int(y_test.sum())
    print(f"Precision@K: {precision_at_k(y_test, cde_score, k)*100:.2f}%")
    print(f"NDCG@K: {ndcg_at_k(y_test, cde_score, k)*100:.2f}%")

def run_method_2_elastic_net():
    print("\n--- Method 2: Elastic Net Regularization (saga solver) ---")
    train, test, target, X_trainC, X_testC, X_trainD, X_testD, X_trainE, X_testE, X_ps_train = get_data()
    seed = 21
    bound = (1, 99)
    C_params = np.logspace(-4, 4, 5)
    l1_ratios = [0.1, 0.5, 0.9]
    
    test_preds = pd.DataFrame(index=test.index)
    scaler = StandardScaler()
    
    for set_name, (X_tr, X_te) in zip(['C', 'D', 'E'], [(X_trainC, X_testC), (X_trainD, X_testD), (X_trainE, X_testE)]):
        y_tr = train[target]
        D_tr = train['research_oriented']
        X_tr_80, X_val, y_tr_80, y_val = train_test_split(X_tr, y_tr, test_size=0.2, random_state=seed)
        
        X_tr_80_s = scaler.fit_transform(X_tr_80)
        X_val_s = scaler.transform(X_val)
        X_tr_s = scaler.fit_transform(X_tr)
        X_te_s = scaler.transform(X_te)
        
        X_ps_80 = X_ps_train.loc[X_tr_80.index]
        D_80 = D_tr.loc[X_tr_80.index]
        weights_80 = calculate_ipw(X_ps_80, D_80, bound, seed)
        
        best_c = None; best_l1 = None; best_auc = -1
        for c in C_params:
            for l1 in l1_ratios:
                model = LogisticRegression(penalty='elasticnet', l1_ratio=l1, solver='saga', C=c, random_state=seed, max_iter=500)
                model.fit(X_tr_80_s, y_tr_80, sample_weight=weights_80)
                y_val_score = model.predict_proba(X_val_s)[:, 1]
                auc = roc_auc_score(y_val, y_val_score)
                if auc > best_auc:
                    best_auc = auc; best_c = c; best_l1 = l1
                    
        weights_100 = calculate_ipw(X_ps_train, D_tr, bound, seed)
        final_model = LogisticRegression(penalty='elasticnet', l1_ratio=best_l1, solver='saga', C=best_c, random_state=seed, max_iter=500)
        final_model.fit(X_tr_s, y_tr, sample_weight=weights_100)
        test_preds[set_name] = final_model.predict_proba(X_te_s)[:, 1]
        
    cde_score = test_preds[['C', 'D', 'E']].mean(axis=1)
    y_test = test[target]
    k = int(y_test.sum())
    print(f"Precision@K: {precision_at_k(y_test, cde_score, k)*100:.2f}%")
    print(f"NDCG@K: {ndcg_at_k(y_test, cde_score, k)*100:.2f}%")

def run_method_3_meta_fusion():
    print("\n--- Method 3: Trained Weighted Late Fusion ---")
    train, test, target, X_trainC, X_testC, X_trainD, X_testD, X_trainE, X_testE, X_ps_train = get_data()
    seed = 21
    bound = (1, 99)
    C_params = np.logspace(-4, 4, 10)
    
    test_preds = pd.DataFrame(index=test.index)
    train_oof = pd.DataFrame(index=train.index)
    
    kf = KFold(n_splits=5, shuffle=True, random_state=seed)
    
    for set_name, (X_tr, X_te) in zip(['C', 'D', 'E'], [(X_trainC, X_testC), (X_trainD, X_testD), (X_trainE, X_testE)]):
        y_tr = train[target]
        D_tr = train['research_oriented']
        
        oof_preds = np.zeros(len(train))
        for train_idx, val_idx in kf.split(X_tr):
            X_tr_fold, X_val_fold = X_tr.iloc[train_idx], X_tr.iloc[val_idx]
            y_tr_fold, y_val_fold = y_tr.iloc[train_idx], y_tr.iloc[val_idx]
            
            X_ps_fold = X_ps_train.iloc[train_idx]
            D_fold = D_tr.iloc[train_idx]
            weights_fold = calculate_ipw(X_ps_fold, D_fold, bound, seed)
            
            # Simplified grid search inside fold
            best_c = 1.0 # fallback
            model = LogisticRegression(C=best_c, random_state=seed, max_iter=1000)
            model.fit(X_tr_fold, y_tr_fold, sample_weight=weights_fold)
            oof_preds[val_idx] = model.predict_proba(X_val_fold)[:, 1]
            
        train_oof[set_name] = oof_preds
        
        # Test predictions
        weights_100 = calculate_ipw(X_ps_train, D_tr, bound, seed)
        final_model = LogisticRegression(C=1.0, random_state=seed, max_iter=1000)
        final_model.fit(X_tr, y_tr, sample_weight=weights_100)
        test_preds[set_name] = final_model.predict_proba(X_te)[:, 1]
        
    meta_model = LogisticRegression(random_state=seed)
    meta_model.fit(train_oof[['C', 'D', 'E']], train[target])
    cde_score = meta_model.predict_proba(test_preds[['C', 'D', 'E']])[:, 1]
    
    y_test = test[target]
    k = int(y_test.sum())
    print(f"Precision@K: {precision_at_k(y_test, cde_score, k)*100:.2f}%")
    print(f"NDCG@K: {ndcg_at_k(y_test, cde_score, k)*100:.2f}%")

if __name__ == '__main__':
    run_method_1_poly_ps()
    run_method_2_elastic_net()
    run_method_3_meta_fusion()
