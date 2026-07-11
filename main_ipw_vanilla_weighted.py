import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from itertools import product

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

def recall_at_k(y_true, y_score, k):
    temp = pd.DataFrame({'y_true': y_true, 'y_score': y_score})
    temp = temp.sort_values('y_score', ascending=False)
    TP = temp.iloc[:k]['y_true'].sum()
    return TP / y_true.sum() if y_true.sum() > 0 else 0.0

def calculate_ipw(X, D):
    # Train PS model
    ps_model = LogisticRegression(random_state=42, max_iter=1000)
    ps_model.fit(X, D)
    prop_scores = ps_model.predict_proba(X)[:, 1]
    
    # Stabilized weights
    p_marginal = D.mean()
    weights = np.where(D == 1, p_marginal / prop_scores, (1 - p_marginal) / (1 - prop_scores))
    
    # 1st and 99th percentile capping
    p01 = np.percentile(weights, 1)
    p99 = np.percentile(weights, 99)
    weights = np.clip(weights, p01, p99)
    
    return weights

def run_vanilla_pipeline():
    print("Loading data...")
    data = pd.read_csv('20260509_2015-2018_rookie_dataset.csv', index_col=0)
    
    # Target
    target = 'pub_w_top_5pct'
    
    # Splits
    train = data[data.year < 2018].copy()
    test = data[data.year == 2018].copy()
    
    # Features
    X_trainC = train.loc[:, 'gender':'multi_language']
    X_testC = test.loc[:, 'gender':'multi_language']
    
    X_trainD = train.loc[:, 'Bachelor_top':'second_language_euro']
    X_testD = test.loc[:, 'Bachelor_top':'second_language_euro']
    
    X_trainE = train.loc[:, '0_dt':'255_dt']
    X_testE = test.loc[:, '0_dt':'255_dt']
    
    X_trainF = train[['Placerank']]
    
    # Basic covariates for PS model (C, D, F)
    X_ps_train = pd.concat([X_trainC, X_trainD, X_trainF], axis=1)
    
    feature_sets = {
        'C': (X_trainC, X_testC),
        'D': (X_trainD, X_testD),
        'E': (X_trainE, X_testE)
    }
    
    test_preds = pd.DataFrame(index=test.index)
    
    C_params = [0.001, 0.01, 0.1, 1]
    
    for set_name, (X_tr, X_te) in feature_sets.items():
        print(f"\nProcessing feature set {set_name}...")
        y_tr = train[target]
        D_tr = train['research_oriented']
        
        # Split train into 80/20 for grid search
        X_tr_80, X_val, y_tr_80, y_val = train_test_split(X_tr, y_tr, test_size=0.2, random_state=42)
        
        # Calculate IPW weights for 80% train
        X_ps_80 = X_ps_train.loc[X_tr_80.index]
        D_80 = D_tr.loc[X_tr_80.index]
        weights_80 = calculate_ipw(X_ps_80, D_80)
        
        # Grid Search for best C based on ROC-AUC
        best_c = None
        best_auc = -1
        for c in C_params:
            model = LogisticRegression(C=c, random_state=42, max_iter=1000)
            model.fit(X_tr_80, y_tr_80, sample_weight=weights_80)
            
            y_val_score = model.predict_proba(X_val)[:, 1]
            auc = roc_auc_score(y_val, y_val_score)
            if auc > best_auc:
                best_auc = auc
                best_c = c
                
        print(f"Best C for {set_name}: {best_c} (Validation AUC: {best_auc:.4f})")
        
        # Retrain on 100% of training data with best C
        weights_100 = calculate_ipw(X_ps_train, D_tr)
        final_model = LogisticRegression(C=best_c, random_state=42, max_iter=1000)
        final_model.fit(X_tr, y_tr, sample_weight=weights_100)
        
        # Predict on Test set
        test_preds[set_name] = final_model.predict_proba(X_te)[:, 1]
        
    print("\n--- Late Fusion (CDE Ensemble) ---")
    # Simple average of raw probabilities
    cde_score = test_preds[['C', 'D', 'E']].mean(axis=1)
    
    # Calculate metrics
    y_test = test[target]
    k = int(y_test.sum())
    
    ndcg = ndcg_at_k(y_test, cde_score, k)
    prec_k = precision_at_k(y_test, cde_score, k)
    auc = roc_auc_score(y_test, cde_score)
    lift = (recall_at_k(y_test, cde_score, k) / y_test.mean()) * 100
    
    print("\nFinal Metrics:")
    print(f"Precision@K: {prec_k * 100:.2f}%")
    print(f"NDCG@K: {ndcg * 100:.2f}%")
    print(f"LIFT: {lift:.2f}%")
    print(f"ROC-AUC: {auc * 100:.2f}%")
    
    # Export predictions
    out_df = pd.DataFrame({
        'Candidate_ID': test.index,
        'Actual_Ground_Truth': y_test.values,
        'Predicted_Probability': cde_score.values
    })
    
    out_df = out_df.sort_values('Predicted_Probability', ascending=False).reset_index(drop=True)
    out_df['Ranked_Position'] = out_df.index + 1
    
    out_df.to_csv('vanilla_ipw_predictions_weighted_2018.csv', index=False)
    print("Exported final predictions to vanilla_ipw_predictions_weighted_2018.csv")
    
if __name__ == '__main__':
    run_vanilla_pipeline()
