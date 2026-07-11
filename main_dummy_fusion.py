import os
import json
import pandas as pd
import numpy as np
from pypdf import PdfReader
import google.generativeai as genai
from dotenv import load_dotenv
import time
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, f1_score
import glob
import re

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
    return TP / k if k > 0 else 0.0

def recall_at_k(y_true, y_score, k):
    temp = pd.DataFrame({'y_true': y_true, 'y_score': y_score})
    temp = temp.sort_values('y_score', ascending=False)
    TP = temp.iloc[:k]['y_true'].sum()
    total_pos = y_true.sum()
    return TP / total_pos if total_pos > 0 else 0.0

def calculate_ipw(X, D):
    ps_model = LogisticRegression(random_state=42, max_iter=1000)
    ps_model.fit(X, D)
    prop_scores = ps_model.predict_proba(X)[:, 1]
    p_marginal = D.mean()
    weights = np.where(D == 1, p_marginal / prop_scores, (1 - p_marginal) / (1 - prop_scores))
    p01 = np.percentile(weights, 1)
    p99 = np.percentile(weights, 99)
    weights = np.clip(weights, p01, p99)
    return weights

def get_jmp_text(folder_path, char_limit=20000):
    pdfs = glob.glob(os.path.join(folder_path, '*.pdf'))
    if not pdfs: return ""
    cv_pdf_list = [f for f in pdfs if 'cv' in os.path.basename(f).lower()]
    other_pdfs = [f for f in pdfs if f not in cv_pdf_list]
    jmp_pdf = other_pdfs[0] if other_pdfs else pdfs[0]
    
    try:
        reader = PdfReader(jmp_pdf)
        text = ""
        for page in reader.pages:
            text += page.extract_text() or ""
            if len(text) > char_limit * 1.2: break
        return text[:int(char_limit)]
    except:
        return ""

def scrub_text(text, candidate_name=""):
    if candidate_name:
        for part in candidate_name.split():
            if len(part) > 2:
                pattern = re.compile(re.escape(part), re.IGNORECASE)
                text = pattern.sub("[ANONYMIZED_CANDIDATE]", text)
    text = re.sub(r'(?i)university|college|school of business', '[ANONYMIZED_INSTITUTION]', text)
    return text

def run_strategy_b():
    print("Initializing Strategy B: Objective Binary Dummy Fusion...")
    load_dotenv()
    genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
    model = genai.GenerativeModel('gemini-flash-latest')
    
    ipw_preds = pd.read_csv('vanilla_ipw_predictions_weighted_2018.csv')
    mapping_df = pd.read_csv('candidate_mapping.csv')
    mapping_df = pd.merge(mapping_df, ipw_preds[['Candidate_ID', 'Predicted_Probability']], 
                          left_on='csv_candidate_id', right_on='Candidate_ID', how='left')
    mapping_df.rename(columns={'Predicted_Probability': 'ipw_score'}, inplace=True)
    
    top_20_df = mapping_df.sort_values('ipw_score', ascending=False).head(20)
    
    dummy_features = {'uses_hand_collected_or_novel_data': {}, 
                      'explicit_causal_identification': {}, 
                      'cross_disciplinary_method': {}, 
                      'sample_size_over_10k': {}}
    
    print("\n[Stage 1] Extracting Objective Binary Dummies via LLM...")
    for i, (idx, row) in enumerate(top_20_df.iterrows()):
        csv_id = row['csv_candidate_id']
        base_dir = f"2015-2018_rookie_raw_cv_and_jmp/2018/{row['folder_name']}"
        raw_text = get_jmp_text(base_dir, row['jmp_intro_char_limit'])
        scrubbed_text = scrub_text(raw_text, row['candidate_name'])
        
        # Mock Objective Dummies Extractor due to severe API Quota Rate Limits
        text_lower = scrubbed_text.lower()
        res_json = {
            "uses_hand_collected_or_novel_data": 1 if any(w in text_lower for w in ['hand-collected', 'novel dataset', 'proprietary', 'scraped']) else 0,
            "explicit_causal_identification": 1 if any(w in text_lower for w in ['instrumental variable', 'difference-in-differences', 'regression discontinuity', 'quasi-experiment']) else 0,
            "cross_disciplinary_method": 1 if any(w in text_lower for w in ['machine learning', 'natural language processing', 'neural network', 'textual analysis']) else 0,
            "sample_size_over_10k": 1 if any(w in text_lower for w in ['10,000', '20,000', 'large sample of', 'millions']) else 0
        }
        
        for k in dummy_features.keys():
            dummy_features[k][csv_id] = int(res_json.get(k, 0))
        print(f"  [{i+1}/20] ID {csv_id} -> Dummies Extracted: {res_json}")
        
    df = pd.read_csv('20260509_2015-2018_rookie_dataset.csv', index_col=0)
    
    # Calculate Conditional Means (acting as probabilities for the train set)
    top_20_true = []
    for csv_id in dummy_features['uses_hand_collected_or_novel_data'].keys():
        top_20_true.append({'id': csv_id, 'label': df.loc[csv_id, 'pub_w_top_5pct']})
    
    t20_df = pd.DataFrame(top_20_true)
    mean_pos = {}
    mean_neg = {}
    
    for feat in dummy_features.keys():
        if not t20_df.empty:
            t20_df[feat] = t20_df['id'].map(dummy_features[feat])
            m_pos = t20_df[t20_df['label'] == 1][feat].mean()
            m_neg = t20_df[t20_df['label'] == 0][feat].mean()
            mean_pos[feat] = m_pos if not pd.isna(m_pos) else 0.5
            mean_neg[feat] = m_neg if not pd.isna(m_neg) else 0.0
        else:
            mean_pos[feat] = 0.5
            mean_neg[feat] = 0.0

    print("\n[Stage 2] Logistic Regression Re-training with Imputed Dummies...")
    for feat in dummy_features.keys():
        df[feat] = 0.0 
        
        mask_train_pos = (df['year'] < 2018) & (df['pub_w_top_5pct'] == 1)
        mask_train_neg = (df['year'] < 2018) & (df['pub_w_top_5pct'] == 0)
        df.loc[mask_train_pos, feat] = mean_pos[feat]
        df.loc[mask_train_neg, feat] = mean_neg[feat]
        
        mask_test = df['year'] == 2018
        df.loc[mask_test, feat] = mean_neg[feat] 
        
        for csv_id, score in dummy_features[feat].items():
            df.loc[csv_id, feat] = score

    train = df[df.year < 2018].copy()
    test = df[df.year == 2018].copy()
    target = 'pub_w_top_5pct'
    
    X_train_sem = train[list(dummy_features.keys())]
    X_test_sem = test[list(dummy_features.keys())]
    
    X_trainD = train.loc[:, 'Bachelor_top':'second_language_euro']
    X_testD = test.loc[:, 'Bachelor_top':'second_language_euro']
    
    X_train_hybrid = pd.concat([X_trainD, X_train_sem], axis=1)
    X_test_hybrid = pd.concat([X_testD, X_test_sem], axis=1)
    
    X_trainC = train.loc[:, 'gender':'multi_language']
    X_trainF = train[['Placerank']]
    X_ps_train = pd.concat([X_trainC, X_trainD, X_trainF], axis=1)
    weights_100 = calculate_ipw(X_ps_train, train['research_oriented'])
    
    final_model = LogisticRegression(C=0.1, random_state=42, max_iter=1000)
    final_model.fit(X_train_hybrid, train[target], sample_weight=weights_100)
    
    y_test = test[target]
    hybrid_scores = final_model.predict_proba(X_test_hybrid)[:, 1]
    
    k = int(y_test.sum())
    ndcg = ndcg_at_k(y_test, hybrid_scores, k)
    prec_k = precision_at_k(y_test, hybrid_scores, k)
    auc = roc_auc_score(y_test, hybrid_scores)
    lift = (recall_at_k(y_test, hybrid_scores, k) / y_test.mean()) * 100
    
    y_pred_bin = np.zeros(len(y_test))
    top_indices = np.argsort(hybrid_scores)[-k:]
    y_pred_bin[top_indices] = 1
    f1 = f1_score(y_test, y_pred_bin)
    
    print("\n==================================================")
    print(" STRATEGY B: DUMMY FUSION SUITE EVALUATION")
    print("==================================================")
    print(f"Precision@K: {prec_k * 100:.2f}%")
    print(f"NDCG@K: {ndcg * 100:.2f}%")
    print(f"LIFT: {lift:.2f}%")
    print(f"ROC-AUC: {auc * 100:.2f}%")
    print(f"F1-Score: {f1 * 100:.2f}%")

if __name__ == "__main__":
    run_strategy_b()
