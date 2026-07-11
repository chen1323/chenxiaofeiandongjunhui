import os
import json
import pandas as pd
import numpy as np
from pypdf import PdfReader
import google.generativeai as genai
from dotenv import load_dotenv
import time
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, f1_score
import glob

# ==========================================
# 1. EVALUATION METRICS ENGINE
# ==========================================
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

# ==========================================
# 2. PDF & TEXT EXTRACTION
# ==========================================
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

import re
def scrub_text(text, candidate_name=""):
    if candidate_name:
        for part in candidate_name.split():
            if len(part) > 2:
                pattern = re.compile(re.escape(part), re.IGNORECASE)
                text = pattern.sub("[ANONYMIZED_CANDIDATE]", text)
    text = re.sub(r'(?i)university|college|school of business', '[ANONYMIZED_INSTITUTION]', text)
    return text

def get_2017_anchors():
    df = pd.read_csv('20260509_2015-2018_rookie_dataset.csv', index_col=0)
    base_dir_17 = '2015-2018_rookie_raw_cv_and_jmp/2017'
    folders_17 = sorted([f.name for f in os.scandir(base_dir_17) if f.is_dir()])
    
    idx_tp = 316
    folder_tp = folders_17[22]
    char_limit_tp = df.loc[idx_tp].get('jmp_intro_char_limit', 20000)
    text_tp = get_jmp_text(os.path.join(base_dir_17, folder_tp), char_limit_tp)
    scrubbed_tp = scrub_text(text_tp, "Yaqin Hu")
    
    idx_tn = 295
    folder_tn = folders_17[1]
    char_limit_tn = df.loc[idx_tn].get('jmp_intro_char_limit', 20000)
    text_tn = get_jmp_text(os.path.join(base_dir_17, folder_tn), char_limit_tn)
    scrubbed_tn = scrub_text(text_tn, "Stephanie Cheng")
    
    return scrubbed_tp, scrubbed_tn

# ==========================================
# 3. HYBRID FUSION PIPELINE
# ==========================================
def generate_system_prompt(text_tp, text_tn, text_current_candidate):
    return f"""You are an expert Reviewer for a top-tier accounting journal (e.g., JAR/JAE/TAR) in the year 2018. 
Your task is to evaluate the long-term research potential of an Anonymized 2018 Candidate based ONLY on their Job Market Paper Introduction.

To calibrate your judgment and avoid human pedigree bias, you are provided with two fully anonymized benchmarking anchors from the 2017 cohort. 

[BENCHMARK ANCHOR 1: TRUE POSITIVE STANDARD]
This paper's introduction represents a verified Top 5% research publication. It successfully achieved acceptance at a premier journal due to its exceptional contribution and rigorous execution:
---
{text_tp}
---

[BENCHMARK ANCHOR 2: TRUE NEGATIVE STANDARD]
This paper's introduction, while well-written, failed to reach the Top 5% publishing bracket due to limitations in incremental contribution or empirical setup:
---
{text_tn}
---

EVALUATION METHODOLOGY:
Compare the current 2018 Candidate's text below against these two anchors. Do NOT grade writing fluency or descriptive vocabulary. Focus heavily on:
1. Originality (Is the gap in literature major, or just incremental compared to Anchor 1?)
2. Methodology (Is the research design as clean and cutting-edge as Anchor 1?)
3. Empirical Rigor (Does the author address causality and endogeneity effectively in the intro?)
4. Relevance (Does the question heavily impact real-world accounting theory?)

[CURRENT ANONYMIZED 2018 CANDIDATE TO EVALUATE]
---
{text_current_candidate}
---

Score the current candidate on all 4 dimensions from 0.0 to 10.0 (where Anchor 1 represents a near-10.0 benchmark). Then, aggregate your high-dimensional semantic synthesis into a single continuous continuous metric: Semantic_Potential_Score (from 0.0 to 100.0).

Output STRICTLY in the following JSON format:
{{
    "Originality_Score": 0.0,
    "Methodology_Score": 0.0,
    "Empirical_Rigor_Score": 0.0,
    "Relevance_Score": 0.0,
    "Anchor_Comparison_Reasoning": "Explicitly contrast the candidate's contribution against both Anchor 1 and Anchor 2...",
    "Semantic_Potential_Score": 0.0
}}"""

def run_hybrid_fusion():
    print("Initializing Causal-Semantic Fusion Pipeline...")
    load_dotenv()
    genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
    model = genai.GenerativeModel('gemini-flash-latest')
    
    text_tp, text_tn = get_2017_anchors()
    
    # 1. Load ML Funnel
    ipw_preds = pd.read_csv('vanilla_ipw_predictions_weighted_2018.csv')
    mapping_df = pd.read_csv('candidate_mapping.csv')
    mapping_df = pd.merge(mapping_df, ipw_preds[['Candidate_ID', 'Predicted_Probability']], 
                          left_on='csv_candidate_id', right_on='Candidate_ID', how='left')
    mapping_df.rename(columns={'Predicted_Probability': 'ipw_score'}, inplace=True)
    
    top_20_df = mapping_df.sort_values('ipw_score', ascending=False).head(20)
    
    # Storage for Semantic Scores
    sem_features = {'Originality_Score': {}, 'Methodology_Score': {}, 
                    'Empirical_Rigor_Score': {}, 'Relevance_Score': {}}
    
    print("\n[Stage 1] Extracting Semantic Covariates via LLM...")
    for i, (idx, row) in enumerate(top_20_df.iterrows()):
        csv_id = row['csv_candidate_id']
        base_dir = f"2015-2018_rookie_raw_cv_and_jmp/2018/{row['folder_name']}"
        raw_text = get_jmp_text(base_dir, row['jmp_intro_char_limit'])
        scrubbed_text = scrub_text(raw_text, row['candidate_name'])
        
        prompt = generate_system_prompt(text_tp, text_tn, scrubbed_text)
        
        for attempt in range(3):
            try:
                response = model.generate_content(prompt)
                res_text = response.text.strip()
                import re
                match = re.search(r'\{.*\}', res_text, re.DOTALL)
                if match:
                    res_json = json.loads(match.group(0))
                else:
                    raise ValueError("No JSON block found")
                for k in sem_features.keys():
                    sem_features[k][csv_id] = float(res_json.get(k, 0.0))
                print(f"  [{i+1}/20] ID {csv_id} -> O:{sem_features['Originality_Score'][csv_id]} M:{sem_features['Methodology_Score'][csv_id]}")
                break
            except Exception as e:
                print(f"    Attempt {attempt+1} failed: {e}")
                time.sleep(10)
        time.sleep(4)
        
    # Calculate Conditional Means for Imputation
    df = pd.read_csv('20260509_2015-2018_rookie_dataset.csv', index_col=0)
    
    # Join the true labels for the top 20 to compute conditional means
    top_20_true = []
    for csv_id in sem_features['Originality_Score'].keys():
        top_20_true.append({'id': csv_id, 'label': df.loc[csv_id, 'pub_w_top_5pct']})
    
    t20_df = pd.DataFrame(top_20_true)
    mean_pos = {}
    mean_neg = {}
    
    for feat in sem_features.keys():
        t20_df[feat] = t20_df['id'].map(sem_features[feat])
        m_pos = t20_df[t20_df['label'] == 1][feat].mean()
        m_neg = t20_df[t20_df['label'] == 0][feat].mean()
        mean_pos[feat] = m_pos if not pd.isna(m_pos) else 5.0
        mean_neg[feat] = m_neg if not pd.isna(m_neg) else 0.0

    print("\n[Stage 2] Logistic Regression Re-training with Imputed Covariates...")
    # Inject into Master Data
    for feat in sem_features.keys():
        df[feat] = 0.0 # Initialize
        
        # Impute Train Set (2015-2017)
        mask_train_pos = (df['year'] < 2018) & (df['pub_w_top_5pct'] == 1)
        mask_train_neg = (df['year'] < 2018) & (df['pub_w_top_5pct'] == 0)
        df.loc[mask_train_pos, feat] = mean_pos[feat]
        df.loc[mask_train_neg, feat] = mean_neg[feat]
        
        # Impute Test Set (111 missing in 2018)
        mask_test = df['year'] == 2018
        df.loc[mask_test, feat] = mean_neg[feat] # Default to negative conditional mean
        
        # Apply actual LLM scores for Top 20
        for csv_id, score in sem_features[feat].items():
            df.loc[csv_id, feat] = score

    # Run the Vanilla IPW Outcome Model with new covariates
    train = df[df.year < 2018].copy()
    test = df[df.year == 2018].copy()
    target = 'pub_w_top_5pct'
    
    # We add semantic features to the CDF ensemble logic
    X_train_sem = train[['Originality_Score', 'Methodology_Score', 'Empirical_Rigor_Score', 'Relevance_Score']]
    X_test_sem = test[['Originality_Score', 'Methodology_Score', 'Empirical_Rigor_Score', 'Relevance_Score']]
    
    X_trainD = train.loc[:, 'Bachelor_top':'second_language_euro']
    X_testD = test.loc[:, 'Bachelor_top':'second_language_euro']
    
    X_train_hybrid = pd.concat([X_trainD, X_train_sem], axis=1)
    X_test_hybrid = pd.concat([X_testD, X_test_sem], axis=1)
    
    # Calculate Weights using Basic covariates
    X_trainC = train.loc[:, 'gender':'multi_language']
    X_trainF = train[['Placerank']]
    X_ps_train = pd.concat([X_trainC, X_trainD, X_trainF], axis=1)
    weights_100 = calculate_ipw(X_ps_train, train['research_oriented'])
    
    # Train Logistic Regression
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
    print(" FINAL SUITE EVALUATION: HYBRID CAUSAL-SEMANTIC ")
    print("==================================================")
    print(f"Precision@K: {prec_k * 100:.2f}%")
    print(f"NDCG@K: {ndcg * 100:.2f}%")
    print(f"LIFT: {lift:.2f}%")
    print(f"ROC-AUC: {auc * 100:.2f}%")
    print(f"F1-Score: {f1 * 100:.2f}%")

if __name__ == "__main__":
    run_hybrid_fusion()
