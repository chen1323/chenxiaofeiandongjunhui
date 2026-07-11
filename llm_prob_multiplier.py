import os
import json
import pandas as pd
import numpy as np
from pypdf import PdfReader
import google.generativeai as genai
from dotenv import load_dotenv
import time
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

def local_semantic_multiplier(text):
    text_lower = text.lower()
    score = 0.5
    if any(w in text_lower for w in ['instrumental variable', 'difference-in-differences', 'regression discontinuity']):
        score += 0.3
    if any(w in text_lower for w in ['hand-collected', 'proprietary', 'novel']):
        score += 0.1
    if len(text_lower) > 5000:
        score += 0.1
    return min(1.0, max(0.5, score))

def run_strategy_a():
    print("Running Refactored Strategy A: Continuous Probability Multiplier...")
    load_dotenv()
    genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
    model = genai.GenerativeModel('gemini-1.5-pro-latest') # Switch to Pro to avoid Flash limit
    
    ipw_preds = pd.read_csv('vanilla_ipw_predictions_weighted_2018.csv')
    mapping_df = pd.read_csv('candidate_mapping.csv')
    mapping_df = pd.merge(mapping_df, ipw_preds[['Candidate_ID', 'Predicted_Probability', 'Actual_Ground_Truth']], 
                          left_on='csv_candidate_id', right_on='Candidate_ID', how='left')
    mapping_df.rename(columns={'Predicted_Probability': 'ipw_score', 'Actual_Ground_Truth': 'label'}, inplace=True)
    
    mapping_df = mapping_df.sort_values('ipw_score', ascending=False).reset_index(drop=True)
    k = 8
    
    mapping_df['hybrid_score'] = mapping_df['ipw_score']
    
    top_20 = mapping_df.head(20).copy()
    
    print("\nExecuting Semantic Confidence Multiplier on Top 20...")
    for idx, row in top_20.iterrows():
        csv_id = row['csv_candidate_id']
        base_dir = f"2015-2018_rookie_raw_cv_and_jmp/2018/{row['folder_name']}"
        raw_text = get_jmp_text(base_dir, row['jmp_intro_char_limit'])
        scrubbed = scrub_text(raw_text, row['candidate_name'])
        
        prompt = f"""You are an Objective Gatekeeper.
Review the following Anonymized Job Market Paper Introduction:
---
{scrubbed}
---
Provide a Semantic Confidence Multiplier from 0.5 to 1.0. A paper with flawless empirical grounding (e.g. strict quasi-experiments, IV) gets 1.0. A paper with descriptive or incremental flaws gets closer to 0.5.
Output strictly JSON: {{"W_sem": 0.8}}"""

        w_sem = 0.5
        for attempt in range(2):
            try:
                response = model.generate_content(prompt)
                res_text = response.text.strip()
                match = re.search(r'\{.*\}', res_text, re.DOTALL)
                if match:
                    res_json = json.loads(match.group(0))
                    w_sem = float(res_json.get("W_sem", 0.5))
                else:
                    raise ValueError("No JSON")
                break
            except Exception as e:
                time.sleep(5)
        
        # Fallback to local heuristic if API completely exhausted
        if w_sem == 0.5:
            w_sem = local_semantic_multiplier(scrubbed)
            
        print(f"  ID {csv_id} (Vanilla Rank {idx+1}) -> W_sem: {w_sem:.2f}")
        mapping_df.loc[idx, 'hybrid_score'] = row['ipw_score'] * w_sem
        
    y_test = mapping_df['label']
    y_score = mapping_df['hybrid_score']
    
    ndcg = ndcg_at_k(y_test, y_score, k)
    prec_k = precision_at_k(y_test, y_score, k)
    auc = roc_auc_score(y_test, y_score)
    lift = (recall_at_k(y_test, y_score, k) / y_test.mean()) * 100
    
    y_pred_bin = np.zeros(len(y_test))
    top_indices = np.argsort(y_score.values)[-k:]
    y_pred_bin[top_indices] = 1
    f1 = f1_score(y_test, y_pred_bin)
    
    print("\n==================================================")
    print(" STRATEGY A: PROBABILITY MULTIPLIER METRICS")
    print("==================================================")
    print(f"Precision@K: {prec_k * 100:.2f}%")
    print(f"NDCG@K: {ndcg * 100:.2f}%")
    print(f"LIFT: {lift:.2f}%")
    print(f"ROC-AUC: {auc * 100:.2f}%")
    print(f"F1-Score: {f1 * 100:.2f}%")

if __name__ == "__main__":
    run_strategy_a()
