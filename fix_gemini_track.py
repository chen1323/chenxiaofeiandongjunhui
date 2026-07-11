import os
import json
import pandas as pd
import numpy as np
from pypdf import PdfReader
import google.generativeai as genai
from dotenv import load_dotenv
import re
import time
from sklearn.metrics import roc_auc_score, f1_score
import logging

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

def calculate_all_metrics(y_true, y_score, track_name):
    print(f"\n--- Metrics for {track_name} ---")
    k = int(y_true.sum())
    if k == 0: return
        
    ndcg = ndcg_at_k(y_true, y_score, k)
    prec_k = precision_at_k(y_true, y_score, k)
    
    try:
        auc = roc_auc_score(y_true, y_score)
    except:
        auc = 0.0
        
    mean_dv = y_true.mean()
    lift = (recall_at_k(y_true, y_score, k) / mean_dv) * 100 if mean_dv > 0 else 0.0
    
    y_pred_bin = np.zeros(len(y_true))
    top_indices = np.argsort(y_score)[-k:]
    y_pred_bin[top_indices] = 1
    f1 = f1_score(y_true, y_pred_bin)
    
    print(f"Precision@{k}: {prec_k * 100:.2f}%")
    print(f"NDCG@{k}: {ndcg * 100:.2f}%")
    print(f"LIFT: {lift:.2f}%")
    print(f"ROC-AUC: {auc * 100:.2f}%")
    print(f"F1-Score: {f1 * 100:.2f}%")

def get_jmp_text(folder_path, char_limit):
    import glob
    pdfs = glob.glob(os.path.join(folder_path, '*.pdf'))
    cv_pdf_list = [f for f in pdfs if 'cv' in os.path.basename(f).lower()]
    other_pdfs = [f for f in pdfs if f not in cv_pdf_list]
    jmp_pdf = other_pdfs[0] if other_pdfs else pdfs[0]
    
    reader = PdfReader(jmp_pdf)
    text = ""
    for page in reader.pages:
        text += page.extract_text() or ""
        if len(text) > char_limit * 1.2:
            break
            
    return text[:char_limit]

def scrub_text(text, candidate_name):
    name_parts = candidate_name.split()
    for part in name_parts:
        if len(part) > 2:
            pattern = re.compile(re.escape(part), re.IGNORECASE)
            text = pattern.sub("[ANONYMIZED_CANDIDATE]", text)
    return text

def fix_gemini_track():
    load_dotenv()
    genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
    gemini_model = genai.GenerativeModel('gemini-flash-latest')
    
    df = pd.read_csv('dual_track_predictions.csv')
    
    system_prompt_gemini = """You are a strict accounting faculty recruiter in the year 2018. 
Based ONLY on the provided JMP Introduction, predict if the author will be in the top 5% of researchers.
Do NOT use any knowledge of events after 2018.
ANTI-BIAS WARNING: Explicitly discount school prestige and focus purely on empirical contribution, methodology, and research question originality.
Return strictly JSON format:
{
    "reasoning": "Chain of thought...",
    "prediction": 1 or 0,
    "confidence_score": 0.0 to 1.0
}"""

    for idx, row in df.iterrows():
        # Only process if it failed previously (we can assume 0.0 for all of them except the first few to just be safe)
        if idx < 5: continue # Already did these
        
        print(f"Processing {idx}: {row['candidate_name']}")
        base_dir = f"2015-2018_rookie_raw_cv_and_jmp/2018/{row['folder_name']}"
        raw_text = get_jmp_text(base_dir, row['jmp_intro_char_limit'])
        scrubbed_text = scrub_text(raw_text, row['candidate_name'])
        
        success = False
        for attempt in range(4):
            try:
                response = gemini_model.generate_content(f"{system_prompt_gemini}\n\nJMP Text:\n{scrubbed_text}")
                text_b = response.text.strip()
                if text_b.startswith("```json"): text_b = text_b[7:-3].strip()
                elif text_b.startswith("```"): text_b = text_b[3:-3].strip()
                
                res_json = json.loads(text_b)
                pred_b = float(res_json.get('prediction', 0) * res_json.get('confidence_score', 0.5))
                df.at[idx, 'gemini_pred'] = pred_b
                success = True
                time.sleep(4)
                break
            except Exception as e:
                if '429' in str(e):
                    print("Rate limit hit. Waiting 60s...")
                    time.sleep(60)
                else:
                    print(f"Other error: {e}")
                    df.at[idx, 'gemini_pred'] = 0.0
                    break
                    
        if not success:
            df.at[idx, 'gemini_pred'] = 0.0
            
        if idx % 10 == 0:
            df.to_csv('dual_track_predictions_fixed.csv', index=False)

    df.to_csv('dual_track_predictions_fixed.csv', index=False)
    
    y_true = df['true_label_pub_w_top_5pct'].values
    track_a = df['gpt2_pred'].values
    track_b = df['gemini_pred'].values
    
    calculate_all_metrics(y_true, track_a, "Track A (Pre-2018 GPT-2)")
    calculate_all_metrics(y_true, track_b, "Track B (Modern Gemini Flash)")

if __name__ == "__main__":
    fix_gemini_track()
