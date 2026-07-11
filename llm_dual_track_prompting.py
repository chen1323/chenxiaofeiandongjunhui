import os
import json
import pandas as pd
import numpy as np
from pypdf import PdfReader
from transformers import pipeline
import torch
import google.generativeai as genai
from dotenv import load_dotenv
import re
from sklearn.metrics import roc_auc_score, f1_score
import logging

# Suppress HuggingFace warnings
logging.getLogger("transformers").setLevel(logging.ERROR)
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# -----------------
# Setup Metrics
# -----------------
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
    if k == 0:
        print("No positive labels in this batch to calculate K-based metrics.")
        return
        
    ndcg = ndcg_at_k(y_true, y_score, k)
    prec_k = precision_at_k(y_true, y_score, k)
    
    try:
        auc = roc_auc_score(y_true, y_score)
    except:
        auc = 0.0
        
    mean_dv = y_true.mean()
    lift = (recall_at_k(y_true, y_score, k) / mean_dv) * 100 if mean_dv > 0 else 0.0
    
    # Binarize top K for F1
    y_pred_bin = np.zeros(len(y_true))
    top_indices = np.argsort(y_score)[-k:]
    y_pred_bin[top_indices] = 1
    f1 = f1_score(y_true, y_pred_bin)
    
    print(f"Precision@{k}: {prec_k * 100:.2f}%")
    print(f"NDCG@{k}: {ndcg * 100:.2f}%")
    print(f"LIFT: {lift:.2f}%")
    print(f"ROC-AUC: {auc * 100:.2f}%")
    print(f"F1-Score: {f1 * 100:.2f}%")

# -----------------
# Extractor logic
# -----------------
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

# -----------------
# Track A & B 
# -----------------
def run_dual_track_batch():
    # API setup for Track B
    load_dotenv()
    genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
    gemini_model = genai.GenerativeModel('gemini-flash-latest')
    
    # Model setup for Track A
    print("Loading GPT-2 Large for Track A...")
    gpt2_pipeline = pipeline('text-generation', model='gpt2-large', max_new_tokens=1, device_map='auto')
    
    print("Loading Candidate Mapping...")
    mapping_df = pd.read_csv('candidate_mapping.csv')
    
    # Run on all 131 candidates
    batch_df = mapping_df
    
    track_a_preds = []
    track_b_preds = []
    y_true = []
    
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

    for idx, row in batch_df.iterrows():
        print(f"\nProcessing Candidate {idx+1}: {row['candidate_name']}")
        base_dir = f"2015-2018_rookie_raw_cv_and_jmp/2018/{row['folder_name']}"
        raw_text = get_jmp_text(base_dir, row['jmp_intro_char_limit'])
        scrubbed_text = scrub_text(raw_text, row['candidate_name'])
        
        y_true.append(row['true_label_pub_w_top_5pct'])
        
        # --- Track A: Pre-2018 GPT-2 ---
        # GPT-2 limit is 1024 tokens (~4000 chars). We truncate strongly to fit prompt + context.
        gpt_context = scrubbed_text[:3000]
        gpt_prompt = f"Review the following academic paper introduction to predict if the author will be a top 5% researcher.\nPaper Introduction:\n{gpt_context}\n\nTop 5% researcher prediction (1 for yes, 0 for no): "
        
        try:
            res_a = gpt2_pipeline(gpt_prompt, return_full_text=False)[0]['generated_text'].strip()
            # Try to parse a number from the generated text
            pred_a = 1.0 if '1' in res_a[:2] else 0.0
        except Exception as e:
            print(f"Track A Error: {e}")
            pred_a = 0.0
            
        track_a_preds.append(pred_a)
        
        # --- Track B: Modern Gemini ---
        try:
            response = gemini_model.generate_content(f"{system_prompt_gemini}\n\nJMP Text:\n{scrubbed_text}")
            text_b = response.text.strip()
            if text_b.startswith("```json"): text_b = text_b[7:-3].strip()
            elif text_b.startswith("```"): text_b = text_b[3:-3].strip()
            
            res_json = json.loads(text_b)
            pred_b = float(res_json.get('prediction', 0) * res_json.get('confidence_score', 0.5))
            print(f"  Gemini CoT: {res_json.get('reasoning')[:100]}...")
        except Exception as e:
            print(f"Track B Error: {e}")
            pred_b = 0.0
            
        track_b_preds.append(pred_b)
        
        print(f"  True Label: {row['true_label_pub_w_top_5pct']} | GPT-2 Pred: {pred_a} | Gemini Pred: {pred_b:.2f}")

    y_true = np.array(y_true)
    track_a_preds = np.array(track_a_preds)
    track_b_preds = np.array(track_b_preds)
    
    calculate_all_metrics(y_true, track_a_preds, "Track A (Pre-2018 GPT-2)")
    calculate_all_metrics(y_true, track_b_preds, "Track B (Modern Gemini Flash)")
    
    # Save predictions
    batch_df['gpt2_pred'] = track_a_preds
    batch_df['gemini_pred'] = track_b_preds
    batch_df.to_csv("dual_track_predictions.csv", index=False)
    print("\nSaved full predictions to dual_track_predictions.csv")
    
if __name__ == "__main__":
    run_dual_track_batch()
