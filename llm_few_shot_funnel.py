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

# ==========================================
# 2. DATA EXTRACTION & SCRUBBING
# ==========================================
def get_jmp_text(folder_path, char_limit=25000):
    import glob
    pdfs = glob.glob(os.path.join(folder_path, '*.pdf'))
    if not pdfs: return ""
    
    cv_pdf_list = [f for f in pdfs if 'cv' in os.path.basename(f).lower()]
    other_pdfs = [f for f in pdfs if f not in cv_pdf_list]
    jmp_pdf = other_pdfs[0] if other_pdfs else pdfs[0]
    
    reader = PdfReader(jmp_pdf)
    text = ""
    for page in reader.pages:
        text += page.extract_text() or ""
        if len(text) > char_limit * 1.2:
            break
            
    return text[:int(char_limit)]

def scrub_text(text, candidate_name=""):
    # Scrub names and general academic affiliations
    if candidate_name:
        for part in candidate_name.split():
            if len(part) > 2:
                pattern = re.compile(re.escape(part), re.IGNORECASE)
                text = pattern.sub("[ANONYMIZED_CANDIDATE]", text)
    
    # Strip common university tags to enforce anti-bias
    text = re.sub(r'(?i)university|college|school of business', '[ANONYMIZED_INSTITUTION]', text)
    return text

# ==========================================
# 3. FEW-SHOT ANCHOR EXTRACTION (2017)
# ==========================================
def get_2017_anchors():
    print("Extracting 2017 Few-Shot Anchors (Empirically Verified Alignment)...")
    df = pd.read_csv('20260509_2015-2018_rookie_dataset.csv', index_col=0)
    
    base_dir_17 = '2015-2018_rookie_raw_cv_and_jmp/2017'
    folders_17 = sorted([f.name for f in os.scandir(base_dir_17) if f.is_dir()])
    
    # True Positive (CSV ID 316: '1387 - Yaqin Hu')
    idx_tp = 316
    folder_tp = folders_17[22] # Empirically mapped to folder index 22
    char_limit_tp = df.loc[idx_tp].get('jmp_intro_char_limit', 20000)
    text_tp = get_jmp_text(os.path.join(base_dir_17, folder_tp), char_limit_tp)
    scrubbed_tp = scrub_text(text_tp, "Yaqin Hu")
    
    # True Negative (CSV ID 295: '1321 - Stephanie Cheng')
    idx_tn = 295
    folder_tn = folders_17[1] # Empirically mapped to folder index 1
    char_limit_tn = df.loc[idx_tn].get('jmp_intro_char_limit', 20000)
    text_tn = get_jmp_text(os.path.join(base_dir_17, folder_tn), char_limit_tn)
    scrubbed_tn = scrub_text(text_tn, "Stephanie Cheng")
    
    return scrubbed_tp, scrubbed_tn

# ==========================================
# 4. SYSTEM PROMPT & PIPELINE LOGIC
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

def run_two_stage_funnel():
    load_dotenv()
    genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
    model = genai.GenerativeModel('gemini-flash-latest')
    
    # Get Anchors
    text_tp, text_tn = get_2017_anchors()
    
    # Load Stage 1 ML Funnel (Top 20 Candidates)
    print("\nExecuting Stage 1: Loading ML Funnel (Top 20 IPW Candidates)")
    ipw_preds = pd.read_csv('vanilla_ipw_predictions_weighted_2018.csv')
    mapping_df = pd.read_csv('candidate_mapping.csv')
    
    # Merge ML predictions with Candidate mapping safely
    mapping_df = pd.merge(mapping_df, ipw_preds[['Candidate_ID', 'Predicted_Probability']], 
                          left_on='csv_candidate_id', right_on='Candidate_ID', how='left')
    mapping_df.rename(columns={'Predicted_Probability': 'ipw_score'}, inplace=True)
    
    # Filter Top 20
    top_20_df = mapping_df.sort_values('ipw_score', ascending=False).head(20)
    print(f"Pre-screened {len(top_20_df)} candidates for Stage 2 LLM Review.")
    
    # Initialize Score Array (Default 0.0 for those who failed Stage 1)
    final_semantic_scores = np.zeros(len(mapping_df))
    
    # Stage 2: LLM Evaluation
    print("\nExecuting Stage 2: LLM Few-Shot Re-Ranking (Gemini Flash)")
    for i, (idx, row) in enumerate(top_20_df.iterrows()):
        print(f"[{i+1}/20] Evaluating Candidate ID {row['csv_candidate_id']}...")
        
        base_dir = f"2015-2018_rookie_raw_cv_and_jmp/2018/{row['folder_name']}"
        raw_text = get_jmp_text(base_dir, row['jmp_intro_char_limit'])
        scrubbed_text = scrub_text(raw_text, row['candidate_name'])
        
        prompt = generate_system_prompt(text_tp, text_tn, scrubbed_text)
        
        # Call Gemini (with basic retry logic)
        for attempt in range(3):
            try:
                response = model.generate_content(prompt)
                res_text = response.text.strip()
                if res_text.startswith("```json"): res_text = res_text[7:-3].strip()
                elif res_text.startswith("```"): res_text = res_text[3:-3].strip()
                
                res_json = json.loads(res_text)
                score = float(res_json.get('Semantic_Potential_Score', 0.0))
                final_semantic_scores[idx] = score
                break
            except Exception as e:
                print(f"  Attempt {attempt+1} failed: {e}. Retrying in 10s...")
                time.sleep(10)
        
        time.sleep(4) # Respect Rate Limits
        
    # Evaluate System
    y_true = mapping_df['true_label_pub_w_top_5pct'].values
    calculate_all_metrics(y_true, final_semantic_scores, "Stage 2 LLM Re-Ranker (Few-Shot Funnel)")
    
    # Save output
    mapping_df['funnel_semantic_score'] = final_semantic_scores
    mapping_df.to_csv("funnel_predictions_2018.csv", index=False)
    print("\nSaved pipeline outputs to funnel_predictions_2018.csv")

if __name__ == "__main__":
    run_two_stage_funnel()
