import os
import json
import pandas as pd
from dotenv import load_dotenv
import google.generativeai as genai
from pypdf import PdfReader
import sys

# 1. API Setup & Security
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise ValueError("GEMINI_API_KEY not found in .env file")

genai.configure(api_key=api_key)
model = genai.GenerativeModel('gemini-flash-latest')

# 2. Look-ahead Bias Prevention Helpers
def anonymize_text(text, name):
    """Replaces candidate name with 'Candidate' to prevent bias."""
    import re
    parts = name.split()
    last_name = parts[-1]
    
    pattern = re.compile(re.escape(name), re.IGNORECASE)
    text = pattern.sub("Candidate", text)
    
    pattern_ln = re.compile(re.escape(last_name), re.IGNORECASE)
    text = pattern_ln.sub("Candidate", text)
    
    return text

def extract_text_from_pdf(pdf_path):
    """Extracts text from a PDF file."""
    try:
        reader = PdfReader(pdf_path)
        text = ""
        for page in reader.pages:
            text += page.extract_text() or ""
        return text
    except Exception as e:
        print(f"Error reading {pdf_path}: {e}", flush=True)
        return ""

# 3. Experiment Setup
candidates = [
    {"name": "Sehwa Kim", "folder": "2015-2018_rookie_raw_cv_and_jmp/2018/2126 - Sehwa Kim", "id": 439},
    {"name": "Joshua Khavis", "folder": "2015-2018_rookie_raw_cv_and_jmp/2018/784 - Khavis, Joshua", "id": 493},
    {"name": "Maximilian Muhn", "folder": "2015-2018_rookie_raw_cv_and_jmp/2018/807 - Muhn, Maximilian", "id": 502},
]

# 4. Advanced CoT System Prompt
system_prompt = (
    "You are a strict accounting faculty recruiter in the year 2018. "
    "You must evaluate this candidate based ONLY on the provided text. "
    "Do NOT use any knowledge of events, publications, or academic success that occurred after 2018. "
    "\n\nANTI-BIAS WARNING: Do NOT overvalue elite PhD program prestige (e.g., Top 10 schools) or famous committee members. "
    "Historical data shows these factors often lead to overestimating a candidate's actual future productivity. "
    "\n\nEVALUATION RUBRIC: Focus heavily on early indicators of research momentum: number of academic awards, "
    "breadth of co-authorship networks, actual publications/R&Rs, and specialized reviewing experience. "
    "\n\nReturn your evaluation in strict JSON using this structure:\n"
    "{\n"
    '  "step_1_pedigree_check": "assess school/advisors but explicitly discount their weight",\n'
    '  "step_2_momentum_check": "assess publications, awards, and network",\n'
    '  "step_3_final_reasoning": "synthesize",\n'
    '  "confidence_score": 1-10,\n'
    '  "top_5_percent_prediction": 1 or 0\n'
    "}"
)

# 5. Run Experiment
dataset_path = '20260422 from longzhen 2015-2018_rookie_dataset_and_demo_code/2015-2018_rookie_dataset.csv'
df = pd.read_csv(dataset_path)

print(f"--- Advanced LLM Prompting Experiment (3 Candidates) ---", flush=True)

for cand in candidates:
    print(f"\nEvaluating: {cand['name']}...", flush=True)
    
    # Robust file selection
    files = [f for f in os.listdir(cand['folder']) if f.lower().endswith('.pdf')]
    cv_pdf_list = [f for f in files if 'cv' in f.lower()]
    cv_pdf = cv_pdf_list[0] if cv_pdf_list else files[0]
    
    other_pdfs = [f for f in files if f != cv_pdf]
    jmp_pdf = other_pdfs[0] if other_pdfs else cv_pdf # Fallback if only one file
    
    print(f"  Files: CV={cv_pdf}, JMP={jmp_pdf}", flush=True)

    # Extraction
    print("  Extracting text...", flush=True)
    cv_text = extract_text_from_pdf(os.path.join(cand['folder'], cv_pdf))
    jmp_text = extract_text_from_pdf(os.path.join(cand['folder'], jmp_pdf))
    
    # Anonymization
    print("  Anonymizing...", flush=True)
    cv_anon = anonymize_text(cv_text, cand['name'])
    jmp_anon = anonymize_text(jmp_text, cand['name'])
    
    user_content = f"CV TEXT:\n{cv_anon}\n\nJOB MARKET PAPER TEXT:\n{jmp_anon}"
    
    # Call API
    print("  Calling Gemini API...", flush=True)
    try:
        response = model.generate_content([system_prompt, user_content])
        raw_json = response.text.strip()
        if raw_json.startswith("```json"):
            raw_json = raw_json[7:-3].strip()
        elif raw_json.startswith("```"):
            raw_json = raw_json[3:-3].strip()
        evaluation = json.loads(raw_json)
    except Exception as e:
        print(f"Error processing {cand['name']}: {e}", flush=True)
        evaluation = {}

    # Ground Truth
    ground_truth = df[df['ID'] == cand['id']]['pub_top_5pct'].values[0]
    
    # Output Results
    print(f"\nRESULTS FOR {cand['name']}:", flush=True)
    print(f"  Ground Truth (pub_top_5pct): {ground_truth}", flush=True)
    print(f"  LLM Prediction: {evaluation.get('top_5_percent_prediction')}", flush=True)
    print(f"  Step 1 (Pedigree): {evaluation.get('step_1_pedigree_check')}", flush=True)
    print(f"  Step 2 (Momentum): {evaluation.get('step_2_momentum_check')}", flush=True)
    print(f"  Step 3 (Final): {evaluation.get('step_3_final_reasoning')}", flush=True)
    print(f"  Confidence: {evaluation.get('confidence_score')}/10", flush=True)
