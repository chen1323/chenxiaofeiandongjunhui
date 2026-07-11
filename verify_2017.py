import os
import glob
import pandas as pd
from pypdf import PdfReader
import google.generativeai as genai
from dotenv import load_dotenv
import json

def get_jmp_text(folder_path):
    pdfs = glob.glob(os.path.join(folder_path, '*.pdf'))
    if not pdfs: return ""
    cv_pdf_list = [f for f in pdfs if 'cv' in os.path.basename(f).lower()]
    other_pdfs = [f for f in pdfs if f not in cv_pdf_list]
    jmp_pdf = other_pdfs[0] if other_pdfs else pdfs[0]
    
    reader = PdfReader(jmp_pdf)
    text = ""
    for i, page in enumerate(reader.pages):
        if i >= 3: break # First 3 pages enough for metadata
        text += page.extract_text() or ""
    return text

def extract_metadata_gemini(text):
    load_dotenv()
    genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
    model = genai.GenerativeModel('gemini-flash-latest')
    prompt = """Extract the following metadata from this research paper text. Return ONLY a JSON object:
{
    "candidate_name": "Full Name",
    "implied_gender": "Male or Female",
    "number_of_coauthors": number (0 if solo author)
}
Paper Text:
""" + text[:5000]
    try:
        response = model.generate_content(prompt)
        res_text = response.text.strip()
        if res_text.startswith("```json"): res_text = res_text[7:-3].strip()
        elif res_text.startswith("```"): res_text = res_text[3:-3].strip()
        return json.loads(res_text)
    except Exception as e:
        return {"error": str(e)}

def run_diagnostic():
    df = pd.read_csv('20260509_2015-2018_rookie_dataset.csv')
    df_17 = df[df['year'] == 2017]
    start_idx = df_17.index.min()
    
    base_dir = "2015-2018_rookie_raw_cv_and_jmp/2017"
    folders = sorted([f for f in os.scandir(base_dir) if f.is_dir()], key=lambda f: f.name)
    
    print("--- 2017 Alignment Diagnostic ---")
    
    # Check folder 0 (expected CSV idx 293)
    idx_0 = 293
    print(f"\n1. Checking TN Candidate - Folder Index 0: {folders[0].name}")
    print(f"   -> Expected CSV Index: {idx_0}")
    print(f"   -> CSV Target Label (pub_w_top_5pct): {df.loc[idx_0, 'pub_w_top_5pct']}")
    print(f"   -> CSV Features: gender={df.loc[idx_0, 'gender']} (1=Female/0=Male), coauthors={df.loc[idx_0, 'number of coauthors']}")
    
    text_0 = get_jmp_text(folders[0].path)
    meta_0 = extract_metadata_gemini(text_0)
    print(f"   -> Extracted from PDF via Gemini: {meta_0}")
    
    # Check folder 13 (expected CSV idx 306)
    idx_13 = 306
    print(f"\n2. Checking TP Candidate - Folder Index 13: {folders[13].name}")
    print(f"   -> Expected CSV Index: {idx_13}")
    print(f"   -> CSV Target Label (pub_w_top_5pct): {df.loc[idx_13, 'pub_w_top_5pct']}")
    print(f"   -> CSV Features: gender={df.loc[idx_13, 'gender']} (1=Female/0=Male), coauthors={df.loc[idx_13, 'number of coauthors']}")
    
    text_13 = get_jmp_text(folders[13].path)
    meta_13 = extract_metadata_gemini(text_13)
    print(f"   -> Extracted from PDF via Gemini: {meta_13}")

if __name__ == "__main__":
    run_diagnostic()
