import os
import json
import glob
import pandas as pd
from pypdf import PdfReader
import google.generativeai as genai
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise ValueError("GEMINI_API_KEY not found in .env")
genai.configure(api_key=api_key)

model = genai.GenerativeModel('gemini-flash-latest')

SYSTEM_PROMPT = """You are an expert academic data extractor. Your task is to extract exact demographic and metadata attributes from the provided candidate's CV and Job Market Paper.
Please extract the following information strictly in JSON format:
{
  "CandidateName": "Full name",
  "Gender": "Male or Female (infer from name or pronouns)",
  "PhD_Institution": "Name of PhD granting institution",
  "PhD_Graduation_Year": "Year of PhD graduation",
  "Has_Coauthor_on_JMP": true/false (Are there co-authors listed on the JMP or is it single-authored?),
  "Total_Number_of_Coauthors_on_all_papers": integer (Count the total number of unique co-authors across all papers in the CV),
  "Mentions_Folder_Number_In_Text": true/false (Does the abstract or introduction explicitly mention the folder number as a sample size or parameter?)
}"""

def extract_pdf_text(pdf_path, max_pages=10):
    try:
        reader = PdfReader(pdf_path)
        text = ""
        for i, page in enumerate(reader.pages):
            if i >= max_pages:
                break
            text += page.extract_text() + "\n"
        return text
    except Exception as e:
        return str(e)

def process_candidate(folder_path):
    folder_name = os.path.basename(folder_path)
    print(f"\nProcessing Folder: {folder_name}")
    
    # Extract folder prefix number if it exists
    folder_num = folder_name.split(' - ')[0] if ' - ' in folder_name else ""
    
    pdfs = glob.glob(os.path.join(folder_path, '*.pdf'))
    if not pdfs:
        print("No PDFs found.")
        return None
        
    combined_text = f"Folder Name / Number: {folder_num}\n\n"
    for pdf in pdfs:
        # Just extract first few pages of each PDF to get metadata (CV and JMP intro)
        combined_text += f"--- Document: {os.path.basename(pdf)} ---\n"
        combined_text += extract_pdf_text(pdf, max_pages=5)
        combined_text += "\n\n"
        
    prompt = f"{SYSTEM_PROMPT}\n\nCandidate Text:\n{combined_text}"
    
    try:
        response = model.generate_content(prompt)
        text = response.text.strip()
        if text.startswith("```json"):
            text = text[7:-3].strip()
        elif text.startswith("```"):
            text = text[3:-3].strip()
        result = json.loads(text)
        result['Folder_Name'] = folder_name
        return result
    except Exception as e:
        print(f"Failed to generate/parse JSON: {e}")
        return None

def run_diagnostic(test_limit=5):
    base_dir = "2015-2018_rookie_raw_cv_and_jmp/2018"
    folders = [f.path for f in os.scandir(base_dir) if f.is_dir()]
    folders = sorted(folders)[:test_limit]
    
    results = []
    for folder in folders:
        res = process_candidate(folder)
        if res:
            results.append(res)
            print(json.dumps(res, indent=2))
            
    # Load CSV to see what we can map to
    print("\n--- CSV Metadata Available (Sample) ---")
    df = pd.read_csv('20260509_2015-2018_rookie_dataset.csv', index_col=0)
    df_2018 = df[df.year == 2018]
    metadata_cols = [c for c in df.columns if not c.endswith('_dt') and not c.endswith('_cv') and 'top' not in c and 'pub' not in c and 'job' not in c and '0_' not in c]
    print(df_2018[metadata_cols].head(3).to_string())

if __name__ == "__main__":
    run_diagnostic()
