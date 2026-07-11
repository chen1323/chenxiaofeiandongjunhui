import os
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()
if "GEMINI_API_KEY" in os.environ:
    genai.configure(api_key=os.environ["GEMINI_API_KEY"])

for m in genai.list_models():
    print(m.name)
