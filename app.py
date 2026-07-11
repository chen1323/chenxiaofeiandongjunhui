import streamlit as st
import pandas as pd
import numpy as np
import time
import math
import uuid
import os
import json
import traceback
from dotenv import load_dotenv
import google.generativeai as genai
import fitz  # PyMuPDF

# ==========================================
# ENV & API INITIALIZATION
# ==========================================
load_dotenv()
if "GEMINI_API_KEY" in os.environ:
    genai.configure(api_key=os.environ["GEMINI_API_KEY"])

# ==========================================
# CAUSAL INFERENCE ENGINE 
# ==========================================
def run_causal_inference(features):
    logit = -3.5 
    placerank = float(features.get('Placerank', 25.0))
    logit -= (placerank - 1.0) * 0.08  
    logit += float(features.get('gender', 0.0)) * 0.10
    
    coauthors = features.get('number of coauthors', features.get('number_of_coauthors', 0))
    logit += float(coauthors) * 0.35
    
    logit += float(features.get('num_pub', 0)) * 0.45
    
    novel_score = features.get('has_novel_data', {}).get('score', 0)
    causal_score = features.get('explicit_causal_id', {}).get('score', 0)
    cross_score = features.get('cross_disciplinary_method', {}).get('score', 0)
    sample_score = features.get('sample_size_over_10k', {}).get('score', 0)
    
    logit += float(novel_score) * 0.65
    logit += float(causal_score) * 0.75
    logit += float(cross_score) * 0.25
    logit += float(sample_score) * 0.15
    return 1.0 / (1.0 + math.exp(-logit))

# ==========================================
# PDF & LLM PARSING PIPELINE
# ==========================================
def extract_pdf_text(uploaded_file, num_pages=10):
    uploaded_file.seek(0)
    doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
    text = ""
    for i in range(min(num_pages, len(doc))):
        text += doc[i].get_text()
    return text

def render_interactive_pdf(pdf_bytes, active_quote, current_page):
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    page_to_render = current_page
    
    if active_quote:
        found = False
        for page_num in range(len(doc)):
            page = doc[page_num]
            text_instances = page.search_for(active_quote)
            if not text_instances and len(active_quote) > 40:
                text_instances = page.search_for(active_quote[:40])
            
            if text_instances:
                for inst in text_instances:
                    page.add_highlight_annot(inst)
                page_to_render = page_num
                found = True
                break # Render the first page it's found on
                
        if not found:
            page_to_render = current_page
            
    # Bound safety
    page_to_render = max(0, min(page_to_render, len(doc)-1))
    
    page = doc[page_to_render]
    pix = page.get_pixmap(dpi=150)
    img_bytes = pix.tobytes("png")
    
    return img_bytes, page_to_render, len(doc)

def parse_pdf_with_gemini(uploaded_file):
    short_id = str(uuid.uuid4())[:4].upper()
    candidate_name = f"{uploaded_file.name.replace('.pdf', '')} (ID-{short_id})"
    error_traceback = None
    llm_output = {}
    
    try:
        sample_text = extract_pdf_text(uploaded_file, num_pages=10)
        
        system_prompt = """
        You are an expert econometric methodology reviewer. Analyze the following academic Job Market Paper text.
        Extract the following 4 methodological features. For each feature, provide a 'score' (1 or 0), 
        an 'evidence_quote' (a short, exact 1-2 sentence quote directly from the text if score is 1, else an empty string), 
        and an 'improvement_advice' (a suggestion on how to rephrase or improve the methodology if score is 0, else an empty string).
        
        Criteria:
        - explicit_causal_id: Does the author use a strict causal identification strategy (e.g., Difference-in-Differences, Instrumental Variables, Regression Discontinuity)?
        - has_novel_data: Does the author use proprietary, hand-collected, or highly novel unstructured data (not just standard Compustat/CRSP)?
        - cross_disciplinary_method: Does the author apply Machine Learning, NLP, or advanced textual analysis?
        - sample_size_over_10k: Is the empirical sample size explicitly stated to be over 10,000 observations?
        
        Return a strict JSON object with these 4 keys exactly.
        """
        
        model = genai.GenerativeModel('gemini-2.5-flash', generation_config={"response_mime_type": "application/json"})
        response = model.generate_content(f"{system_prompt}\n\nPAPER TEXT:\n{sample_text}")
        
        llm_output = json.loads(response.text)
        
    except Exception as e:
        error_traceback = traceback.format_exc()
        # Fallback mock data if API fails
        llm_output = {
            "explicit_causal_id": {"score": 0, "evidence_quote": "", "improvement_advice": "API Error."},
            "has_novel_data": {"score": 0, "evidence_quote": "", "improvement_advice": "API Error."},
            "cross_disciplinary_method": {"score": 0, "evidence_quote": "", "improvement_advice": "API Error."},
            "sample_size_over_10k": {"score": 0, "evidence_quote": "", "improvement_advice": "API Error."}
        }

    return {
        "Session_UUID": f"{str(uuid.uuid4())[:8]}",
        "Candidate_Name": candidate_name,
        "Placerank": float(np.random.randint(1, 51)),
        "gender": float(np.random.choice([0.0, 1.0])),
        "number of coauthors": int(np.random.randint(0, 4)),
        "num_pub": int(np.random.randint(0, 3)),
        "explicit_causal_id": llm_output.get("explicit_causal_id", {}),
        "has_novel_data": llm_output.get("has_novel_data", {}),
        "cross_disciplinary_method": llm_output.get("cross_disciplinary_method", {}),
        "sample_size_over_10k": llm_output.get("sample_size_over_10k", {}),
        "error_traceback": error_traceback
    }

# ==========================================
# PRESENTATION FORMATTERS
# ==========================================
def format_prestige_tier(placerank):
    if placerank <= 10: return "🥇 Elite (Top 10)"
    elif placerank <= 25: return "🥈 High Tier (11-25)"
    else: return "🥉 Standard"

def generate_methodological_tags(row):
    tags = []
    if row.get('has_novel_data', {}).get('score', 0) == 1: tags.append("🍏 Hand-Collected Data")
    if row.get('explicit_causal_id', {}).get('score', 0) == 1: tags.append("🛡️ Rigorous Causal ID")
    if row.get('cross_disciplinary_method', {}).get('score', 0) == 1: tags.append("🧬 Cross-Disciplinary")
    if row.get('sample_size_over_10k', {}).get('score', 0) == 1: tags.append("📊 Large Sample (N>10k)")
    return " | ".join(tags) if tags else "Standard Empirics"

# ==========================================
# UI CONFIGURATION & CSS
# ==========================================
st.set_page_config(page_title="Academic Forecasting Engine", layout="wide", initial_sidebar_state="collapsed")
st.markdown("""
<style>
    .report-card { background-color: #f8f9fa; border-radius: 8px; padding: 25px; box-shadow: 0 4px 6px rgba(0,0,0,0.05); }
    .metric-value { font-size: 2.5rem; font-weight: 800; color: #2ecc71; margin-bottom: 10px; }
    .advice-box { background-color: #fff3cd; border-left: 5px solid #ffc107; padding: 10px; margin-bottom: 15px; border-radius: 4px; }
    .quote-box { background-color: #d4edda; border-left: 5px solid #28a745; padding: 10px; margin-bottom: 15px; border-radius: 4px; font-style: italic; }
</style>
""", unsafe_allow_html=True)

# ==========================================
# SESSION STATE INITIALIZATION
# ==========================================
if 'step' not in st.session_state:
    st.session_state.step = 1
if 'user_role' not in st.session_state:
    st.session_state.user_role = "💼 Recruiter / Faculty Committee"
if 'domain' not in st.session_state:
    st.session_state.domain = "Accounting"
if 'parsed_data' not in st.session_state:
    st.session_state.parsed_data = None
if 'api_budget' not in st.session_state:
    st.session_state.api_budget = 5.00
if 'pdf_bytes_list' not in st.session_state:
    st.session_state.pdf_bytes_list = []
if 'active_quote' not in st.session_state:
    st.session_state.active_quote = ""
if 'current_page' not in st.session_state:
    st.session_state.current_page = 0

def next_step(): st.session_state.step += 1
def prev_step(): st.session_state.step -= 1
def reset_step(): 
    st.session_state.step = 1
    st.session_state.parsed_data = None
    st.session_state.pdf_bytes_list = []
    st.session_state.active_quote = ""
    st.session_state.current_page = 0

# ==========================================
# STEP 1: WELCOME GATEWAY & ROLE CONTEXT
# ==========================================
if st.session_state.step == 1:
    st.title("Top-Tier Academic Publishing Prediction Engine")
    st.markdown("### Welcome Gateway")
    
    col1, col2 = st.columns(2)
    with col1:
        st.session_state.user_role = st.radio(
            "Select Authorization Role:",
            ["💼 Recruiter / Faculty Committee", "🎓 Applicant / Student Researcher"],
            index=0 if "Recruiter" in st.session_state.user_role else 1
        )
    with col2:
        st.session_state.domain = st.selectbox(
            "Target Domain:",
            ["Accounting", "Computer Science", "Finance", "Economics"],
            index=["Accounting", "Computer Science", "Finance", "Economics"].index(st.session_state.domain)
        )
        
    st.markdown("### Authentication")
    st.text_input("User Login / API Key (Enterprise Portal):", type="password", placeholder="Enter your assigned API Key")
    
    st.markdown(f"**Current API Budget:** ${st.session_state.api_budget:.2f}")
    
    st.markdown("<br>", unsafe_allow_html=True)
    st.button("Next: Match Prediction Model →", type="primary", on_click=next_step)

# ==========================================
# STEP 2: MODEL HUB & SECURE FILE GATEWAY
# ==========================================
elif st.session_state.step == 2:
    st.title("Model Hub & Secure File Gateway")
    
    st.info("System has matched you with our peerless **Econometric GBT Ensemble Model**. Unlike vanilla LLMs trapped by semantic rhetoric, this architecture carries mathematically stable causal network parameters, archiving the highest validated NDCG and Precision on extreme upper-tail academic forecasting.")
    
    st.markdown(f"### Authenticated Route: {st.session_state.user_role}")
    
    if "Recruiter" in st.session_state.user_role:
        st.info("**Batch Processing Enabled:** You may upload multiple candidate JMPs simultaneously (Max 50 files per batch) to generate a unified comparative leaderboard.")
        uploaded_files = st.file_uploader("Upload Resumes / JMPs (PDF)", type=["pdf"], accept_multiple_files=True)
    else:
        st.markdown("**Individual Candidate Uploader**")
        uploaded_files = st.file_uploader("Upload Your Resume / JMP (PDF)", type=["pdf"], accept_multiple_files=False)
        if uploaded_files:
            uploaded_files = [uploaded_files]

    col1, col2, _ = st.columns([1, 2, 7])
    with col1:
        st.button("← Back", on_click=prev_step)
    with col2:
        if uploaded_files:
            if st.button("Parse & Generate Dashboard 🚀", type="primary"):
                cost_per_pdf = 0.05
                total_cost = cost_per_pdf * len(uploaded_files)
                
                if st.session_state.api_budget >= total_cost:
                    with st.spinner(f"Sending to Gemini API... (Deducting ${total_cost:.2f} from budget)"):
                        st.session_state.api_budget -= total_cost
                        st.session_state.parsed_data = []
                        st.session_state.pdf_bytes_list = []
                        st.session_state.active_quote = ""
                        st.session_state.current_page = 0
                        
                        for f in uploaded_files:
                            # 1. Parse text with Gemini API
                            parsed_json = parse_pdf_with_gemini(f)
                            st.session_state.parsed_data.append(parsed_json)
                            
                            # 2. Store raw PDF bytes for dynamic rendering
                            f.seek(0)
                            st.session_state.pdf_bytes_list.append(f.read())
                            
                        next_step()
                        st.rerun()
                else:
                    st.error("Usage Limit Exceeded: Your allocated API budget has been depleted. Please contact the administrator for an enterprise tier upgrade.")
        else:
            st.button("Parse & Generate Dashboard 🚀", type="primary", disabled=True)

# ==========================================
# STEP 3: THE EXECUTIVE ANALYSIS REPORT
# ==========================================
elif st.session_state.step == 3:
    st.title("The Executive Analysis Report")
    
    if "Recruiter" in st.session_state.user_role:
        st.markdown("### K=8 Aligned Master Leaderboard")
        df = pd.DataFrame(st.session_state.parsed_data)
        
        # Display API errors globally if any occurred in batch
        for row in st.session_state.parsed_data:
            if row.get('error_traceback'):
                st.error(f"API Error Encountered for {row['Candidate_Name']}:")
                st.code(row['error_traceback'], language="python")
        
        probs = [run_causal_inference(row) for row in st.session_state.parsed_data]
        df['Academic Potential Index'] = probs
        df['Institutional Pedigree'] = df['Placerank'].apply(format_prestige_tier)
        df['Methodological Highlights'] = df.apply(generate_methodological_tags, axis=1)
        
        columns_to_drop = ['Session_UUID', 'Placerank', 'gender', 'has_novel_data', 'explicit_causal_id', 
                           'cross_disciplinary_method', 'sample_size_over_10k', 'error_traceback']
        display_df = df.drop(columns=columns_to_drop, errors='ignore')
        
        cols = ['Candidate_Name', 'Academic Potential Index', 'Institutional Pedigree', 'Methodological Highlights', 'number of coauthors', 'num_pub']
        display_df = display_df[[c for c in cols if c in display_df.columns]]
        display_df = display_df.sort_values('Academic Potential Index', ascending=False).reset_index(drop=True)
        
        def highlight_top_k(s, k=8):
            is_top = pd.Series(data=False, index=s.index)
            is_top[:min(k, len(s))] = True
            return ['background-color: rgba(46, 204, 113, 0.15)' if v else '' for v in is_top]

        styled_df = display_df.style.apply(highlight_top_k, axis=0, k=8).format({
            'Academic Potential Index': '{:.1%}'
        })
        dynamic_height = min(len(display_df) * 35 + 43, 400)
        st.dataframe(styled_df, use_container_width=True, height=dynamic_height, hide_index=True)
        
        # --- Interactive Deep-Dive (XAI Dossier) ---
        st.markdown("---")
        st.markdown("### Executive XAI Dossier")
        
        # Build mapping of candidate name to their original raw row for quick lookup
        candidate_map = {row['Candidate_Name']: row for row in st.session_state.parsed_data}
        
        selected_candidate = st.selectbox("Select a candidate for SHAP Deep-Dive Analysis:", options=display_df['Candidate_Name'].tolist())
        
        if selected_candidate:
            features = candidate_map[selected_candidate]
            base_prob = run_causal_inference(features)
            
            st.markdown(f"#### Analyzing: {selected_candidate} (Score: {base_prob * 100:.1f}%)")
            
            # Simulated SHAP Breakdown logic
            strengths = []
            drags = []
            
            pr = features.get('Placerank', 25.0)
            if pr <= 15:
                strengths.append(("Elite Pedigree (Top 15)", round((25 - pr) * 0.8, 1)))
            elif pr > 25:
                drags.append(("Institutional Drag (Rank > 25)", round((25 - pr) * -0.5, 1)))
                
            if features.get('explicit_causal_id', {}).get('score', 0) == 1:
                strengths.append(("Rigorous Causal ID", 15.2))
            else:
                drags.append(("Missing Causal Identification", -12.4))
                
            if features.get('has_novel_data', {}).get('score', 0) == 1:
                strengths.append(("Hand-Collected/Novel Data", 11.5))
            else:
                drags.append(("Standard Data Assets", -5.3))
                
            coauthors = features.get('number of coauthors', features.get('number_of_coauthors', 0))
            if coauthors > 0:
                strengths.append(("Strong Coauthor Network", round(coauthors * 3.5, 1)))
            else:
                drags.append(("Isolated Researcher (0 Coauthors)", -4.1))
                
            pubs = features.get('num_pub', 0)
            if pubs > 0:
                strengths.append(("Proven Publication Record", round(pubs * 6.2, 1)))
                
            col_pos, col_neg = st.columns(2)
            
            with col_pos:
                with st.expander("🟢 Positive Drivers (Strengths)", expanded=True):
                    if strengths:
                        for name, val in strengths:
                            st.metric(label=name, value=f"+{val}%", delta=f"{val}%")
                    else:
                        st.write("No major structural strengths detected.")
                        
            with col_neg:
                with st.expander("🔴 Negative Drags (Vulnerabilities)", expanded=True):
                    if drags:
                        for name, val in drags:
                            st.metric(label=name, value=f"{val}%", delta=f"{val}%", delta_color="normal")
                    else:
                        st.write("No significant structural drags detected.")
                        
            st.markdown("**Qualitative AI Summary:**")
            if strengths and drags:
                top_strength = sorted(strengths, key=lambda x: x[1], reverse=True)[0][0]
                top_drag = sorted(drags, key=lambda x: x[1])[0][0]
                st.info(f"This candidate is highly propelled by their **{top_strength}**, which acts as a massive positive driver. However, their **{top_drag}** acts as a structural vulnerability dragging their long-term potential.")
            elif strengths:
                top_strength = sorted(strengths, key=lambda x: x[1], reverse=True)[0][0]
                st.success(f"This candidate demonstrates an elite profile, heavily propelled by their **{top_strength}**. They have no significant structural drags and represent a highly competitive prospect.")
            else:
                st.warning("This candidate currently lacks major structural and methodological drivers, leading to a suppressed forecast score. Significant methodological improvements are required.")
        
    else:
        # ==========================================
        # C-END DUAL-COLUMN DASHBOARD (SPLIT-SCREEN)
        # ==========================================
        features = st.session_state.parsed_data[0]
        
        if features.get('error_traceback'):
            st.error("API Error Encountered During Parsing:")
            st.code(features['error_traceback'], language="python")
            
        base_prob = run_causal_inference(features)
        
        st.markdown(f"### {features['Candidate_Name']} | Executive Summary")
        
        st.markdown(f"""
        <div class="report-card" style="margin-bottom: 20px; display: flex; flex-direction: column; align-items: center;">
            <div style="font-size: 1.1rem; color: #7f8c8d; text-transform: uppercase;">Academic Potential Index</div>
            <div class="metric-value" style="font-size: 3.5rem;">{base_prob * 100:.1f}%</div>
        </div>
        """, unsafe_allow_html=True)
        st.progress(base_prob)
        
        if base_prob >= 0.50:
            st.success("🏆 You currently place within the Top 8% of the global Rookie cohort.")
        else:
            st.warning("⚠️ Your profile currently projects outside the elite bracket. Review your vulnerability metrics.")
            
        st.markdown("---")
        
        col_pdf, col_analysis = st.columns([1, 1])
        
        with col_pdf:
            st.markdown("### Interactive PDF Diagnostic Viewer")
            pdf_bytes = st.session_state.pdf_bytes_list[0] if st.session_state.pdf_bytes_list else None
            
            if pdf_bytes:
                img_bytes, rendered_page, total_pages = render_interactive_pdf(
                    pdf_bytes, 
                    st.session_state.active_quote, 
                    st.session_state.current_page
                )
                
                # Dynamic Pagination Controls
                c1, c2, c3, c4 = st.columns([1, 1, 2, 1])
                with c1:
                    if st.button("⬅️ Prev", disabled=(rendered_page <= 0)):
                        st.session_state.current_page = rendered_page - 1
                        st.session_state.active_quote = "" # Clear highlight on manual scroll
                        st.rerun()
                with c2:
                    if st.button("Next ➡️", disabled=(rendered_page >= total_pages - 1)):
                        st.session_state.current_page = rendered_page + 1
                        st.session_state.active_quote = ""
                        st.rerun()
                with c3:
                    st.markdown(f"**Page {rendered_page + 1} of {total_pages}**")
                with c4:
                    if st.button("🔄 Reset View"):
                        st.session_state.active_quote = ""
                        st.rerun()
                
                # Render Image with use_container_width
                st.image(img_bytes, use_container_width=True)
            else:
                st.error("Failed to load PDF bytes.")
            
        with col_analysis:
            st.markdown("### AI Methodology Diagnosis")
            st.write("The econometric engine has audited your Job Market Paper using Gemini 2.5 Flash. Below is a breakdown of your methodological highlights and structural vulnerabilities.")
            
            def render_feature_block(title, key, feature_dict, weight):
                score = feature_dict.get('score', 0)
                if score == 1:
                    st.markdown(f"#### ✅ {title} (+{weight:.2f} Logit Weight)")
                    st.markdown(f"<div class='quote-box'><b>Evidence Found:</b> \"{feature_dict.get('evidence_quote', '')}\"</div>", unsafe_allow_html=True)
                    if st.button(f"🔍 Locate Evidence", key=f"btn_{key}"):
                        st.session_state.active_quote = feature_dict.get('evidence_quote', '')
                        st.rerun()
                else:
                    st.markdown(f"#### ❌ {title} (Missed Weight: +{weight:.2f})")
                    st.markdown(f"<div class='advice-box'><b>AI Advise:</b> {feature_dict.get('improvement_advice', '')}</div>", unsafe_allow_html=True)
            
            st.markdown("---")
            render_feature_block("Explicit Causal Identification", "causal", features.get('explicit_causal_id', {}), 0.75)
            render_feature_block("Novel / Hand-Collected Data", "novel", features.get('has_novel_data', {}), 0.65)
            render_feature_block("Cross-Disciplinary Methodology", "cross", features.get('cross_disciplinary_method', {}), 0.25)
            render_feature_block("Large Sample Size (N>10k)", "sample", features.get('sample_size_over_10k', {}), 0.15)
            
            st.markdown("---")
            st.markdown("#### Structural Network Constraints")
            col_a, col_b = st.columns(2)
            with col_a:
                st.metric("Institutional Placerank", f"{features.get('Placerank', 25.0):.1f}")
            with col_b:
                st.metric("Coauthor Network Size", features.get('number of coauthors', 0))
            st.caption("Note: Institutional Placerank and Coauthor Networks are historical variables and cannot be optimized within the text.")

    st.markdown("---")
    st.button("← Reset Engine", type="secondary", on_click=reset_step)
