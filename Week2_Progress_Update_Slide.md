# Project Update: AI for Research Talent Identification
**Status Report: End of Week 2 (Academic Year 2026)**

### **1. Progress Achieved (Weeks 1-2)**
*   **Environment & Replication:** Successfully configured a cross-platform Python environment and replicated the paper’s 'Late-Fusion Logistic Regression' baseline on the 2015-2018 rookie dataset.
*   **Robust Evaluation Framework:** Implemented a comprehensive multi-metric suite (Precision@K, NDCG@K, ROC-AUC, F1-Score) to ensure rigorous performance benchmarking beyond the original paper's reporting.
*   **Automated LLM Pipeline:** Engineered an end-to-end experimental pipeline:
    *   **PDF Extraction:** Automated parsing of raw CV and Job Market Paper (JMP) documents.
    *   **Anonymization:** Implemented name-redaction logic to mitigate identity-based 'look-ahead bias'.
    *   **Evaluation Engine:** Integrated Gemini API for zero-shot candidate evaluation with structured JSON output.

### **2. Key Challenges & Insights**
*   **Methodological Reconciliation:** Discovered a significant performance gap (NDCG 30.41%) when strictly implementing the paper’s stated IPW procedure due to missing granular institutional classification data. Empirically determined that undocumented SMOTE oversampling was critical to achieving the paper’s 50% Precision benchmark (Restored Baseline NDCG: 51.49%).
*   **Mitigating the 'Prestige Trap':** Initial testing revealed that vanilla zero-shot LLMs mirror human recruiter bias by over-indexing on elite PhD school prestige.
*   **Advanced Prompt Engineering:** Successfully developed a Chain-of-Thought (CoT) system prompt that de-biases the model, forcing an objective assessment of 'Research Momentum' (e.g., R&R status at Top 3 journals, reviewing experience, and doctoral awards).

### **3. Strategic Next Steps (Weeks 3-5)**
*   **Quantitative LLM Benchmarking:** Execute the full batch evaluation of the 2018 rookie cohort through the de-biased CoT pipeline to calculate comparative metrics.
*   **Performance Comparison:** Direct head-to-head comparison of LLM zero-shot/few-shot accuracy against the optimized SMOTE-ML baseline.
*   **Integration & Iteration:** Begin transitioning extracted LLM qualitative features into Gradient Boosted Trees (XGBoost/LightGBM) and initiate fine-tuning experiments on the rookie corpus.
