# Web Deployment PRD: Academic Forecasting Prediction Engine

## 0. User Session Authentication & View Routing Matrix
**The Gateway Selector:** Upon rendering the root page, the application must deploy a high-visibility, stateful selector (e.g., `st.radio`) forcing the user to define their authorization role: 
- `[ 💼 Recruiter / Faculty Committee ]`
- `[ 🎓 Applicant / Student Researcher ]`

**B-End Route (Recruiter Mode):** 
- Activates the multi-file parsing architecture.
- Renders the interactive `st.data_editor` batch grid for manual overrides.
- Outputs the $K=8$ color-coded Top Candidate Leaderboard.
- **Privacy Lock:** Explicitly hides all counterfactual optimization diagnostic panels to protect interface density and maintain a sterile hiring environment.

**C-End Route (Applicant Mode):** 
- Activates single-file parsing and diagnostics.
- Renders a continuous publishing gauge chart showing current placement odds.
- Opens the interactive **What-If Simulation Panel**. This allows applicants to toggle sliders (e.g., `number of coauthors`, data novelty, causal identification) and receive real-time delta probability metrics via `st.metric`.

## 1. Data Schema & In-Memory Parsing Pipeline
### 1.1 Explicit Feature Schema
The backend extraction engine must extract and pass a complete, structured mathematical feature array to the core IPW prediction classifier. The schema must strictly enforce the following column definitions and data types:
1. `Candidate_Name` (String): Extracted from the PDF header or metadata.
2. `Placerank` (Float): Continuous variable (1.0 to 50.0) representing PhD institution pedigree.
3. `gender` (Float): Binary representation (`1.0` for Male, `0.0` for Female).
4. `number of coauthors` (Integer): Represents scholarly network density.
5. `num_pub` (Integer): Baseline historical publication count prior to the JMP.
6. `has_novel_data` (Binary Integer): `1` if the JMP contains explicitly proprietary/hand-collected data, `0` otherwise.
7. `explicit_causal_id` (Binary Integer): `1` if the text explicitly outlines IV, DiD, or quasi-experimental setups.
8. `cross_disciplinary_method` (Binary Integer): `1` if the paper leverages ML, NLP, or computer science frameworks.
9. `sample_size_over_10k` (Binary Integer): `1` if the empirical sample $N$ is stated as $>10,000$.

### 1.2 State Management & Async Pipeline
- Text parsing and LLM extraction must execute as isolated background processes. If a PDF extraction fails, exceeds timeout, or returns `null`, the UI must not crash. It must surface a validation alert, inject default schema values, and prompt manual override via the UI grid.

## 2. B-End Batch Alignment Matrix (Corporate Route)
### 2.1 UI Data Editor Specification
- **Batch Upload:** Uses `st.file_uploader(accept_multiple_files=True)`.
- **Session_UUID Binding:** The backend must bind a unique internal `Session_UUID` to every uploaded file row to prevent index collisions if identical or abbreviated candidate names are batch-processed.
- **Validation Grid:** Post-parsing, data is piped into a mutable `st.data_editor` grid displaying the full schema. Recruiters possess the explicit capability to override any incorrectly parsed feature manually before executing inference.

### 2.2 Top-K Leaderboard Logic
- **Evaluation Lock:** The leaderboard output is mathematically locked to a strict **K=8 bracket**.
- **Presentation:** Renders a Pandas DataFrame sorted by the computed `Final_Probability` in descending order. The Top 8 rows are strictly color-coded (e.g., highlighted green) to demarcate the elite bracket.

## 3. C-End Counterfactual Simulation Engine (Diagnostic Route)
### 3.1 Post-Hoc Multi-Variable Counterfactual Simulator
- If the applicant's base probability is below `0.50`, the UI renders an interactive "What-If Simulation Panel".
- **Boundary Constraints:** The UI must enforce mathematical safety limits on the counterfactual sliders (e.g., locking `Coauthor_Count` max at 10, `num_pub` max at 5) to keep simulated variables strictly within the model's legitimate feature distribution and prevent OOD (Out-of-Distribution) variance.
- **Algorithmic Execution:** When a user manipulates a widget (e.g., adding a coauthor or setting causal identification text to 1), the script programmatically clones their feature vector, re-runs the IPW classification function, and dynamically renders the delta metric using `st.metric(label="Predicted Probability", value=new_prob, delta=prob_shift)` to quantify the immediate percentage shift in their potential.

## 4. Cross-Domain Standardization Dictionary
### 4.1 Explicit Translation Logic
- A domain selection dropdown (Accounting, Computer Science, Finance) applies a Zero-Retraining Translation Strategy mapping external domains to the baseline causal topology.
- **Dynamic UI Form Refactoring:** Changing the domain dropdown to `Computer Science` must programmatically use `st.empty()` to clear the page state, physically hiding the traditional institutional rank slider and dynamically rendering the "Top-Tier Conference Publications" number input.
- **Example Implementation (CS Domain):** If `Computer Science` is active, traditional institutional ranking inputs are replaced by "Top-Tier Conference Publications".
- $\ge 3$ CS Papers maps to `Placerank = 10.0` and `has_novel_data = 1`.
- $1-2$ CS Papers maps to `Placerank = 25.0` and `has_novel_data = 0`.
- $0$ CS Papers maps to `Placerank = 50.0`.
- This structural equivalence allows the core econometric baseline to operate reliably across multiple scientific fields.
