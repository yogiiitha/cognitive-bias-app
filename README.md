# CognitiveLens — AI-Based Cognitive Bias Detection System
### IIM Ranchi Summer Project 2025

> **Project Title:** AI-Based Identification of Cognitive Biases in Human Decision-Making
> and Their Role in Errors and Accidents

---

## 📁 Folder Structure

```
cognitive_bias_app/
│
├── app.py              ← The entire application (single file)
├── requirements.txt    ← Python package dependencies
└── README.md           ← This file
```

> ℹ️ The app is intentionally kept as a **single file** for simplicity,
> beginner-friendliness, and easy editing.

---

## ⚡ Quick Setup (Local)

### Step 1 — Install Python
Make sure Python 3.9 or above is installed.
Download from: https://www.python.org/downloads/

### Step 2 — Create a virtual environment (recommended)
```bash
python -m venv venv
source venv/bin/activate        # Mac / Linux
venv\Scripts\activate           # Windows
```

### Step 3 — Install dependencies
```bash
pip install -r requirements.txt
```

### Step 4 — Run the app
```bash
streamlit run app.py
```

The app will open in your browser at: http://localhost:8501

---

## ☁️ Deploy on Streamlit Community Cloud (Free)

Streamlit Cloud lets you host this app publicly for free.

### Step 1 — Push your code to GitHub
1. Create a new GitHub repository (e.g., `cognitive-bias-app`)
2. Upload `app.py` and `requirements.txt` to the repo root

### Step 2 — Deploy
1. Go to https://share.streamlit.io
2. Sign in with your GitHub account
3. Click **"New app"**
4. Select your repository, branch (`main`), and main file (`app.py`)
5. Click **"Deploy"**

Your app will be live at:
`https://your-username-cognitive-bias-app-app-XXXXX.streamlit.app`

---

## 🧠 How the App Works

### Question Selection
8 questions were selected from a 20-question behavioral survey administered
to 42 IIM Ranchi students. Selection criteria:
- High discriminative power (spread across all answer options)
- Strong behavioral signal per bias category
- Realistic, relatable scenarios (finance, workplace, travel, academics)
- Low social-desirability bias

### Scoring Framework
| Score | Signal Level | Meaning |
|-------|-------------|---------|
| 0     | Neutral     | Rational/unbiased response |
| 1     | Mild        | Slight bias tendency |
| 2     | Moderate    | Noticeable bias pattern |
| 3     | Strong      | Clear bias-driven decision |

### Biases Detected
| Code | Bias | Questions |
|------|------|-----------|
| CB   | Confirmation Bias     | Q1, Q2 |
| AB   | Anchoring Bias        | Q3, Q4 |
| OB   | Overconfidence Bias   | Q5, Q6 |
| AV   | Availability Heuristic | Q7, Q8 |

### Classification Logic
- Each bias gets a maximum possible raw score of 6 (2 questions × 3 max)
- Raw scores are normalized to a 0–100% intensity scale
- The bias with the highest % is classified as **dominant**
- The second-highest becomes the **secondary** bias
- Intensity label: Low (<25%), Mild (25–50%), Moderate (50–75%), Strong (75%+)

---

## 🚀 Future Improvements (AI/ML Roadmap)

### Phase 2 — Machine Learning Integration
- Train a **Naive Bayes or Logistic Regression** classifier on the
  collected survey response dataset
- Replace rule-based scoring with probabilistic classification
- Add confidence intervals to bias estimates

### Phase 3 — NLP Question Generation
- Use a language model to generate adaptive follow-up questions
  based on early response patterns (dynamic questionnaire)

### Phase 4 — Multi-Session Tracking
- Add user accounts and store response history
- Show bias trend over time (is the user improving?)
- Build a comparison dashboard (individual vs peer group)

### Phase 5 — Domain-Specific Profiles
- Separate bias profiles for financial decisions, medical decisions,
  HR decisions, engineering safety decisions
- Each domain has a curated question bank

---

## ⚠️ Limitations of This Prototype

1. **Small calibration dataset** — scoring logic is based on 42 student
   responses. Larger datasets would improve classification accuracy.

2. **Self-report limitation** — users may answer questions based on
   how they *want* to see themselves, not actual behavior.

3. **Context sensitivity** — cognitive biases are domain-specific;
   a person may be overconfident in tech but not in finance.

4. **Binary classification** — real cognitive profiles are continuous
   and overlapping; this tool produces a simplified categorical output.

5. **Cultural calibration** — the scenarios use Indian currency and
   workplace contexts; international generalizability is limited.

6. **No longitudinal tracking** — the current version is stateless;
   repeated assessments do not build a behavioral history.

---

## 📚 References

- Kahneman, D. (2011). *Thinking, Fast and Slow*. Farrar, Straus and Giroux.
- Tversky, A. & Kahneman, D. (1974). Judgment under uncertainty: Heuristics
  and biases. *Science*, 185(4157), 1124–1131.
- Gilovich, T., Griffin, D., & Kahneman, D. (2002).
  *Heuristics and Biases: The Psychology of Intuitive Judgment*. Cambridge.

---

*IIM Ranchi AI Summer Project 2025 | CognitiveLens v1.0*
