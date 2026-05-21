# =============================================================================
# AI-Based Cognitive Bias Detection & Behavioral Analysis System
# IIM Ranchi AI Summer Project | Annexure C / ACM-87
#
# Author: [Your Name] | Roll No: [Your Roll No]
# Supervisor: [Faculty Supervisor Name]
#
# ABOUT THE DATASET:
# The 8 questions below were selected from a 20-question behavioral survey
# administered to 42 IIM Ranchi students. Questions were selected based on:
#   1. High discriminative power (spread of responses across options)
#   2. Clear behavioral signal for the target bias
#   3. Realistic, relatable scenarios
#   4. Low social-desirability bias in the response options
# =============================================================================

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# ─────────────────────────────────────────────────────────────
# PAGE CONFIG — must be the very first Streamlit call
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="CognitiveLens | Bias Detection",
    page_icon="🧠",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# ─────────────────────────────────────────────────────────────
# CUSTOM CSS  — clean, minimal, academic-grade UI
# ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* Import Google Font */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    /* Remove default Streamlit padding */
    .block-container { padding-top: 1.5rem; padding-bottom: 2rem; max-width: 820px; }

    /* ── Gradient hero banner ── */
    .hero-banner {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
        border-radius: 16px;
        padding: 2.5rem 2rem;
        margin-bottom: 1.5rem;
        text-align: center;
        color: white;
    }
    .hero-banner h1 { font-size: 1.7rem; font-weight: 700; margin: 0.4rem 0; letter-spacing: -0.3px; }
    .hero-banner p  { font-size: 0.95rem; opacity: 0.82; margin: 0.2rem 0; }
    .hero-tag {
        display: inline-block;
        background: rgba(255,255,255,0.12);
        border: 1px solid rgba(255,255,255,0.2);
        border-radius: 20px;
        padding: 0.25rem 0.9rem;
        font-size: 0.78rem;
        margin-bottom: 0.8rem;
        letter-spacing: 0.5px;
        text-transform: uppercase;
    }

    /* ── Bias info cards ── */
    .bias-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 0.9rem; margin: 1.2rem 0; }
    .bias-card {
        border-radius: 12px;
        padding: 1.1rem 1.2rem;
        border-left: 4px solid;
    }
    .bias-card h4 { font-size: 0.9rem; font-weight: 600; margin: 0 0 0.3rem 0; }
    .bias-card p  { font-size: 0.8rem; margin: 0; line-height: 1.5; opacity: 0.85; }
    .card-cb { background: #fff8f0; border-color: #e67e22; color: #7d4500; }
    .card-ab { background: #f0f4ff; border-color: #3498db; color: #1a3a5c; }
    .card-ob { background: #f0fff4; border-color: #27ae60; color: #1a4d2e; }
    .card-av { background: #fdf0ff; border-color: #9b59b6; color: #4a1a6b; }

    /* ── Question card ── */
    .q-card {
        background: #f8f9fb;
        border-radius: 12px;
        padding: 1.4rem 1.5rem;
        margin-bottom: 1rem;
        border: 1px solid #e8ecf2;
    }
    .q-number {
        display: inline-block;
        background: #0f3460;
        color: white;
        border-radius: 6px;
        padding: 0.15rem 0.55rem;
        font-size: 0.75rem;
        font-weight: 600;
        margin-bottom: 0.6rem;
        letter-spacing: 0.3px;
    }
    .q-text { font-size: 1.0rem; font-weight: 500; color: #1a1a2e; line-height: 1.55; }

    /* ── Progress bar override ── */
    .stProgress > div > div { background: #0f3460; border-radius: 4px; }

    /* ── Result section cards ── */
    .result-primary {
        background: linear-gradient(135deg, #0f3460, #1a5276);
        color: white;
        border-radius: 14px;
        padding: 1.8rem;
        text-align: center;
        margin-bottom: 1rem;
    }
    .result-primary h2 { font-size: 1.4rem; margin: 0.3rem 0; }
    .result-primary .bias-name { font-size: 2rem; font-weight: 700; margin: 0.5rem 0; }
    .result-primary p { opacity: 0.88; font-size: 0.92rem; }

    .insight-card {
        background: white;
        border-radius: 12px;
        padding: 1.2rem 1.4rem;
        margin-bottom: 0.8rem;
        border: 1px solid #e8ecf2;
        box-shadow: 0 1px 4px rgba(0,0,0,0.05);
    }
    .insight-card h4 { font-size: 0.85rem; font-weight: 600; color: #666; text-transform: uppercase; letter-spacing: 0.5px; margin: 0 0 0.4rem 0; }
    .insight-card p  { font-size: 0.95rem; color: #2c3e50; margin: 0; line-height: 1.6; }

    .score-pill {
        display: inline-block;
        border-radius: 20px;
        padding: 0.3rem 1rem;
        font-size: 0.82rem;
        font-weight: 600;
        margin: 0.2rem;
    }

    /* ── Buttons ── */
    .stButton > button {
        background: linear-gradient(135deg, #0f3460, #1a5276);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 0.65rem 2.5rem;
        font-size: 1rem;
        font-weight: 600;
        transition: all 0.2s;
        width: 100%;
    }
    .stButton > button:hover { transform: translateY(-1px); box-shadow: 0 4px 15px rgba(15,52,96,0.35); }

    /* ── Divider ── */
    .section-divider { height: 1px; background: #e8ecf2; margin: 1.5rem 0; }

    /* ── Footer ── */
    .footer { text-align: center; font-size: 0.78rem; color: #888; margin-top: 2rem; padding-top: 1rem; border-top: 1px solid #eee; }

    /* Radio label override */
    .stRadio label { font-size: 0.95rem !important; }
    .stRadio > div { gap: 0.4rem; }

    /* Hide Streamlit default header */
    header[data-testid="stHeader"] { display: none; }
    #MainMenu, footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════
#  DATA LAYER — Questions, Options, and Scoring Schema
#
#  Question selection rationale (from 42-response survey):
#    Q1  (Survey Q1 ) — Confirmation Bias: best spread (A:20, C:11, D:6, B:5). Classic
#                       belief-preservation trigger. High signal.
#    Q5  (Survey Q5 ) — Confirmation Bias: stock falling scenario. B:25 vs D:9 shows
#                       strong commitment bias. Real financial stakes.
#    Q6  (Survey Q6 ) — Anchoring Bias: salary negotiation. First offer (₹40k) anchors
#                       counter-offers. Spread: A:16, C:12, D:10, B:4.
#    Q10 (Survey Q10) — Anchoring Bias: class average anchors personal target. Clear
#                       numeric anchor. A:17 vs B:16 — excellent discrimination.
#    Q17 (Survey Q17) — Overconfidence Bias: technical fix scenario. "Fix it yourself"
#                       response chosen by 24/42 — strong overconfidence signal.
#    Q18 (Survey Q18) — Overconfidence Bias: investor pitch projection. Mentor vs own
#                       estimate. D:28 shows most hedge, but B+C reveal overconfidence.
#    Q4  (Survey Q4 ) — Availability Heuristic: plane crash fear. Salient vivid event
#                       overrides statistics. B:27 vs A:12 — strong availability effect.
#    Q20 (Survey Q20) — Availability Heuristic: vivid board-room story overrides data.
#                       Best availability question. All 4 options well-distributed.
# ═══════════════════════════════════════════════════════════════════════════

QUESTIONS = [
    {
        "id": 1,
        "bias": "Confirmation Bias",
        "bias_code": "CB",
        "scenario_tag": "Health & Belief",
        "text": "You have been following a specific diet for 3 months and genuinely believe it is working for you. A close friend sends you a well-researched article arguing that the diet may be ineffective. You most likely:",
        "options": [
            ("A", "Read it carefully and genuinely reconsider whether the diet is working"),
            ("B", "Skim through it, mentally note the flaws, and stick with your diet"),
            ("C", "Search for other articles that support your diet to counter this one"),
            ("D", "Dismiss it — a single article is not enough to change your mind"),
        ],
        # Scoring: A = 0 (rational), B = 2 (mild CB), C = 3 (strong CB), D = 3 (strong CB)
        "scores": {"A": 0, "B": 2, "C": 3, "D": 3},
    },
    {
        "id": 2,
        "bias": "Confirmation Bias",
        "bias_code": "CB",
        "scenario_tag": "Finance & Investment",
        "text": "You bought shares of a company you strongly believe in. The stock has been declining for three consecutive weeks. You:",
        "options": [
            ("A", "Cut your losses, reassess objectively, and decide based on current data"),
            ("B", "Hold on — short-term dips are normal for fundamentally good stocks"),
            ("C", "Buy more shares since the price drop is a buying opportunity"),
            ("D", "Search for analyst opinions and news that confirm it will recover"),
        ],
        # A = 0 (rational), B = 1 (mild CB), C = 2 (moderate), D = 3 (strong CB — selective evidence-seeking)
        "scores": {"A": 0, "B": 1, "C": 2, "D": 3},
    },
    {
        "id": 3,
        "bias": "Anchoring Bias",
        "bias_code": "AB",
        "scenario_tag": "Salary Negotiation",
        "text": "A company opens a negotiation by offering you ₹40,000/month. Your research shows the role is worth ₹55,000. When asked for your counter-offer, you say:",
        "options": [
            ("A", "₹55,000 — that is the market rate and what you researched"),
            ("B", "₹45,000 — you do not want to seem too far from their offer"),
            ("C", "₹50,000 — split the difference between their offer and your target"),
            ("D", "Ask for time — you want to think before committing to a number"),
        ],
        # A = 0 (ignores anchor), B = 3 (strong anchor effect — pulled toward ₹40k), C = 2 (moderate), D = 1 (cautious)
        "scores": {"A": 0, "B": 3, "C": 2, "D": 1},
    },
    {
        "id": 4,
        "bias": "Anchoring Bias",
        "bias_code": "AB",
        "scenario_tag": "Academic Target-Setting",
        "text": "During orientation, a professor casually mentions that last year's class average on the final exam was 62%. When setting your personal target for the same exam, you aim for:",
        "options": [
            ("A", "90%+ — the class average is irrelevant to your personal goals"),
            ("B", "Around 70% — a bit above average sounds realistic and safe"),
            ("C", "Whatever you can manage given your actual study time and effort"),
            ("D", "65–68% — just comfortably above the mentioned average"),
        ],
        # A = 0, B = 2 (anchored to 62%), C = 1, D = 3 (strong anchor — very close to 62%)
        "scores": {"A": 0, "B": 2, "C": 1, "D": 3},
    },
    {
        "id": 5,
        "bias": "Overconfidence Bias",
        "bias_code": "OB",
        "scenario_tag": "Technical Problem-Solving",
        "text": "A technical issue arises at work that seems similar to one you successfully resolved six months ago. You:",
        "options": [
            ("A", "Research the current issue independently for at least 15 minutes before acting"),
            ("B", "Outline a potential fix but run it by a colleague before implementing"),
            ("C", "Fix it yourself immediately — you have done this before and know what to do"),
            ("D", "Escalate it to your team lead; it is better to be safe than sorry"),
        ],
        # A = 1, B = 0, C = 3 (strong OC — direct action based on past success), D = 1
        "scores": {"A": 1, "B": 0, "C": 3, "D": 1},
    },
    {
        "id": 6,
        "bias": "Overconfidence Bias",
        "bias_code": "OB",
        "scenario_tag": "Entrepreneurship & Estimation",
        "text": "You are pitching your app idea to investors. You project 50,000 users in 6 months. A mentor who has launched 3 apps before tells you that 8,000–10,000 is more realistic. You:",
        "options": [
            ("A", "Revise your projection — the mentor has real-world experience you lack"),
            ("B", "Stick to 50,000 — your research and analysis support that number"),
            ("C", "Present 25,000 as a 'conservative estimate' in the pitch deck"),
            ("D", "Run a small beta launch first to gather real data before the final pitch"),
        ],
        # A = 0, B = 3 (strong OC), C = 2 (moderate OC — still inflated), D = 1 (rational hedge)
        "scores": {"A": 0, "B": 3, "C": 2, "D": 1},
    },
    {
        "id": 7,
        "bias": "Availability Heuristic",
        "bias_code": "AV",
        "scenario_tag": "Risk Perception (Travel)",
        "text": "You just saw extensive news coverage of a plane crash. You have a scheduled work trip by air tomorrow. Your reaction is:",
        "options": [
            ("A", "Travel as planned — flying remains one of the statistically safest modes of transport"),
            ("B", "Feel uneasy throughout check-in and the flight but go anyway"),
            ("C", "Seriously consider switching to a train, even if it takes much longer"),
            ("D", "Postpone the trip until the feeling of anxiety fades"),
        ],
        # A = 0, B = 2 (moderate AV), C = 3 (strong AV), D = 3 (strong AV)
        "scores": {"A": 0, "B": 2, "C": 3, "D": 3},
    },
    {
        "id": 8,
        "bias": "Availability Heuristic",
        "bias_code": "AV",
        "scenario_tag": "Group Behavior & Decision-Making",
        "text": "In a board meeting, a senior member shares a vivid, detailed story about a competitor's recent failure caused by a risky expansion. The room becomes cautious and risk-averse. You:",
        "options": [
            ("A", "Point out that one anecdote may not represent a broader trend and ask for data"),
            ("B", "Factor in the story alongside other quantitative data before drawing conclusions"),
            ("C", "Agree that caution is warranted — real-world examples matter more than statistics"),
            ("D", "Ask for more case examples before forming any opinion on the matter"),
        ],
        # A = 0, B = 1, C = 3 (strong AV — vivid story overrides data), D = 2
        "scores": {"A": 0, "B": 1, "C": 3, "D": 2},
    },
]

# ─────────────────────────────────────────────────────────────
# BIAS METADATA — color palette, descriptions, interpretations
# ─────────────────────────────────────────────────────────────

BIAS_META = {
    "CB": {
        "name": "Confirmation Bias",
        "emoji": "🔍",
        "color": "#e67e22",
        "bg": "#fff8f0",
        "short_desc": "Tendency to favor information that confirms existing beliefs",
        "behavioral_profile": (
            "You tend to process information selectively, gravitating toward evidence that "
            "reinforces your existing beliefs while unconsciously minimizing or dismissing "
            "contradictory data. This pattern is especially active when your identity or "
            "prior decisions are invested in an outcome."
        ),
        "decision_tendencies": [
            "Seeks validating evidence before considering alternatives",
            "Interprets ambiguous information in a self-confirming way",
            "May maintain losing positions (financial or otherwise) too long",
            "Prone to echo chambers in information consumption",
        ],
        "real_world": (
            "In financial decisions, this may lead you to hold onto underperforming "
            "investments longer than optimal. In professional settings, it may cause you "
            "to under-evaluate critical feedback. In research and analysis, it can introduce "
            "systematic bias into your conclusions."
        ),
        "mitigation": (
            "Practice 'steel-manning' opposing views. Before making a decision, list at least "
            "3 strong reasons why you might be wrong. Seek out a 'red team' or devil's advocate."
        ),
    },
    "AB": {
        "name": "Anchoring Bias",
        "emoji": "⚓",
        "color": "#3498db",
        "bg": "#f0f4ff",
        "short_desc": "Over-reliance on the first piece of information encountered",
        "behavioral_profile": (
            "You show a strong tendency to use the first number, reference point, or piece of "
            "information you encounter as a mental anchor, allowing it to disproportionately "
            "influence subsequent judgments and decisions — even when the anchor is arbitrary "
            "or irrelevant to the actual value in question."
        ),
        "decision_tendencies": [
            "Negotiations are heavily influenced by whoever makes the first offer",
            "Estimates (time, cost, performance) cluster around initial reference points",
            "Discounts and 'original vs sale price' framing strongly affects purchasing",
            "Adjustments from anchors tend to be insufficient",
        ],
        "real_world": (
            "In salary negotiations, you may accept sub-optimal offers because the initial "
            "number anchors your counter. In project planning, early estimates become implicit "
            "commitments. In pricing decisions, list prices set the perceived 'baseline value.'"
        ),
        "mitigation": (
            "Before any negotiation or estimate, independently determine your target value. "
            "Delay responding to first offers — write down your own number before hearing theirs. "
            "Always ask: 'What would I say if I had never heard that number?'"
        ),
    },
    "OB": {
        "name": "Overconfidence Bias",
        "emoji": "🎯",
        "color": "#27ae60",
        "bg": "#f0fff4",
        "short_desc": "Overestimating one's abilities, accuracy, or control over outcomes",
        "behavioral_profile": (
            "You display a tendency to overestimate the accuracy of your knowledge, the "
            "quality of your predictions, or your ability to control outcomes. This often "
            "manifests as underestimating risks, overestimating the transferability of past "
            "experience to new situations, and setting overly optimistic projections."
        ),
        "decision_tendencies": [
            "Consistently underestimates task duration (planning fallacy)",
            "Applies past solutions to new problems without adequate re-evaluation",
            "Sets ambitious targets that do not account for unknown variables",
            "Tends to attribute past successes to skill rather than context or luck",
        ],
        "real_world": (
            "In entrepreneurship, this often leads to over-optimistic financial projections. "
            "In project management, it manifests as chronic under-estimation of timelines. "
            "In technical roles, it can cause skipped verification steps that lead to errors."
        ),
        "mitigation": (
            "Use 'reference class forecasting' — look at base rates for similar projects. "
            "Maintain a decision journal to track calibration over time. "
            "Build in explicit uncertainty buffers and red-team your assumptions."
        ),
    },
    "AV": {
        "name": "Availability Heuristic",
        "emoji": "📰",
        "color": "#9b59b6",
        "bg": "#fdf0ff",
        "short_desc": "Judging probability by how easily examples come to mind",
        "behavioral_profile": (
            "You tend to judge the likelihood or importance of events based on how easily "
            "vivid examples or recent experiences come to mind, rather than on statistical "
            "base rates. Emotionally striking or widely-publicized events distort your "
            "risk perception and decision-making in predictable ways."
        ),
        "decision_tendencies": [
            "Risk perception is strongly influenced by recent news and vivid stories",
            "Overestimates frequency of dramatic but rare events (plane crashes, crimes)",
            "Peer experiences and anecdotes carry disproportionate evidential weight",
            "Narrative and emotional resonance can override statistical reasoning",
        ],
        "real_world": (
            "In investment decisions, fear from a friend's loss or a viral news story may "
            "lead to avoidance of objectively sound opportunities. In strategic planning, "
            "a vivid competitor failure story in a boardroom can trigger excessive caution "
            "even when data suggests the risk is low."
        ),
        "mitigation": (
            "Develop a habit of asking 'What does the data actually say?' before acting on "
            "a vivid impression. Use base-rate statistics as your first frame of reference. "
            "Pause 24 hours after emotionally charged information before making decisions."
        ),
    },
}

# ─────────────────────────────────────────────────────────────
# CLASSIFICATION ENGINE
# ─────────────────────────────────────────────────────────────

def compute_scores(answers: dict) -> dict:
    """
    Rule-based AI classification engine.
    
    Each answer is scored 0–3 against its target bias.
    Score of 0 = rational/unbiased response
    Score of 3 = strong bias signal
    
    Returns a dict: {bias_code: raw_score, ...}
    """
    bias_totals = {"CB": 0, "AB": 0, "OB": 0, "AV": 0}
    for q in QUESTIONS:
        qid = q["id"]
        if qid in answers:
            selected_letter = answers[qid]
            raw_score = q["scores"].get(selected_letter, 0)
            bias_totals[q["bias_code"]] += raw_score
    return bias_totals


def normalize_scores(raw: dict) -> dict:
    """
    Convert raw scores (0–6 per bias, 2 questions × max 3 pts) to
    percentages for display. Max possible per bias = 6.
    """
    MAX_PER_BIAS = 6  # 2 questions × max score of 3
    return {k: round((v / MAX_PER_BIAS) * 100, 1) for k, v in raw.items()}


def classify_bias(raw_scores: dict):
    """
    Returns (dominant_bias_code, secondary_bias_code, intensity_label).
    Intensity is based on dominant score percentage.
    """
    norm = normalize_scores(raw_scores)
    sorted_biases = sorted(norm.items(), key=lambda x: -x[1])
    
    dominant_code   = sorted_biases[0][0]
    secondary_code  = sorted_biases[1][0]
    dominant_pct    = sorted_biases[0][1]
    
    if dominant_pct >= 75:
        intensity = "Strong"
    elif dominant_pct >= 50:
        intensity = "Moderate"
    elif dominant_pct >= 25:
        intensity = "Mild"
    else:
        intensity = "Low"
    
    return dominant_code, secondary_code, intensity, norm


def generate_interpretation(dominant_code: str, secondary_code: str,
                             intensity: str, norm_scores: dict) -> dict:
    """
    Generates personalized textual interpretations for the results page.
    """
    d = BIAS_META[dominant_code]
    s = BIAS_META[secondary_code]

    # Intensity-modulated opener
    intensity_phrases = {
        "Strong":   "Your behavioral profile reveals a pronounced tendency toward",
        "Moderate": "Your response pattern shows a clear inclination toward",
        "Mild":     "Your decisions show a noticeable but moderate lean toward",
        "Low":      "Your responses suggest relatively balanced decision-making, with some traces of",
    }
    opener = intensity_phrases[intensity]

    headline = f"{opener} **{d['name']}** ({d['emoji']})"
    
    dominant_pct   = norm_scores[dominant_code]
    secondary_pct  = norm_scores[secondary_code]
    
    combined_note = ""
    if secondary_pct >= 40:
        combined_note = (
            f"Additionally, your responses also reflect elements of **{s['name']}** "
            f"({s['emoji']}), suggesting a compound bias pattern that may compound "
            f"decision errors in high-stakes contexts."
        )

    return {
        "headline":         headline,
        "behavioral_profile": d["behavioral_profile"],
        "decision_tendencies": d["decision_tendencies"],
        "real_world":       d["real_world"],
        "mitigation":       d["mitigation"],
        "combined_note":    combined_note,
        "dominant_pct":     dominant_pct,
        "secondary_pct":    secondary_pct,
    }


# ─────────────────────────────────────────────────────────────
# CHART RENDERER
# ─────────────────────────────────────────────────────────────

def render_chart(norm_scores: dict, dominant_code: str):
    """
    Renders a horizontal bar chart with per-bias color coding.
    Returns a Matplotlib figure.
    """
    bias_labels = [BIAS_META[k]["name"] for k in ["CB", "AB", "OB", "AV"]]
    values      = [norm_scores[k] for k in ["CB", "AB", "OB", "AV"]]
    colors      = [BIAS_META[k]["color"] for k in ["CB", "AB", "OB", "AV"]]
    alphas      = [1.0 if k == dominant_code else 0.55 for k in ["CB", "AB", "OB", "AV"]]

    fig, ax = plt.subplots(figsize=(7, 3.2))
    fig.patch.set_facecolor("#f8f9fb")
    ax.set_facecolor("#f8f9fb")

    bars = ax.barh(
        bias_labels, values,
        color=[c for c in colors],
        height=0.52,
        edgecolor="none",
        alpha=0.9,
    )

    # Apply individual alphas
    for bar, alpha, val in zip(bars, alphas, values):
        bar.set_alpha(alpha)
        # Inline score label
        ax.text(val + 1.5, bar.get_y() + bar.get_height() / 2,
                f"{val:.0f}%",
                va='center', ha='left',
                fontsize=9, color='#555', fontweight='500')

    # Dominant bias star marker
    dom_idx = ["CB", "AB", "OB", "AV"].index(dominant_code)
    ax.text(-3.5, dom_idx, "★", va='center', ha='center',
            fontsize=11, color=BIAS_META[dominant_code]["color"])

    ax.set_xlim(-5, 115)
    ax.set_xlabel("Bias Intensity Score (%)", fontsize=9, color='#777', labelpad=8)
    ax.tick_params(axis='y', labelsize=9, colors='#333')
    ax.tick_params(axis='x', labelsize=8, colors='#999')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_color('#ddd')
    ax.xaxis.set_tick_params(length=0)
    ax.axvline(x=50, color='#ddd', linewidth=0.8, linestyle='--')

    ax.set_title("Cognitive Bias Intensity Profile", fontsize=11,
                 fontweight='600', color='#1a1a2e', pad=12)

    plt.tight_layout(pad=1.2)
    return fig


# ═══════════════════════════════════════════════════════════════
#  SESSION STATE INIT
# ═══════════════════════════════════════════════════════════════

if "page" not in st.session_state:
    st.session_state.page = "home"      # pages: home | quiz | results
if "answers" not in st.session_state:
    st.session_state.answers = {}
if "current_q" not in st.session_state:
    st.session_state.current_q = 0      # 0-indexed


# ═══════════════════════════════════════════════════════════════
#  PAGE 1: HOME
# ═══════════════════════════════════════════════════════════════

def page_home():
    # Hero banner
    st.markdown("""
    <div class="hero-banner">
        <div class="hero-tag">🎓 IIM Ranchi · AI Summer Project</div>
        <h1>🧠 CognitiveLens</h1>
        <h1 style="font-size:1.1rem; font-weight:400; opacity:0.85;">
            AI-Based Cognitive Bias Detection &amp; Behavioral Analysis System
        </h1>
        <p style="margin-top:0.8rem; font-size:0.82rem; opacity:0.65;">
            A rule-based behavioral profiling prototype · 8 scenario questions · ~5 minutes
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="insight-card">
        <h4>About this tool</h4>
        <p>
            CognitiveLens uses <strong>scenario-based behavioral questions</strong> to identify which
            cognitive biases most influence your decisions. It does not measure intelligence or personality —
            it maps your <em>systematic reasoning patterns</em> under realistic, contextual pressure.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("#### The Four Biases Assessed")
    st.markdown("""
    <div class="bias-grid">
        <div class="bias-card card-cb">
            <h4>🔍 Confirmation Bias</h4>
            <p>Favoring information that validates what you already believe, while unconsciously discounting contradictory evidence.</p>
        </div>
        <div class="bias-card card-ab">
            <h4>⚓ Anchoring Bias</h4>
            <p>Over-relying on the first number or fact you encounter, letting it disproportionately influence subsequent judgments.</p>
        </div>
        <div class="bias-card card-ob">
            <h4>🎯 Overconfidence Bias</h4>
            <p>Overestimating the accuracy of your knowledge, skills, or projections — especially when experience feels sufficient.</p>
        </div>
        <div class="bias-card card-av">
            <h4>📰 Availability Heuristic</h4>
            <p>Judging likelihood by how easily vivid examples come to mind, rather than by actual statistical frequency.</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="insight-card" style="background:#f0f4ff; border-color:#c8d8f0;">
        <h4>⚙️ How the AI Classification Works</h4>
        <p>
            Your responses are mapped against a <strong>behavioral scoring framework</strong> calibrated on
            42 survey responses collected from IIM Ranchi students. Each answer carries a bias-signal weight (0–3).
            The system aggregates signals across questions to produce a bias intensity profile, then identifies
            your dominant behavioral tendency using a rule-based classification engine.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("")
    if st.button("▶  Start the Assessment", key="start_btn"):
        st.session_state.page = "quiz"
        st.session_state.answers = {}
        st.session_state.current_q = 0
        st.rerun()

    st.markdown("""
    <div class="footer">
        "AI-Based Identification of Cognitive Biases in Human Decision-Making" ·
        IIM Ranchi Summer Project 2025 · Built with Python + Streamlit
    </div>
    """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════
#  PAGE 2: QUIZ
# ═══════════════════════════════════════════════════════════════

def page_quiz():
    total = len(QUESTIONS)
    current = st.session_state.current_q
    q = QUESTIONS[current]

    # ── Header ──
    st.markdown("""
    <div style="text-align:center; margin-bottom:1rem;">
        <span style="font-size:0.85rem; color:#888;">🧠 CognitiveLens · Behavioral Assessment</span>
    </div>
    """, unsafe_allow_html=True)

    # ── Progress ──
    progress_pct = current / total
    st.progress(progress_pct)
    col1, col2 = st.columns([1, 1])
    with col1:
        st.markdown(f"<span style='font-size:0.82rem; color:#666;'>Question {current + 1} of {total}</span>",
                    unsafe_allow_html=True)
    with col2:
        bias_color = BIAS_META[q["bias_code"]]["color"]
        st.markdown(
            f"<span style='font-size:0.78rem; color:{bias_color}; "
            f"background:{BIAS_META[q['bias_code']]['bg']}; "
            f"padding:0.15rem 0.5rem; border-radius:8px; float:right;'>"
            f"{BIAS_META[q['bias_code']]['emoji']} {q['scenario_tag']}</span>",
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Question card ──
    st.markdown(f"""
    <div class="q-card">
        <div class="q-number">Q{q['id']} / {total}</div>
        <div class="q-text">{q['text']}</div>
    </div>
    """, unsafe_allow_html=True)

    # ── Radio options ──
    # Build labels with letter prefix for clarity
    option_labels = [f"{letter})  {text}" for letter, text in q["options"]]
    letter_map    = {f"{letter})  {text}": letter for letter, text in q["options"]}

    # Pre-select if already answered
    default_idx = None
    if q["id"] in st.session_state.answers:
        prev_letter = st.session_state.answers[q["id"]]
        for i, (letter, text) in enumerate(q["options"]):
            if letter == prev_letter:
                default_idx = i
                break

    selected_label = st.radio(
        "Choose the response that best reflects your natural tendency:",
        options=option_labels,
        index=default_idx,
        key=f"q_{q['id']}",
        label_visibility="visible",
    )

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Navigation ──
    nav_col1, nav_col2, nav_col3 = st.columns([1, 2, 1])

    with nav_col1:
        if current > 0:
            if st.button("← Back", key="back_btn"):
                st.session_state.current_q -= 1
                st.rerun()

    with nav_col3:
        is_last = (current == total - 1)
        btn_label = "View Results →" if is_last else "Next →"

        if st.button(btn_label, key="next_btn"):
            if selected_label is None:
                st.warning("Please select an option before continuing.")
            else:
                chosen_letter = letter_map[selected_label]
                st.session_state.answers[q["id"]] = chosen_letter
                if is_last:
                    st.session_state.page = "results"
                    st.rerun()
                else:
                    st.session_state.current_q += 1
                    st.rerun()

    # ── Answered tracker ──
    answered = len(st.session_state.answers)
    st.markdown(f"""
    <div style="text-align:center; margin-top:1.5rem;">
        <span style="font-size:0.78rem; color:#aaa;">{answered} of {total} questions answered</span>
    </div>
    """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════
#  PAGE 3: RESULTS
# ═══════════════════════════════════════════════════════════════

def page_results():
    # ── Compute scores ──
    raw_scores = compute_scores(st.session_state.answers)
    dom_code, sec_code, intensity, norm_scores = classify_bias(raw_scores)
    interp = generate_interpretation(dom_code, sec_code, intensity, norm_scores)
    d = BIAS_META[dom_code]
    s = BIAS_META[sec_code]

    # ── Primary result banner ──
    dom_pct = interp["dominant_pct"]
    sec_pct = interp["secondary_pct"]

    st.markdown(f"""
    <div class="result-primary">
        <div style="font-size:0.78rem; opacity:0.7; text-transform:uppercase; letter-spacing:1px;">
            Dominant Cognitive Bias Detected
        </div>
        <div class="bias-name">{d['emoji']} {d['name']}</div>
        <div style="font-size:0.9rem; opacity:0.85; margin-bottom:0.6rem;">
            {intensity} signal · {dom_pct:.0f}% intensity score
        </div>
        <div style="font-size:0.82rem; opacity:0.7;">
            Secondary: {s['emoji']} {s['name']} ({sec_pct:.0f}%)
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Score breakdown chart ──
    st.markdown("#### Bias Intensity Profile")
    fig = render_chart(norm_scores, dom_code)
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

    # ── Score pills ──
    st.markdown("<div style='margin:0.5rem 0 1.2rem 0;'>", unsafe_allow_html=True)
    for code in ["CB", "AB", "OB", "AV"]:
        color = BIAS_META[code]["color"]
        pct   = norm_scores[code]
        is_dom = "★ " if code == dom_code else ""
        st.markdown(
            f"<span class='score-pill' style='background:{BIAS_META[code]['bg']}; "
            f"color:{color}; border:1.5px solid {color};'>"
            f"{is_dom}{BIAS_META[code]['emoji']} {BIAS_META[code]['name']}: {pct:.0f}%</span>",
            unsafe_allow_html=True,
        )
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div class='section-divider'></div>", unsafe_allow_html=True)

    # ── Behavioral Profile ──
    st.markdown("#### 🧬 Your Behavioral Profile")
    st.markdown(f"""
    <div class="insight-card">
        <h4>Profile Summary</h4>
        <p>{interp['behavioral_profile']}</p>
    </div>
    """, unsafe_allow_html=True)

    if interp["combined_note"]:
        st.markdown(f"""
        <div class="insight-card" style="background:{s['bg']}; border-left: 3px solid {s['color']};">
            <h4>Compound Bias Pattern</h4>
            <p>{interp['combined_note']}</p>
        </div>
        """, unsafe_allow_html=True)

    # ── Decision tendencies ──
    st.markdown("#### 📊 Key Decision-Making Tendencies")
    for tendency in d["decision_tendencies"]:
        st.markdown(f"""
        <div class="insight-card" style="padding:0.8rem 1.2rem;">
            <p>▸ &nbsp; {tendency}</p>
        </div>
        """, unsafe_allow_html=True)

    # ── Real-world implications ──
    st.markdown("#### 🌐 Real-World Implications")
    st.markdown(f"""
    <div class="insight-card" style="background:{d['bg']}; border-left: 3px solid {d['color']};">
        <h4>How this affects your decisions</h4>
        <p>{d['real_world']}</p>
    </div>
    """, unsafe_allow_html=True)

    # ── Mitigation ──
    st.markdown("#### 🛡️ Evidence-Based Mitigation Strategies")
    st.markdown(f"""
    <div class="insight-card">
        <h4>What you can do</h4>
        <p>{d['mitigation']}</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<div class='section-divider'></div>", unsafe_allow_html=True)

    # ── Answer review table ──
    st.markdown("#### 📋 Your Response Summary")
    review_data = []
    for q in QUESTIONS:
        qid = q["id"]
        ans_letter = st.session_state.answers.get(qid, "—")
        ans_text = next((txt for l, txt in q["options"] if l == ans_letter), "—")
        raw_score = q["scores"].get(ans_letter, 0)
        signal = ["⬜ Neutral", "🟡 Mild", "🟠 Moderate", "🔴 Strong"][raw_score]
        review_data.append({
            "Q#": f"Q{qid}",
            "Bias Tested": BIAS_META[q["bias_code"]]["emoji"] + " " + q["bias"],
            "Your Answer": f"{ans_letter}) {ans_text[:55]}...",
            "Signal": signal,
        })

    df = pd.DataFrame(review_data)
    st.dataframe(df, use_container_width=True, hide_index=True)

    st.markdown("<div class='section-divider'></div>", unsafe_allow_html=True)

    # ── Retake button ──
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("🔄  Retake Assessment", key="retake_btn"):
            st.session_state.page = "home"
            st.session_state.answers = {}
            st.session_state.current_q = 0
            st.rerun()

    st.markdown("""
    <div class="footer">
        <strong>CognitiveLens</strong> · AI Summer Project · IIM Ranchi 2025<br>
        "AI-Based Identification of Cognitive Biases in Human Decision-Making and Their Role in Errors and Accidents"<br>
        <span style="color:#bbb;">Results are for academic and reflective purposes only.</span>
    </div>
    """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════
#  ROUTER — render the correct page based on session state
# ═══════════════════════════════════════════════════════════════

if st.session_state.page == "home":
    page_home()
elif st.session_state.page == "quiz":
    page_quiz()
elif st.session_state.page == "results":
    page_results()
