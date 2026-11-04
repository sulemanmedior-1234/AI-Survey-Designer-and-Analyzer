"""
AI Survey Designer and Analyzer
Part 1 - Design: describe your research objective and the AI drafts survey
         questions. The app checks them for common wording problems and exports the
         survey and a blank response sheet.
Part 2 - Analyze: upload the responses as a CSV. Pandas calculates the statistics,
         charts show the results, and the AI finds themes in the written answers.

Design notes:
  - All numbers (averages, counts, percentages) are calculated by pandas, never by the AI.
  - The AI only designs questions and groups written feedback into themes. The theme
    counts and example quotes are produced by the code from the AI's assignments.
  - Statistics and charts work without an API key. Only the AI parts need one.

Uses the Groq REST API directly through `requests`, so the `groq` package is
not needed.

Run locally:  streamlit run app.py
"""

import json
import os
import random
import re
from datetime import datetime

import pandas as pd
import requests
import streamlit as st

# --------------------------------------------------------------------------- #
# Page configuration
# --------------------------------------------------------------------------- #
st.set_page_config(page_title="AI Survey Designer & Analyzer", page_icon="📋", layout="wide")

GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
MODELS = ["llama-3.3-70b-versatile", "llama-3.1-8b-instant"]

QUESTION_TYPES = ["Single choice", "Multiple choice", "Likert (5-point)", "Rating (0-10)", "Yes/No", "Open text"]
COLUMN_TYPES = ["Number", "Scale", "Choice", "Multi-select", "Text", "Skip"]
LIKERT_DEFAULT = ["Strongly disagree", "Disagree", "Neutral", "Agree", "Strongly agree"]
MAX_TEXT_RESPONSES = 100

SAMPLE_OBJECTIVE = (
    "Find out how satisfied university students are with the campus cafeteria, "
    "what they value most, and what should be improved."
)
SAMPLE_AUDIENCE = "University students"

# HELPERS START
# ----------------------------- Survey design -------------------------------- #
LEADING_PHRASES = [
    "don't you", "do not you", "wouldn't you", "isn't it", "do you agree that", "obviously",
    "clearly", "everyone knows", "how great", "how amazing", "terrible", "excellent",
]
ABSOLUTE_WORDS = ["always", "never", "all", "every", "none", "everyone", "nobody"]


def match_question_type(value):
    v = str(value).strip().lower()
    if "likert" in v or "agree" in v or "scale" in v:
        return "Likert (5-point)"
    if "multi" in v or "select all" in v or "checkbox" in v:
        return "Multiple choice"
    if "rating" in v or "0-10" in v or "nps" in v or "number" in v:
        return "Rating (0-10)"
    if "yes" in v:
        return "Yes/No"
    if "open" in v or "text" in v or "free" in v or "comment" in v:
        return "Open text"
    if "single" in v or "choice" in v or "multiple choice" in v or "radio" in v:
        return "Single choice"
    return "Single choice"


def default_options(qtype):
    if qtype == "Likert (5-point)":
        return list(LIKERT_DEFAULT)
    if qtype == "Yes/No":
        return ["Yes", "No"]
    if qtype == "Rating (0-10)":
        return [str(i) for i in range(11)]
    return []


def split_options(text):
    return [o.strip() for o in re.split(r"[;\n]", str(text or "")) if o.strip()]


def clean_options(qtype, options):
    """Fill in standard options for scales, keep the AI's own for choice questions."""
    if isinstance(options, str):
        options = split_options(options)
    options = [str(o).strip() for o in (options or []) if str(o).strip()]
    if qtype in ("Yes/No", "Rating (0-10)", "Open text"):
        return default_options(qtype)
    if qtype == "Likert (5-point)" and len(options) != 5:
        return default_options(qtype)
    return options


def validate_design(data, wanted):
    """Turn the AI's reply into a clean survey. Returns dict with title, intro, rows, notes."""
    if not isinstance(data, dict):
        raise ValueError("The AI reply was not in the expected format.")
    rows = []
    for q in data.get("questions") or []:
        if not isinstance(q, dict):
            continue
        text = str(q.get("text", "") or "").strip()
        if not text:
            continue
        qtype = match_question_type(q.get("type", ""))
        rows.append(
            {
                "Question": text,
                "Type": qtype,
                "Options": "; ".join(clean_options(qtype, q.get("options"))) if qtype != "Rating (0-10)" else "0 to 10",
                "Required": bool(q.get("required", True)),
                "Purpose": str(q.get("purpose", "") or "").strip(),
            }
        )
    if not rows:
        raise ValueError("The AI reply was not in the expected format.")
    notes = [str(n).strip() for n in (data.get("design_notes") or []) if str(n).strip()]
    return {
        "title": str(data.get("title", "") or "").strip() or "Survey",
        "intro": str(data.get("intro", "") or "").strip(),
        "rows": rows[:wanted + 3],
        "notes": notes,
    }


def lint_questions(rows):
    """Plain checks for common survey-writing problems. Returns a list of (question number, message)."""
    issues = []
    seen = {}
    open_count = 0
    for i, r in enumerate(rows, start=1):
        text = str(r.get("Question", "")).strip()
        low = text.lower()
        qtype = r.get("Type", "")
        words = len(text.split())
        options = split_options(r.get("Options", ""))

        if re.search(r"\band\b", low) and qtype != "Open text":
            issues.append((i, "May be double-barreled (it contains 'and'). Ask about one thing at a time."))
        for phrase in LEADING_PHRASES:
            if phrase in low:
                issues.append((i, "May be a leading question ('" + phrase + "'). Use neutral wording."))
                break
        found = [w for w in ABSOLUTE_WORDS if re.search(r"\b" + w + r"\b", low)]
        if found and qtype in ("Likert (5-point)", "Yes/No"):
            issues.append((i, "Absolute word '" + found[0] + "' makes it hard to answer honestly."))
        if words > 25:
            issues.append((i, "Long question (" + str(words) + " words). Shorter questions get better answers."))
        if qtype in ("Single choice", "Multiple choice"):
            if len(options) < 2:
                issues.append((i, "Needs at least two answer options."))
            elif not any(o.lower().startswith(("other", "none", "not sure", "prefer not")) for o in options):
                issues.append((i, "Consider adding an 'Other' or 'None of these' option."))
        if qtype == "Open text":
            open_count += 1
        key = re.sub(r"\W+", " ", low).strip()
        if key in seen:
            issues.append((i, "Repeats question " + str(seen[key]) + "."))
        else:
            seen[key] = i

    if len(rows) > 15:
        issues.append((0, "This is a long survey (" + str(len(rows)) + " questions). Completion rates drop after about 10 to 15."))
    if open_count > 3:
        issues.append((0, "Many open-text questions (" + str(open_count) + "). They tire respondents. Keep to 2 or 3."))
    if rows and rows[0].get("Type") == "Open text":
        issues.append((1, "Start with an easy closed question, not an open-text one."))
    return issues


def build_design_messages(objective, audience, n_questions, types, tone):
    system_msg = (
        "You are an expert survey methodologist. You write clear, neutral, unbiased survey "
        "questions that each measure one thing. Reply with valid JSON only."
    )
    user_msg = (
        "Design a survey.\n\n"
        "RESEARCH OBJECTIVE: " + objective + "\n"
        "AUDIENCE: " + (audience or "general") + "\n"
        "NUMBER OF QUESTIONS: " + str(n_questions) + "\n"
        "ALLOWED QUESTION TYPES: " + ", ".join(types) + "\n"
        "TONE: " + tone + "\n\n"
        "Return a JSON object with exactly this shape:\n"
        "{\n"
        '  "title": "short survey title",\n'
        '  "intro": "2-3 sentence introduction for respondents (purpose, time needed, anonymity)",\n'
        '  "questions": [\n'
        "    {\n"
        '      "text": "the question",\n'
        '      "type": one of the allowed types,\n'
        '      "options": ["answer options. Leave empty for Open text, Rating (0-10), Yes/No and Likert (5-point)"],\n'
        '      "required": true or false,\n'
        '      "purpose": "what this question helps to learn, in a few words"\n'
        "    }\n"
        "  ],\n"
        '  "design_notes": ["1-3 short tips about how to run this survey well"]\n'
        "}\n\n"
        "Rules:\n"
        "- Every question must help answer the research objective.\n"
        "- Ask about ONE thing per question. No double-barreled questions.\n"
        "- Use neutral wording. No leading questions and no absolute words like always or never.\n"
        "- Start with easy questions and put sensitive ones later. Put open-text questions last.\n"
        "- Use at most 2 open-text questions.\n"
        "- Likert (5-point) questions must be statements to agree or disagree with.\n"
        "- Choice options must not overlap and should include an 'Other' option when needed.\n"
        "- Write for the audience's reading level.\n"
    )
    return [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": user_msg},
    ]


def survey_markdown(title, intro, rows):
    lines = ["# " + title, ""]
    if intro:
        lines += [intro, ""]
    for i, r in enumerate(rows, start=1):
        req = "" if r.get("Required", True) else " (optional)"
        lines.append(str(i) + ". **" + str(r["Question"]) + "**" + req)
        qtype = r.get("Type", "")
        options = split_options(r.get("Options", ""))
        if qtype == "Open text":
            lines.append("   - _Write your answer_")
        elif qtype == "Rating (0-10)":
            lines.append("   - Choose a number from 0 (lowest) to 10 (highest)")
        else:
            box = "[ ]" if qtype == "Multiple choice" else "( )"
            if qtype == "Multiple choice":
                lines.append("   - _Select all that apply_")
            for o in options:
                lines.append("   - " + box + " " + o)
        lines.append("")
    return "\n".join(lines)


def response_template_csv(rows):
    """A blank response sheet: one column per question."""
    cols = ["Respondent ID"] + [str(r["Question"]) for r in rows]
    return pd.DataFrame(columns=cols).to_csv(index=False).encode("utf-8-sig")


# ----------------------------- Response analysis ---------------------------- #
LIKERT_SCALES = {
    "Agreement": ["strongly disagree", "disagree", "neutral", "agree", "strongly agree"],
    "Satisfaction": ["very dissatisfied", "dissatisfied", "neutral", "satisfied", "very satisfied"],
    "Frequency": ["never", "rarely", "sometimes", "often", "always"],
    "Quality": ["very poor", "poor", "average", "good", "excellent"],
    "Likelihood": ["very unlikely", "unlikely", "neutral", "likely", "very likely"],
    "Importance": ["not at all important", "slightly important", "moderately important", "very important", "extremely important"],
}
NEUTRAL_VARIANTS = {
    "neither agree nor disagree": "neutral",
    "neither satisfied nor dissatisfied": "neutral",
    "neither likely nor unlikely": "neutral",
    "neutral": "neutral",
}
SKIP_NAME_RE = re.compile(r"\b(id|respondent|timestamp|submitted|submission|email|e-mail|phone|name|ip)\b", re.IGNORECASE)


def clean_label(value):
    v = str(value).strip().lower()
    return NEUTRAL_VARIANTS.get(v, v)


def detect_scale(values):
    """If every answer is a standard rating label, return the scale name."""
    labels = {clean_label(v) for v in values}
    if len(labels) < 2:
        return ""
    best, best_hits = "", 0
    for name, scale in LIKERT_SCALES.items():
        if labels <= set(scale):
            hits = len(labels)
            if hits > best_hits:
                best, best_hits = name, hits
    return best


def answered(series):
    s = series.dropna()
    return s[s.astype(str).str.strip() != ""]


def detect_column_type(name, series):
    """Guess what kind of question a column is. Returns (type, scale name)."""
    s = answered(series)
    if s.empty:
        return "Skip", ""
    if len(str(name)) <= 25 and SKIP_NAME_RE.search(str(name)):
        return "Skip", ""
    if pd.api.types.is_numeric_dtype(s):
        return "Number", ""
    text = s.astype(str).str.strip()
    as_number = pd.to_numeric(text, errors="coerce")
    if as_number.notna().mean() >= 0.9:
        return "Number", ""
    scale = detect_scale(text)
    if scale:
        return "Scale", scale
    if text.str.contains(";").mean() >= 0.3:
        return "Multi-select", ""
    n = len(text)
    avg_len = text.str.len().mean()
    unique = text.nunique()
    if avg_len >= 30 or (n >= 8 and unique / n > 0.7 and avg_len >= 15):
        return "Text", ""
    if unique <= max(12, int(0.25 * n)):
        return "Choice", ""
    return ("Text" if avg_len >= 15 else "Choice"), ""


def to_numbers(series):
    return pd.to_numeric(answered(series), errors="coerce").dropna()


def numeric_summary(series):
    x = to_numbers(series)
    if x.empty:
        return None
    return {
        "Answered": int(len(x)),
        "Mean": round(float(x.mean()), 2),
        "Median": round(float(x.median()), 2),
        "Std dev": round(float(x.std(ddof=1)), 2) if len(x) > 1 else 0.0,
        "Min": round(float(x.min()), 2),
        "Max": round(float(x.max()), 2),
    }


def numeric_distribution(series):
    x = to_numbers(series)
    if x.empty:
        return pd.Series(dtype=float)
    if (x % 1 == 0).all() and x.nunique() <= 12:
        return x.astype(int).value_counts().sort_index()
    bins = pd.cut(x, bins=min(8, max(2, x.nunique())))
    counts = bins.value_counts(sort=False)
    counts.index = [round((iv.left + iv.right) / 2, 1) for iv in counts.index]
    return counts


def scale_scores(series, scale_name):
    """Turn rating labels into numbers 1 to 5. Unknown labels become missing."""
    order = LIKERT_SCALES[scale_name]
    mapping = {label: i + 1 for i, label in enumerate(order)}
    return answered(series).map(lambda v: mapping.get(clean_label(v))).dropna().astype(int)


def scale_distribution(series, scale_name):
    """Counts in scale order, with labels that keep their order in charts."""
    order = LIKERT_SCALES[scale_name]
    scores = scale_scores(series, scale_name)
    counts = scores.value_counts().reindex(range(1, 6), fill_value=0)
    counts.index = [str(i) + " - " + order[i - 1].capitalize() for i in range(1, 6)]
    return counts


def scale_summary(series, scale_name):
    scores = scale_scores(series, scale_name)
    if scores.empty:
        return None
    return {
        "Answered": int(len(scores)),
        "Mean (1-5)": round(float(scores.mean()), 2),
        "Favourable % (4-5)": round(float((scores >= 4).mean() * 100), 1),
        "Unfavourable % (1-2)": round(float((scores <= 2).mean() * 100), 1),
    }


def choice_counts(series):
    s = answered(series).astype(str).str.strip()
    return s.value_counts()


def multi_counts(series, sep=";"):
    """Counts of each option; percent is of respondents who answered."""
    s = answered(series).astype(str)
    parts = s.str.split(sep).explode().str.strip()
    parts = parts[parts != ""]
    counts = parts.value_counts()
    return counts, int(len(s))


def group_means(df, group_col, measure, measure_kind, scale_name=""):
    """Average of a Number or Scale column for each group."""
    if measure_kind == "Scale":
        order = LIKERT_SCALES[scale_name]
        mapping = {label: i + 1 for i, label in enumerate(order)}
        values = df[measure].map(lambda v: mapping.get(clean_label(v)) if pd.notna(v) else None)
    else:
        values = pd.to_numeric(df[measure], errors="coerce")
    frame = pd.DataFrame({"Group": df[group_col].astype(str).str.strip(), "Value": values})
    frame = frame[(frame["Group"] != "") & (frame["Group"].str.lower() != "nan")].dropna(subset=["Value"])
    out = frame.groupby("Group")["Value"].agg(["count", "mean"]).reset_index()
    out.columns = ["Group", "Answered", "Average"]
    out["Average"] = out["Average"].round(2)
    return out.sort_values("Average", ascending=False).reset_index(drop=True)


# ----------------------------- Written feedback ----------------------------- #
SENTIMENTS = ["Positive", "Neutral", "Negative"]


def build_theme_messages(question, responses):
    system_msg = (
        "You are a careful qualitative researcher. You group written survey answers into "
        "themes. You only use what the respondents wrote. Reply with valid JSON only."
    )
    lines = [str(i) + ": " + r.replace("\n", " ")[:300] for i, r in enumerate(responses, start=1)]
    user_msg = (
        "Survey question: " + question + "\n\n"
        "Group the numbered responses below.\n\n"
        "Return a JSON object with exactly this shape:\n"
        "{\n"
        '  "themes": [{"name": "short theme name", "description": "one sentence"}],\n'
        '  "assignments": [{"id": 1, "theme": "a theme name from the list", "sentiment": "Positive" or "Neutral" or "Negative"}],\n'
        '  "summary": "3-5 sentences summarising what respondents said",\n'
        '  "recommendations": ["2-4 specific actions suggested by the feedback"]\n'
        "}\n\n"
        "Rules:\n"
        "- Create 3 to 8 themes. Name them clearly and do not overlap them.\n"
        "- Give every response exactly one theme (its main point) and one sentiment.\n"
        "- Include every response id once.\n"
        "- Do not state numbers or percentages in the summary. The app counts them.\n"
        "- Do not invent things respondents did not say.\n\n"
        "RESPONSES:\n" + "\n".join(lines)
    )
    return [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": user_msg},
    ]


def extract_json(text):
    """Parse JSON even if the model wrapped it in code fences or extra text."""
    try:
        return json.loads(text)
    except Exception:
        pass
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        return json.loads(match.group(0))
    raise ValueError("The AI did not return valid JSON.")


def match_label(value, allowed, default):
    v = str(value).strip().lower()
    for item in allowed:
        if v == item.lower():
            return item
    return default


def validate_themes(data, responses):
    """Count themes and sentiments in code, and pick real quotes as examples."""
    if not isinstance(data, dict):
        raise ValueError("The AI reply was not in the expected format.")
    theme_info = {}
    for t in data.get("themes") or []:
        if isinstance(t, dict) and str(t.get("name", "")).strip():
            theme_info[str(t["name"]).strip()] = str(t.get("description", "") or "").strip()
    if not theme_info:
        raise ValueError("The AI reply was not in the expected format.")
    lower_to_name = {n.lower(): n for n in theme_info}

    assigned, sentiments = {}, {}
    for a in data.get("assignments") or []:
        if not isinstance(a, dict):
            continue
        try:
            rid = int(a.get("id"))
        except Exception:
            continue
        if 1 <= rid <= len(responses):
            name = lower_to_name.get(str(a.get("theme", "")).strip().lower())
            assigned[rid] = name or "Other"
            sentiments[rid] = match_label(a.get("sentiment", ""), SENTIMENTS, "Neutral")
    if not assigned:
        raise ValueError("The AI reply was not in the expected format.")

    total = len(responses)
    themes = []
    names = list(theme_info) + (["Other"] if "Other" in assigned.values() and "Other" not in theme_info else [])
    for name in names:
        ids = [i for i, t in assigned.items() if t == name]
        if not ids:
            continue
        quotes = sorted((responses[i - 1] for i in ids), key=lambda r: abs(len(r) - 90))[:2]
        themes.append(
            {
                "name": name,
                "description": theme_info.get(name, "Responses that did not fit another theme."),
                "count": len(ids),
                "percent": round(len(ids) / total * 100, 1),
                "examples": quotes,
            }
        )
    themes.sort(key=lambda t: -t["count"])
    counts = {s: sum(1 for v in sentiments.values() if v == s) for s in SENTIMENTS}
    return {
        "summary": str(data.get("summary", "") or "").strip(),
        "recommendations": [str(r).strip() for r in (data.get("recommendations") or []) if str(r).strip()],
        "themes": themes,
        "sentiment": counts,
        "classified": len(assigned),
        "total": total,
    }


# ----------------------------- Report --------------------------------------- #
def build_stats_table(df, types, scales):
    """One tidy row per question with the headline numbers."""
    rows = []
    for col, kind in types.items():
        if kind == "Skip":
            continue
        n_answered = int(len(answered(df[col])))
        row = {"Question": col, "Type": kind, "Answered": n_answered, "Missing": int(len(df) - n_answered)}
        if kind == "Number":
            s = numeric_summary(df[col]) or {}
            row.update({"Mean": s.get("Mean"), "Median": s.get("Median"), "Std dev": s.get("Std dev"),
                        "Min": s.get("Min"), "Max": s.get("Max")})
        elif kind == "Scale" and scales.get(col):
            s = scale_summary(df[col], scales[col]) or {}
            row.update({"Mean": s.get("Mean (1-5)"), "Favourable %": s.get("Favourable % (4-5)"),
                        "Unfavourable %": s.get("Unfavourable % (1-2)")})
        elif kind == "Choice":
            c = choice_counts(df[col])
            if len(c):
                row.update({"Top answer": c.index[0], "Top answer %": round(c.iloc[0] / max(n_answered, 1) * 100, 1)})
        elif kind == "Multi-select":
            c, base = multi_counts(df[col])
            if len(c):
                row.update({"Top answer": c.index[0], "Top answer %": round(c.iloc[0] / max(base, 1) * 100, 1)})
        rows.append(row)
    return rows


def build_report(source, n_responses, stats_rows, theme_results):
    stamp = datetime.now().strftime("%Y-%m-%d")
    lines = ["# Survey Results", "", "_" + source + " | " + str(n_responses) + " responses | " + stamp + "_", "", "## Statistics", ""]
    for r in stats_rows:
        extras = ", ".join(str(k) + ": " + str(v) for k, v in r.items()
                           if k not in ("Question", "Type", "Answered", "Missing") and v is not None and str(v) != "nan")
        lines.append("- **" + r["Question"] + "** (" + r["Type"] + ", " + str(r["Answered"]) + " answered)" + (": " + extras if extras else ""))
    for question, res in theme_results.items():
        lines += ["", "## Written feedback: " + question, "", res["summary"] or "-", "", "**Themes**"]
        for t in res["themes"]:
            lines.append("- " + t["name"] + ": " + str(t["count"]) + " (" + str(t["percent"]) + "%). " + t["description"])
            for q in t["examples"]:
                lines.append('  - "' + q + '"')
        s = res["sentiment"]
        lines += ["", "**Sentiment:** " + ", ".join(k + " " + str(v) for k, v in s.items())]
        if res["recommendations"]:
            lines += ["", "**Suggested actions**"] + ["- " + r for r in res["recommendations"]]
    lines.append("")
    return "\n".join(lines)


def friendly_error(exc):
    msg = str(exc)
    low = msg.lower()
    if "401" in msg or "invalid api key" in low or "invalid_api_key" in low:
        return "Your Groq API key looks invalid. Please check it and try again."
    if "429" in msg or "rate limit" in low:
        return "Groq rate limit reached. Please wait a minute and try again."
    if "decommissioned" in low or "does not exist" in low or "404" in msg:
        return (
            "That model is not available on your account. Choose another model in the "
            "sidebar. Details: " + msg[:300]
        )
    if "connection" in low or "timeout" in low or "timed out" in low:
        return "Could not reach the Groq API. Check your internet connection and retry."
    if "json" in low or "expected format" in low:
        return "The AI reply could not be read. Please try again."
    return "Something went wrong: " + msg
# HELPERS END


def sample_responses_df():
    """Fictional cafeteria survey responses, so the analyzer can be tried straight away."""
    rng = random.Random(7)
    departments = ["Computer Science", "Business", "Engineering", "Psychology"]
    sat_labels = LIKERT_SCALES["Satisfaction"]
    agree_labels = LIKERT_SCALES["Agreement"]
    meals = ["Breakfast", "Lunch", "Dinner", "Snacks"]
    comments = [
        "Prices are too high for students, a simple lunch costs almost as much as a restaurant meal.",
        "The queue at lunch time is very long. I often wait 20 minutes and miss part of my next class.",
        "Please add more variety. The menu is the same every week.",
        "More healthy options like salads and fruit would be great.",
        "The tables and dining area are not always clean.",
        "Opening hours are too short. It closes before my evening classes end.",
        "Staff are friendly but the service is slow when it is busy.",
        "Portions are small compared to the price.",
        "I would like vegetarian and vegan choices every day.",
        "Not enough seating at lunch, we end up eating standing up.",
        "The food is tasty and I like the fresh bread, please keep it.",
        "Prices keep going up every semester. Some student discount would help.",
        "Need a faster way to order, maybe an app, to skip the line.",
        "The hygiene of the serving area should be improved.",
        "Great staff and good taste. Just add more drink options.",
        "Breakfast options are very limited and often sold out by 9 am.",
    ]
    rows = []
    for i in range(1, 49):
        dept = rng.choices(departments, weights=[3, 2, 3, 1])[0]
        base = {"Computer Science": 3, "Business": 2, "Engineering": 1, "Psychology": 3}[dept]
        sat = min(4, max(0, int(rng.gauss(base + 0.6, 1.0))))
        price = min(4, max(0, int(rng.gauss(1.6, 1.0))))
        days = min(7, max(0, int(rng.gauss(3.2, 1.6))))
        picked = [m for m in meals if rng.random() < (0.85 if m == "Lunch" else 0.35)] or ["Lunch"]
        recommend = min(10, max(0, int(rng.gauss(4 + sat * 1.2, 1.8))))
        comment = rng.choice(comments) if rng.random() > 0.08 else None
        rows.append(
            {
                "Respondent ID": i,
                "Department": dept,
                "How satisfied are you with the campus cafeteria?": sat_labels[sat],
                "The prices are reasonable": agree_labels[price],
                "How many days per week do you eat at the cafeteria?": days,
                "Which meals do you usually buy? (select all that apply)": "; ".join(picked),
                "How likely are you to recommend the cafeteria to a friend? (0-10)": recommend,
                "What is the one thing you would improve?": comment,
            }
        )
    return pd.DataFrame(rows)


def get_api_key(sidebar_value):
    """Priority: sidebar input -> st.secrets -> environment variable."""
    if sidebar_value and sidebar_value.strip():
        return sidebar_value.strip()
    try:
        secret = st.secrets["GROQ_API_KEY"]
        if secret:
            return str(secret).strip()
    except Exception:
        pass
    return os.environ.get("GROQ_API_KEY", "").strip()


@st.cache_data(ttl=600, show_spinner=False)
def fetch_models(api_key):
    """Ask Groq which chat models this API key can use right now."""
    try:
        r = requests.get(
            "https://api.groq.com/openai/v1/models",
            headers={"Authorization": "Bearer " + api_key},
            timeout=20,
        )
        if r.status_code != 200:
            return []
        skip = ("whisper", "guard", "tts", "playai", "orpheus", "safeguard", "embed")
        ids = [
            m["id"]
            for m in r.json().get("data", [])
            if m.get("active", True) and not any(s in m["id"].lower() for s in skip)
        ]
        return sorted(ids)
    except Exception:
        return []


def call_groq(api_key, model, messages):
    payload = {
        "model": model,
        "messages": messages,
        "temperature": 0.3,
        "response_format": {"type": "json_object"},
    }
    headers = {"Authorization": "Bearer " + api_key, "Content-Type": "application/json"}
    resp = requests.post(GROQ_URL, headers=headers, json=payload, timeout=120)

    # Some models do not support JSON mode: retry once without it.
    if resp.status_code == 400 and "response_format" in resp.text:
        payload.pop("response_format")
        resp = requests.post(GROQ_URL, headers=headers, json=payload, timeout=120)

    if resp.status_code != 200:
        raise RuntimeError("Groq API error " + str(resp.status_code) + ": " + resp.text[:300])
    choices = resp.json().get("choices") or []
    if not choices:
        return ""
    return (choices[0].get("message", {}).get("content") or "").strip()


def read_csv(uploaded):
    """Read an uploaded CSV, trying common encodings."""
    last_error = None
    for enc in ("utf-8", "utf-8-sig", "latin-1"):
        try:
            uploaded.seek(0)
            return pd.read_csv(uploaded, encoding=enc)
        except Exception as exc:
            last_error = exc
    raise ValueError("Could not read the CSV file: " + str(last_error))


# --------------------------------------------------------------------------- #
# Session state and callbacks
# --------------------------------------------------------------------------- #
for key, value in {
    "objective": "",
    "audience": "",
    "design": None,
    "design_run": 0,
    "use_sample": False,
    "themes": {},
    "themes_sig": "",
}.items():
    if key not in st.session_state:
        st.session_state[key] = value


def load_design_example():
    st.session_state["objective"] = SAMPLE_OBJECTIVE
    st.session_state["audience"] = SAMPLE_AUDIENCE


def load_sample_responses():
    st.session_state["use_sample"] = True


# --------------------------------------------------------------------------- #
# Sidebar
# --------------------------------------------------------------------------- #
with st.sidebar:
    st.header("⚙️ Settings")
    key_input = st.text_input(
        "Groq API key (optional)",
        type="password",
        help="Leave empty to use st.secrets or the GROQ_API_KEY environment variable. "
        "Statistics and charts work without a key. Only the AI parts need one.",
    )
    api_key = get_api_key(key_input)
    if api_key:
        st.success("API key found ✅")
    else:
        st.warning("No API key found yet")

    live_models = fetch_models(api_key) if api_key else []
    model_options = live_models or MODELS
    default_index = 0
    for preferred in ("llama-3.3-70b-versatile", "llama-3.1-8b-instant"):
        if preferred in model_options:
            default_index = model_options.index(preferred)
            break
    model = st.selectbox("Groq model", model_options, index=default_index)
    if live_models:
        st.caption("Models loaded from your Groq account.")

    st.info(
        "Privacy: written answers are sent to Groq for theme analysis. "
        "Remove names and personal details from your data first."
    )

# --------------------------------------------------------------------------- #
# Main page
# --------------------------------------------------------------------------- #
st.title("📋 AI Survey Designer and Analyzer")
st.write(
    "Design a survey from your research objective, then upload the responses to get "
    "statistics, charts and a summary of the written feedback."
)

tab_design, tab_analyze = st.tabs(["1️⃣ Design a survey", "2️⃣ Analyze responses"])

# =========================================================================== #
# DESIGN
# =========================================================================== #
with tab_design:
    st.button("📋 Load an example", on_click=load_design_example, key="ex_design")
    st.text_area(
        "Research objective",
        height=110,
        key="objective",
        placeholder="What do you want to find out? For example: understand why customers stop using our app.",
    )
    c1, c2, c3 = st.columns(3)
    with c1:
        st.text_input("Audience", key="audience", placeholder="Who will answer? e.g. university students")
    with c2:
        n_questions = st.slider("Number of questions", 5, 20, 10)
    with c3:
        tone = st.selectbox("Tone", ["Friendly", "Professional", "Simple (easy reading)"])
    wanted_types = st.multiselect("Question types to use", QUESTION_TYPES, default=QUESTION_TYPES)

    if st.button("✨ Generate survey", type="primary"):
        objective = st.session_state["objective"].strip()
        if not api_key:
            st.error(
                "No Groq API key found. Enter it in the sidebar, or add GROQ_API_KEY "
                "to your Streamlit secrets."
            )
        elif len(objective) < 20:
            st.warning("Please describe your research objective in a sentence or two.")
        elif not wanted_types:
            st.warning("Please choose at least one question type.")
        else:
            try:
                with st.spinner("Designing your survey..."):
                    raw = call_groq(
                        api_key, model,
                        build_design_messages(objective, st.session_state["audience"].strip(), n_questions, wanted_types, tone),
                    )
                    st.session_state["design"] = validate_design(extract_json(raw), n_questions)
                    st.session_state["design_run"] += 1
            except Exception as exc:
                st.error(friendly_error(exc))

    design = st.session_state["design"]
    if design:
        run = str(st.session_state["design_run"])
        st.divider()
        st.subheader("Your survey")
        st.warning("This is an AI draft. Read every question and edit anything that is unclear or biased.")

        title = st.text_input("Survey title", value=design["title"], key="d_title_" + run)
        intro = st.text_area("Introduction for respondents", value=design["intro"], height=90, key="d_intro_" + run)

        st.caption("Click a cell to edit. Use the + row at the bottom to add a question.")
        edited = st.data_editor(
            pd.DataFrame(design["rows"], columns=["Question", "Type", "Options", "Required", "Purpose"]),
            num_rows="dynamic",
            key="d_editor_" + run,
            column_config={
                "Question": st.column_config.TextColumn("Question", width="large"),
                "Type": st.column_config.SelectboxColumn("Type", options=QUESTION_TYPES, width="medium"),
                "Options": st.column_config.TextColumn("Options (separate with ;)", width="large"),
                "Required": st.column_config.CheckboxColumn("Required"),
                "Purpose": st.column_config.TextColumn("Purpose", width="medium"),
            },
        )
        rows = [r for r in edited.fillna("").to_dict("records") if str(r.get("Question", "")).strip()]
        for r in rows:
            r["Type"] = r["Type"] if r["Type"] in QUESTION_TYPES else "Single choice"
            if r["Type"] in ("Yes/No", "Rating (0-10)", "Open text") or (r["Type"] == "Likert (5-point)" and not str(r["Options"]).strip()):
                r["Options"] = "; ".join(default_options(r["Type"])) if r["Type"] != "Rating (0-10)" else "0 to 10"

        st.markdown("#### Question quality checks")
        issues = lint_questions(rows)
        if not issues:
            st.success("No common wording problems found.")
        for num, message in issues:
            st.warning(("Question " + str(num) + ": " if num else "") + message)
        for note in design["notes"]:
            st.info("💡 " + note)

        d1, d2 = st.columns(2)
        d1.download_button(
            "⬇️ Download survey (.md)",
            data=survey_markdown(title, intro, rows),
            file_name="survey.md",
            mime="text/markdown",
        )
        d2.download_button(
            "⬇️ Download blank response sheet (CSV)",
            data=response_template_csv(rows),
            file_name="survey_responses_template.csv",
            mime="text/csv",
        )
        st.caption(
            "Collect answers in Google Forms, Microsoft Forms or a spreadsheet, using the blank sheet's columns, "
            "then upload the finished file in the next tab."
        )

# =========================================================================== #
# ANALYZE
# =========================================================================== #
with tab_analyze:
    uploaded = st.file_uploader("Upload survey responses (CSV: one row per respondent, one column per question)", type=["csv"])
    st.button("📋 Use sample responses", on_click=load_sample_responses, key="ex_resp")

    df, source = None, ""
    if uploaded is not None:
        try:
            df = read_csv(uploaded)
            source = uploaded.name
        except Exception as exc:
            st.error(str(exc))
    elif st.session_state["use_sample"]:
        df = sample_responses_df()
        source = "Sample cafeteria survey"
        st.caption("Showing fictional sample data. Upload a file to analyze your own.")

    if df is None or df.empty:
        st.info("Upload a CSV, or click **Use sample responses** to try the analyzer.")
    else:
        df = df.dropna(how="all").reset_index(drop=True)
        signature = source + "|" + str(df.shape)
        if st.session_state["themes_sig"] != signature:
            st.session_state["themes"] = {}
            st.session_state["themes_sig"] = signature

        st.subheader("Overview")
        o1, o2, o3 = st.columns(3)
        o1.metric("Responses", len(df))
        o2.metric("Questions (columns)", len(df.columns))
        o3.metric("Blank cells", f"{round(df.isna().mean().mean() * 100, 1)}%")
        with st.expander("Preview of your data"):
            st.dataframe(df.head(15))

        # --- Column types the teacher can correct ---
        detected = {c: detect_column_type(c, df[c]) for c in df.columns}
        st.subheader("Question types")
        st.caption("The app guessed the type of each column. Change any that are wrong.")
        type_table = st.data_editor(
            pd.DataFrame(
                {
                    "Question": list(df.columns),
                    "Type": [detected[c][0] for c in df.columns],
                    "Answered": [int(len(answered(df[c]))) for c in df.columns],
                }
            ),
            key="types_" + signature,
            hide_index=True,
            disabled=["Question", "Answered"],
            column_config={"Type": st.column_config.SelectboxColumn("Type", options=COLUMN_TYPES)},
        )
        types = dict(zip(type_table["Question"], type_table["Type"]))

        scales = {}
        for col, kind in types.items():
            if kind == "Scale":
                name = detect_scale(answered(df[col]).astype(str))
                if name:
                    scales[col] = name
                else:
                    types[col] = "Choice"
                    st.caption("'" + str(col) + "' does not use a standard rating scale, so it is shown as a Choice.")

        t_stats, t_compare, t_text, t_export = st.tabs(
            ["📊 Statistics", "⚖️ Compare groups", "💬 Written feedback", "⬇️ Download"]
        )

        # ----------------------------- Statistics ----------------------------- #
        with t_stats:
            shown = 0
            for col, kind in types.items():
                if kind in ("Skip", "Text"):
                    continue
                shown += 1
                with st.expander(str(col) + "  ·  " + kind, expanded=shown <= 3):
                    if kind == "Number":
                        s = numeric_summary(df[col])
                        if s:
                            cols = st.columns(len(s))
                            for c, (k, v) in zip(cols, s.items()):
                                c.metric(k, v)
                            st.bar_chart(numeric_distribution(df[col]))
                        else:
                            st.info("No numeric answers found.")
                    elif kind == "Scale":
                        s = scale_summary(df[col], scales[col])
                        if s:
                            cols = st.columns(len(s))
                            for c, (k, v) in zip(cols, s.items()):
                                c.metric(k, v)
                            st.bar_chart(scale_distribution(df[col], scales[col]))
                            st.caption(scales[col] + " scale, scored 1 (lowest) to 5 (highest).")
                    elif kind == "Choice":
                        counts = choice_counts(df[col])
                        base = int(counts.sum()) or 1
                        st.bar_chart(counts)
                        st.dataframe(
                            pd.DataFrame({"Answer": counts.index, "Responses": counts.values,
                                          "Percent": (counts.values / base * 100).round(1)}),
                            hide_index=True,
                        )
                    elif kind == "Multi-select":
                        counts, base = multi_counts(df[col])
                        st.bar_chart(counts)
                        st.dataframe(
                            pd.DataFrame({"Option": counts.index, "Chosen by": counts.values,
                                          "% of respondents": (counts.values / max(base, 1) * 100).round(1)}),
                            hide_index=True,
                        )
                        st.caption("Respondents could pick several options, so percentages add up to more than 100.")
            if shown == 0:
                st.info("No numeric or choice questions were found.")

        # ----------------------------- Compare -------------------------------- #
        with t_compare:
            group_cols = [c for c, k in types.items() if k == "Choice"]
            measure_cols = [c for c, k in types.items() if k in ("Number", "Scale")]
            if not group_cols or not measure_cols:
                st.info("Comparing groups needs at least one Choice question (for example Department) and one Number or Scale question.")
            else:
                g1, g2 = st.columns(2)
                group_col = g1.selectbox("Compare groups of", group_cols)
                measure_col = g2.selectbox("Average of", measure_cols)
                result = group_means(df, group_col, measure_col, types[measure_col], scales.get(measure_col, ""))
                if result.empty:
                    st.info("No data to compare.")
                else:
                    st.bar_chart(result.set_index("Group")["Average"])
                    st.dataframe(result, hide_index=True)
                    if (result["Answered"] < 5).any():
                        st.warning("Some groups have fewer than 5 answers. Treat those averages with care.")
                    st.caption("These are descriptive averages. Small differences may be due to chance.")

        # ----------------------------- Written feedback ----------------------- #
        with t_text:
            text_cols = [c for c, k in types.items() if k == "Text"]
            if not text_cols:
                st.info("No written-answer questions were found. Set a column's type to Text above if it is one.")
            if text_cols and not api_key:
                st.warning("Add your Groq API key in the sidebar to analyze written feedback.")
            for col in text_cols:
                responses = [str(x).strip() for x in answered(df[col])]
                with st.expander(str(col) + "  ·  " + str(len(responses)) + " answers", expanded=True):
                    if st.button("✨ Find themes", key="themes_btn_" + signature + "_" + str(col)):
                        if not api_key:
                            st.error("No Groq API key found. Enter it in the sidebar.")
                        elif len(responses) < 3:
                            st.warning("Need at least 3 written answers.")
                        else:
                            used = responses[:MAX_TEXT_RESPONSES]
                            try:
                                with st.spinner("Reading the answers..."):
                                    raw = call_groq(api_key, model, build_theme_messages(str(col), used))
                                    st.session_state["themes"][col] = validate_themes(extract_json(raw), used)
                                if len(responses) > MAX_TEXT_RESPONSES:
                                    st.info("Only the first " + str(MAX_TEXT_RESPONSES) + " answers were analyzed.")
                            except Exception as exc:
                                st.error(friendly_error(exc))

                    res = st.session_state["themes"].get(col)
                    if res:
                        st.markdown(res["summary"])
                        s = res["sentiment"]
                        s1, s2, s3 = st.columns(3)
                        s1.metric("Positive", s["Positive"])
                        s2.metric("Neutral", s["Neutral"])
                        s3.metric("Negative", s["Negative"])
                        st.markdown("**Themes**")
                        st.bar_chart({t["name"]: t["count"] for t in res["themes"]})
                        for t in res["themes"]:
                            st.markdown(
                                "**" + t["name"] + "** · " + str(t["count"]) + " answers (" + str(t["percent"]) + "%)  \n"
                                + t["description"]
                            )
                            for q in t["examples"]:
                                st.caption("“" + q + "”")
                        if res["recommendations"]:
                            st.markdown("**Suggested actions**")
                            for r in res["recommendations"]:
                                st.markdown("- " + r)
                        st.caption(
                            "Counts and quotes are produced by the app from the AI's grouping. "
                            "The quotes are real answers, not AI-written."
                        )

        # ----------------------------- Download ------------------------------- #
        with t_export:
            stats_rows = build_stats_table(df, types, scales)
            st.dataframe(pd.DataFrame(stats_rows), hide_index=True)
            e1, e2 = st.columns(2)
            e1.download_button(
                "⬇️ Download statistics (CSV)",
                data=pd.DataFrame(stats_rows).to_csv(index=False).encode("utf-8-sig"),
                file_name="survey_statistics.csv",
                mime="text/csv",
            )
            e2.download_button(
                "⬇️ Download full report (.md)",
                data=build_report(source, len(df), stats_rows, st.session_state["themes"]),
                file_name="survey_report.md",
                mime="text/markdown",
            )
            st.caption("The full report includes the written-feedback themes you have generated.")
