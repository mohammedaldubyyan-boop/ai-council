import os
import re
import json
import glob
import time
import math
import difflib
import streamlit as st

from groq import Groq
from fpdf import FPDF


# =========================================================
# BUILD
# =========================================================

BUILD_ID = "V5.1-RESEARCH-COUNCIL"


# =========================================================
# PAGE
# =========================================================

st.set_page_config(
    page_title="MD Investment Council",
    page_icon="🧠",
    layout="wide",
)

st.markdown(
    """
    <style>
    .stApp { direction: rtl; }
    h1, h2, h3, h4, h5, p { text-align: right; }
    div[data-testid="stMarkdownContainer"] { direction: rtl; text-align: right; }
    textarea { direction: rtl !important; text-align: right !important; }
    div[data-baseweb="textarea"] textarea { direction: rtl !important; text-align: right !important; }
    </style>
    """,
    unsafe_allow_html=True,
)


# =========================================================
# GROQ CLIENT
# =========================================================

client = Groq(
    api_key=st.secrets["GROQ_API_KEY"],
    default_headers={"Groq-Model-Version": "latest"},
)

HUNTER_MODEL = "openai/gpt-oss-120b"
KILLER_MODEL = "qwen/qwen3.8-27b"
OPERATOR_MODEL = "openai/gpt-oss-20b"
RESEARCH_MODEL = "groq/compound"

MODEL_LAST_USED = {}
MODEL_MIN_GAP = {
    HUNTER_MODEL: 28,
    KILLER_MODEL: 28,
    OPERATOR_MODEL: 28,
    RESEARCH_MODEL: 3,
}


# =========================================================
# REJECTED IDEAS
# =========================================================

REJECTED_IDEAS = [
    "Ask Mohammed",
    "Company Profile Generator",
    "Proposal contract signature deposit tools",
    "Ad spend monitoring optimization",
    "WPS file validator generator",
    "Saudi cosmetic compliance checker",
    "New business commercial registration lead alerts",
    "General tender RFP AI",
    "Vendor prequalification autofill",
    "Amazon Noon profitability dashboard",
    "E-commerce compliance scanner",
    "Generic SEO content refresh tools",
    "Instant quote widgets",
    "Generic invoice collection agents",
    "Warranty management",
    "COD reconciliation",
    "Freight invoice audit",
    "LC document pre-check",
    "Generic RFQ comparison",
    "Gmail to Sheets extraction",
    "Stripe MRR verification",
    "App store privacy declaration tools",
    "General SaaS subscription monitoring",
    "Generic scope creep detection",
    "Noon warranty workflow automation",
]

REJECTED_CONCEPTS = {
    "freight_invoice_audit": [
        "freight invoice", "shipping invoice", "shipment invoice",
        "فواتير الشحن", "فاتورة الشحن", "تدقيق الشحن", "shipping audit",
    ],
    "invoice_collection": [
        "invoice collection", "collect invoices", "invoice chasing",
        "تحصيل الفواتير", "مطاردة الفواتير",
    ],
    "rfq_comparison": [
        "rfq comparison", "compare quotations", "quotation comparison",
        "مقارنة عروض الأسعار", "مقارنة rfq",
    ],
    "tender_rfp": [
        "tender ai", "rfp ai", "tender analysis", "تحليل المناقصات",
    ],
    "warranty": [
        "warranty management", "warranty workflow", "إدارة الضمان", "ضمان المنتجات",
    ],
    "seo_content": [
        "seo content", "content refresh", "seo refresh", "تحديث المحتوى",
    ],
    "saas_monitoring": [
        "subscription monitoring", "saas monitoring", "مراقبة الاشتراكات",
    ],
    "scope_creep": [
        "scope creep", "تغير نطاق المشروع", "تجاوز نطاق المشروع",
    ],
}


# =========================================================
# GENERIC HELPERS
# =========================================================

def normalize_text(text):
    if not text:
        return ""
    text = str(text).lower()
    text = re.sub(r"[^a-z0-9\u0600-\u06ff\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def compact_text(text, max_chars):
    text = (text or "").strip()
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "\n\n[تم اختصار بقية النص لتقليل استهلاك الـtokens]"


def retry_seconds(error_text):
    patterns = [
        r"try again in\s+([0-9.]+)s",
        r"retry after\s+([0-9.]+)",
        r"retry-after[^0-9]*([0-9.]+)",
    ]
    for pattern in patterns:
        match = re.search(pattern, error_text, re.IGNORECASE)
        if match:
            try:
                return max(3, math.ceil(float(match.group(1))) + 3)
            except Exception:
                pass
    return 12


def wait_for_model(model, stage, status):
    last = MODEL_LAST_USED.get(model)
    if last is None:
        return
    gap = MODEL_MIN_GAP.get(model, 10)
    remaining = gap - (time.time() - last)
    if remaining > 0:
        seconds = math.ceil(remaining)
        status.warning(f"⏳ {stage}: انتظار {seconds} ثانية لحماية الحد المجاني لـGroq...")
        time.sleep(seconds)


def model_reasoning(model):
    if model == KILLER_MODEL:
        return "none"
    if model in (HUNTER_MODEL, OPERATOR_MODEL):
        return "low"
    return None


def prepare_case_brief(original):
    text = original.strip()
    protocol_markers = [
        "\n# الوكلاء",
        "\n## الوكيل الأول",
        "\n# قواعد المناظرة",
        "\n# نظام التقييم",
        "\n# شرط النجاح",
        "\n# اختبار الحقيقة",
        "\n# المرحلة النهائية",
    ]
    cuts = [text.find(marker) for marker in protocol_markers if text.find(marker) != -1]
    if cuts:
        text = text[:min(cuts)]
    text = re.sub(r"\n{4,}", "\n\n\n", text)
    if len(text) > 13000:
        text = text[:10000] + "\n\n[تم اختصار جزء من النص محلياً]\n\n" + text[-2500:]
    return text


def local_blacklist_check(idea):
    combined = normalize_text(" ".join([
        idea.get("name", ""),
        idea.get("one_liner", ""),
        idea.get("problem", ""),
        idea.get("product", ""),
        idea.get("distribution", ""),
    ]))

    for rejected in REJECTED_IDEAS:
        r = normalize_text(rejected)
        if r and r in combined:
            return True, f"تشابه مباشر مع فكرة مرفوضة: {rejected}"

        # Similarity only on idea name to avoid false positives from long descriptions.
        name_ratio = difflib.SequenceMatcher(
            None,
            normalize_text(idea.get("name", "")),
            r,
        ).ratio()
        if name_ratio >= 0.78:
            return True, f"اسم قريب جداً من فكرة مرفوضة: {rejected}"

    for concept, phrases in REJECTED_CONCEPTS.items():
        hits = [phrase for phrase in phrases if normalize_text(phrase) in combined]
        if hits:
            return True, f"تشابه مفاهيمي مع فكرة مرفوضة ({concept}): {', '.join(hits)}"

    if idea.get("similar_to_rejected"):
        return True, idea.get("rejected_similarity_explanation") or "Hunter صنفها كقريبة من فكرة مرفوضة."

    return False, ""


# =========================================================
# STRICT JSON CALLS
# =========================================================

def call_json(model, system_prompt, user_prompt, schema_name, schema, max_tokens, stage, status, retries=4):
    wait_for_model(model, stage, status)
    errors = []

    for _ in range(retries):
        try:
            kwargs = {
                "model": model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                "temperature": 0.3,
                "max_completion_tokens": max_tokens,
                "response_format": {
                    "type": "json_schema",
                    "json_schema": {
                        "name": schema_name,
                        "strict": True,
                        "schema": schema,
                    },
                },
                "reasoning_format": "hidden",
            }
            reasoning = model_reasoning(model)
            if reasoning:
                kwargs["reasoning_effort"] = reasoning

            response = client.chat.completions.create(**kwargs)
            MODEL_LAST_USED[model] = time.time()
            content = response.choices[0].message.content
            if not content:
                errors.append(f"{model}: empty structured response")
                continue
            return {"ok": True, "data": json.loads(content), "error": None}

        except Exception as exc:
            error_text = str(exc)
            errors.append(error_text)
            if "429" in error_text or "rate_limit" in error_text.lower():
                seconds = retry_seconds(error_text)
                status.warning(f"⏳ {stage}: Groq طلب انتظار {seconds} ثانية...")
                time.sleep(seconds)
                continue
            if "413" in error_text or "too large" in error_text.lower():
                break
            time.sleep(2)

    return {"ok": False, "data": None, "error": "\n\n".join(errors)}


def call_text(model, system_prompt, user_prompt, max_tokens, stage, status, retries=4):
    wait_for_model(model, stage, status)
    errors = []

    for _ in range(retries):
        try:
            kwargs = {
                "model": model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                "temperature": 0.3,
                "max_completion_tokens": max_tokens,
                "reasoning_format": "hidden",
            }
            reasoning = model_reasoning(model)
            if reasoning:
                kwargs["reasoning_effort"] = reasoning

            response = client.chat.completions.create(**kwargs)
            MODEL_LAST_USED[model] = time.time()
            content = response.choices[0].message.content
            if content:
                return {"ok": True, "text": content.strip(), "error": None}
            errors.append(f"{model}: empty response")

        except Exception as exc:
            error_text = str(exc)
            errors.append(error_text)
            if "429" in error_text or "rate_limit" in error_text.lower():
                seconds = retry_seconds(error_text)
                status.warning(f"⏳ {stage}: انتظار {seconds} ثانية...")
                time.sleep(seconds)
                continue
            time.sleep(2)

    return {"ok": False, "text": "", "error": "\n\n".join(errors)}


# =========================================================
# HUNTER
# =========================================================

HUNTER_SCHEMA = {
    "type": "object",
    "properties": {
        "ideas": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "one_liner": {"type": "string"},
                    "buyer": {"type": "string"},
                    "problem": {"type": "string"},
                    "product": {"type": "string"},
                    "why_pay": {"type": "string"},
                    "price": {"type": "string"},
                    "current_alternative": {"type": "string"},
                    "distribution": {"type": "string"},
                    "first_10_customers": {"type": "string"},
                    "automation": {"type": "integer"},
                    "human_work": {"type": "string"},
                    "why_now": {"type": "string"},
                    "similar_to_rejected": {"type": "boolean"},
                    "rejected_similarity_explanation": {"type": "string"},
                },
                "required": [
                    "name", "one_liner", "buyer", "problem", "product", "why_pay",
                    "price", "current_alternative", "distribution", "first_10_customers",
                    "automation", "human_work", "why_now", "similar_to_rejected",
                    "rejected_similarity_explanation",
                ],
                "additionalProperties": False,
            },
        }
    },
    "required": ["ideas"],
    "additionalProperties": False,
}


def run_hunter(case_brief, status):
    system = """
أنت THE HUNTER. ابحث عن أماكن تتحرك فيها الأموال فعلياً، لا عن فكرة AI مثيرة.
الأولوية: ألم مالي، willingness-to-pay، قناة توزيع واضحة، أتمتة عالية، تكرار، وسرعة لأول دفعة.
ممنوع AI wrapper أو dashboard عام أو أداة يستطيع ChatGPT العادي تنفيذها بما يكفي.
ممنوع إعادة تغليف فكرة رفضها المستخدم. أخرج 3 أفكار فقط، مختلفة اقتصادياً، ولا تختر Winner.
إذا كانت الفكرة قريبة من المرفوضات اجعل similar_to_rejected=true.
"""
    prompt = f"""
حالة المستخدم:

{case_brief}

الأفكار المرفوضة صراحة:
{json.dumps(REJECTED_IDEAS, ensure_ascii=False)}

أخرج 3 أفكار فقط. لا تعامل أي ادعاء سوقي غير متحقق كحقيقة.
"""
    result = call_json(
        HUNTER_MODEL, system, prompt, "hunter_ideas", HUNTER_SCHEMA,
        1700, "THE HUNTER", status,
    )
    if result["ok"]:
        ideas = result["data"].get("ideas", [])
        if len(ideas) != 3:
            return {"ok": False, "data": None, "error": f"Hunter returned {len(ideas)} ideas instead of 3."}
        for idx, idea in enumerate(ideas, start=1):
            idea["id"] = f"idea-{idx}"
            idea["automation"] = max(0, min(100, int(idea.get("automation", 0))))
    return result


def filter_ideas(hunter_data):
    passed, blocked = [], []
    for idea in hunter_data["ideas"]:
        is_blocked, reason = local_blacklist_check(idea)
        if is_blocked:
            blocked.append({"idea": idea, "reason": reason})
        else:
            passed.append(idea)
    return passed, blocked


# =========================================================
# WEB RESEARCH — ONE CALL, WITH AUTOMATIC FALLBACK
# =========================================================

def _model_to_plain(value):
    if value is None:
        return None
    if hasattr(value, "model_dump"):
        try:
            return value.model_dump()
        except Exception:
            pass
    if isinstance(value, dict):
        return {k: _model_to_plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_model_to_plain(v) for v in value]
    return value


def extract_sources(response):
    sources = []
    try:
        tools = getattr(response.choices[0].message, "executed_tools", None)
        plain = _model_to_plain(tools)
    except Exception:
        plain = None

    def walk(obj):
        if isinstance(obj, dict):
            url = obj.get("url")
            if isinstance(url, str) and url.startswith("http"):
                sources.append({
                    "title": str(obj.get("title") or obj.get("name") or "Source"),
                    "url": url,
                    "snippet": str(obj.get("content") or obj.get("snippet") or ""),
                })
            for value in obj.values():
                walk(value)
        elif isinstance(obj, list):
            for item in obj:
                walk(item)

    walk(plain)
    unique, seen = [], set()
    for source in sources:
        if source["url"] not in seen:
            seen.add(source["url"])
            unique.append(source)
    return unique[:20]


def parse_research_blocks(text, ideas):
    research = {}
    for idea in ideas:
        idea_id = re.escape(idea["id"])
        pattern = rf"===\s*IDEA\s*:\s*{idea_id}\s*===\s*(.*?)\s*===\s*END\s*IDEA\s*==="
        match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
        if match:
            research[idea["id"]] = match.group(1).strip()
    return research


def research_prompt_for_ideas(ideas):
    compact = [
        {
            "idea_id": idea["id"],
            "name": idea["name"],
            "buyer": idea["buyer"],
            "problem": idea["problem"],
            "product": idea["product"],
            "price_hypothesis": idea["price"],
            "distribution_hypothesis": idea["distribution"],
            "why_now_hypothesis": idea["why_now"],
        }
        for idea in ideas
    ]
    return f"""
You are performing current investment due diligence. Use web_search and visit_website.
Research EACH idea independently. Do not trust the idea author's claims.

IDEAS:
{json.dumps(compact, ensure_ascii=False)}

For each idea investigate:
1. Direct competitors doing the same job-to-be-done.
2. Public competitor pricing where available.
3. Free or platform-native alternatives.
4. Evidence buyers actually pay for this problem.
5. Evidence against willingness-to-pay.
6. Concrete distribution channels to reach the first paying buyers.
7. Regulation/licensing risk.
8. Platform/API dependency.
9. Liability, privacy, and security risk.
10. Evidence the idea is already commoditized or replaceable by a generic AI tool.
11. Evidence supporting or contradicting the "why now" claim.

Use current sources. If evidence cannot be found, write UNKNOWN. If a claim is contradicted, say CONTRADICTED.

OUTPUT FORMAT IS MANDATORY. For every supplied idea_id output exactly one block:

=== IDEA: idea-1 ===
VERIFIED:
- ...
UNKNOWN:
- ...
CONTRADICTED:
- ...
COMPETITORS:
- ...
PRICING EVIDENCE:
- ...
WTP EVIDENCE:
- ...
DISTRIBUTION EVIDENCE:
- ...
REGULATORY / PLATFORM RISKS:
- ...
BOTTOM LINE:
...
=== END IDEA ===

Repeat the same structure for every idea_id. Keep each block concise but evidence-driven.
"""


def research_compound_call(ideas, status, stage, retries=4):
    wait_for_model(RESEARCH_MODEL, stage, status)
    prompt = research_prompt_for_ideas(ideas)
    errors = []

    for _ in range(retries):
        try:
            response = client.chat.completions.create(
                model=RESEARCH_MODEL,
                messages=[{"role": "user", "content": prompt}],
                max_completion_tokens=2600 if len(ideas) > 1 else 1300,
                temperature=0.1,
                compound_custom={
                    "tools": {"enabled_tools": ["web_search", "visit_website"]}
                },
            )
            MODEL_LAST_USED[RESEARCH_MODEL] = time.time()
            content = response.choices[0].message.content or ""
            parsed = parse_research_blocks(content, ideas)
            sources = extract_sources(response)
            if len(parsed) == len(ideas):
                return {
                    "ok": True,
                    "research": {
                        idea["id"]: {
                            "ok": True,
                            "text": parsed[idea["id"]],
                            "sources": sources,
                            "error": None,
                        }
                        for idea in ideas
                    },
                    "error": None,
                }
            errors.append(
                f"Research output missing blocks. Expected {len(ideas)}, parsed {len(parsed)}.\nRaw output:\n{content[:3000]}"
            )
        except Exception as exc:
            error_text = str(exc)
            errors.append(error_text)
            if "429" in error_text or "rate_limit" in error_text.lower():
                seconds = retry_seconds(error_text)
                status.warning(f"⏳ {stage}: وصل حد Groq. انتظار {seconds} ثانية ثم نكمل...")
                time.sleep(seconds)
                continue
            if "413" in error_text or "request_too_large" in error_text.lower():
                break
            time.sleep(3)

    return {"ok": False, "research": {}, "error": "\n\n".join(errors)}


def research_all_ideas(ideas, status):
    # First try: one Compound request for all surviving ideas.
    result = research_compound_call(ideas, status, "Web Research")
    if result["ok"]:
        return result

    # Automatic fallback: research one idea at a time if the combined call fails.
    status.warning("⚠️ البحث الجماعي لم يكتمل؛ سأنتقل تلقائياً إلى بحث مستقل لكل فكرة.")
    combined = {}
    errors = ["Combined research failed:", result["error"]]

    for index, idea in enumerate(ideas, start=1):
        status.info(f"🔎 بحث مستقل {index}/{len(ideas)}: {idea['name']}")
        single = research_compound_call([idea], status, f"بحث {idea['name']}")
        if not single["ok"]:
            errors.append(f"\n{idea['id']} failed:\n{single['error']}")
            continue
        combined.update(single["research"])

    if len(combined) != len(ideas):
        missing = [idea["id"] for idea in ideas if idea["id"] not in combined]
        errors.append("\nMissing ideas: " + ", ".join(missing))
        return {"ok": False, "research": combined, "error": "\n".join(errors)}

    return {"ok": True, "research": combined, "error": None}


# =========================================================
# KILLER — FIRST ATTACK
# =========================================================

KILLER_SCHEMA = {
    "type": "object",
    "properties": {
        "reviews": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "idea_id": {"type": "string"},
                    "failure_reason_1": {"type": "string"},
                    "failure_reason_2": {"type": "string"},
                    "failure_reason_3": {"type": "string"},
                    "kill_shot": {"type": "string"},
                    "immediate_rejection_evidence": {"type": "string"},
                    "research_supports": {"type": "string"},
                    "research_hurts": {"type": "string"},
                    "score_out_of_10": {"type": "integer"},
                    "decision": {"type": "string", "enum": ["SURVIVES", "KILL IT"]},
                },
                "required": [
                    "idea_id", "failure_reason_1", "failure_reason_2", "failure_reason_3",
                    "kill_shot", "immediate_rejection_evidence", "research_supports",
                    "research_hurts", "score_out_of_10", "decision",
                ],
                "additionalProperties": False,
            },
        }
    },
    "required": ["reviews"],
    "additionalProperties": False,
}


def run_killer(ideas, research, status):
    system = """
أنت THE KILLER. أنت مستثمر متشائم هدفه منعنا من بناء المشروع الخطأ.
لا تقترح أفكاراً جديدة. استخدم البحث كدليل؛ ادعاء Hunter ليس حقيقة إذا لم يدعمه البحث.
افحص: المنافسين، البدائل المجانية، ChatGPT substitution، WTP، CAC، churn، distribution، regulation،
platform risk، privacy/security، liability، الدعم، moat، Feature vs Company، والتكرار.
إذا الفكرة ماتت اكتب KILL IT. لا تجامل.
"""
    payload = [
        {
            "idea": idea,
            "research": compact_text(research[idea["id"]]["text"], 7000),
        }
        for idea in ideas
    ]
    result = call_json(
        KILLER_MODEL, system, json.dumps(payload, ensure_ascii=False),
        "killer_reviews", KILLER_SCHEMA, 1500, "THE KILLER", status,
    )
    if result["ok"]:
        expected = {idea["id"] for idea in ideas}
        returned = {row.get("idea_id") for row in result["data"].get("reviews", [])}
        if returned != expected:
            return {"ok": False, "data": None, "error": f"Killer IDs mismatch. expected={expected}, returned={returned}"}
        for row in result["data"]["reviews"]:
            row["score_out_of_10"] = max(0, min(10, int(row["score_out_of_10"])))
    return result


# =========================================================
# HUNTER REBUTTAL
# =========================================================

REBUTTAL_SCHEMA = {
    "type": "object",
    "properties": {
        "responses": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "idea_id": {"type": "string"},
                    "valid_objection": {"type": "string"},
                    "disputed_objection": {"type": "string"},
                    "evidence_needed": {"type": "string"},
                    "position": {"type": "string", "enum": ["DEFEND", "DROP"]},
                },
                "required": ["idea_id", "valid_objection", "disputed_objection", "evidence_needed", "position"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["responses"],
    "additionalProperties": False,
}


def run_rebuttal(ideas, killer_data, research, status):
    system = """
أنت THE HUNTER ولديك رد واحد فقط. لا تضف أفكاراً جديدة.
إذا كشف البحث أو Killer عيباً حقيقياً اعترف به. DROP أفضل من ترقيع فكرة ميتة.
"""
    payload = {
        "ideas": ideas,
        "killer": killer_data,
        "research": {idea["id"]: compact_text(research[idea["id"]]["text"], 5000) for idea in ideas},
    }
    return call_json(
        HUNTER_MODEL, system, json.dumps(payload, ensure_ascii=False),
        "hunter_rebuttal", REBUTTAL_SCHEMA, 900, "HUNTER REBUTTAL", status,
    )


# =========================================================
# KILLER — FINAL ATTACK
# =========================================================

FINAL_KILLER_SCHEMA = {
    "type": "object",
    "properties": {
        "decisions": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "idea_id": {"type": "string"},
                    "decision": {"type": "string", "enum": ["SURVIVES", "KILL IT"]},
                    "remaining_problem": {"type": "string"},
                    "wtp_real": {"type": "boolean"},
                    "distribution_real": {"type": "boolean"},
                    "feature_or_company": {"type": "string", "enum": ["FEATURE", "COMPANY", "UNCLEAR"]},
                    "final_score_out_of_10": {"type": "integer"},
                },
                "required": [
                    "idea_id", "decision", "remaining_problem", "wtp_real", "distribution_real",
                    "feature_or_company", "final_score_out_of_10",
                ],
                "additionalProperties": False,
            },
        }
    },
    "required": ["decisions"],
    "additionalProperties": False,
}


def run_killer_final(ideas, first_killer, rebuttal, research, status):
    system = """
أنت THE KILLER في الجولة الأخيرة. لا تولد أفكاراً ولا تعيد إحياء فكرة ضعيفة.
إذا WTP أو Distribution غير مثبتة فلا تعاملها كحقيقة. القرار SURVIVES أو KILL IT فقط.
"""
    payload = {
        "ideas": ideas,
        "first_attack": first_killer,
        "hunter_rebuttal": rebuttal,
        "research": {idea["id"]: compact_text(research[idea["id"]]["text"], 4200) for idea in ideas},
    }
    result = call_json(
        KILLER_MODEL, system, json.dumps(payload, ensure_ascii=False),
        "killer_final", FINAL_KILLER_SCHEMA, 1100, "KILLER FINAL", status,
    )
    if result["ok"]:
        for row in result["data"].get("decisions", []):
            row["final_score_out_of_10"] = max(0, min(10, int(row["final_score_out_of_10"])))
    return result


# =========================================================
# OPERATOR
# =========================================================

OPERATOR_SCHEMA = {
    "type": "object",
    "properties": {
        "evaluations": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "idea_id": {"type": "string"},
                    "idea_name": {"type": "string"},
                    "severity": {"type": "integer"},
                    "willingness_to_pay": {"type": "integer"},
                    "distribution": {"type": "integer"},
                    "automation": {"type": "integer"},
                    "recurring": {"type": "integer"},
                    "competition": {"type": "integer"},
                    "moat": {"type": "integer"},
                    "speed_to_revenue": {"type": "integer"},
                    "stack_fit": {"type": "integer"},
                    "price": {"type": "string"},
                    "gross_margin": {"type": "string"},
                    "ltv": {"type": "string"},
                    "cac": {"type": "string"},
                    "customers_for_1k_mrr": {"type": "string"},
                    "customers_for_5k_mrr": {"type": "string"},
                    "customers_for_10k_mrr": {"type": "string"},
                    "automation_percent": {"type": "integer"},
                    "first_buyer": {"type": "string"},
                    "fastest_test": {"type": "string"},
                    "biggest_risk": {"type": "string"},
                    "truth_test": {"type": "string"},
                },
                "required": [
                    "idea_id", "idea_name", "severity", "willingness_to_pay", "distribution",
                    "automation", "recurring", "competition", "moat", "speed_to_revenue", "stack_fit",
                    "price", "gross_margin", "ltv", "cac", "customers_for_1k_mrr",
                    "customers_for_5k_mrr", "customers_for_10k_mrr", "automation_percent",
                    "first_buyer", "fastest_test", "biggest_risk", "truth_test",
                ],
                "additionalProperties": False,
            },
        }
    },
    "required": ["evaluations"],
    "additionalProperties": False,
}

SCORE_LIMITS = {
    "severity": 15,
    "willingness_to_pay": 15,
    "distribution": 15,
    "automation": 15,
    "recurring": 10,
    "competition": 10,
    "moat": 5,
    "speed_to_revenue": 10,
    "stack_fit": 5,
}


def clamp_score(value, maximum):
    return max(0, min(maximum, int(value)))


def run_operator(surviving_ideas, research, killer_final_data, status):
    system = """
أنت THE OPERATOR / ECONOMIST: CTO + CFO + Growth Operator.
قيّم فقط الأفكار التي نجت من Killer. لا تولد أفكاراً جديدة.
الأوزان: Severity 15, WTP 15, Distribution 15, Automation 15, Recurring 10,
Competition 10, Moat 5, Speed to Revenue 10, Stack Fit 5.
لا ترفع الدرجات لتحقيق 85. لا تختر Winner؛ البرنامج سيجمع الدرجات بنفسه.
إذا CAC/LTV غير مثبتين اكتب تقديراً واضحاً بأنه تقدير، ولا تخترع حقيقة.
"""
    payload = {
        "surviving_ideas": surviving_ideas,
        "research": {idea["id"]: compact_text(research[idea["id"]]["text"], 5200) for idea in surviving_ideas},
        "killer_final": killer_final_data,
    }
    result = call_json(
        OPERATOR_MODEL, system, json.dumps(payload, ensure_ascii=False),
        "operator_evaluations", OPERATOR_SCHEMA, 1800, "THE OPERATOR", status,
    )
    if result["ok"]:
        for row in result["data"].get("evaluations", []):
            total = 0
            for key, max_value in SCORE_LIMITS.items():
                row[key] = clamp_score(row[key], max_value)
                total += row[key]
            row["total_score"] = total
            row["automation_percent"] = max(0, min(100, int(row["automation_percent"])))
    return result


# =========================================================
# FINAL OBJECTION + VERDICT
# =========================================================

def final_objection(winner, operator_row, research, status):
    system = """
أنت THE KILLER. هذه آخر فرصة لقتل الفكرة الفائزة.
اكتب أقوى حجة واحدة ممكنة تجعل المشروع ينتهي عند $0 MRR. استخدم البحث الموجود. لا تقترح فكرة جديدة.
"""
    payload = {
        "winner": winner,
        "operator_evaluation": operator_row,
        "research": compact_text(research[winner["id"]]["text"], 6500),
    }
    return call_text(
        KILLER_MODEL, system, json.dumps(payload, ensure_ascii=False),
        450, "FINAL OBJECTION", status,
    )


def final_verdict(winner, operator_row, objection, status):
    system = """
أنت THE OPERATOR. راجع تقييمك واعتراض Killer الأخير.
أخرج فقط:
## FINAL VERDICT
BUILD أو KILL
ثم السبب، اختبار 7 أيام، BUILD criterion، KILL criterion.
لا ترفع الدرجة ولا تغير القرار إلا إذا كشف الاعتراض مشكلة جوهرية.
"""
    payload = {
        "winner": winner,
        "operator_evaluation": operator_row,
        "final_objection": objection,
    }
    return call_text(
        OPERATOR_MODEL, system, json.dumps(payload, ensure_ascii=False),
        600, "FINAL VERDICT", status,
    )


# =========================================================
# DISPLAY HELPERS
# =========================================================

def render_idea(idea):
    st.subheader(idea["name"])
    st.write(idea["one_liner"])
    st.markdown(f"""
**المشتري:** {idea['buyer']}

**المشكلة:** {idea['problem']}

**المنتج:** {idea['product']}

**لماذا سيدفع؟** {idea['why_pay']}

**السعر:** {idea['price']}

**البديل الحالي:** {idea['current_alternative']}

**Distribution:** {idea['distribution']}

**أول 10 عملاء:** {idea['first_10_customers']}

**Automation:** {idea['automation']}%

**العمل البشري:** {idea['human_work']}

**لماذا الآن؟** {idea['why_now']}
""")


def idea_name_by_id(ideas, idea_id):
    for idea in ideas:
        if idea["id"] == idea_id:
            return idea["name"]
    return idea_id


# =========================================================
# REPORT + PDF
# =========================================================

def build_report(ideas, blocked, research, killer, rebuttal, killer_final, operator_rows, objection, verdict):
    lines = ["MD INVESTMENT RESEARCH COUNCIL", ""]

    lines.append("IDEAS")
    for idea in ideas:
        lines.extend([
            "",
            idea["name"],
            f"Buyer: {idea['buyer']}",
            f"Problem: {idea['problem']}",
            f"Price: {idea['price']}",
            f"Distribution: {idea['distribution']}",
            f"Automation: {idea['automation']}%",
        ])

    if blocked:
        lines.extend(["", "BLOCKED IDEAS"])
        for item in blocked:
            lines.extend([item["idea"]["name"], f"Reason: {item['reason']}"])

    lines.extend(["", "WEB RESEARCH"])
    for idea in ideas:
        result = research.get(idea["id"], {})
        lines.extend(["", idea["name"], result.get("text", "")])
        for source in result.get("sources", []):
            lines.append(f"SOURCE: {source.get('title', '')} - {source.get('url', '')}")

    lines.extend([
        "", "KILLER FIRST ATTACK", json.dumps(killer, ensure_ascii=False, indent=2),
        "", "HUNTER REBUTTAL", json.dumps(rebuttal, ensure_ascii=False, indent=2),
        "", "KILLER FINAL", json.dumps(killer_final, ensure_ascii=False, indent=2),
        "", "OPERATOR", json.dumps(operator_rows, ensure_ascii=False, indent=2),
        "", "FINAL OBJECTION", objection or "",
        "", "FINAL VERDICT", verdict or "",
    ])
    return "\n".join(lines).strip()


def clean_pdf_text(text):
    text = re.sub(r"#{1,6}\s*", "", text)
    for emoji in ["🧠", "😈", "💡", "🏛️", "⚔️", "✅", "❌", "⚠️", "🚀", "📥", "📄", "📝", "🗑️", "🔎", "🎯", "🔪"]:
        text = text.replace(emoji, "")
    return text


def find_font():
    candidates = [
        "/usr/share/fonts/truetype/noto/NotoNaskhArabic-Regular.ttf",
        "/usr/share/fonts/truetype/noto/NotoSansArabic-Regular.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    for font in glob.glob("/usr/share/fonts/**/*.ttf", recursive=True):
        lower = font.lower()
        if "naskh" in lower or "arabic" in lower or "dejavusans" in lower:
            return font
    return None


def create_pdf(report):
    font = find_font()
    if not font:
        raise RuntimeError("Arabic font not found on server.")

    pdf = FPDF(orientation="P", unit="mm", format="A4")
    pdf.set_auto_page_break(True, 15)
    pdf.set_margins(15, 15, 15)
    pdf.add_page()
    pdf.add_font("Arabic", fname=font)
    pdf.set_font("Arabic", size=10)

    try:
        pdf.set_text_shaping(
            use_shaping_engine=True,
            direction="rtl",
            script="arab",
            language="ara",
        )
    except Exception:
        pass

    for line in clean_pdf_text(report).split("\n"):
        line = line.strip()
        if not line:
            pdf.ln(3)
            continue
        pdf.multi_cell(0, 6, line, align="R", new_x="LMARGIN", new_y="NEXT")

    return bytes(pdf.output())


# =========================================================
# SESSION STATE
# =========================================================

DEFAULT_STATE = {
    "original": "",
    "case_brief": "",
    "ideas": None,
    "blocked": None,
    "research": None,
    "killer": None,
    "rebuttal": None,
    "killer_final": None,
    "operator_rows": None,
    "winner": None,
    "objection": None,
    "verdict": None,
    "error": None,
}

for key, value in DEFAULT_STATE.items():
    if key not in st.session_state:
        st.session_state[key] = value


# =========================================================
# UI
# =========================================================

st.title("🧠 MD Investment Research Council")
st.caption(f"Build: {BUILD_ID}")
st.write("""
**THE HUNTER** يولد 3 فرص فقط → فلتر للمرفوضات → **Web Research حقيقي** →
**THE KILLER** → رد واحد من Hunter → Killer النهائي → **THE OPERATOR** → Final Objection → Final Verdict.
""")

original = st.text_area(
    "اكتب الحالة أو الصق البرومبت السابق:",
    height=330,
    value=st.session_state.original,
)

start = st.button("🚀 ابدأ Research Council", type="primary", use_container_width=True)


# =========================================================
# RUN WORKFLOW
# =========================================================

if start:
    if not original.strip():
        st.warning("اكتب الحالة أولاً.")
    else:
        for key, value in DEFAULT_STATE.items():
            st.session_state[key] = value
        st.session_state.original = original
        status = st.empty()

        # 1) Local brief
        status.info("📋 تجهيز الحالة محلياً...")
        case_brief = prepare_case_brief(original)
        st.session_state.case_brief = case_brief

        # 2) Hunter
        status.info("🎯 THE HUNTER يولد 3 أفكار...")
        hunter_result = run_hunter(case_brief, status)
        if not hunter_result["ok"]:
            st.session_state.error = hunter_result["error"]
        else:
            ideas, blocked = filter_ideas(hunter_result["data"])
            st.session_state.ideas = ideas
            st.session_state.blocked = blocked

            if not ideas:
                st.session_state.verdict = (
                    "## FINAL VERDICT\n\nKILL\n\n"
                    "كل الأفكار التي ولّدها Hunter سقطت بسبب تشابهها مع أفكار مرفوضة مسبقاً."
                )
                status.success("✅ انتهى المجلس: كل الأفكار سقطت في الفلتر.")
            else:
                # 3) Research
                status.info("🔎 Web Research على الأفكار الناجية...")
                research_result = research_all_ideas(ideas, status)
                st.session_state.research = research_result.get("research", {})

                if not research_result["ok"]:
                    st.session_state.error = "فشل Web Research.\n\n" + research_result["error"]
                else:
                    research = research_result["research"]

                    # 4) Killer first attack
                    status.info("🔪 THE KILLER يراجع الأدلة ويحاول قتل الأفكار...")
                    killer_result = run_killer(ideas, research, status)
                    if not killer_result["ok"]:
                        st.session_state.error = killer_result["error"]
                    else:
                        killer_data = killer_result["data"]
                        st.session_state.killer = killer_data

                        # 5) Hunter rebuttal
                        status.info("🎯 Hunter يرد مرة واحدة...")
                        rebuttal_result = run_rebuttal(ideas, killer_data, research, status)
                        if not rebuttal_result["ok"]:
                            st.session_state.error = rebuttal_result["error"]
                        else:
                            rebuttal_data = rebuttal_result["data"]
                            st.session_state.rebuttal = rebuttal_data

                            # 6) Killer final
                            status.info("🔪 Killer يصدر الحكم الأخير...")
                            killer_final_result = run_killer_final(
                                ideas, killer_data, rebuttal_data, research, status
                            )
                            if not killer_final_result["ok"]:
                                st.session_state.error = killer_final_result["error"]
                            else:
                                killer_final_data = killer_final_result["data"]
                                st.session_state.killer_final = killer_final_data

                                surviving_ids = {
                                    row["idea_id"]
                                    for row in killer_final_data.get("decisions", [])
                                    if row["decision"] == "SURVIVES"
                                }
                                surviving = [idea for idea in ideas if idea["id"] in surviving_ids]

                                if not surviving:
                                    st.session_state.operator_rows = []
                                    st.session_state.objection = (
                                        "لا يوجد Winner يمكن الاعتراض عليه؛ لم تنج أي فكرة من Red Team."
                                    )
                                    st.session_state.verdict = (
                                        "## FINAL VERDICT\n\nKILL\n\n"
                                        "لم تنج أي فكرة من THE KILLER بعد مراجعة البحث."
                                    )
                                    status.success("✅ انتهى المجلس: NO WINNER")
                                else:
                                    # 7) Operator
                                    status.info("📊 THE OPERATOR يحسب الاقتصاديات والدرجات...")
                                    operator_result = run_operator(
                                        surviving, research, killer_final_data, status
                                    )
                                    if not operator_result["ok"]:
                                        st.session_state.error = operator_result["error"]
                                    else:
                                        rows = operator_result["data"].get("evaluations", [])
                                        st.session_state.operator_rows = rows

                                        # Winner is calculated LOCALLY, not chosen by the model.
                                        ranked = sorted(rows, key=lambda x: x["total_score"], reverse=True)
                                        top = ranked[0] if ranked else None

                                        if not top or top["total_score"] <= 85:
                                            st.session_state.winner = None
                                            st.session_state.objection = (
                                                "لا يوجد Winner فوق 85/100؛ اختيار مشروع سيخالف قواعد التقييم."
                                            )
                                            best_text = f" أفضل نتيجة كانت {top['total_score']}/100." if top else ""
                                            st.session_state.verdict = (
                                                "## FINAL VERDICT\n\nKILL\n\n"
                                                "لا توجد فكرة تجاوزت 85/100 بعد Red Team." + best_text
                                            )
                                            status.success("✅ انتهى المجلس: NO WINNER")
                                        else:
                                            winner = next(
                                                idea for idea in surviving if idea["id"] == top["idea_id"]
                                            )
                                            st.session_state.winner = winner

                                            # 8) Final objection
                                            status.info("🔪 FINAL OBJECTION...")
                                            objection_result = final_objection(
                                                winner, top, research, status
                                            )
                                            if not objection_result["ok"]:
                                                st.session_state.error = objection_result["error"]
                                            else:
                                                objection = objection_result["text"]
                                                st.session_state.objection = objection

                                                # 9) Final verdict
                                                status.info("🏛️ FINAL VERDICT...")
                                                verdict_result = final_verdict(
                                                    winner, top, objection, status
                                                )
                                                if not verdict_result["ok"]:
                                                    st.session_state.error = verdict_result["error"]
                                                else:
                                                    st.session_state.verdict = verdict_result["text"]
                                                    status.success("✅ انتهى Research Council")


# =========================================================
# ERROR
# =========================================================

if st.session_state.error:
    st.error("حدث خطأ، ولم يسمح التطبيق للمجلس بإصدار حكم ناقص.")
    with st.expander("🔧 التفاصيل التقنية", expanded=True):
        st.code(st.session_state.error)


# =========================================================
# BLOCKED IDEAS
# =========================================================

if st.session_state.blocked:
    st.divider()
    st.header("🚫 أفكار أسقطها الفلتر")
    for item in st.session_state.blocked:
        st.warning(f"**{item['idea']['name']}**\n\n{item['reason']}")


# =========================================================
# IDEAS
# =========================================================

if st.session_state.ideas:
    st.divider()
    st.header("🎯 أفكار THE HUNTER")
    for idea in st.session_state.ideas:
        with st.expander(idea["name"], expanded=True):
            render_idea(idea)


# =========================================================
# WEB RESEARCH
# =========================================================

if st.session_state.research:
    st.divider()
    st.header("🔎 Web Research")
    for idea in st.session_state.ideas or []:
        result = st.session_state.research.get(idea["id"], {})
        with st.expander(f"🔎 {idea['name']}", expanded=False):
            st.markdown(result.get("text", ""))
            sources = result.get("sources", [])
            if sources:
                st.subheader("المصادر التي استخدمها البحث")
                for source in sources:
                    title = source.get("title") or "Source"
                    url = source.get("url") or ""
                    if url:
                        st.markdown(f"- [{title}]({url})")


# =========================================================
# KILLER FIRST
# =========================================================

if st.session_state.killer:
    st.divider()
    st.header("🔪 THE KILLER")
    for review in st.session_state.killer.get("reviews", []):
        name = idea_name_by_id(st.session_state.ideas or [], review["idea_id"])
        with st.expander(f"{name} — {review['decision']}", expanded=True):
            st.markdown(f"""
**سبب الفشل 1:** {review['failure_reason_1']}

**سبب الفشل 2:** {review['failure_reason_2']}

**سبب الفشل 3:** {review['failure_reason_3']}

**Kill Shot:** {review['kill_shot']}

**الدليل الذي يقتلها فوراً:** {review['immediate_rejection_evidence']}

**ما يدعمها من البحث:** {review['research_supports']}

**ما يضرها من البحث:** {review['research_hurts']}

**Score:** {review['score_out_of_10']}/10

**Decision:** `{review['decision']}`
""")


# =========================================================
# REBUTTAL
# =========================================================

if st.session_state.rebuttal:
    st.divider()
    st.header("🎯 HUNTER REBUTTAL")
    for row in st.session_state.rebuttal.get("responses", []):
        name = idea_name_by_id(st.session_state.ideas or [], row["idea_id"])
        st.markdown(f"""
### {name}
**اعتراض صحيح:** {row['valid_objection']}

**اعتراض يرفضه Hunter:** {row['disputed_objection']}

**الدليل المطلوب:** {row['evidence_needed']}

**Position:** `{row['position']}`
""")


# =========================================================
# KILLER FINAL
# =========================================================

if st.session_state.killer_final:
    st.divider()
    st.header("🔪 KILLER FINAL")
    for row in st.session_state.killer_final.get("decisions", []):
        name = idea_name_by_id(st.session_state.ideas or [], row["idea_id"])
        st.markdown(f"""
### {name}
**Decision:** `{row['decision']}`

**المشكلة المتبقية:** {row['remaining_problem']}

**WTP مثبتة؟** {row['wtp_real']}

**Distribution واقعية؟** {row['distribution_real']}

**Feature / Company:** {row['feature_or_company']}

**Red-Team Score:** {row['final_score_out_of_10']}/10
""")


# =========================================================
# OPERATOR
# =========================================================

if st.session_state.operator_rows is not None:
    st.divider()
    st.header("📊 THE OPERATOR")
    rows = st.session_state.operator_rows or []
    if not rows:
        st.warning("لم تصل أي فكرة إلى مرحلة التقييم الاقتصادي النهائية.")
    else:
        ranked = sorted(rows, key=lambda x: x["total_score"], reverse=True)
        table = [
            {
                "Rank": idx,
                "Idea": row["idea_name"],
                "Score": row["total_score"],
                "First Buyer": row["first_buyer"],
                "Price": row["price"],
                "Automation": f"{row['automation_percent']}%",
                "Fastest Test": row["fastest_test"],
                "Biggest Risk": row["biggest_risk"],
            }
            for idx, row in enumerate(ranked, start=1)
        ]
        st.dataframe(table, use_container_width=True, hide_index=True)

        for row in ranked:
            with st.expander(f"{row['idea_name']} — {row['total_score']}/100"):
                st.markdown(f"""
**Severity:** {row['severity']}/15  
**WTP:** {row['willingness_to_pay']}/15  
**Distribution:** {row['distribution']}/15  
**Automation:** {row['automation']}/15  
**Recurring:** {row['recurring']}/10  
**Competition:** {row['competition']}/10  
**Moat:** {row['moat']}/5  
**Speed to Revenue:** {row['speed_to_revenue']}/10  
**Stack Fit:** {row['stack_fit']}/5

**Gross Margin:** {row['gross_margin']}  
**LTV:** {row['ltv']}  
**CAC:** {row['cac']}  
**$1k MRR:** {row['customers_for_1k_mrr']}  
**$5k MRR:** {row['customers_for_5k_mrr']}  
**$10k MRR:** {row['customers_for_10k_mrr']}  
**Truth Test:** {row['truth_test']}
""")

        top = ranked[0]
        if top["total_score"] > 85:
            st.success(f"PROVISIONAL WINNER: {top['idea_name']} — {top['total_score']}/100")
        else:
            st.warning(f"NO WINNER — أعلى نتيجة {top['total_score']}/100")


# =========================================================
# FINAL
# =========================================================

if st.session_state.objection:
    st.divider()
    st.header("🔪 FINAL OBJECTION")
    st.markdown(st.session_state.objection)

if st.session_state.verdict:
    st.divider()
    st.header("🏛️ FINAL VERDICT")
    st.markdown(st.session_state.verdict)

    report = build_report(
        st.session_state.ideas or [],
        st.session_state.blocked or [],
        st.session_state.research or {},
        st.session_state.killer or {},
        st.session_state.rebuttal or {},
        st.session_state.killer_final or {},
        st.session_state.operator_rows or [],
        st.session_state.objection or "",
        st.session_state.verdict or "",
    )

    st.divider()
    st.header("📥 التقرير")
    st.download_button(
        "📝 تحميل التقرير TXT",
        data=report.encode("utf-8"),
        file_name="MD_Investment_Research.txt",
        mime="text/plain",
        use_container_width=True,
    )

    try:
        pdf = create_pdf(report)
        st.download_button(
            "📄 تحميل التقرير PDF",
            data=pdf,
            file_name="MD_Investment_Research.pdf",
            mime="application/pdf",
            use_container_width=True,
        )
    except Exception as exc:
        with st.expander("🔧 مشكلة PDF"):
            st.code(str(exc))


# =========================================================
# RESET
# =========================================================

if st.session_state.ideas or st.session_state.error or st.session_state.verdict:
    st.divider()
    if st.button("🗑️ مسح كل شيء وبدء بحث جديد"):
        for key, value in DEFAULT_STATE.items():
            st.session_state[key] = value
        st.rerun()
