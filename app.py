import os
import re
import json
import glob
import time
import math
import difflib
from urllib.parse import urlparse

import streamlit as st
from groq import Groq
from ddgs import DDGS
from fpdf import FPDF


# =========================================================
# BUILD
# =========================================================

BUILD_ID = "V8.3-EXACT-JTBD-EVIDENCE-PROOF"


# =========================================================
# PAGE
# =========================================================

st.set_page_config(
    page_title="MD Investment Research Council",
    page_icon="🧠",
    layout="wide",
)

st.markdown(
    """
    <style>
    .stApp { direction: rtl; }
    h1,h2,h3,h4,h5,p { text-align:right; }
    div[data-testid="stMarkdownContainer"] { direction:rtl; text-align:right; }
    textarea, div[data-baseweb="textarea"] textarea {
        direction:rtl !important;
        text-align:right !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# =========================================================
# CLIENT / MODELS
# =========================================================

client = Groq(
    api_key=st.secrets["GROQ_API_KEY"],
    timeout=120.0,
)

HUNTER_MODEL = "openai/gpt-oss-120b"
KILLER_MODEL = "qwen/qwen3.8-27b"
OPERATOR_MODEL = "openai/gpt-oss-20b"

MODEL_LAST_USED = {}

MIN_MODEL_GAP = {
    HUNTER_MODEL: 25,
    KILLER_MODEL: 25,
    OPERATOR_MODEL: 25,
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
    "App store privacy declaration",
    "General SaaS subscription monitoring",
    "Generic scope creep detection",
    "Noon warranty workflow automation",
]

REJECTED_CONCEPTS = {
    "freight_invoice_audit": [
        "freight invoice",
        "shipping invoice",
        "shipment invoice",
        "freight audit",
        "shipping audit",
        "فواتير الشحن",
        "فاتورة الشحن",
        "تدقيق الشحن",
    ],
    "invoice_collection": [
        "invoice collection",
        "collect invoices",
        "invoice chasing",
        "تحصيل الفواتير",
    ],
    "rfq_comparison": [
        "rfq comparison",
        "compare quotations",
        "مقارنة عروض الأسعار",
    ],
    "tender_rfp": [
        "tender ai",
        "rfp ai",
        "tender analysis",
        "تحليل المناقصات",
    ],
    "warranty": [
        "warranty management",
        "warranty workflow",
        "إدارة الضمان",
    ],
    "seo_content": [
        "seo content refresh",
        "content refresh",
        "تحديث المحتوى",
    ],
    "saas_monitoring": [
        "subscription monitoring",
        "saas monitoring",
        "مراقبة الاشتراكات",
    ],
    "scope_creep": [
        "scope creep",
        "تجاوز نطاق المشروع",
        "تغير نطاق المشروع",
    ],
}


# =========================================================
# GENERIC HELPERS
# =========================================================

def norm(text):
    text = (text or "").lower()
    text = re.sub(r"[^a-z0-9\u0600-\u06ff\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def cut(text, n):
    text = str(text or "").strip()
    if len(text) <= n:
        return text
    return text[:n] + "\n[تم اختصار الباقي لتقليل الـtokens]"


def retry_seconds(msg):
    match = re.search(r"try again in\s+([0-9.]+)s", msg, re.I)
    if match:
        return max(3, math.ceil(float(match.group(1))) + 3)

    match = re.search(r"try again in\s+([0-9.]+)ms", msg, re.I)
    if match:
        return max(2, math.ceil(float(match.group(1)) / 1000) + 2)

    return 12


def wait_model(model, stage, status):
    last = MODEL_LAST_USED.get(model)
    if not last:
        return

    remaining = MIN_MODEL_GAP.get(model, 10) - (time.time() - last)
    if remaining > 0:
        seconds = math.ceil(remaining)
        status.warning(
            f"⏳ {stage}: انتظار {seconds} ثانية لحماية الحد المجاني لـGroq..."
        )
        time.sleep(seconds)


def obj(props):
    return {
        "type": "object",
        "properties": props,
        "required": list(props.keys()),
        "additionalProperties": False,
    }


def arr(items, min_items=None, max_items=None):
    result = {"type": "array", "items": items}
    if min_items is not None:
        result["minItems"] = min_items
    if max_items is not None:
        result["maxItems"] = max_items
    return result


# =========================================================
# GROQ CALLS
# =========================================================

def call_json(
    model,
    system,
    prompt,
    schema_name,
    schema,
    max_tokens,
    stage,
    status,
    reasoning="none",
    retries=4,
):
    wait_model(model, stage, status)
    errors = []

    for _ in range(retries):
        try:
            kwargs = {
                "model": model,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": prompt},
                ],
                "temperature": 0.25,
                "max_completion_tokens": max_tokens,
                "response_format": {
                    "type": "json_schema",
                    "json_schema": {
                        "name": schema_name,
                        "strict": True,
                        "schema": schema,
                    },
                },
            }

            if reasoning:
                kwargs["reasoning_effort"] = reasoning
                kwargs["reasoning_format"] = "hidden"

            response = client.chat.completions.create(**kwargs)
            MODEL_LAST_USED[model] = time.time()

            content = response.choices[0].message.content
            if content:
                return {"ok": True, "data": json.loads(content), "error": None}

            errors.append(f"{model}: empty JSON response")

        except Exception as e:
            msg = str(e)
            errors.append(msg)

            if "429" in msg or "rate_limit" in msg.lower():
                seconds = retry_seconds(msg)
                status.warning(f"⏳ {stage}: انتظار {seconds} ثانية...")
                time.sleep(seconds)
                continue

            if "413" in msg or "request_too_large" in msg.lower():
                break

            time.sleep(2)

    return {"ok": False, "data": None, "error": "\n\n".join(errors)}


def call_text(
    model,
    system,
    prompt,
    max_tokens,
    stage,
    status,
    reasoning="none",
    retries=4,
):
    wait_model(model, stage, status)
    errors = []

    for _ in range(retries):
        try:
            kwargs = {
                "model": model,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": prompt},
                ],
                "temperature": 0.25,
                "max_completion_tokens": max_tokens,
            }

            if reasoning:
                kwargs["reasoning_effort"] = reasoning
                kwargs["reasoning_format"] = "hidden"

            response = client.chat.completions.create(**kwargs)
            MODEL_LAST_USED[model] = time.time()

            content = response.choices[0].message.content
            if content:
                return {"ok": True, "text": content.strip(), "error": None}

            errors.append(f"{model}: empty response")

        except Exception as e:
            msg = str(e)
            errors.append(msg)

            if "429" in msg or "rate_limit" in msg.lower():
                seconds = retry_seconds(msg)
                status.warning(f"⏳ {stage}: انتظار {seconds} ثانية...")
                time.sleep(seconds)
                continue

            time.sleep(2)

    return {"ok": False, "text": "", "error": "\n\n".join(errors)}


# =========================================================
# INPUT PREPARATION
# =========================================================

def prepare_case_brief(text):
    text = text.strip()

    markers = [
        "\n# الوكلاء",
        "\n## الوكيل الأول",
        "\n# قواعد المناظرة",
        "\n# نظام التقييم",
        "\n# شرط النجاح",
        "\n# اختبار الحقيقة",
        "\n# المرحلة النهائية",
    ]

    cuts = [text.find(m) for m in markers if text.find(m) != -1]
    if cuts:
        text = text[: min(cuts)]

    if len(text) > 13000:
        text = text[:10000] + "\n[...اختصار محلي...]\n" + text[-2500:]

    return text


# =========================================================
# BLACKLIST
# =========================================================

def blacklist_check(idea):
    combined = norm(
        " ".join(
            [
                idea.get("name", ""),
                idea.get("one_liner", ""),
                idea.get("product", ""),
                idea.get("problem", ""),
                idea.get("job_to_be_done", ""),
            ]
        )
    )

    short = norm(f"{idea.get('name','')} {idea.get('one_liner','')}")

    for rejected in REJECTED_IDEAS:
        rejected_norm = norm(rejected)
        if rejected_norm and (
            rejected_norm in combined
            or difflib.SequenceMatcher(None, short, rejected_norm).ratio() >= 0.72
        ):
            return True, f"تشابه مع فكرة مرفوضة: {rejected}"

    for concept, phrases in REJECTED_CONCEPTS.items():
        hits = [p for p in phrases if norm(p) and norm(p) in combined]
        if hits:
            return True, f"تشابه مفاهيمي مع ({concept}): {', '.join(hits)}"

    return False, ""


def filter_ideas(data):
    passed, blocked = [], []

    for idea in data["ideas"]:
        local_block, reason = blacklist_check(idea)

        if local_block or idea.get("similar_to_rejected"):
            blocked.append(
                {
                    "idea": idea,
                    "reason": reason
                    or idea.get(
                        "rejected_similarity_explanation",
                        "قريبة من فكرة مرفوضة",
                    ),
                }
            )
        else:
            passed.append(idea)

    return passed, blocked


# =========================================================
# HUNTER SCHEMA + TOP-UP
# =========================================================

idea_schema = obj(
    {
        "id": {"type": "string"},
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
        "automation": {"type": "integer", "minimum": 0, "maximum": 100},
        "human_work": {"type": "string"},
        "why_now": {"type": "string"},
        "job_to_be_done": {"type": "string"},
        "direct_competitor_definition": {"type": "string"},
        "not_a_direct_competitor": {"type": "string"},
        "search_term_1": {"type": "string"},
        "search_term_2": {"type": "string"},
        "search_term_3": {"type": "string"},
        "pricing_search_term": {"type": "string"},
        "wtp_search_term": {"type": "string"},
        "distribution_search_term": {"type": "string"},
        "regulatory_sensitive": {"type": "boolean"},
        "regulatory_search_term": {"type": "string"},
        "similar_to_rejected": {"type": "boolean"},
        "rejected_similarity_explanation": {"type": "string"},
    }
)

# IMPORTANT:
# لا نجبر الموديل على إخراج 3 أفكار طويلة في طلب واحد.
# الدفعة الأولى تسمح بـ 1-2 فقط، ثم نكمل بفكرة واحدة في كل Top-Up.
HUNTER_BATCH_SCHEMA = obj({"ideas": arr(idea_schema, 1, 2)})
HUNTER_TOPUP_SCHEMA = obj({"idea": idea_schema})


def idea_signature(idea):
    return norm(
        " ".join(
            [
                idea.get("name", ""),
                idea.get("one_liner", ""),
                idea.get("job_to_be_done", ""),
                idea.get("buyer", ""),
            ]
        )
    )


def is_duplicate_idea(candidate, existing):
    candidate_sig = idea_signature(candidate)
    candidate_name = norm(candidate.get("name", ""))
    candidate_jtbd = norm(candidate.get("job_to_be_done", ""))

    for old in existing:
        old_sig = idea_signature(old)
        old_name = norm(old.get("name", ""))
        old_jtbd = norm(old.get("job_to_be_done", ""))

        if candidate_name and old_name:
            if candidate_name == old_name:
                return True

            if difflib.SequenceMatcher(
                None,
                candidate_name,
                old_name,
            ).ratio() >= 0.80:
                return True

        if candidate_jtbd and old_jtbd:
            if difflib.SequenceMatcher(
                None,
                candidate_jtbd,
                old_jtbd,
            ).ratio() >= 0.82:
                return True

        if candidate_sig and old_sig:
            if difflib.SequenceMatcher(
                None,
                candidate_sig,
                old_sig,
            ).ratio() >= 0.78:
                return True

    return False


def renumber_ideas(ideas):
    for index, idea in enumerate(ideas, 1):
        idea["id"] = f"idea-{index}"
    return ideas


def hunter_core_rules():
    return """
أنت THE HUNTER.

ابحث عن أماكن تتحرك فيها الأموال فعلياً.
الأولوية:
- ألم مالي حقيقي
- willingness-to-pay
- distribution
- automation
- repeat usage
- speed to revenue

ممنوع:
- AI wrapper بلا قيمة مستقلة
- Dashboard عام
- أداة يستطيع ChatGPT تنفيذها بما يكفي
- إعادة تغليف فكرة رفضها المستخدم

لكل فكرة:
- حدد Job-to-be-Done بدقة وبصيغة فعل/نتيجة واضحة.
- عرّف المنافس المباشر بدقة: يجب أن يطابق نفس المشتري تقريباً، نفس trigger/input،
  نفس المهمة الأساسية، ونفس output/action النهائي.
- إذا كان المنتج في نفس المجال لكنه يحل جزءاً مختلفاً من المهمة فهو ADJACENT وليس Direct.
- اختلاف نوع الضريبة/نوع الامتثال/نوع التدفق المالي اختلاف جوهري وليس Feature صغيرة.
- اشرح ما الذي لا يُعد منافساً مباشراً.
- Search Terms يجب أن تبحث عن نفس الفعل النهائي، لا عن المجال العام فقط.
- لا تخترع أرقام سوق أو قوانين كحقائق.

إذا كانت الفكرة قريبة من فكرة مرفوضة:
similar_to_rejected=true.

لا تختر WINNER.
اجعل الحقول مختصرة ومباشرة حتى يكتمل JSON.
"""


def run_hunter(case_brief, status):
    # -----------------------------------------------------
    # 1) الدفعة الأولى: 1-2 أفكار فقط
    # -----------------------------------------------------
    system = hunter_core_rules() + """

في هذه الدفعة أخرج فكرة أو فكرتين فقط.
لا تحاول إخراج 3 أفكار الآن.
"""

    prompt = f"""
حالة المستخدم:

{case_brief}

القائمة المرفوضة:

{json.dumps(REJECTED_IDEAS, ensure_ascii=False)}

أنشئ الآن أفضل فكرة أو فكرتين مختلفتين اقتصادياً.
"""

    initial = call_json(
        HUNTER_MODEL,
        system,
        prompt,
        "hunter_batch_v81",
        HUNTER_BATCH_SCHEMA,
        1450,
        "THE HUNTER — الدفعة الأولى",
        status,
        "low",
    )

    if not initial["ok"]:
        return initial

    ideas = []

    for candidate in initial["data"].get("ideas", []):
        if not is_duplicate_idea(candidate, ideas):
            ideas.append(candidate)

    # -----------------------------------------------------
    # 2) Top-Up: فكرة واحدة في كل مرة حتى نصل إلى 3
    # -----------------------------------------------------
    attempts = 0
    max_topup_attempts = 5

    while len(ideas) < 3 and attempts < max_topup_attempts:
        attempts += 1

        existing_summary = [
            {
                "name": idea.get("name", ""),
                "buyer": cut(idea.get("buyer", ""), 180),
                "job_to_be_done": cut(
                    idea.get("job_to_be_done", ""),
                    260,
                ),
            }
            for idea in ideas
        ]

        topup_system = hunter_core_rules() + """

أخرج فكرة واحدة فقط في الحقل idea.
يجب أن تكون مختلفة اقتصادياً عن الأفكار الموجودة.
لا تعيد نفس Job-to-be-Done باسم جديد.
"""

        topup_prompt = f"""
حالة المستخدم:

{case_brief}

الأفكار الموجودة بالفعل:

{json.dumps(existing_summary, ensure_ascii=False)}

القائمة المرفوضة:

{json.dumps(REJECTED_IDEAS, ensure_ascii=False)}

نحتاج فكرة واحدة جديدة فقط حتى نصل إلى 3 أفكار.
لا تكرر أي فكرة موجودة.
"""

        status.info(
            f"🎯 THE HUNTER Top-Up: "
            f"لدينا {len(ideas)}/3 — توليد فكرة إضافية..."
        )

        topup = call_json(
            HUNTER_MODEL,
            topup_system,
            topup_prompt,
            "hunter_topup_v81",
            HUNTER_TOPUP_SCHEMA,
            900,
            f"THE HUNTER TOP-UP {attempts}",
            status,
            "low",
        )

        if not topup["ok"]:
            # لا نفشل مباشرة؛ نجرب محاولة Top-Up أخرى.
            continue

        candidate = topup["data"].get("idea")
        if not candidate:
            continue

        if is_duplicate_idea(candidate, ideas):
            status.warning(
                "⚠️ Hunter أعاد فكرة قريبة من الموجودة؛ "
                "سيحاول فكرة مختلفة."
            )
            continue

        ideas.append(candidate)

    # -----------------------------------------------------
    # 3) تحقق برمجي قبل الانتقال
    # -----------------------------------------------------
    if len(ideas) < 3:
        names = [idea.get("name", "بدون اسم") for idea in ideas]
        return {
            "ok": False,
            "data": None,
            "error": (
                "Hunter لم يتمكن من إنشاء 3 أفكار مختلفة بعد "
                f"{max_topup_attempts} محاولات Top-Up.\n\n"
                f"تم جمع {len(ideas)} فقط: {names}\n\n"
                "لم يبدأ Web Research حتى لا يصدر المجلس حكماً ناقصاً."
            ),
        }

    ideas = renumber_ideas(ideas[:3])

    return {
        "ok": True,
        "data": {"ideas": ideas},
        "error": None,
    }


# =========================================================
# DDGS SEARCH
# =========================================================

def domain(url):
    try:
        return urlparse(url).netloc.lower().replace("www.", "")
    except Exception:
        return ""


def search_item(raw, query_type):
    if not isinstance(raw, dict):
        return None

    url = raw.get("href") or raw.get("url") or ""
    if not url:
        return None

    return {
        "query_type": query_type,
        "title": str(raw.get("title") or "").strip(),
        "url": str(url).strip(),
        "snippet": cut(
            str(
                raw.get("body")
                or raw.get("snippet")
                or raw.get("content")
                or ""
            ).strip(),
            600,
        ),
        "page_excerpt": "",
    }


def ddgs_search(query, query_type, max_results=5):
    ddgs = DDGS(timeout=12)
    rows = ddgs.text(query, max_results=max_results) or []
    output = []

    for raw in rows:
        item = search_item(raw, query_type)
        if item:
            output.append(item)

    return output


def extract_best_pages(sources, max_pages=3):
    ddgs = DDGS(timeout=12)
    extracted = 0

    for source in sources:
        if extracted >= max_pages:
            break

        try:
            result = ddgs.extract(source["url"], fmt="text_plain")
            content = (
                result.get("content", "")
                if isinstance(result, dict)
                else ""
            )

            if content:
                source["page_excerpt"] = cut(str(content).strip(), 1500)
                extracted += 1

        except Exception:
            pass

        time.sleep(0.6)

    return extracted


def dedupe_sources(sources, limit=14):
    unique = []
    seen_urls = set()
    per_domain = {}

    for source in sources:
        url = source["url"]
        dm = domain(url)

        if url in seen_urls:
            continue

        if dm and per_domain.get(dm, 0) >= 3:
            continue

        seen_urls.add(url)
        per_domain[dm] = per_domain.get(dm, 0) + 1
        unique.append(source)

        if len(unique) >= limit:
            break

    return unique


# =========================================================
# RELEVANCE JUDGE — V8.3 EXACT JTBD
# =========================================================

SOURCE_CATEGORIES = [
    "DIRECT_COMPETITOR",
    "ADJACENT_COMPETITOR",
    "PRICING_EVIDENCE",
    "PROBLEM_EVIDENCE",
    "WTP_EVIDENCE",
    "DISTRIBUTION_SURFACE",
    "DISTRIBUTION_PROOF",
    "REGULATORY_EVIDENCE",
    "PLATFORM_RISK",
    "BACKGROUND",
    "IRRELEVANT",
]

source_eval_schema = obj(
    {
        "source_id": {"type": "string"},
        "relevance_score": {"type": "integer", "minimum": 0, "maximum": 100},
        "job_match_score": {"type": "integer", "minimum": 0, "maximum": 100},
        "same_job_to_be_done": {"type": "boolean"},
        "critical_job_difference": {"type": "string"},
        "material_job_difference": {"type": "boolean"},
        "authority_type": {
            "type": "string",
            "enum": [
                "OFFICIAL_GOVERNMENT",
                "OFFICIAL_VENDOR",
                "INDUSTRY_PRIMARY",
                "SECONDARY",
                "UNKNOWN",
            ],
        },
        "pricing_explicit": {"type": "boolean"},
        "wtp_signal": {
            "type": "string",
            "enum": ["NONE", "REAL_PAYMENT_SIGNAL"],
        },
        "distribution_signal": {
            "type": "string",
            "enum": ["NONE", "SURFACE_ONLY", "PROOF"],
        },
        "categories": arr(
            {"type": "string", "enum": SOURCE_CATEGORIES},
            1,
            5,
        ),
        "why_relevant": {"type": "string"},
    }
)

RELEVANCE_SCHEMA = obj({"evaluations": arr(source_eval_schema)})


def normalize_source_evaluation(ev):
    ev = dict(ev or {})
    categories = list(dict.fromkeys(ev.get("categories", []) or []))

    relevance = int(ev.get("relevance_score", 0) or 0)
    job_match = int(ev.get("job_match_score", 0) or 0)
    same_job = bool(ev.get("same_job_to_be_done", False))
    material_difference = bool(ev.get("material_job_difference", False))
    authority = ev.get("authority_type", "UNKNOWN")
    pricing_explicit = bool(ev.get("pricing_explicit", False))
    wtp_signal = ev.get("wtp_signal", "NONE")
    distribution_signal = ev.get("distribution_signal", "NONE")

    # Exact JTBD lock:
    # DIRECT means same core job, not merely same industry/domain.
    if "DIRECT_COMPETITOR" in categories:
        if not (
            same_job
            and not material_difference
            and relevance >= 80
            and job_match >= 80
        ):
            categories = [c for c in categories if c != "DIRECT_COMPETITOR"]
            if relevance >= 50 and "ADJACENT_COMPETITOR" not in categories:
                categories.append("ADJACENT_COMPETITOR")

    if "DIRECT_COMPETITOR" in categories and "ADJACENT_COMPETITOR" in categories:
        categories = [c for c in categories if c != "ADJACENT_COMPETITOR"]

    # Explicit pricing only.
    if "PRICING_EVIDENCE" in categories and not pricing_explicit:
        categories = [c for c in categories if c != "PRICING_EVIDENCE"]

    # Paying taxes / having a problem is NOT WTP.
    if "WTP_EVIDENCE" in categories and wtp_signal != "REAL_PAYMENT_SIGNAL":
        categories = [c for c in categories if c != "WTP_EVIDENCE"]

    # A directory/community is only a surface, not proof the channel converts.
    if distribution_signal == "SURFACE_ONLY":
        categories = [c for c in categories if c != "DISTRIBUTION_PROOF"]
        if "DISTRIBUTION_SURFACE" not in categories:
            categories.append("DISTRIBUTION_SURFACE")
    elif distribution_signal == "PROOF":
        if "DISTRIBUTION_PROOF" not in categories:
            categories.append("DISTRIBUTION_PROOF")
    else:
        categories = [
            c
            for c in categories
            if c not in ["DISTRIBUTION_SURFACE", "DISTRIBUTION_PROOF"]
        ]

    # Regulatory proof must come from an official authority/law source.
    if "REGULATORY_EVIDENCE" in categories:
        if authority != "OFFICIAL_GOVERNMENT":
            categories = [c for c in categories if c != "REGULATORY_EVIDENCE"]

    # Platform risk must be evidenced by the platform/vendor itself and closely match the job.
    if "PLATFORM_RISK" in categories:
        if not (
            authority == "OFFICIAL_VENDOR"
            and not material_difference
            and relevance >= 75
            and job_match >= 70
        ):
            categories = [c for c in categories if c != "PLATFORM_RISK"]

    if relevance < 25:
        categories = ["IRRELEVANT"]
    else:
        if "IRRELEVANT" in categories and len(categories) > 1:
            categories = [c for c in categories if c != "IRRELEVANT"]

        if not categories:
            categories = ["BACKGROUND"]

    ev["categories"] = list(dict.fromkeys(categories))
    ev["relevance_score"] = relevance
    ev["job_match_score"] = job_match
    ev["same_job_to_be_done"] = same_job
    ev["material_job_difference"] = material_difference
    ev["authority_type"] = authority
    ev["pricing_explicit"] = pricing_explicit
    ev["wtp_signal"] = wtp_signal
    ev["distribution_signal"] = distribution_signal
    ev["critical_job_difference"] = str(
        ev.get("critical_job_difference", "")
    ).strip()
    ev["why_relevant"] = str(ev.get("why_relevant", "")).strip()
    return ev


def evaluate_source_relevance(idea, sources, status):
    packet = []

    for idx, source in enumerate(sources, 1):
        source["source_id"] = f"S{idx}"
        packet.append(
            {
                "source_id": source["source_id"],
                "query_type": source["query_type"],
                "title": source["title"],
                "url": source["url"],
                "snippet": cut(source["snippet"], 420),
                "page_excerpt": cut(source.get("page_excerpt", ""), 550),
            }
        )

    system = """
أنت Research Relevance Judge شديد الصرامة.

لا تحكم على جاذبية فكرة المشروع. صنّف الأدلة فقط.

أولاً قارن Job-to-be-Done بدقة عبر:
1) من هو المشتري/المستخدم،
2) ما الـtrigger أو input،
3) ما المهمة الأساسية أو الالتزام الذي يُنفذ،
4) ما الـoutput/action النهائي،
5) هل يستطيع المصدر/المنتج أن يحل محل المنتج المقترح فعلياً.

DIRECT_COMPETITOR:
يؤدي نفس المهمة الجوهرية تقريباً ويمكن أن يحل محل المنتج.
يجب أن يكون same_job_to_be_done=true و job_match_score مرتفعاً.
وجود المنتج في نفس المجال لا يكفي.
مثال عام: sales-tax/VAT calculation ليس نفس income-tax withholding/remittance
إذا كانت الفكرة تتطلب حجز ضريبة دخل وتحويلها.

ADJACENT_COMPETITOR:
نفس المجال أو workflow قريب، لكنه لا ينفذ نفس المهمة النهائية كاملة.

PROBLEM_EVIDENCE:
يثبت أن الألم/الالتزام/التكلفة موجودة.
مهم: وجود التزام ضريبي أو مشكلة لا يثبت willingness-to-pay.

WTP_EVIDENCE:
يحتاج REAL PAYMENT SIGNAL:
- عميل مدفوع أو case study لعميل،
- transaction volume مدفوع،
- survey/interview صريح عن الدفع،
- buyer spend فعلي،
- evidence أن نفس المشتري يشتري نفس الحل.
مجرد صفحة أسعار أو وجود المشكلة لا يكفي وحده.

PRICING_EVIDENCE:
سعر/رسوم/نسبة/خطة/نطاق سعر صريح لنفس الحل أو حل قريب جداً.
pricing_explicit=true فقط عند وجود رقم/نموذج سعر واضح في المصدر.

DISTRIBUTION_SURFACE:
وجود directory, marketplace, community, association أو ecosystem يمكن الوصول من خلاله للمشتري.
هذا لا يثبت أن القناة تنتج عملاء.

DISTRIBUTION_PROOF:
دليل أقوى مثل:
- شراكة/تكامل فعلي مع قناة،
- marketplace listing لمنتج مماثل،
- case study يذكر channel/acquisition،
- measurable referral/affiliate acquisition،
- دليل أن buyers adopt integrations عبر هذه القناة.
distribution_signal=PROOF فقط عند وجود هذا النوع من الدليل.

REGULATORY_EVIDENCE:
قانون/جهة حكومية/منظم رسمي فقط.
authority_type يجب أن يكون OFFICIAL_GOVERNMENT.

PLATFORM_RISK:
مصدر رسمي من المنصة الأساسية نفسها يثبت أنها تقدم أو أعلنت وظيفة شديدة القرب من JTBD.
مجرد توقع أنها "قد تبنيها" لا يكفي.

authority_type:
OFFICIAL_GOVERNMENT = جهة حكومية/منظم/نص رسمي.
OFFICIAL_VENDOR = صفحة أو docs رسمية لشركة/منصة المنتج.
INDUSTRY_PRIMARY = شركة/مؤسسة تعمل في المجال تتحدث عن منتجها/عملائها.
SECONDARY = مدونة/صحافة/مراجعة/دليل طرف ثالث.
UNKNOWN = غير واضح.

إذا وجدت اختلافاً جوهرياً، اكتبه صراحة في critical_job_difference
واجعل material_job_difference=true. اختلاف جوهري واحد يمنع DIRECT_COMPETITOR.
"""

    prompt = f"""
IDEA:

Name: {idea["name"]}
Buyer: {idea["buyer"]}
Problem: {idea["problem"]}
Product: {idea["product"]}
Job-to-be-Done: {idea["job_to_be_done"]}

DIRECT COMPETITOR MUST:
{idea["direct_competitor_definition"]}

NOT A DIRECT COMPETITOR:
{idea["not_a_direct_competitor"]}

SOURCES:
{json.dumps(packet, ensure_ascii=False)}
"""

    result = call_json(
        OPERATOR_MODEL,
        system,
        prompt,
        "source_relevance_v83",
        RELEVANCE_SCHEMA,
        2100,
        f"Relevance Judge — {idea['name']}",
        status,
        "low",
    )

    if not result["ok"]:
        return result

    mapping = {
        x["source_id"]: normalize_source_evaluation(x)
        for x in result["data"]["evaluations"]
    }

    for source in sources:
        source["evaluation"] = mapping.get(
            source["source_id"],
            normalize_source_evaluation(
                {
                    "source_id": source["source_id"],
                    "relevance_score": 0,
                    "job_match_score": 0,
                    "same_job_to_be_done": False,
                    "critical_job_difference": "لم يرجع تقييم للمصدر.",
                    "material_job_difference": True,
                    "authority_type": "UNKNOWN",
                    "pricing_explicit": False,
                    "wtp_signal": "NONE",
                    "distribution_signal": "NONE",
                    "categories": ["IRRELEVANT"],
                    "why_relevant": "لم يرجع تقييم للمصدر.",
                }
            ),
        )

    return {"ok": True, "sources": sources, "error": None}

# =========================================================
# RESEARCH GATE — V8.3
# =========================================================

def source_qualifies_for_category(source, category):
    ev = source.get("evaluation", {})
    categories = ev.get("categories", [])
    relevance = ev.get("relevance_score", 0)
    job_match = ev.get("job_match_score", 0)
    same_job = ev.get("same_job_to_be_done", False)
    material_difference = ev.get("material_job_difference", False)

    if "IRRELEVANT" in categories:
        return False

    if category == "DIRECT_COMPETITOR":
        return (
            "DIRECT_COMPETITOR" in categories
            and relevance >= 80
            and job_match >= 80
            and same_job
            and not material_difference
        )

    if category == "ADJACENT_COMPETITOR":
        return (
            "ADJACENT_COMPETITOR" in categories
            and relevance >= 50
        )

    if category == "PRICING_EVIDENCE":
        return (
            "PRICING_EVIDENCE" in categories
            and relevance >= 60
            and ev.get("pricing_explicit", False)
        )

    if category == "PROBLEM_EVIDENCE":
        return (
            "PROBLEM_EVIDENCE" in categories
            and relevance >= 55
        )

    if category == "WTP_EVIDENCE":
        return (
            "WTP_EVIDENCE" in categories
            and relevance >= 65
            and ev.get("wtp_signal") == "REAL_PAYMENT_SIGNAL"
        )

    if category == "DISTRIBUTION_SURFACE":
        return (
            "DISTRIBUTION_SURFACE" in categories
            and relevance >= 50
            and ev.get("distribution_signal") in ["SURFACE_ONLY", "PROOF"]
        )

    if category == "DISTRIBUTION_PROOF":
        return (
            "DISTRIBUTION_PROOF" in categories
            and relevance >= 65
            and ev.get("distribution_signal") == "PROOF"
        )

    if category == "REGULATORY_EVIDENCE":
        return (
            "REGULATORY_EVIDENCE" in categories
            and relevance >= 65
            and ev.get("authority_type") == "OFFICIAL_GOVERNMENT"
        )

    if category == "PLATFORM_RISK":
        return (
            "PLATFORM_RISK" in categories
            and relevance >= 75
            and job_match >= 70
            and not material_difference
            and ev.get("authority_type") == "OFFICIAL_VENDOR"
        )

    return relevance >= 60


def category_sources(sources, category):
    return [
        source
        for source in sources
        if source_qualifies_for_category(source, category)
    ]


def research_gate(idea, sources):
    direct = category_sources(sources, "DIRECT_COMPETITOR")
    adjacent = category_sources(sources, "ADJACENT_COMPETITOR")
    pricing = category_sources(sources, "PRICING_EVIDENCE")
    problem = category_sources(sources, "PROBLEM_EVIDENCE")
    wtp = category_sources(sources, "WTP_EVIDENCE")
    distribution_surface = category_sources(sources, "DISTRIBUTION_SURFACE")
    distribution_proof = category_sources(sources, "DISTRIBUTION_PROOF")
    regulatory = category_sources(sources, "REGULATORY_EVIDENCE")

    relevant_domains = {
        domain(s["url"])
        for s in sources
        if s.get("evaluation", {}).get("relevance_score", 0) >= 60
        and "IRRELEVANT" not in s.get("evaluation", {}).get("categories", [])
        and domain(s.get("url", ""))
    }

    checks = {
        "2_exact_direct_competitors": len(direct) >= 2,
        "problem_evidence": len(problem) >= 1,
        "pricing_evidence": len(pricing) >= 1,
        "real_wtp_evidence": len(wtp) >= 1,
        "distribution_proof": len(distribution_proof) >= 1,
        "3_relevant_domains": len(relevant_domains) >= 3,
        "official_regulatory_evidence_if_needed": (
            len(regulatory) >= 1 if idea["regulatory_sensitive"] else True
        ),
    }

    score = sum(1 for value in checks.values() if value)
    max_score = len(checks)

    status = "SUFFICIENT" if score == max_score else "INSUFFICIENT_EVIDENCE"

    return {
        "status": status,
        "score": score,
        "max_score": max_score,
        "checks": checks,
        "direct_competitors": [s["source_id"] for s in direct],
        "adjacent_competitors": [s["source_id"] for s in adjacent],
        "problem_sources": [s["source_id"] for s in problem],
        "pricing_sources": [s["source_id"] for s in pricing],
        "wtp_sources": [s["source_id"] for s in wtp],
        "distribution_surface_sources": [
            s["source_id"] for s in distribution_surface
        ],
        "distribution_proof_sources": [
            s["source_id"] for s in distribution_proof
        ],
        "regulatory_sources": [s["source_id"] for s in regulatory],
        "relevant_domain_count": len(relevant_domains),
    }


def missing_query_types(gate):
    missing = []

    if not gate["checks"]["2_exact_direct_competitors"]:
        missing.append("direct_competitors")
    if not gate["checks"]["problem_evidence"]:
        missing.append("problem")
    if not gate["checks"]["pricing_evidence"]:
        missing.append("pricing")
    if not gate["checks"]["real_wtp_evidence"]:
        missing.append("wtp")
    if not gate["checks"]["distribution_proof"]:
        missing.append("distribution_proof")
    if not gate["checks"]["official_regulatory_evidence_if_needed"]:
        missing.append("regulatory")

    return missing

# =========================================================
# TARGETED RESEARCH — V8.3
# =========================================================

MAX_RESEARCH_ROUNDS = 3


def base_queries(idea):
    queries = [
        ("direct_competitors", idea["search_term_1"]),
        ("direct_competitors", idea["search_term_2"]),
        ("direct_competitors", idea["search_term_3"]),
        (
            "problem",
            f'"{idea["buyer"]}" "{cut(idea["problem"], 180)}" cost pain manual',
        ),
        ("pricing", idea["pricing_search_term"]),
        ("wtp", idea["wtp_search_term"]),
        ("distribution_proof", idea["distribution_search_term"]),
    ]

    if idea["regulatory_sensitive"] and idea["regulatory_search_term"].strip():
        queries.append(("regulatory", idea["regulatory_search_term"]))

    return queries


def brand_hint(title):
    title = str(title or "").strip()
    if not title:
        return ""

    first = re.split(r"\s*[|–—:]\s*|\s+-\s+", title, maxsplit=1)[0]
    return cut(first, 70).replace("\n", " ").strip()


def direct_competitor_hints(sources, limit=4):
    hints = []

    for source in sources:
        if source_qualifies_for_category(source, "DIRECT_COMPETITOR"):
            hint = brand_hint(source.get("title", ""))
            if hint and hint not in hints:
                hints.append(hint)

        if len(hints) >= limit:
            break

    return hints


def retry_queries_for(idea, missing_types, sources=None, round_no=2):
    sources = sources or []
    competitors = direct_competitor_hints(sources)
    output = []

    base = {
        "direct_competitors": [
            f'"{idea["job_to_be_done"]}" software API',
            f'"{idea["job_to_be_done"]}" service competitor',
            f'"{idea["direct_competitor_definition"]}"',
            f'"{idea["name"]}" alternatives exact workflow',
        ],
        "problem": [
            f'"{idea["buyer"]}" "{cut(idea["problem"], 160)}" cost',
            f'"{idea["job_to_be_done"]}" manual process errors delays',
            f'"{idea["job_to_be_done"]}" compliance risk cost',
        ],
        "pricing": [
            f'"{idea["job_to_be_done"]}" pricing fees',
            f'"{idea["job_to_be_done"]}" API pricing transaction fee',
        ],
        "wtp": [
            f'"{idea["job_to_be_done"]}" customer case study paid',
            f'"{idea["job_to_be_done"]}" customers transaction volume',
            f'"{idea["buyer"]}" paid solution "{idea["job_to_be_done"]}"',
            f'"{idea["job_to_be_done"]}" survey willingness to pay',
        ],
        "distribution_proof": [
            f'"{idea["job_to_be_done"]}" integration partner case study',
            f'"{idea["job_to_be_done"]}" marketplace app listing integration',
            f'"{idea["buyer"]}" adopted integration case study',
            f'"{idea["job_to_be_done"]}" referral affiliate customer acquisition',
        ],
        "regulatory": [
            idea.get("regulatory_search_term", ""),
            f'site:.gov "{idea["job_to_be_done"]}"',
            f'"{idea["job_to_be_done"]}" regulator official law',
        ],
    }

    for competitor in competitors:
        if "pricing" in missing_types:
            base["pricing"].extend(
                [
                    f'"{competitor}" pricing fees',
                    f'"{competitor}" transaction fee pricing',
                ]
            )

        if "wtp" in missing_types:
            base["wtp"].extend(
                [
                    f'"{competitor}" customer case study paid customers',
                    f'"{competitor}" customers transaction volume',
                    f'"{competitor}" reviews pricing customers',
                ]
            )

        if "distribution_proof" in missing_types:
            base["distribution_proof"].extend(
                [
                    f'"{competitor}" integration partner case study',
                    f'"{competitor}" marketplace app listing',
                    f'"{competitor}" referral affiliate partner',
                    f'"{competitor}" customer acquisition channel case study',
                ]
            )

    if round_no >= 3:
        if "direct_competitors" in missing_types:
            base["direct_competitors"].append(
                f'"{idea["job_to_be_done"]}" exact alternative platform'
            )

        if "pricing" in missing_types:
            base["pricing"].append(
                f'"{idea["name"]}" alternatives pricing fee'
            )

        if "wtp" in missing_types:
            base["wtp"].append(
                f'"{idea["job_to_be_done"]}" buyer spend paid pilot'
            )

        if "distribution_proof" in missing_types:
            base["distribution_proof"].extend(
                [
                    f'"{idea["buyer"]}" integration adoption case study',
                    f'"{idea["job_to_be_done"]}" partner announcement integration',
                    f'"{idea["job_to_be_done"]}" app marketplace customers',
                ]
            )

    seen = set()

    for missing in missing_types:
        for query in base.get(missing, []):
            query = str(query or "").strip()
            if query and query not in seen:
                seen.add(query)
                output.append((missing, query))

    return output[:12]


def run_search_queries(queries, status, idea_name):
    found = []
    errors = []

    for idx, (query_type, query) in enumerate(queries, 1):
        status.info(
            f"🔎 {idea_name}: بحث {idx}/{len(queries)} — {query_type}"
        )

        try:
            found.extend(ddgs_search(query, query_type, max_results=5))
        except Exception as e:
            errors.append(f"{query_type}: {query}: {e}")

        time.sleep(0.8)

    return [x for x in found if x.get("url")], errors


def research_one_idea(idea, status):
    all_errors = []

    phase1_sources, phase1_errors = run_search_queries(
        base_queries(idea),
        status,
        idea["name"],
    )

    all_errors.extend(phase1_errors)
    sources = dedupe_sources(phase1_sources, limit=14)

    if not sources:
        return {
            "ok": False,
            "research_status": "RESEARCH_FAILED",
            "gate": None,
            "sources": [],
            "error": (
                "DDGS returned no usable sources.\n"
                + "\n".join(all_errors[-8:])
            ),
        }

    extract_best_pages(sources, max_pages=3)
    judged = evaluate_source_relevance(idea, sources, status)

    if not judged["ok"]:
        return {
            "ok": False,
            "research_status": "RESEARCH_FAILED",
            "gate": None,
            "sources": sources,
            "error": "Relevance Judge failed.\n" + str(judged["error"]),
        }

    sources = judged["sources"]
    gate = research_gate(idea, sources)

    for round_no in range(2, MAX_RESEARCH_ROUNDS + 1):
        if gate["status"] == "SUFFICIENT":
            break

        missing = missing_query_types(gate)
        retry_qs = retry_queries_for(
            idea,
            missing,
            sources=sources,
            round_no=round_no,
        )

        if not retry_qs:
            break

        status.warning(
            f"🔁 {idea['name']}: Research Round "
            f"{round_no}/{MAX_RESEARCH_ROUNDS}. "
            f"المفقود فقط: {', '.join(missing)}"
        )

        extra, retry_errors = run_search_queries(
            retry_qs,
            status,
            idea["name"],
        )

        all_errors.extend(retry_errors)

        if not extra:
            continue

        combined = dedupe_sources(sources + extra, limit=18)
        extract_best_pages(combined, max_pages=4)

        judged_retry = evaluate_source_relevance(
            idea,
            combined,
            status,
        )

        if not judged_retry["ok"]:
            all_errors.append(
                f"Round {round_no} relevance judge failed: "
                f"{judged_retry['error']}"
            )
            continue

        sources = judged_retry["sources"]
        gate = research_gate(idea, sources)

    return {
        "ok": True,
        "research_status": gate["status"],
        "gate": gate,
        "sources": sources,
        "error": "\n".join(all_errors[-8:]) if all_errors else None,
    }


def research_all(ideas, status):
    research = {}

    for idx, idea in enumerate(ideas, 1):
        status.info(
            f"🌐 Evidence Research {idx}/{len(ideas)}: {idea['name']}"
        )

        research[idea["id"]] = research_one_idea(
            idea,
            status,
        )

        time.sleep(1)

    return research

# =========================================================
# EVIDENCE PACKET
# =========================================================

def source_packet(source):
    ev = source["evaluation"]

    return {
        "source_id": source["source_id"],
        "title": source["title"],
        "url": source["url"],
        "query_type": source["query_type"],
        "snippet": cut(source["snippet"], 500),
        "page_excerpt": cut(source.get("page_excerpt", ""), 700),
        "relevance_score": ev["relevance_score"],
        "job_match_score": ev.get("job_match_score", 0),
        "same_job_to_be_done": ev["same_job_to_be_done"],
        "critical_job_difference": ev.get("critical_job_difference", ""),
        "material_job_difference": ev.get("material_job_difference", False),
        "authority_type": ev.get("authority_type", "UNKNOWN"),
        "pricing_explicit": ev.get("pricing_explicit", False),
        "wtp_signal": ev.get("wtp_signal", "NONE"),
        "distribution_signal": ev.get("distribution_signal", "NONE"),
        "categories": ev["categories"],
        "why_relevant": ev["why_relevant"],
    }


def evidence_packet(research_result):
    sources = [
        source
        for source in research_result.get("sources", [])
        if source.get("evaluation", {}).get("relevance_score", 0) >= 50
        and "IRRELEVANT"
        not in source.get("evaluation", {}).get("categories", [])
    ]

    sources = sorted(
        sources,
        key=lambda source: (
            source["evaluation"].get("job_match_score", 0),
            source["evaluation"].get("relevance_score", 0),
        ),
        reverse=True,
    )

    return [source_packet(source) for source in sources[:12]]

# =========================================================
# EVIDENCE CATEGORY LOCK — V8.3
# =========================================================

CLAIM_TYPES = [
    "COMPETITION",
    "PRICING",
    "WTP",
    "DISTRIBUTION",
    "REGULATION",
    "PLATFORM",
    "OTHER",
]

CLAIM_TO_CATEGORY = {
    "COMPETITION": "DIRECT_COMPETITOR",
    "PRICING": "PRICING_EVIDENCE",
    "WTP": "WTP_EVIDENCE",
    "DISTRIBUTION": "DISTRIBUTION_PROOF",
    "REGULATION": "REGULATORY_EVIDENCE",
    "PLATFORM": "PLATFORM_RISK",
}


def evidence_source_map(research_result):
    return {
        source.get("source_id"): source
        for source in research_result.get("sources", [])
        if source.get("source_id")
    }


def evidence_ids_by_category(research_result):
    output = {
        "DIRECT_COMPETITOR": [],
        "ADJACENT_COMPETITOR": [],
        "PRICING_EVIDENCE": [],
        "PROBLEM_EVIDENCE": [],
        "WTP_EVIDENCE": [],
        "DISTRIBUTION_SURFACE": [],
        "DISTRIBUTION_PROOF": [],
        "REGULATORY_EVIDENCE": [],
        "PLATFORM_RISK": [],
    }

    for source in research_result.get("sources", []):
        for category in output:
            if source_qualifies_for_category(source, category):
                output[category].append(source.get("source_id"))

    return output


def validate_evidence_ids(ids, claim_type, research_result):
    ids = [str(value) for value in (ids or [])]
    source_map = evidence_source_map(research_result)
    required_category = CLAIM_TO_CATEGORY.get(claim_type)

    valid = []

    for source_id in ids:
        source = source_map.get(source_id)
        if not source:
            continue

        if required_category:
            if not source_qualifies_for_category(
                source,
                required_category,
            ):
                continue
        else:
            ev = source.get("evaluation", {})
            if (
                ev.get("relevance_score", 0) < 60
                or "IRRELEVANT" in ev.get("categories", [])
            ):
                continue

        valid.append(source_id)

    return valid


def claim_can_be_fatal(claim_type, evidence_ids, research_result):
    # Missing WTP/distribution is an evidence gap, not a fatal proof.
    if claim_type in ["OTHER", "WTP", "DISTRIBUTION"]:
        return False

    valid = validate_evidence_ids(
        evidence_ids,
        claim_type,
        research_result,
    )

    if not valid:
        return False

    source_map = evidence_source_map(research_result)

    # Competition kill requires exact JTBD.
    if claim_type == "COMPETITION":
        return any(
            source_qualifies_for_category(
                source_map[source_id],
                "DIRECT_COMPETITOR",
            )
            for source_id in valid
            if source_id in source_map
        )

    # Pricing can be fatal only when the price comparison is for the same/substitutable job.
    if claim_type == "PRICING":
        return any(
            source_map[source_id]
            .get("evaluation", {})
            .get("same_job_to_be_done", False)
            and not source_map[source_id]
            .get("evaluation", {})
            .get("material_job_difference", False)
            and source_map[source_id]
            .get("evaluation", {})
            .get("job_match_score", 0) >= 75
            for source_id in valid
            if source_id in source_map
        )

    # Regulation and platform risk are already locked to official primary evidence.
    if claim_type in ["REGULATION", "PLATFORM"]:
        return True

    return False

# =========================================================
# KILLER PER-IDEA SCHEMA
# =========================================================

claim_schema = obj(
    {
        "claim": {"type": "string"},
        "claim_type": {"type": "string", "enum": CLAIM_TYPES},
        "status": {
            "type": "string",
            "enum": [
                "VERIFIED_RISK",
                "UNVERIFIED_RISK",
                "EVIDENCE_GAP",
                "UNKNOWN",
            ],
        },
        "evidence_ids": arr({"type": "string"}, 0, 5),
    }
)

KILLER_ONE_SCHEMA = obj(
    {
        "idea_id": {"type": "string"},
        "research_status": {
            "type": "string",
            "enum": ["SUFFICIENT", "INSUFFICIENT_EVIDENCE", "RESEARCH_FAILED"],
        },
        "top_risks": arr(claim_schema, 3, 3),
        "kill_shot": {"type": "string"},
        "kill_shot_claim_type": {"type": "string", "enum": CLAIM_TYPES},
        "kill_shot_status": {
            "type": "string",
            "enum": [
                "VERIFIED_RISK",
                "UNVERIFIED_RISK",
                "EVIDENCE_GAP",
                "UNKNOWN",
            ],
        },
        "kill_shot_evidence_ids": arr({"type": "string"}, 0, 5),
        "evidence_for": {"type": "string"},
        "evidence_against": {"type": "string"},
        "unknowns": {"type": "string"},
        "score_out_of_10": {"type": "integer", "minimum": 0, "maximum": 10},
        "decision": {
            "type": "string",
            "enum": ["SURVIVES", "KILL IT", "INSUFFICIENT EVIDENCE", "RESEARCH FAILED"],
        },
    }
)


def enforce_killer_evidence(idea, research_result, data):
    actual_status = research_result["research_status"]
    data["research_status"] = actual_status

    if actual_status == "RESEARCH_FAILED":
        data["decision"] = "RESEARCH FAILED"
        return data

    for risk in data["top_risks"]:
        risk["evidence_ids"] = validate_evidence_ids(
            risk.get("evidence_ids", []),
            risk.get("claim_type", "OTHER"),
            research_result,
        )
        if risk["status"] == "VERIFIED_RISK" and not risk["evidence_ids"]:
            risk["status"] = "UNVERIFIED_RISK"
        if risk["status"] == "EVIDENCE_GAP":
            risk["evidence_ids"] = []

    data["kill_shot_evidence_ids"] = validate_evidence_ids(
        data.get("kill_shot_evidence_ids", []),
        data.get("kill_shot_claim_type", "OTHER"),
        research_result,
    )

    if data["kill_shot_status"] == "VERIFIED_RISK" and not data["kill_shot_evidence_ids"]:
        data["kill_shot_status"] = "UNVERIFIED_RISK"
    if data["kill_shot_status"] == "EVIDENCE_GAP":
        data["kill_shot_evidence_ids"] = []

    if data["decision"] == "KILL IT":
        if not (
            data["kill_shot_status"] == "VERIFIED_RISK"
            and claim_can_be_fatal(
                data.get("kill_shot_claim_type", "OTHER"),
                data.get("kill_shot_evidence_ids", []),
                research_result,
            )
        ):
            data["decision"] = "INSUFFICIENT EVIDENCE"

    if actual_status == "INSUFFICIENT_EVIDENCE" and data["decision"] == "SURVIVES":
        data["decision"] = "INSUFFICIENT EVIDENCE"

    return data


def run_killer_one(idea, research_result, status):
    system = """
أنت THE KILLER.

هاجم فكرة واحدة فقط.

قواعد Evidence Lock الإلزامية:
1. COMPETITION يحتاج DIRECT_COMPETITOR حقيقي:
   same_job_to_be_done=true + job_match_score>=80 + relevance>=80.
   ADJACENT_COMPETITOR لا يكفي لقتل الفكرة.
2. Pricing يحتاج PRICING_EVIDENCE صريح.
3. WTP يحتاج WTP_EVIDENCE مع REAL_PAYMENT_SIGNAL.
   وجود المشكلة أو الالتزام لا يثبت الدفع.
4. Distribution يحتاج DISTRIBUTION_PROOF.
   وجود directory/community = DISTRIBUTION_SURFACE فقط ولا يثبت CAC أو adoption.
5. Regulation يحتاج REGULATORY_EVIDENCE من OFFICIAL_GOVERNMENT.
6. Platform risk يحتاج PLATFORM_RISK من OFFICIAL_VENDOR وبـjob match قوي.
7. اختلاف جوهري في المهمة مثل sales tax مقابل contractor income-tax withholding
   يعني Adjacent وليس Direct.
8. غياب Pricing/WTP/Distribution = EVIDENCE_GAP وليس VERIFIED_RISK.
9. KILL IT فقط إذا Kill Shot = VERIFIED_RISK مع Evidence IDs مطابقة فعلاً.
10. RESEARCH FAILED يعني فشل تقني فعلي فقط.
"""

    packet = {
        "idea": idea,
        "research_gate": research_result["gate"],
        "research_status": research_result["research_status"],
        "allowed_evidence_ids_by_category": evidence_ids_by_category(research_result),
        "evidence": evidence_packet(research_result),
    }

    result = call_json(
        KILLER_MODEL,
        system,
        json.dumps(packet, ensure_ascii=False),
        "killer_one_v83",
        KILLER_ONE_SCHEMA,
        1400,
        f"KILLER — {idea['name']}",
        status,
        "none",
    )

    if result["ok"]:
        result["data"] = enforce_killer_evidence(idea, research_result, result["data"])
    return result


# =========================================================
# HUNTER REBUTTAL EVIDENCE LOCK
# =========================================================

support_claim_schema = obj(
    {
        "claim": {"type": "string"},
        "claim_type": {"type": "string", "enum": CLAIM_TYPES},
        "status": {"type": "string", "enum": ["SUPPORTED", "UNVERIFIED", "EVIDENCE_GAP"]},
        "evidence_ids": arr({"type": "string"}, 0, 5),
    }
)

REBUTTAL_ONE_SCHEMA = obj(
    {
        "idea_id": {"type": "string"},
        "valid_objection": {"type": "string"},
        "valid_objection_claim_type": {"type": "string", "enum": CLAIM_TYPES},
        "valid_objection_evidence_ids": arr({"type": "string"}, 0, 5),
        "disputed_objection": {"type": "string"},
        "disputed_objection_claim_type": {"type": "string", "enum": CLAIM_TYPES},
        "disputed_objection_evidence_ids": arr({"type": "string"}, 0, 5),
        "disputed_reason": {"type": "string"},
        "defense_claims": arr(support_claim_schema, 0, 3),
        "evidence_needed": {"type": "string"},
        "position": {"type": "string", "enum": ["DEFEND", "DROP", "NEEDS MORE EVIDENCE"]},
    }
)


def enforce_hunter_rebuttal(research_result, data):
    data["valid_objection_evidence_ids"] = validate_evidence_ids(
        data.get("valid_objection_evidence_ids", []),
        data.get("valid_objection_claim_type", "OTHER"),
        research_result,
    )
    data["disputed_objection_evidence_ids"] = validate_evidence_ids(
        data.get("disputed_objection_evidence_ids", []),
        data.get("disputed_objection_claim_type", "OTHER"),
        research_result,
    )

    for claim in data.get("defense_claims", []):
        claim["evidence_ids"] = validate_evidence_ids(
            claim.get("evidence_ids", []),
            claim.get("claim_type", "OTHER"),
            research_result,
        )
        if claim["status"] == "SUPPORTED" and not claim["evidence_ids"]:
            claim["status"] = "UNVERIFIED"
        if claim["status"] == "EVIDENCE_GAP":
            claim["evidence_ids"] = []
    return data


def run_rebuttal_one(idea, research_result, killer_data, status):
    system = """
أنت THE HUNTER.

هذه فرصتك الوحيدة للرد على Killer لفكرة واحدة.

Evidence Lock:
- ممنوع إدخال حقيقة سوقية/سعرية/تنظيمية جديدة بلا Source ID مناسب.
- Competition claim يحتاج DIRECT_COMPETITOR exact-JTBD، وليس Adjacent.
- Pricing claim يحتاج PRICING_EVIDENCE صريح.
- WTP claim يحتاج REAL_PAYMENT_SIGNAL.
- Distribution claim يحتاج DISTRIBUTION_PROOF؛ surface فقط لا يكفي.
- Regulatory claim يحتاج OFFICIAL_GOVERNMENT.
- إذا لا يوجد الدليل: UNVERIFIED أو EVIDENCE_GAP.
- لا تستخدم وجود المشكلة كدليل على willingness-to-pay.
- disputed_reason تفسير منطقي للأدلة فقط، وليس مكاناً لاختراع facts.
- DROP أفضل من الدفاع عن عيب قاتل موثق.
- NEEDS MORE EVIDENCE أفضل من التخمين.
"""

    packet = {
        "idea": idea,
        "research_status": research_result["research_status"],
        "research_gate": research_result["gate"],
        "allowed_evidence_ids_by_category": evidence_ids_by_category(research_result),
        "evidence": evidence_packet(research_result),
        "killer": killer_data,
    }

    result = call_json(
        HUNTER_MODEL,
        system,
        json.dumps(packet, ensure_ascii=False),
        "hunter_rebuttal_one_v83",
        REBUTTAL_ONE_SCHEMA,
        850,
        f"HUNTER REBUTTAL — {idea['name']}",
        status,
        "low",
    )
    if result["ok"]:
        result["data"] = enforce_hunter_rebuttal(research_result, result["data"])
    return result


# =========================================================
# KILLER FINAL
# =========================================================

KILLER_FINAL_ONE_SCHEMA = obj(
    {
        "idea_id": {"type": "string"},
        "decision": {
            "type": "string",
            "enum": ["SURVIVES", "KILL IT", "INSUFFICIENT EVIDENCE", "RESEARCH FAILED"],
        },
        "remaining_problem": {"type": "string"},
        "wtp_real": {"type": "boolean"},
        "distribution_real": {"type": "boolean"},
        "feature_or_company": {"type": "string", "enum": ["FEATURE", "COMPANY", "UNCLEAR"]},
        "final_score_out_of_10": {"type": "integer", "minimum": 0, "maximum": 10},
        "decisive_claim": {"type": "string"},
        "decisive_claim_type": {"type": "string", "enum": CLAIM_TYPES},
        "decisive_claim_status": {
            "type": "string",
            "enum": ["VERIFIED_RISK", "UNVERIFIED_RISK", "EVIDENCE_GAP", "UNKNOWN"],
        },
        "decisive_evidence_ids": arr({"type": "string"}, 0, 5),
        "evidence_gap": {"type": "string"},
    }
)


def enforce_final_decision(research_result, data):
    actual_status = research_result["research_status"]
    if actual_status == "RESEARCH_FAILED":
        data["decision"] = "RESEARCH FAILED"
        return data

    data["decisive_evidence_ids"] = validate_evidence_ids(
        data.get("decisive_evidence_ids", []),
        data.get("decisive_claim_type", "OTHER"),
        research_result,
    )

    if data["decisive_claim_status"] == "VERIFIED_RISK" and not data["decisive_evidence_ids"]:
        data["decisive_claim_status"] = "UNVERIFIED_RISK"
    if data["decisive_claim_status"] == "EVIDENCE_GAP":
        data["decisive_evidence_ids"] = []

    if data["decision"] == "KILL IT":
        if not (
            data["decisive_claim_status"] == "VERIFIED_RISK"
            and claim_can_be_fatal(
                data.get("decisive_claim_type", "OTHER"),
                data.get("decisive_evidence_ids", []),
                research_result,
            )
        ):
            data["decision"] = "INSUFFICIENT EVIDENCE"

    if actual_status == "INSUFFICIENT_EVIDENCE" and data["decision"] == "SURVIVES":
        data["decision"] = "INSUFFICIENT EVIDENCE"
    return data


def run_killer_final_one(idea, research_result, killer_data, rebuttal_data, status):
    system = """
أنت THE KILLER في الجولة النهائية لفكرة واحدة.

Evidence Lock:
- KILL IT يحتاج Decisive Claim = VERIFIED_RISK.
- COMPETITION kill يحتاج DIRECT_COMPETITOR exact-JTBD، وليس Adjacent.
- WTP يحتاج REAL_PAYMENT_SIGNAL، لا مجرد وجود المشكلة.
- Distribution kill يحتاج DISTRIBUTION_PROOF، لا مجرد directory/community.
- Regulation يحتاج OFFICIAL_GOVERNMENT.
- Platform kill يحتاج official vendor source + strong job match.
- Missing Distribution/Pricing/WTP = EVIDENCE_GAP وليس Verified Failure.
- إذا البحث غير كافٍ ولا يوجد عيب قاتل موثق: INSUFFICIENT EVIDENCE.
- إذا البحث فشل تقنياً: RESEARCH FAILED.
- SURVIVES يحتاج Research Status = SUFFICIENT وعدم وجود Kill Shot موثق.
- لا تحول فرضية أو توقع إلى حقيقة.
"""

    packet = {
        "idea": idea,
        "research_status": research_result["research_status"],
        "research_gate": research_result["gate"],
        "allowed_evidence_ids_by_category": evidence_ids_by_category(research_result),
        "evidence": evidence_packet(research_result),
        "killer_first": killer_data,
        "hunter_rebuttal": rebuttal_data,
    }

    result = call_json(
        KILLER_MODEL,
        system,
        json.dumps(packet, ensure_ascii=False),
        "killer_final_one_v83",
        KILLER_FINAL_ONE_SCHEMA,
        950,
        f"KILLER FINAL — {idea['name']}",
        status,
        "none",
    )
    if result["ok"]:
        result["data"] = enforce_final_decision(research_result, result["data"])
    return result

# =========================================================
# PER-IDEA RED TEAM
# =========================================================

def red_team_all(ideas, research, status):
    output = {}

    for idx, idea in enumerate(ideas, 1):
        idea_id = idea["id"]
        r = research[idea_id]

        status.info(
            f"⚔️ Red Team {idx}/{len(ideas)}: {idea['name']}"
        )

        killer = run_killer_one(idea, r, status)
        if not killer["ok"]:
            return {
                "ok": False,
                "error": f"{idea_id} Killer failed:\n{killer['error']}",
                "data": output,
            }

        rebuttal = run_rebuttal_one(idea, r, killer["data"], status)
        if not rebuttal["ok"]:
            return {
                "ok": False,
                "error": f"{idea_id} Hunter rebuttal failed:\n{rebuttal['error']}",
                "data": output,
            }

        killer_final = run_killer_final_one(
            idea,
            r,
            killer["data"],
            rebuttal["data"],
            status,
        )

        if not killer_final["ok"]:
            return {
                "ok": False,
                "error": f"{idea_id} Killer final failed:\n{killer_final['error']}",
                "data": output,
            }

        output[idea_id] = {
            "killer_first": killer["data"],
            "hunter_rebuttal": rebuttal["data"],
            "killer_final": killer_final["data"],
        }

    return {"ok": True, "error": None, "data": output}


# =========================================================
# OPERATOR
# =========================================================

operator_item_schema = obj(
    {
        "idea_id": {"type": "string"},
        "idea_name": {"type": "string"},
        "severity": {"type": "integer", "minimum": 0, "maximum": 15},
        "willingness_to_pay": {"type": "integer", "minimum": 0, "maximum": 15},
        "distribution": {"type": "integer", "minimum": 0, "maximum": 15},
        "automation": {"type": "integer", "minimum": 0, "maximum": 15},
        "recurring": {"type": "integer", "minimum": 0, "maximum": 10},
        "competition": {"type": "integer", "minimum": 0, "maximum": 10},
        "moat": {"type": "integer", "minimum": 0, "maximum": 5},
        "speed_to_revenue": {"type": "integer", "minimum": 0, "maximum": 10},
        "stack_fit": {"type": "integer", "minimum": 0, "maximum": 5},
        "price": {"type": "string"},
        "gross_margin": {"type": "string"},
        "ltv": {"type": "string"},
        "cac": {"type": "string"},
        "customers_for_1k_mrr": {"type": "string"},
        "customers_for_5k_mrr": {"type": "string"},
        "customers_for_10k_mrr": {"type": "string"},
        "automation_percent": {"type": "integer", "minimum": 0, "maximum": 100},
        "first_buyer": {"type": "string"},
        "fastest_test": {"type": "string"},
        "biggest_risk": {"type": "string"},
        "truth_test": {"type": "string"},
        "evidence_quality_note": {"type": "string"},
    }
)

OPERATOR_SCHEMA = obj(
    {
        "evaluations": arr(operator_item_schema),
    }
)


def recalc_operator(data):
    evaluations = data.get("evaluations", [])

    for item in evaluations:
        item["total_score"] = (
            item["severity"]
            + item["willingness_to_pay"]
            + item["distribution"]
            + item["automation"]
            + item["recurring"]
            + item["competition"]
            + item["moat"]
            + item["speed_to_revenue"]
            + item["stack_fit"]
        )

    evaluations.sort(
        key=lambda x: x["total_score"],
        reverse=True,
    )

    winner = (
        evaluations[0]
        if evaluations and evaluations[0]["total_score"] > 85
        else None
    )

    highest = evaluations[0]["total_score"] if evaluations else None

    return {
        "evaluations": evaluations,
        "winner_exists": bool(winner),
        "winner_idea_id": winner["idea_id"] if winner else "",
        "winner_reason": (
            f"أعلى نتيجة بعد إعادة الحساب محلياً: "
            f"{winner['total_score']}/100"
            if winner
            else (
                f"لا توجد فكرة تجاوزت 85/100. أعلى نتيجة: {highest}/100"
                if highest is not None
                else "لا توجد فكرة ناجية للتقييم."
            )
        ),
    }


def run_operator(survivors, research, redteam, status):
    system = """
أنت THE OPERATOR / ECONOMIST.

قيّم فقط الأفكار التي انتهت SURVIVES.

النقاط:
Severity 15
WTP 15
Distribution 15
Automation 15
Recurring 10
Competition 10
Moat 5
Speed 10
Stack Fit 5

لا ترفع الدرجة للوصول إلى 85.
لا تستخدم معلومة غير مدعومة كمعلومة مؤكدة.
إذا WTP أو CAC أو LTV تقديرية، اذكر أنها تقدير.
"""

    payload = {
        "surviving_ideas": survivors,
        "research": {
            idea["id"]: {
                "gate": research[idea["id"]]["gate"],
                "evidence": evidence_packet(research[idea["id"]]),
            }
            for idea in survivors
        },
        "redteam": {
            idea["id"]: redteam[idea["id"]]["killer_final"]
            for idea in survivors
        },
    }

    result = call_json(
        OPERATOR_MODEL,
        system,
        json.dumps(payload, ensure_ascii=False),
        "operator_v8",
        OPERATOR_SCHEMA,
        1700,
        "THE OPERATOR",
        status,
        "low",
    )

    if result["ok"]:
        result["data"] = recalc_operator(result["data"])

    return result


# =========================================================
# FINAL OBJECTION / VERDICT
# =========================================================

def final_objection(operator, survivors, research, status):
    if not operator["winner_exists"]:
        return {
            "ok": True,
            "text": (
                "## FINAL OBJECTION\n\n"
                "لا يوجد Winner فوق 85/100؛ "
                "إجبار النظام على اختيار مشروع يخالف قواعد التقييم."
            ),
        }

    winner_id = operator["winner_idea_id"]
    winner = next(
        (idea for idea in survivors if idea["id"] == winner_id),
        None,
    )

    system = """
أنت THE KILLER.

هذه آخر فرصة لقتل الـWinner.
استخدم الأدلة فقط.
إذا كانت أقوى حجة مجرد فرضية غير موثقة، قل إنها UNVERIFIED.
لا تقترح فكرة جديدة.
"""

    prompt = json.dumps(
        {
            "winner": winner,
            "operator": operator,
            "research": {
                "gate": research[winner_id]["gate"],
                "evidence": evidence_packet(research[winner_id]),
            },
        },
        ensure_ascii=False,
    )

    return call_text(
        KILLER_MODEL,
        system,
        prompt,
        450,
        "FINAL OBJECTION",
        status,
        "none",
    )


def final_verdict(operator, objection, survivors, status):
    if not operator["winner_exists"]:
        return {
            "ok": True,
            "text": (
                "## FINAL VERDICT\n\n"
                "**KILL FOR NOW**\n\n"
                f"{operator['winner_reason']}\n\n"
                "لا نبني مشروعاً فقط لأننا نريد الخروج بفكرة."
            ),
        }

    system = """
أنت THE OPERATOR.

اتخذ القرار النهائي:
BUILD أو KILL.

راجع FINAL OBJECTION بجدية.
لا تغير الدرجة عشوائياً.
لا تعتمد على ادعاء غير موثق كسبب قاتل.
اكتب:
- FINAL VERDICT
- BUILD/KILL
- السبب
- اختبار 7 أيام
- BUILD criterion
- KILL criterion
"""

    prompt = json.dumps(
        {
            "operator": operator,
            "final_objection": objection,
            "surviving_ideas": survivors,
        },
        ensure_ascii=False,
    )

    return call_text(
        OPERATOR_MODEL,
        system,
        prompt,
        600,
        "FINAL VERDICT",
        status,
        "low",
    )


# =========================================================
# UI RENDERING
# =========================================================

def render_idea(idea):
    st.subheader(idea["name"])
    st.write(idea["one_liner"])

    st.markdown(
        f"""
**المشتري:** {idea["buyer"]}

**المشكلة:** {idea["problem"]}

**المنتج:** {idea["product"]}

**لماذا سيدفع؟** {idea["why_pay"]}

**السعر:** {idea["price"]}

**البديل الحالي:** {idea["current_alternative"]}

**Distribution:** {idea["distribution"]}

**أول 10 عملاء:** {idea["first_10_customers"]}

**Automation:** {idea["automation"]}%

**العمل البشري:** {idea["human_work"]}

**لماذا الآن؟** {idea["why_now"]}

**Job-to-be-Done:** {idea["job_to_be_done"]}

**المنافس المباشر يجب أن:** {idea["direct_competitor_definition"]}

**ليس منافساً مباشراً:** {idea["not_a_direct_competitor"]}
"""
    )


def render_gate(gate):
    label = {
        "SUFFICIENT": "✅ SUFFICIENT",
        "INSUFFICIENT_EVIDENCE": "⚠️ INSUFFICIENT EVIDENCE",
        "RESEARCH_FAILED": "❌ RESEARCH FAILED",
    }[gate["status"]]

    st.markdown(
        f"""
**Research Status:** `{label}`

**Gate Score:** {gate["score"]}/{gate["max_score"]}

- 2+ Exact Direct Competitors: {gate["checks"]["2_exact_direct_competitors"]}
- Problem Evidence: {gate["checks"]["problem_evidence"]}
- Explicit Pricing Evidence: {gate["checks"]["pricing_evidence"]}
- Real WTP Evidence: {gate["checks"]["real_wtp_evidence"]}
- Distribution Proof: {gate["checks"]["distribution_proof"]}
- 3+ Relevant Domains: {gate["checks"]["3_relevant_domains"]}
- Official Regulatory Evidence if needed: {gate["checks"]["official_regulatory_evidence_if_needed"]}

**Adjacent competitors:** {", ".join(gate.get("adjacent_competitors", [])) or "None"}

**Distribution surfaces only:** {", ".join(gate.get("distribution_surface_sources", [])) or "None"}
"""
    )

# =========================================================
# REPORT
# =========================================================

def build_report(
    ideas,
    blocked,
    research,
    redteam,
    operator,
    objection,
    verdict,
):
    output = ["MD INVESTMENT RESEARCH COUNCIL — V8.3", "\nIDEAS"]

    for idea in ideas:
        output += [
            f"\n{idea['name']}",
            f"ID: {idea['id']}",
            f"Buyer: {idea['buyer']}",
            f"Problem: {idea['problem']}",
            f"Product: {idea['product']}",
            f"Price: {idea['price']}",
            f"Distribution: {idea['distribution']}",
            f"Automation: {idea['automation']}%",
            f"Job-to-be-Done: {idea['job_to_be_done']}",
        ]

    if blocked:
        output.append("\nBLOCKED IDEAS")
        for item in blocked:
            output += [
                item["idea"]["name"],
                f"Reason: {item['reason']}",
            ]

    output.append("\nPER-IDEA EVIDENCE + RED TEAM")

    for idea in ideas:
        idea_id = idea["id"]
        r = research.get(idea_id, {})

        output += [
            f"\n===== {idea['name']} ({idea_id}) =====",
            "RESEARCH GATE:",
            json.dumps(r.get("gate", {}), ensure_ascii=False, indent=2),
            "\nEVIDENCE SOURCES:",
        ]

        for source in r.get("sources", []):
            output += [
                f"\n[{source.get('source_id','')}] {source.get('title','')}",
                f"URL: {source.get('url','')}",
                f"Query type: {source.get('query_type','')}",
                f"Relevance: {source.get('evaluation',{}).get('relevance_score',0)}/100",
                f"Job Match: {source.get('evaluation',{}).get('job_match_score',0)}/100",
                f"Same JTBD: {source.get('evaluation',{}).get('same_job_to_be_done',False)}",
                f"Critical Difference: {source.get('evaluation',{}).get('critical_job_difference','')}",
                f"Material Difference: {source.get('evaluation',{}).get('material_job_difference',False)}",
                f"Authority: {source.get('evaluation',{}).get('authority_type','UNKNOWN')}",
                f"Pricing Explicit: {source.get('evaluation',{}).get('pricing_explicit',False)}",
                f"WTP Signal: {source.get('evaluation',{}).get('wtp_signal','NONE')}",
                f"Distribution Signal: {source.get('evaluation',{}).get('distribution_signal','NONE')}",
                f"Categories: {source.get('evaluation',{}).get('categories',[])}",
                f"Why relevant: {source.get('evaluation',{}).get('why_relevant','')}",
                f"Snippet: {source.get('snippet','')}",
                f"Excerpt: {source.get('page_excerpt','')}",
            ]

        if idea_id in redteam:
            output += [
                "\nKILLER FIRST ATTACK:",
                json.dumps(
                    redteam[idea_id]["killer_first"],
                    ensure_ascii=False,
                    indent=2,
                ),
                "\nHUNTER REBUTTAL:",
                json.dumps(
                    redteam[idea_id]["hunter_rebuttal"],
                    ensure_ascii=False,
                    indent=2,
                ),
                "\nKILLER FINAL:",
                json.dumps(
                    redteam[idea_id]["killer_final"],
                    ensure_ascii=False,
                    indent=2,
                ),
            ]

    output += [
        "\nOPERATOR",
        json.dumps(operator, ensure_ascii=False, indent=2),
        "\nFINAL OBJECTION",
        objection,
        "\nFINAL VERDICT",
        verdict,
    ]

    return "\n".join(output)


# =========================================================
# PDF
# =========================================================

def find_font():
    paths = [
        "/usr/share/fonts/truetype/noto/NotoNaskhArabic-Regular.ttf",
        "/usr/share/fonts/truetype/noto/NotoSansArabic-Regular.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]

    for path in paths:
        if os.path.exists(path):
            return path

    for path in glob.glob("/usr/share/fonts/**/*.ttf", recursive=True):
        if any(
            word in path.lower()
            for word in ["naskh", "arabic", "dejavusans"]
        ):
            return path

    return None


def create_pdf(report):
    font = find_font()
    if not font:
        raise RuntimeError("Arabic font not found")

    pdf = FPDF("P", "mm", "A4")
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

    text = re.sub(r"#{1,6}\s*", "", report)

    emojis = [
        "🧠",
        "😈",
        "💡",
        "🏛️",
        "⚔️",
        "✅",
        "❌",
        "⚠️",
        "🚀",
        "📥",
        "📄",
        "📝",
        "🗑️",
        "🔎",
        "🎯",
        "🌐",
        "📋",
        "📊",
        "🔁",
    ]

    for emoji in emojis:
        text = text.replace(emoji, "")

    for line in text.splitlines():
        line = line.strip()
        if not line:
            pdf.ln(3)
            continue

        pdf.multi_cell(
            0,
            6,
            line,
            align="R",
            new_x="LMARGIN",
            new_y="NEXT",
        )

    return bytes(pdf.output())


# =========================================================
# SESSION STATE
# =========================================================

DEFAULT = {
    "original": "",
    "ideas": None,
    "blocked": None,
    "research": None,
    "redteam": None,
    "operator": None,
    "objection": None,
    "verdict": None,
    "error": None,
}

for key, value in DEFAULT.items():
    if key not in st.session_state:
        st.session_state[key] = value


# =========================================================
# UI
# =========================================================

st.title("🧠 MD Investment Research Council")
st.caption(f"Build: {BUILD_ID}")

st.info(
    """
**V8.3 — Exact JTBD + Evidence Proof**

الجديد:
- المنافس المباشر يجب أن يطابق نفس Job-to-be-Done فعلياً.
- ADJACENT_COMPETITOR لا يُستخدم كـ Kill Shot للمنافسة.
- material_job_difference يمنع تصنيف المصدر كمنافس مباشر.
- PROBLEM_EVIDENCE منفصل تماماً عن WTP_EVIDENCE.
- WTP يحتاج Real Payment Signal، وليس مجرد وجود المشكلة أو الالتزام.
- DISTRIBUTION_SURFACE منفصل عن DISTRIBUTION_PROOF.
- غياب WTP أو Distribution لا يستطيع قتل الفكرة؛ يبقى Evidence Gap.
- Pricing Kill يحتاج سعراً صريحاً لحل قابل للاستبدال لنفس المهمة.
- Regulation يحتاج مصدر حكومي رسمي، وPlatform Risk يحتاج مصدر المنصة الرسمي.
- Targeted Research يعيد البحث فقط عن الفئات الناقصة.
"""
)

original = st.text_area(
    "اكتب الحالة أو الصق البرومبت السابق:",
    height=330,
    value=st.session_state.original,
)

start = st.button(
    "🚀 ابدأ Evidence Research Council",
    type="primary",
    use_container_width=True,
)


# =========================================================
# RUN
# =========================================================

if start:
    if not original.strip():
        st.warning("اكتب الحالة أولاً.")
    else:
        for key, value in DEFAULT.items():
            st.session_state[key] = value

        st.session_state.original = original
        status = st.empty()

        case = prepare_case_brief(original)

        # HUNTER
        status.info("🎯 THE HUNTER يولد 3 أفكار مع Search Definitions...")
        result = run_hunter(case, status)

        if not result["ok"]:
            st.session_state.error = result["error"]

        else:
            ideas, blocked = filter_ideas(result["data"])
            st.session_state.ideas = ideas
            st.session_state.blocked = blocked

            if not ideas:
                st.session_state.verdict = (
                    "## FINAL VERDICT\n\n"
                    "**KILL**\n\n"
                    "كل الأفكار تشبه أفكاراً مرفوضة مسبقاً."
                )
                status.error("❌ لا توجد فكرة بعد الفلتر.")

            else:
                # RESEARCH
                status.info("🌐 Evidence Research + Relevance Scoring...")
                research = research_all(ideas, status)
                st.session_state.research = research

                # RED TEAM
                status.info("⚔️ بدء Red Team مستقل لكل فكرة...")
                redteam_result = red_team_all(
                    ideas,
                    research,
                    status,
                )

                if not redteam_result["ok"]:
                    st.session_state.error = redteam_result["error"]
                    st.session_state.redteam = redteam_result["data"]

                else:
                    redteam = redteam_result["data"]
                    st.session_state.redteam = redteam

                    survivor_ids = [
                        idea_id
                        for idea_id, pack in redteam.items()
                        if pack["killer_final"]["decision"] == "SURVIVES"
                    ]

                    survivors = [
                        idea
                        for idea in ideas
                        if idea["id"] in survivor_ids
                    ]

                    unresolved = [
                        idea_id
                        for idea_id, pack in redteam.items()
                        if pack["killer_final"]["decision"]
                        in [
                            "INSUFFICIENT EVIDENCE",
                            "RESEARCH FAILED",
                        ]
                    ]

                    if not survivors:
                        st.session_state.operator = {
                            "evaluations": [],
                            "winner_exists": False,
                            "winner_idea_id": "",
                            "winner_reason": (
                                "لا توجد فكرة ناجية."
                                + (
                                    " توجد أفكار تحتاج بحثاً إضافياً: "
                                    + ", ".join(unresolved)
                                    if unresolved
                                    else ""
                                )
                            ),
                        }

                        st.session_state.objection = (
                            "لا يوجد Winner يمكن الاعتراض على اختياره."
                        )

                        if unresolved:
                            st.session_state.verdict = (
                                "## FINAL VERDICT\n\n"
                                "**KILL FOR NOW**\n\n"
                                "لا توجد فكرة نجت إلى Operator.\n\n"
                                "غير محسومة بسبب نقص/فشل البحث: "
                                + ", ".join(unresolved)
                            )
                        else:
                            st.session_state.verdict = (
                                "## FINAL VERDICT\n\n"
                                "**KILL**\n\n"
                                "جميع الأفكار قُتلت بأدلة موثقة."
                            )

                        status.success("✅ انتهى المجلس: NO WINNER")

                    else:
                        status.info("📊 THE OPERATOR يقيّم الناجين فقط...")
                        op_result = run_operator(
                            survivors,
                            research,
                            redteam,
                            status,
                        )

                        if not op_result["ok"]:
                            st.session_state.error = op_result["error"]

                        else:
                            operator = op_result["data"]
                            st.session_state.operator = operator

                            status.info("🔪 FINAL OBJECTION...")
                            objection_result = final_objection(
                                operator,
                                survivors,
                                research,
                                status,
                            )

                            if not objection_result["ok"]:
                                st.session_state.error = objection_result["error"]

                            else:
                                objection = objection_result["text"]
                                st.session_state.objection = objection

                                status.info("🏛️ FINAL VERDICT...")
                                verdict_result = final_verdict(
                                    operator,
                                    objection,
                                    survivors,
                                    status,
                                )

                                if not verdict_result["ok"]:
                                    st.session_state.error = verdict_result["error"]

                                else:
                                    st.session_state.verdict = verdict_result["text"]
                                    status.success("✅ انتهى Evidence Research Council")


# =========================================================
# ERROR
# =========================================================

if st.session_state.error:
    st.error("حدث خطأ ولم يسمح التطبيق بإصدار حكم ناقص.")

    with st.expander("🔧 التفاصيل التقنية", expanded=True):
        st.code(st.session_state.error)


# =========================================================
# BLOCKED
# =========================================================

if st.session_state.blocked:
    st.divider()
    st.header("🚫 أفكار أسقطها الفلتر")

    for item in st.session_state.blocked:
        st.warning(
            f"**{item['idea']['name']}**\n\n"
            f"{item['reason']}"
        )


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
# RESEARCH
# =========================================================

if st.session_state.research:
    st.divider()
    st.header("🌐 Evidence Research")

    for idea in st.session_state.ideas or []:
        result = st.session_state.research.get(idea["id"], {})

        with st.expander(
            f"🔎 {idea['name']} — {result.get('research_status','')}",
            expanded=False,
        ):
            if result.get("gate"):
                render_gate(result["gate"])

            st.markdown("### الأدلة")

            for source in result.get("sources", []):
                ev = source.get("evaluation", {})
                score = ev.get("relevance_score", 0)
                job_match = ev.get("job_match_score", 0)
                categories = ", ".join(ev.get("categories", []))

                st.markdown(
                    f"""
**[{source.get("source_id","")}] {source.get("title","")}**

- Relevance: **{score}/100**
- Job Match: **{job_match}/100**
- Same JTBD: `{ev.get("same_job_to_be_done", False)}`
- Critical Difference: {ev.get("critical_job_difference","")}
- Material Difference: `{ev.get("material_job_difference", False)}`
- Authority: `{ev.get("authority_type", "UNKNOWN")}`
- Pricing Explicit: `{ev.get("pricing_explicit", False)}`
- WTP Signal: `{ev.get("wtp_signal", "NONE")}`
- Distribution Signal: `{ev.get("distribution_signal", "NONE")}`
- Categories: `{categories}`
- Why relevant: {ev.get("why_relevant","")}
- Query type: `{source.get("query_type","")}`
- URL: {source.get("url","")}
"""
                )

                if source.get("snippet"):
                    st.caption(source["snippet"])


# =========================================================
# RED TEAM
# =========================================================

if st.session_state.redteam:
    st.divider()
    st.header("⚔️ Per-Idea Red Team")

    for idea in st.session_state.ideas or []:
        idea_id = idea["id"]

        if idea_id not in st.session_state.redteam:
            continue

        pack = st.session_state.redteam[idea_id]

        with st.expander(
            f"{idea['name']} — {pack['killer_final']['decision']}",
            expanded=True,
        ):
            st.markdown("### 🔪 KILLER FIRST")
            st.json(pack["killer_first"])

            st.markdown("### 🎯 HUNTER REBUTTAL")
            st.json(pack["hunter_rebuttal"])

            st.markdown("### 🔪 KILLER FINAL")
            st.json(pack["killer_final"])


# =========================================================
# OPERATOR
# =========================================================

if st.session_state.operator:
    st.divider()
    st.header("📊 THE OPERATOR")

    operator = st.session_state.operator
    evaluations = operator.get("evaluations", [])

    if evaluations:
        table = []

        for item in evaluations:
            table.append(
                {
                    "Idea": item["idea_name"],
                    "Score": item["total_score"],
                    "First Buyer": item["first_buyer"],
                    "Price": item["price"],
                    "Automation": f"{item['automation_percent']}%",
                    "Fastest Test": item["fastest_test"],
                    "Biggest Risk": item["biggest_risk"],
                }
            )

        st.dataframe(
            table,
            use_container_width=True,
            hide_index=True,
        )

        for item in evaluations:
            with st.expander(
                f"{item['idea_name']} — {item['total_score']}/100"
            ):
                st.markdown(
                    f"""
**Severity:** {item["severity"]}/15

**WTP:** {item["willingness_to_pay"]}/15

**Distribution:** {item["distribution"]}/15

**Automation:** {item["automation"]}/15

**Recurring:** {item["recurring"]}/10

**Competition:** {item["competition"]}/10

**Moat:** {item["moat"]}/5

**Speed:** {item["speed_to_revenue"]}/10

**Stack Fit:** {item["stack_fit"]}/5

---

**Gross Margin:** {item["gross_margin"]}

**LTV:** {item["ltv"]}

**CAC:** {item["cac"]}

**$1k MRR:** {item["customers_for_1k_mrr"]}

**$5k MRR:** {item["customers_for_5k_mrr"]}

**$10k MRR:** {item["customers_for_10k_mrr"]}

**Truth Test:** {item["truth_test"]}

**Evidence Quality:** {item["evidence_quality_note"]}
"""
                )

    if operator.get("winner_exists"):
        st.success(
            f"WINNER: {operator['winner_idea_id']}\n\n"
            f"{operator['winner_reason']}"
        )
    else:
        st.warning(
            "NO WINNER\n\n"
            + operator.get("winner_reason", "")
        )


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
        st.session_state.redteam or {},
        st.session_state.operator or {},
        st.session_state.objection or "",
        st.session_state.verdict or "",
    )

    st.divider()
    st.header("📥 التقرير")

    st.download_button(
        "📝 تحميل التقرير TXT",
        report.encode("utf-8"),
        "MD_Investment_Research_V8_3.txt",
        "text/plain",
        use_container_width=True,
    )

    try:
        st.download_button(
            "📄 تحميل التقرير PDF",
            create_pdf(report),
            "MD_Investment_Research_V8_3.pdf",
            "application/pdf",
            use_container_width=True,
        )
    except Exception as e:
        with st.expander("🔧 مشكلة PDF"):
            st.code(str(e))


# =========================================================
# RESET
# =========================================================

if (
    st.session_state.ideas
    or st.session_state.error
    or st.session_state.verdict
):
    st.divider()

    if st.button("🗑️ مسح كل شيء وبدء بحث جديد"):
        for key, value in DEFAULT.items():
            st.session_state[key] = value
        st.rerun()
