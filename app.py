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

BUILD_ID = "V7-PER-IDEA-REDTEAM"


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
.stApp{direction:rtl}
h1,h2,h3,h4,h5,p{text-align:right}
div[data-testid="stMarkdownContainer"]{direction:rtl;text-align:right}
textarea,div[data-baseweb="textarea"] textarea{
    direction:rtl!important;
    text-align:right!important
}
</style>
""",
    unsafe_allow_html=True,
)


# =========================================================
# GROQ
# =========================================================

client = Groq(
    api_key=st.secrets["GROQ_API_KEY"],
    timeout=120.0,
)

HUNTER_MODEL = "openai/gpt-oss-120b"
KILLER_MODEL = "qwen/qwen3.8-27b"
OPERATOR_MODEL = "openai/gpt-oss-20b"

MODEL_LAST_USED = {}

# الطلبات في V7 أصغر من V6، لكن نبقي فجوة لحماية Free Tier.
MIN_MODEL_GAP = {
    HUNTER_MODEL: 24,
    KILLER_MODEL: 24,
    OPERATOR_MODEL: 24,
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
# BASIC HELPERS
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

    cuts = [text.find(marker) for marker in markers if text.find(marker) != -1]
    if cuts:
        text = text[: min(cuts)]

    if len(text) > 13000:
        text = text[:10000] + "\n[...اختصار محلي...]\n" + text[-2500:]

    return text


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
    validator=None,
):
    errors = []

    for attempt in range(retries):
        wait_model(model, stage, status)

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
            if not content:
                errors.append(f"{model}: empty JSON response")
                continue

            data = json.loads(content)

            if validator is not None:
                valid, why = validator(data)
                if not valid:
                    errors.append(f"Validation failed: {why}")
                    prompt = (
                        prompt
                        + "\n\nIMPORTANT CORRECTION: Your previous output failed validation: "
                        + why
                        + "\nReturn a complete corrected object only."
                    )
                    continue

            return {"ok": True, "data": data, "error": None}

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
    errors = []

    for _ in range(retries):
        wait_model(model, stage, status)

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
        hits = [phrase for phrase in phrases if norm(phrase) and norm(phrase) in combined]
        if hits:
            return True, f"تشابه مفاهيمي مع ({concept}): " + ", ".join(hits)

    return False, ""


def filter_ideas(data):
    passed = []
    blocked = []

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
# SCHEMA HELPERS
# =========================================================


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
        "similar_to_rejected": {"type": "boolean"},
        "rejected_similarity_explanation": {"type": "string"},
    }
)

HUNTER_SCHEMA = obj({"ideas": arr(idea_schema, 3, 3)})

killer_one_schema = obj(
    {
        "idea_id": {"type": "string"},
        "top_failure_reason_1": {"type": "string"},
        "top_failure_reason_2": {"type": "string"},
        "top_failure_reason_3": {"type": "string"},
        "kill_shot": {"type": "string"},
        "evidence_for": {"type": "string"},
        "evidence_against": {"type": "string"},
        "unknowns": {"type": "string"},
        "score_out_of_10": {"type": "integer", "minimum": 0, "maximum": 10},
        "decision": {
            "type": "string",
            "enum": ["SURVIVES", "KILL IT", "INSUFFICIENT EVIDENCE"],
        },
    }
)

rebuttal_one_schema = obj(
    {
        "idea_id": {"type": "string"},
        "valid_objection": {"type": "string"},
        "disputed_objection": {"type": "string"},
        "evidence_needed": {"type": "string"},
        "position": {
            "type": "string",
            "enum": ["DEFEND", "DROP", "NEEDS MORE EVIDENCE"],
        },
    }
)

killer_final_one_schema = obj(
    {
        "idea_id": {"type": "string"},
        "decision": {
            "type": "string",
            "enum": ["SURVIVES", "KILL IT", "INSUFFICIENT EVIDENCE"],
        },
        "remaining_problem": {"type": "string"},
        "wtp_real": {"type": "boolean"},
        "distribution_real": {"type": "boolean"},
        "feature_or_company": {
            "type": "string",
            "enum": ["FEATURE", "COMPANY", "UNCLEAR"],
        },
        "final_score_out_of_10": {
            "type": "integer",
            "minimum": 0,
            "maximum": 10,
        },
        "evidence_gap": {"type": "string"},
    }
)

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
    }
)

OPERATOR_SCHEMA = obj({"evaluations": arr(operator_item_schema)})


# =========================================================
# VALIDATORS
# =========================================================


def validate_hunter(data):
    ideas = data.get("ideas", [])
    if len(ideas) != 3:
        return False, f"Expected exactly 3 ideas, got {len(ideas)}"

    ids = [str(x.get("id", "")) for x in ideas]
    if len(set(ids)) != 3 or any(not x for x in ids):
        return False, f"Idea IDs must be unique and non-empty: {ids}"

    return True, ""


def validator_for_idea(expected_id):
    def _validate(data):
        actual = str(data.get("idea_id", ""))
        if actual != str(expected_id):
            return False, f"Expected idea_id={expected_id}, got {actual}"
        return True, ""

    return _validate


def operator_validator(expected_ids):
    expected = {str(x) for x in expected_ids}

    def _validate(data):
        rows = data.get("evaluations", [])
        actual = {str(x.get("idea_id", "")) for x in rows}
        if actual != expected:
            return False, f"Expected evaluation IDs {sorted(expected)}, got {sorted(actual)}"
        return True, ""

    return _validate


# =========================================================
# HUNTER
# =========================================================


def run_hunter(case_brief, status):
    system = """
أنت THE HUNTER.

ابحث عن أماكن تتحرك فيها الأموال فعلياً.

الأولوية:
- ألم مالي حقيقي.
- willingness-to-pay.
- distribution واضحة.
- automation مرتفعة.
- recurring usage.
- speed to revenue.

ممنوع:
- AI wrapper بلا قيمة مستقلة.
- dashboard عام.
- أداة يستطيع ChatGPT تنفيذها بما يكفي.
- إعادة تغليف فكرة مرفوضة.

أخرج 3 أفكار فقط ومختلفة اقتصادياً.
لا تختر WINNER.
لا تعامل الافتراضات كحقائق.
"""

    prompt = f"""
حالة المستخدم:

{case_brief}

القائمة المرفوضة:

{json.dumps(REJECTED_IDEAS, ensure_ascii=False)}
"""

    return call_json(
        HUNTER_MODEL,
        system,
        prompt,
        "hunter_ideas_v7",
        HUNTER_SCHEMA,
        1500,
        "THE HUNTER",
        status,
        "low",
        validator=validate_hunter,
    )


# =========================================================
# DDGS WEB RESEARCH
# =========================================================


def search_item(raw, query_type, query):
    if not isinstance(raw, dict):
        return None

    url = raw.get("href") or raw.get("url") or ""
    if not url:
        return None

    return {
        "title": str(raw.get("title") or "").strip(),
        "url": str(url).strip(),
        "snippet": cut(
            str(raw.get("body") or raw.get("snippet") or raw.get("content") or "").strip(),
            550,
        ),
        "query_type": query_type,
        "query": query,
    }


def domain(url):
    try:
        return urlparse(url).netloc.lower().replace("www.", "")
    except Exception:
        return ""


def queries_for(idea):
    # أكثر تحديداً من V6 حتى لا نخلط "السوق العام" مع "المنافس المباشر".
    return [
        (
            "direct_competitors",
            f'"{idea["name"]}" competitors alternative software pricing',
        ),
        (
            "job_to_be_done",
            f'{cut(idea["buyer"],160)} {cut(idea["problem"],180)} solution software competitors',
        ),
        (
            "pricing",
            f'{cut(idea["product"],180)} pricing subscription fee competitor',
        ),
        (
            "wtp_distribution",
            f'{cut(idea["buyer"],160)} {cut(idea["problem"],160)} cost pain budget buying software',
        ),
        (
            "risk_regulation",
            f'{cut(idea["product"],180)} regulation legal risk API platform compliance',
        ),
    ]


def research_gate(result):
    sources = result.get("sources", [])

    domains = {domain(x.get("url", "")) for x in sources if domain(x.get("url", ""))}
    extracted = sum(1 for x in sources if x.get("page_excerpt"))
    query_types = {x.get("query_type") for x in sources}

    checks = {
        "at_least_4_sources": len(sources) >= 4,
        "at_least_3_domains": len(domains) >= 3,
        "at_least_2_extracted_pages": extracted >= 2,
        "direct_competitor_search": "direct_competitors" in query_types,
        "pricing_search": "pricing" in query_types,
        "wtp_distribution_search": "wtp_distribution" in query_types,
    }

    passed = sum(1 for value in checks.values() if value)
    sufficient = passed >= 5

    return {
        "sufficient": sufficient,
        "score": passed,
        "max_score": len(checks),
        "checks": checks,
        "source_count": len(sources),
        "domain_count": len(domains),
        "extracted_pages": extracted,
    }


def research_idea(idea, status):
    errors = []
    found = []

    try:
        ddgs = DDGS(timeout=12)
    except Exception as e:
        return {
            "ok": False,
            "text": "",
            "sources": [],
            "gate": {
                "sufficient": False,
                "score": 0,
                "max_score": 6,
                "checks": {},
                "source_count": 0,
                "domain_count": 0,
                "extracted_pages": 0,
            },
            "error": f"DDGS init failed: {e}",
        }

    queries = queries_for(idea)

    for index, (query_type, query) in enumerate(queries, 1):
        status.info(
            f"🔎 {idea['name']}: بحث {index}/{len(queries)} — {query_type}"
        )

        try:
            results = ddgs.text(query, max_results=4) or []
            for raw in results:
                item = search_item(raw, query_type, query)
                if item:
                    found.append(item)
        except Exception as e:
            errors.append(f"Search failed [{query_type}] {query}: {e}")

        time.sleep(0.8)

    # dedupe + domain cap
    unique = []
    seen_urls = set()
    per_domain = {}

    for item in found:
        url = item["url"]
        dm = domain(url)

        if url in seen_urls:
            continue
        if dm and per_domain.get(dm, 0) >= 2:
            continue

        seen_urls.add(url)
        per_domain[dm] = per_domain.get(dm, 0) + 1
        unique.append(item)

        if len(unique) >= 9:
            break

    # اقرأ أفضل 3 مصادر فقط لتجنب البطء.
    for index, source in enumerate(unique[:3], 1):
        status.info(f"📄 قراءة مصدر {index}/3: {idea['name']}")

        try:
            extracted = ddgs.extract(source["url"], fmt="text_plain")
            content = extracted.get("content", "") if isinstance(extracted, dict) else ""
            source["page_excerpt"] = cut(str(content).strip(), 1400)
        except Exception as e:
            source["page_excerpt"] = ""
            errors.append(f"Extract failed {source['url']}: {e}")

        time.sleep(0.6)

    packets = []
    for number, source in enumerate(unique, 1):
        packets.append(
            f"""
SOURCE {number}
Query type: {source['query_type']}
Title: {source['title']}
URL: {source['url']}
Search snippet: {source['snippet']}
Page excerpt: {source.get('page_excerpt','')}
"""
        )

    result = {
        "ok": bool(unique),
        "text": cut("\n\n---\n\n".join(packets), 6500),
        "sources": unique,
        "error": "\n".join(errors[-8:]),
    }
    result["gate"] = research_gate(result)
    return result


# =========================================================
# PER-IDEA KILLER / HUNTER / KILLER
# =========================================================


def run_killer_one(idea, research, status):
    system = """
أنت THE KILLER.
أنت مستثمر متشائم هدفه منعنا من بناء المشروع الخطأ.

هذه جلسة لفكرة واحدة فقط.
لا تتحدث عن أي فكرة أخرى.

افحص:
- direct competitors
- platform-native alternatives
- WTP
- CAC / distribution
- churn / repeat usage
- regulation / liability / privacy
- platform/API risk
- ChatGPT substitution
- Feature vs Company
- moat

قواعد مهمة:
1) لا تعامل ادعاء Hunter كحقيقة إذا لم يدعمه البحث.
2) إذا البحث غير كافٍ للحكم، استخدم INSUFFICIENT EVIDENCE بدلاً من KILL IT.
3) KILL IT يحتاج سبباً اقتصادياً أو تنافسياً واضحاً، وليس مجرد غياب دليل.
4) لا تقترح أفكاراً جديدة.
"""

    prompt = f"""
IDEA:
{json.dumps(idea, ensure_ascii=False)}

RESEARCH QUALITY GATE:
{json.dumps(research['gate'], ensure_ascii=False)}

WEB RESEARCH:
{cut(research['text'], 5600)}

يجب أن يكون idea_id في الإجابة بالضبط: {idea['id']}
"""

    return call_json(
        KILLER_MODEL,
        system,
        prompt,
        f"killer_one_{re.sub(r'[^a-zA-Z0-9_]', '_', idea['id'])}",
        killer_one_schema,
        900,
        f"THE KILLER — {idea['name']}",
        status,
        "none",
        validator=validator_for_idea(idea["id"]),
    )


def run_rebuttal_one(idea, research, killer, status):
    system = """
أنت THE HUNTER.
هذه فرصتك الوحيدة للرد على Killer في هذه الفكرة فقط.

لا تضف فكرة جديدة.
إذا Killer أثبت أن الفكرة سيئة: DROP.
إذا المشكلة هي نقص دليل فقط: NEEDS MORE EVIDENCE.
إذا يوجد دفاع قوي ومدعوم: DEFEND.
ممنوع ترقيع فكرة ميتة.
"""

    prompt = f"""
IDEA:
{json.dumps(idea, ensure_ascii=False)}

RESEARCH GATE:
{json.dumps(research['gate'], ensure_ascii=False)}

KILLER FIRST ATTACK:
{json.dumps(killer, ensure_ascii=False)}

RESEARCH EXCERPTS:
{cut(research['text'], 3300)}

يجب أن يكون idea_id بالضبط: {idea['id']}
"""

    return call_json(
        HUNTER_MODEL,
        system,
        prompt,
        f"hunter_rebuttal_{re.sub(r'[^a-zA-Z0-9_]', '_', idea['id'])}",
        rebuttal_one_schema,
        600,
        f"HUNTER REBUTTAL — {idea['name']}",
        status,
        "low",
        validator=validator_for_idea(idea["id"]),
    )


def run_killer_final_one(idea, research, killer, rebuttal, status):
    system = """
أنت THE KILLER.
هذه آخر فرصة للحكم على فكرة واحدة فقط.

أصدر أحد الأحكام:
- SURVIVES
- KILL IT
- INSUFFICIENT EVIDENCE

قواعد:
- غياب الدليل وحده لا يساوي KILL IT.
- إذا Research Gate ضعيف ولا يوجد Kill Shot مستقل قوي، استخدم INSUFFICIENT EVIDENCE.
- لا تقترح فكرة جديدة.
- لا تحكم على أي فكرة أخرى.
"""

    prompt = f"""
IDEA:
{json.dumps(idea, ensure_ascii=False)}

RESEARCH GATE:
{json.dumps(research['gate'], ensure_ascii=False)}

FIRST KILLER ATTACK:
{json.dumps(killer, ensure_ascii=False)}

HUNTER REBUTTAL:
{json.dumps(rebuttal, ensure_ascii=False)}

RESEARCH EXCERPTS:
{cut(research['text'], 3000)}

يجب أن يكون idea_id بالضبط: {idea['id']}
"""

    return call_json(
        KILLER_MODEL,
        system,
        prompt,
        f"killer_final_{re.sub(r'[^a-zA-Z0-9_]', '_', idea['id'])}",
        killer_final_one_schema,
        650,
        f"KILLER FINAL — {idea['name']}",
        status,
        "none",
        validator=validator_for_idea(idea["id"]),
    )


# =========================================================
# OPERATOR
# =========================================================


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

    evaluations.sort(key=lambda x: x["total_score"], reverse=True)

    winner = evaluations[0] if evaluations and evaluations[0]["total_score"] > 85 else None
    highest = evaluations[0]["total_score"] if evaluations else None

    return {
        "evaluations": evaluations,
        "winner_exists": bool(winner),
        "winner_idea_id": winner["idea_id"] if winner else "",
        "winner_reason": (
            f"أعلى نتيجة بعد إعادة الحساب محلياً: {winner['total_score']}/100"
            if winner
            else (
                f"لا توجد فكرة تجاوزت 85/100. أعلى نتيجة: {highest}/100"
                if highest is not None
                else "لا توجد فكرة ناجية للتقييم."
            )
        ),
    }


def run_operator(survivors, research, rounds, status):
    expected_ids = [idea["id"] for idea in survivors]

    system = """
أنت THE OPERATOR / ECONOMIST.
أنت CTO + CFO + Growth Operator.

قيّم فقط الأفكار التي نجت فعلياً بكلمة SURVIVES.
لا تقيّم KILL IT ولا INSUFFICIENT EVIDENCE.

النقاط:
Severity = 15
WTP = 15
Distribution = 15
Automation = 15
Recurring = 10
Competition = 10
Moat = 5
Speed to Revenue = 10
Stack Fit = 5

لا ترفع الدرجة للوصول إلى 85.
لا تخترع CAC/LTV كحقائق؛ سمّها تقديرات إذا لم يوجد دليل.
ركز على أول عميل يدفع.
"""

    payload = []
    for idea in survivors:
        payload.append(
            {
                "idea": idea,
                "research_gate": research[idea["id"]]["gate"],
                "research": cut(research[idea["id"]]["text"], 2600),
                "red_team": rounds[idea["id"]],
            }
        )

    return call_json(
        OPERATOR_MODEL,
        system,
        json.dumps(payload, ensure_ascii=False),
        "operator_v7",
        OPERATOR_SCHEMA,
        1500,
        "THE OPERATOR",
        status,
        "low",
        validator=operator_validator(expected_ids),
    )


# =========================================================
# FINAL OBJECTION / VERDICT
# =========================================================


def final_objection(operator, survivors, research, rounds, status):
    if not operator["winner_exists"]:
        return {
            "ok": True,
            "text": (
                "## FINAL OBJECTION\n\n"
                "لا يوجد Winner فوق 85/100؛ إجبار النظام على الاختيار يخالف قواعد التقييم."
            ),
        }

    winner_id = operator["winner_idea_id"]
    winner = next(idea for idea in survivors if idea["id"] == winner_id)

    system = """
أنت THE KILLER.
هذه آخر فرصة لقتل الـWinner.
اكتب أقوى حجة واحدة يمكن أن تجعله ينتهي عند $0 MRR.
استخدم الأدلة فقط.
لا تقترح فكرة جديدة.
"""

    prompt = json.dumps(
        {
            "winner": winner,
            "operator": operator,
            "research_gate": research[winner_id]["gate"],
            "research": cut(research[winner_id]["text"], 3000),
            "red_team": rounds[winner_id],
        },
        ensure_ascii=False,
    )

    return call_text(
        KILLER_MODEL,
        system,
        prompt,
        420,
        "FINAL OBJECTION",
        status,
        "none",
    )


def final_verdict(operator, objection, survivors, insufficient_ids, status):
    if not operator["winner_exists"]:
        extra = ""
        if insufficient_ids:
            extra = (
                "\n\nيوجد أيضاً أفكار لم تُقتل اقتصادياً لكنها بقيت تحت "
                "INSUFFICIENT EVIDENCE: "
                + ", ".join(insufficient_ids)
                + ". لا تُعتبر Winners حتى يكتمل الدليل."
            )

        return {
            "ok": True,
            "text": (
                "## FINAL VERDICT\n\n"
                "**KILL FOR NOW**\n\n"
                f"{operator['winner_reason']}"
                + extra
            ),
        }

    system = """
أنت THE OPERATOR.
اتخذ القرار النهائي: BUILD أو KILL.
راجع FINAL OBJECTION بجدية.
لا تغير الدرجات عشوائياً.
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
        560,
        "FINAL VERDICT",
        status,
        "low",
    )


# =========================================================
# DISPLAY
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
"""
    )


def render_gate(gate):
    label = "✅ كافٍ" if gate["sufficient"] else "⚠️ غير كافٍ"
    st.markdown(
        f"**Research Gate:** {label} — {gate['score']}/{gate['max_score']}"
    )

    cols = st.columns(3)
    checks = list(gate["checks"].items())
    for i, (name, ok) in enumerate(checks):
        with cols[i % 3]:
            st.write(("✅ " if ok else "❌ ") + name)


# =========================================================
# REPORT
# =========================================================


def build_report(ideas, blocked, research, rounds, operator, objection, verdict):
    output = ["MD INVESTMENT RESEARCH COUNCIL — V7", "\nIDEAS"]

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
        ]

    if blocked:
        output.append("\nBLOCKED IDEAS")
        for item in blocked:
            output += [item["idea"]["name"], f"Reason: {item['reason']}"]

    output.append("\nPER-IDEA RESEARCH + RED TEAM")

    for idea in ideas:
        idea_id = idea["id"]
        result = research.get(idea_id, {})
        round_data = rounds.get(idea_id, {})

        output += [
            f"\n===== {idea['name']} ({idea_id}) =====",
            "RESEARCH GATE:",
            json.dumps(result.get("gate", {}), ensure_ascii=False, indent=2),
            "WEB RESEARCH:",
            result.get("text", ""),
            "SOURCES:",
        ]

        for source in result.get("sources", []):
            output.append(f"- {source.get('title','')}: {source.get('url','')}")

        output += [
            "KILLER FIRST ATTACK:",
            json.dumps(round_data.get("killer_first", {}), ensure_ascii=False, indent=2),
            "HUNTER REBUTTAL:",
            json.dumps(round_data.get("hunter_rebuttal", {}), ensure_ascii=False, indent=2),
            "KILLER FINAL:",
            json.dumps(round_data.get("killer_final", {}), ensure_ascii=False, indent=2),
        ]

    output += [
        "\nOPERATOR",
        json.dumps(operator or {}, ensure_ascii=False, indent=2),
        "\nFINAL OBJECTION",
        objection or "",
        "\nFINAL VERDICT",
        verdict or "",
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
        if any(word in path.lower() for word in ["naskh", "arabic", "dejavusans"]):
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
    "rounds": None,
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
**V7**

كل فكرة الآن تحصل على جلسة مستقلة كاملة:

Research → Killer → Hunter Rebuttal → Killer Final

ولا يمكن للـOperator تقييم فكرة إلا إذا انتهت إلى `SURVIVES`.

إذا البحث غير كافٍ، النتيجة تصبح `INSUFFICIENT EVIDENCE` بدلاً من قتل الفكرة لمجرد نقص البيانات.
"""
)

original = st.text_area(
    "اكتب الحالة أو الصق البرومبت السابق:",
    height=330,
    value=st.session_state.original,
)

start = st.button(
    "🚀 ابدأ Research Council",
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

        # 1) HUNTER
        status.info("🎯 THE HUNTER يولد 3 أفكار...")
        result = run_hunter(case, status)

        if not result["ok"]:
            st.session_state.error = result["error"]
        else:
            ideas, blocked = filter_ideas(result["data"])
            st.session_state.ideas = ideas
            st.session_state.blocked = blocked

            if not ideas:
                st.session_state.operator = {
                    "evaluations": [],
                    "winner_exists": False,
                    "winner_idea_id": "",
                    "winner_reason": "كل الأفكار سقطت في فلتر المرفوضات.",
                }
                st.session_state.objection = "لا توجد فكرة ناجية."
                st.session_state.verdict = (
                    "## FINAL VERDICT\n\n**KILL FOR NOW**\n\n"
                    "كل الأفكار التي ولدها Hunter تشبه أفكاراً مرفوضة مسبقاً."
                )
                status.error("❌ لا توجد فكرة بعد الفلتر.")
            else:
                research = {}
                rounds = {}
                fatal_error = None

                # 2) EACH IDEA GETS A FULL ROUND
                for index, idea in enumerate(ideas, 1):
                    idea_id = idea["id"]

                    status.info(
                        f"🌐 الفكرة {index}/{len(ideas)} — Web Research: {idea['name']}"
                    )
                    research_result = research_idea(idea, status)
                    research[idea_id] = research_result

                    # البحث إذا فشل تماماً لا يوقف البرنامج؛ البوابة ستعتبره غير كافٍ.
                    if not research_result["ok"]:
                        research_result["gate"]["sufficient"] = False

                    status.info(
                        f"🔪 الفكرة {index}/{len(ideas)} — Killer First Attack"
                    )
                    killer_first_result = run_killer_one(
                        idea, research_result, status
                    )
                    if not killer_first_result["ok"]:
                        fatal_error = (
                            f"Killer First Attack failed for {idea_id}:\n\n"
                            + killer_first_result["error"]
                        )
                        break

                    killer_first = killer_first_result["data"]

                    status.info(
                        f"🎯 الفكرة {index}/{len(ideas)} — Hunter Rebuttal"
                    )
                    rebuttal_result = run_rebuttal_one(
                        idea, research_result, killer_first, status
                    )
                    if not rebuttal_result["ok"]:
                        fatal_error = (
                            f"Hunter Rebuttal failed for {idea_id}:\n\n"
                            + rebuttal_result["error"]
                        )
                        break

                    rebuttal = rebuttal_result["data"]

                    status.info(
                        f"🔪 الفكرة {index}/{len(ideas)} — Killer Final"
                    )
                    killer_final_result = run_killer_final_one(
                        idea,
                        research_result,
                        killer_first,
                        rebuttal,
                        status,
                    )
                    if not killer_final_result["ok"]:
                        fatal_error = (
                            f"Killer Final failed for {idea_id}:\n\n"
                            + killer_final_result["error"]
                        )
                        break

                    rounds[idea_id] = {
                        "killer_first": killer_first,
                        "hunter_rebuttal": rebuttal,
                        "killer_final": killer_final_result["data"],
                    }

                st.session_state.research = research
                st.session_state.rounds = rounds

                if fatal_error:
                    st.session_state.error = fatal_error
                else:
                    survivor_ids = [
                        idea_id
                        for idea_id, data in rounds.items()
                        if data["killer_final"]["decision"] == "SURVIVES"
                    ]

                    insufficient_ids = [
                        idea_id
                        for idea_id, data in rounds.items()
                        if data["killer_final"]["decision"]
                        == "INSUFFICIENT EVIDENCE"
                    ]

                    survivors = [
                        idea for idea in ideas if idea["id"] in survivor_ids
                    ]

                    # 3) OPERATOR ONLY SEES SURVIVORS
                    if survivors:
                        status.info("📊 THE OPERATOR يحسب الاقتصاديات...")
                        operator_result = run_operator(
                            survivors, research, rounds, status
                        )

                        if not operator_result["ok"]:
                            st.session_state.error = operator_result["error"]
                        else:
                            operator = recalc_operator(operator_result["data"])
                            st.session_state.operator = operator

                            status.info("🔪 FINAL OBJECTION...")
                            objection_result = final_objection(
                                operator,
                                survivors,
                                research,
                                rounds,
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
                                    insufficient_ids,
                                    status,
                                )

                                if not verdict_result["ok"]:
                                    st.session_state.error = verdict_result["error"]
                                else:
                                    st.session_state.verdict = verdict_result["text"]
                                    status.success("✅ انتهى Research Council")
                    else:
                        # لا يوجد Survivor: لا نستخدم Operator ليخترع تقييماً.
                        st.session_state.operator = {
                            "evaluations": [],
                            "winner_exists": False,
                            "winner_idea_id": "",
                            "winner_reason": (
                                "لم تنج أي فكرة إلى مرحلة Operator."
                                if not insufficient_ids
                                else "لا توجد فكرة ناجية، وتوجد أفكار تحتاج أدلة إضافية."
                            ),
                        }
                        st.session_state.objection = (
                            "لا يوجد Winner يمكن الاعتراض على اختياره."
                        )

                        extra = ""
                        if insufficient_ids:
                            extra = (
                                "\n\nINSUFFICIENT EVIDENCE: "
                                + ", ".join(insufficient_ids)
                                + ". هذه ليست Winners ولا تُعتبر ميتة اقتصادياً حتى يكتمل البحث."
                            )

                        st.session_state.verdict = (
                            "## FINAL VERDICT\n\n**KILL FOR NOW**\n\n"
                            "لا توجد فكرة نجت من Red Team إلى مرحلة Operator."
                            + extra
                        )
                        status.success("✅ انتهى المجلس: NO WINNER")


# =========================================================
# ERROR
# =========================================================

if st.session_state.error:
    st.error("حدث خطأ ولم يسمح التطبيق بإصدار حكم ناقص.")

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
# IDEAS + PER-IDEA ROUNDS
# =========================================================

if st.session_state.ideas:
    st.divider()
    st.header("🎯 أفكار THE HUNTER")

    for idea in st.session_state.ideas:
        with st.expander(idea["name"], expanded=True):
            render_idea(idea)


if st.session_state.research:
    st.divider()
    st.header("🌐 Research + Red Team لكل فكرة")

    for idea in st.session_state.ideas or []:
        idea_id = idea["id"]
        research = st.session_state.research.get(idea_id, {})
        round_data = (st.session_state.rounds or {}).get(idea_id, {})

        with st.expander(f"🧪 {idea['name']} — {idea_id}", expanded=False):
            gate = research.get("gate")
            if gate:
                render_gate(gate)

            st.markdown("### 🔎 Web Research")
            if research.get("text"):
                st.code(research["text"], language="text")
            else:
                st.warning("لا يوجد Research Packet كافٍ لهذه الفكرة.")

            if research.get("sources"):
                st.markdown("### المصادر")
                for source in research["sources"]:
                    title = source.get("title") or source.get("url")
                    st.markdown(f"- [{title}]({source.get('url','')})")

            if round_data:
                killer_first = round_data["killer_first"]
                rebuttal = round_data["hunter_rebuttal"]
                killer_final = round_data["killer_final"]

                st.markdown("### 🔪 Killer First Attack")
                st.json(killer_first)

                st.markdown("### 🎯 Hunter Rebuttal")
                st.json(rebuttal)

                st.markdown("### 🔪 Killer Final")
                st.json(killer_final)


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

        st.dataframe(table, use_container_width=True, hide_index=True)

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
"""
                )

    if operator.get("winner_exists"):
        st.success(
            f"WINNER: {operator['winner_idea_id']}\n\n{operator['winner_reason']}"
        )
    else:
        st.warning("NO WINNER\n\n" + operator.get("winner_reason", ""))


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
        st.session_state.rounds or {},
        st.session_state.operator or {},
        st.session_state.objection or "",
        st.session_state.verdict or "",
    )

    st.divider()
    st.header("📥 التقرير")

    st.download_button(
        "📝 تحميل التقرير TXT",
        report.encode("utf-8"),
        "MD_Investment_Research_V7.txt",
        "text/plain",
        use_container_width=True,
    )

    try:
        st.download_button(
            "📄 تحميل التقرير PDF",
            create_pdf(report),
            "MD_Investment_Research_V7.pdf",
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
