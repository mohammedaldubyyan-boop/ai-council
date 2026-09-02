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

BUILD_ID = "V8.1-HUNTER-TOPUP"


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
- حدد Job-to-be-Done بدقة.
- عرّف المنافس المباشر بدقة.
- اشرح ما الذي لا يُعد منافساً مباشراً.
- Search Terms يجب أن تكون قصيرة ومحددة.
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
# RELEVANCE JUDGE
# =========================================================

source_eval_schema = obj(
    {
        "source_id": {"type": "string"},
        "relevance_score": {"type": "integer", "minimum": 0, "maximum": 100},
        "same_job_to_be_done": {"type": "boolean"},
        "authoritative": {"type": "boolean"},
        "categories": arr(
            {
                "type": "string",
                "enum": [
                    "DIRECT_COMPETITOR",
                    "PRICING_EVIDENCE",
                    "WTP_EVIDENCE",
                    "DISTRIBUTION_EVIDENCE",
                    "REGULATORY_EVIDENCE",
                    "PLATFORM_RISK",
                    "BACKGROUND",
                    "IRRELEVANT",
                ],
            },
            1,
            4,
        ),
        "why_relevant": {"type": "string"},
    }
)

RELEVANCE_SCHEMA = obj(
    {
        "evaluations": arr(source_eval_schema),
    }
)


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
                "snippet": source["snippet"],
                "page_excerpt": cut(source.get("page_excerpt", ""), 900),
            }
        )

    system = """
أنت Research Relevance Judge.

لا تحكم على جودة فكرة المشروع.
وظيفتك فقط تقييم هل كل مصدر فعلاً متعلق بنفس Job-to-be-Done.

DIRECT_COMPETITOR:
شركة/منتج يؤدي نفس المهمة الأساسية لنفس نوع المشتري تقريباً.

PRICING_EVIDENCE:
مصدر يقدم سعراً حقيقياً أو نموذج تسعير متعلق بنفس المهمة.

WTP_EVIDENCE:
دليل أن المشتري يدفع أو أن المشكلة لها تكلفة مالية واضحة.

DISTRIBUTION_EVIDENCE:
دليل على قناة وصول فعلية للمشتري: marketplace, directory, ecosystem, public list, buyer community, etc.

REGULATORY_EVIDENCE:
مصدر تنظيمي/رسمي أو وثيقة قوية توضح قيداً قانونياً أو API/ترخيصاً متعلقاً مباشرة بالمشروع.

PLATFORM_RISK:
دليل أن منصة أساسية/مزود بنية يقدم نفس الوظيفة أو قد يبتلعها.

BACKGROUND:
مفيد لفهم المجال لكنه ليس دليلاً على منافس/سعر/WTP/توزيع/تنظيم.

IRRELEVANT:
لا يتكلم عن نفس Job-to-be-Done.

لا تعتبر Software directory العام منافساً مباشراً.
لا تعتبر أداة تؤدي وظيفة مختلفة منافساً مباشراً.
"""

    prompt = f"""
IDEA:

Name: {idea["name"]}
Buyer: {idea["buyer"]}
Job-to-be-Done: {idea["job_to_be_done"]}

Direct competitor MUST mean:
{idea["direct_competitor_definition"]}

NOT a direct competitor:
{idea["not_a_direct_competitor"]}

SOURCES:
{json.dumps(packet, ensure_ascii=False)}
"""

    result = call_json(
        OPERATOR_MODEL,
        system,
        prompt,
        "source_relevance_v8",
        RELEVANCE_SCHEMA,
        1500,
        f"Relevance Judge — {idea['name']}",
        status,
        "low",
    )

    if not result["ok"]:
        return result

    mapping = {x["source_id"]: x for x in result["data"]["evaluations"]}

    for source in sources:
        source["evaluation"] = mapping.get(
            source["source_id"],
            {
                "source_id": source["source_id"],
                "relevance_score": 0,
                "same_job_to_be_done": False,
                "authoritative": False,
                "categories": ["IRRELEVANT"],
                "why_relevant": "لم يرجع تقييم للمصدر.",
            },
        )

    return {"ok": True, "sources": sources, "error": None}


# =========================================================
# RESEARCH GATE
# =========================================================

def category_sources(sources, category, min_score=60):
    return [
        s
        for s in sources
        if s["evaluation"]["relevance_score"] >= min_score
        and category in s["evaluation"]["categories"]
    ]


def research_gate(idea, sources):
    direct = category_sources(sources, "DIRECT_COMPETITOR", 65)
    pricing = category_sources(sources, "PRICING_EVIDENCE", 60)
    wtp = category_sources(sources, "WTP_EVIDENCE", 60)
    distribution = category_sources(sources, "DISTRIBUTION_EVIDENCE", 60)
    regulatory = category_sources(sources, "REGULATORY_EVIDENCE", 65)

    relevant_domains = {
        domain(s["url"])
        for s in sources
        if s["evaluation"]["relevance_score"] >= 60
        and "IRRELEVANT" not in s["evaluation"]["categories"]
    }

    checks = {
        "2_direct_competitors": len(direct) >= 2,
        "pricing_evidence": len(pricing) >= 1,
        "wtp_evidence": len(wtp) >= 1,
        "distribution_evidence": len(distribution) >= 1,
        "3_relevant_domains": len(relevant_domains) >= 3,
        "regulatory_evidence_if_needed": (
            len(regulatory) >= 1 if idea["regulatory_sensitive"] else True
        ),
    }

    score = sum(1 for v in checks.values() if v)
    max_score = len(checks)

    if score <= 2:
        status = "RESEARCH_FAILED"
    elif score < max_score:
        status = "INSUFFICIENT_EVIDENCE"
    else:
        status = "SUFFICIENT"

    return {
        "status": status,
        "score": score,
        "max_score": max_score,
        "checks": checks,
        "direct_competitors": [s["source_id"] for s in direct],
        "pricing_sources": [s["source_id"] for s in pricing],
        "wtp_sources": [s["source_id"] for s in wtp],
        "distribution_sources": [s["source_id"] for s in distribution],
        "regulatory_sources": [s["source_id"] for s in regulatory],
        "relevant_domain_count": len(relevant_domains),
    }


def missing_query_types(gate):
    missing = []

    if not gate["checks"]["2_direct_competitors"]:
        missing.append("direct_competitors")
    if not gate["checks"]["pricing_evidence"]:
        missing.append("pricing")
    if not gate["checks"]["wtp_evidence"]:
        missing.append("wtp")
    if not gate["checks"]["distribution_evidence"]:
        missing.append("distribution")
    if not gate["checks"]["regulatory_evidence_if_needed"]:
        missing.append("regulatory")

    return missing


# =========================================================
# TARGETED RESEARCH
# =========================================================

def base_queries(idea):
    queries = [
        ("direct_competitors", idea["search_term_1"]),
        ("direct_competitors", idea["search_term_2"]),
        ("direct_competitors", idea["search_term_3"]),
        ("pricing", idea["pricing_search_term"]),
        ("wtp", idea["wtp_search_term"]),
        ("distribution", idea["distribution_search_term"]),
    ]

    if idea["regulatory_sensitive"] and idea["regulatory_search_term"].strip():
        queries.append(("regulatory", idea["regulatory_search_term"]))

    return queries


def retry_queries_for(idea, missing_types):
    mapping = {
        "direct_competitors": [
            f'"{idea["job_to_be_done"]}" software',
            f'"{idea["job_to_be_done"]}" API pricing',
        ],
        "pricing": [
            f'"{idea["job_to_be_done"]}" pricing',
            f'"{idea["job_to_be_done"]}" price API SaaS',
        ],
        "wtp": [
            f'"{idea["buyer"]}" "{idea["problem"]}" cost',
            f'"{idea["job_to_be_done"]}" ROI case study',
        ],
        "distribution": [
            f'"{idea["buyer"]}" marketplace directory association',
            f'"{idea["buyer"]}" software marketplace ecosystem',
        ],
        "regulatory": [
            idea["regulatory_search_term"],
            f'"{idea["job_to_be_done"]}" regulation API license official',
        ],
    }

    output = []
    for missing in missing_types:
        for q in mapping.get(missing, []):
            if q and q.strip():
                output.append((missing, q))
    return output


def run_search_queries(queries, status, idea_name):
    found = []

    for idx, (query_type, query) in enumerate(queries, 1):
        status.info(
            f"🔎 {idea_name}: بحث {idx}/{len(queries)} — {query_type}"
        )

        try:
            found.extend(ddgs_search(query, query_type, max_results=5))
        except Exception as e:
            found.append(
                {
                    "query_type": query_type,
                    "title": "",
                    "url": "",
                    "snippet": f"SEARCH_ERROR: {e}",
                    "page_excerpt": "",
                }
            )

        time.sleep(0.8)

    return [x for x in found if x.get("url")]


def research_one_idea(idea, status):
    # Phase 1
    sources = dedupe_sources(run_search_queries(base_queries(idea), status, idea["name"]))
    extract_best_pages(sources, max_pages=3)

    judged = evaluate_source_relevance(idea, sources, status)
    if not judged["ok"]:
        return {
            "ok": False,
            "research_status": "RESEARCH_FAILED",
            "gate": None,
            "sources": sources,
            "error": judged["error"],
        }

    sources = judged["sources"]
    gate = research_gate(idea, sources)

    # Phase 2: automatic re-search if needed
    if gate["status"] != "SUFFICIENT":
        missing = missing_query_types(gate)
        retry_qs = retry_queries_for(idea, missing)

        if retry_qs:
            status.warning(
                f"🔁 {idea['name']}: البحث غير كافٍ. إعادة بحث مستهدفة: "
                + ", ".join(missing)
            )

            extra = run_search_queries(retry_qs, status, idea["name"])
            combined = dedupe_sources(sources + extra, limit=18)
            extract_best_pages(combined, max_pages=4)

            judged2 = evaluate_source_relevance(idea, combined, status)
            if judged2["ok"]:
                sources = judged2["sources"]
                gate = research_gate(idea, sources)

    return {
        "ok": True,
        "research_status": gate["status"],
        "gate": gate,
        "sources": sources,
        "error": None,
    }


def research_all(ideas, status):
    research = {}

    for idx, idea in enumerate(ideas, 1):
        status.info(
            f"🌐 Evidence Research {idx}/{len(ideas)}: {idea['name']}"
        )
        research[idea["id"]] = research_one_idea(idea, status)
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
        "snippet": source["snippet"],
        "page_excerpt": cut(source.get("page_excerpt", ""), 800),
        "relevance_score": ev["relevance_score"],
        "same_job_to_be_done": ev["same_job_to_be_done"],
        "authoritative": ev["authoritative"],
        "categories": ev["categories"],
        "why_relevant": ev["why_relevant"],
    }


def evidence_packet(research_result):
    sources = [
        s
        for s in research_result.get("sources", [])
        if s.get("evaluation", {}).get("relevance_score", 0) >= 55
        and "IRRELEVANT" not in s.get("evaluation", {}).get("categories", [])
    ]

    sources = sorted(
        sources,
        key=lambda s: s["evaluation"]["relevance_score"],
        reverse=True,
    )

    return [source_packet(s) for s in sources[:12]]


# =========================================================
# KILLER PER-IDEA SCHEMA
# =========================================================

claim_schema = obj(
    {
        "claim": {"type": "string"},
        "status": {
            "type": "string",
            "enum": [
                "VERIFIED_RISK",
                "UNVERIFIED_RISK",
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
            "enum": [
                "SUFFICIENT",
                "INSUFFICIENT_EVIDENCE",
                "RESEARCH_FAILED",
            ],
        },
        "top_risks": arr(claim_schema, 3, 3),
        "kill_shot": {"type": "string"},
        "kill_shot_status": {
            "type": "string",
            "enum": [
                "VERIFIED_RISK",
                "UNVERIFIED_RISK",
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
            "enum": [
                "SURVIVES",
                "KILL IT",
                "INSUFFICIENT EVIDENCE",
                "RESEARCH FAILED",
            ],
        },
    }
)


def enforce_killer_evidence(idea, research_result, data):
    # Keep research state aligned with actual gate
    actual_status = research_result["research_status"]
    data["research_status"] = actual_status

    # If research failed, model cannot kill or survive
    if actual_status == "RESEARCH_FAILED":
        data["decision"] = "RESEARCH FAILED"
        return data

    # If evidence is insufficient, KILL requires a verified kill shot with evidence IDs
    if actual_status == "INSUFFICIENT_EVIDENCE":
        if not (
            data["kill_shot_status"] == "VERIFIED_RISK"
            and len(data["kill_shot_evidence_ids"]) >= 1
        ):
            data["decision"] = "INSUFFICIENT EVIDENCE"

    # Any VERIFIED claim must include evidence
    for risk in data["top_risks"]:
        if risk["status"] == "VERIFIED_RISK" and not risk["evidence_ids"]:
            risk["status"] = "UNVERIFIED_RISK"

    if (
        data["kill_shot_status"] == "VERIFIED_RISK"
        and not data["kill_shot_evidence_ids"]
    ):
        data["kill_shot_status"] = "UNVERIFIED_RISK"
        if data["decision"] == "KILL IT":
            data["decision"] = "INSUFFICIENT EVIDENCE"

    return data


def run_killer_one(idea, research_result, status):
    system = """
أنت THE KILLER.

هاجم فكرة واحدة فقط.

قواعد الأدلة:
1. لا تكتب أي ادعاء تنظيمي أو تنافسي أو سعري كحقيقة إلا إذا دعمه Source ID.
2. إذا لم يوجد Source ID مناسب: status = UNVERIFIED_RISK أو UNKNOWN.
3. VERIFIED_RISK يجب أن يحتوي evidence_ids.
4. لا تجعل غياب الدليل دليلاً على الفشل.
5. إذا Research Status = INSUFFICIENT_EVIDENCE ولم يوجد Kill Shot موثق فعلاً:
   decision = INSUFFICIENT EVIDENCE.
6. RESEARCH FAILED يعني لا يجوز الحكم KILL/SURVIVES.

KILL IT فقط إذا يوجد عيب بنيوي موثق يمكنه وحده تدمير الاقتصاديات أو قابلية التنفيذ.
"""

    packet = {
        "idea": idea,
        "research_gate": research_result["gate"],
        "research_status": research_result["research_status"],
        "evidence": evidence_packet(research_result),
    }

    result = call_json(
        KILLER_MODEL,
        system,
        json.dumps(packet, ensure_ascii=False),
        "killer_one_v8",
        KILLER_ONE_SCHEMA,
        1300,
        f"KILLER — {idea['name']}",
        status,
        "none",
    )

    if result["ok"]:
        result["data"] = enforce_killer_evidence(
            idea, research_result, result["data"]
        )

    return result


# =========================================================
# HUNTER REBUTTAL
# =========================================================

REBUTTAL_ONE_SCHEMA = obj(
    {
        "idea_id": {"type": "string"},
        "valid_objection": {"type": "string"},
        "valid_objection_evidence_ids": arr({"type": "string"}, 0, 5),
        "disputed_objection": {"type": "string"},
        "disputed_reason": {"type": "string"},
        "evidence_needed": {"type": "string"},
        "position": {
            "type": "string",
            "enum": ["DEFEND", "DROP", "NEEDS MORE EVIDENCE"],
        },
    }
)


def run_rebuttal_one(idea, research_result, killer_data, status):
    system = """
أنت THE HUNTER.

هذه فرصتك الوحيدة للرد على Killer لفكرة واحدة.

لا تضف فكرة جديدة.
لا تنكر دليلاً موثقاً بلا سبب.
لا تعتبر ادعاء غير موثق حقيقة.
DROP أفضل من الدفاع عن عيب قاتل موثق.
NEEDS MORE EVIDENCE أفضل من التخمين.
"""

    packet = {
        "idea": idea,
        "research_status": research_result["research_status"],
        "research_gate": research_result["gate"],
        "evidence": evidence_packet(research_result),
        "killer": killer_data,
    }

    return call_json(
        HUNTER_MODEL,
        system,
        json.dumps(packet, ensure_ascii=False),
        "hunter_rebuttal_one_v8",
        REBUTTAL_ONE_SCHEMA,
        700,
        f"HUNTER REBUTTAL — {idea['name']}",
        status,
        "low",
    )


# =========================================================
# KILLER FINAL
# =========================================================

KILLER_FINAL_ONE_SCHEMA = obj(
    {
        "idea_id": {"type": "string"},
        "decision": {
            "type": "string",
            "enum": [
                "SURVIVES",
                "KILL IT",
                "INSUFFICIENT EVIDENCE",
                "RESEARCH FAILED",
            ],
        },
        "remaining_problem": {"type": "string"},
        "wtp_real": {"type": "boolean"},
        "distribution_real": {"type": "boolean"},
        "feature_or_company": {
            "type": "string",
            "enum": ["FEATURE", "COMPANY", "UNCLEAR"],
        },
        "final_score_out_of_10": {"type": "integer", "minimum": 0, "maximum": 10},
        "decisive_claim": {"type": "string"},
        "decisive_claim_status": {
            "type": "string",
            "enum": ["VERIFIED_RISK", "UNVERIFIED_RISK", "UNKNOWN"],
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

    if (
        data["decisive_claim_status"] == "VERIFIED_RISK"
        and not data["decisive_evidence_ids"]
    ):
        data["decisive_claim_status"] = "UNVERIFIED_RISK"

    if data["decision"] == "KILL IT":
        if not (
            data["decisive_claim_status"] == "VERIFIED_RISK"
            and len(data["decisive_evidence_ids"]) >= 1
        ):
            data["decision"] = "INSUFFICIENT EVIDENCE"

    if actual_status == "INSUFFICIENT_EVIDENCE" and data["decision"] == "SURVIVES":
        # Insufficient evidence cannot become a true survivor.
        data["decision"] = "INSUFFICIENT EVIDENCE"

    return data


def run_killer_final_one(
    idea,
    research_result,
    killer_data,
    rebuttal_data,
    status,
):
    system = """
أنت THE KILLER في الجولة النهائية لفكرة واحدة.

قواعد الحكم:
- KILL IT يحتاج Decisive Claim موثقاً بـ evidence_ids.
- إذا البحث غير كافٍ ولا يوجد عيب قاتل موثق: INSUFFICIENT EVIDENCE.
- إذا البحث فشل: RESEARCH FAILED.
- SURVIVES يحتاج بحثاً كافياً وعدم وجود Kill Shot موثق.
- لا تحول فرضية أو توقع ("قد تضيف Stripe الميزة") إلى حقيقة.
"""

    packet = {
        "idea": idea,
        "research_status": research_result["research_status"],
        "research_gate": research_result["gate"],
        "evidence": evidence_packet(research_result),
        "killer_first": killer_data,
        "hunter_rebuttal": rebuttal_data,
    }

    result = call_json(
        KILLER_MODEL,
        system,
        json.dumps(packet, ensure_ascii=False),
        "killer_final_one_v8",
        KILLER_FINAL_ONE_SCHEMA,
        850,
        f"KILLER FINAL — {idea['name']}",
        status,
        "none",
    )

    if result["ok"]:
        result["data"] = enforce_final_decision(
            research_result, result["data"]
        )

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

- 2+ Direct Competitors: {gate["checks"]["2_direct_competitors"]}
- Pricing Evidence: {gate["checks"]["pricing_evidence"]}
- WTP Evidence: {gate["checks"]["wtp_evidence"]}
- Distribution Evidence: {gate["checks"]["distribution_evidence"]}
- 3+ Relevant Domains: {gate["checks"]["3_relevant_domains"]}
- Regulatory Evidence if needed: {gate["checks"]["regulatory_evidence_if_needed"]}
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
    output = ["MD INVESTMENT RESEARCH COUNCIL — V8.1", "\nIDEAS"]

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
**V8.1 — Evidence First + Hunter Top-Up**

الجديد:
- تعريف Job-to-be-Done لكل فكرة.
- Relevance Judge لكل مصدر.
- Research Gate يعتمد على جودة الأدلة لا عدد الروابط فقط.
- إعادة بحث تلقائية إذا كانت فئات الأدلة ناقصة.
- Killer يجب أن يربط الادعاءات الخطرة بـ Source IDs.
- لا يسمح KILL IT بسبب ادعاء غير موثق.
- حالات البحث: SUFFICIENT / INSUFFICIENT EVIDENCE / RESEARCH FAILED.
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
                categories = ", ".join(ev.get("categories", []))

                st.markdown(
                    f"""
**[{source.get("source_id","")}] {source.get("title","")}**

- Relevance: **{score}/100**
- Categories: `{categories}`
- Same JTBD: `{ev.get("same_job_to_be_done", False)}`
- Authoritative: `{ev.get("authoritative", False)}`
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
        "MD_Investment_Research_V8_1.txt",
        "text/plain",
        use_container_width=True,
    )

    try:
        st.download_button(
            "📄 تحميل التقرير PDF",
            create_pdf(report),
            "MD_Investment_Research_V8_1.pdf",
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
