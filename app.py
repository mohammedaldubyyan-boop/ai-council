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

BUILD_ID = "V5-RESEARCH-COUNCIL"


# =========================================================
# PAGE
# =========================================================

st.set_page_config(
    page_title="MD Investment Council",
    page_icon="🧠",
    layout="wide"
)


# =========================================================
# RTL
# =========================================================

st.markdown(
    """
    <style>

    .stApp {
        direction: rtl;
    }

    h1, h2, h3, h4, h5, p {
        text-align: right;
    }

    div[data-testid="stMarkdownContainer"] {
        direction: rtl;
        text-align: right;
    }

    textarea {
        direction: rtl !important;
        text-align: right !important;
    }

    div[data-baseweb="textarea"] textarea {
        direction: rtl !important;
        text-align: right !important;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# =========================================================
# GROQ
# =========================================================

client = Groq(
    api_key=st.secrets["GROQ_API_KEY"]
)


# =========================================================
# MODELS
#
# 3 وكلاء = 3 موديلات مختلفة
# =========================================================

HUNTER_MODEL = "openai/gpt-oss-120b"

KILLER_MODEL = "qwen/qwen3.8-27b"

OPERATOR_MODEL = "openai/gpt-oss-20b"

RESEARCH_MODEL = "groq/compound-mini"


# =========================================================
# MODEL USAGE CONTROL
# =========================================================

MODEL_LAST_USED = {}

MIN_MODEL_GAP = {
    HUNTER_MODEL: 25,
    KILLER_MODEL: 25,
    OPERATOR_MODEL: 25,
    RESEARCH_MODEL: 3
}


# =========================================================
# BLACKLIST
#
# الأفكار التي رفضتها مسبقاً
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
    "Noon warranty workflow automation"
]


# =========================================================
# مفاهيم ممنوعة
#
# تمنع تغيير الاسم فقط للهروب من القائمة
# =========================================================

REJECTED_CONCEPTS = {

    "freight_invoice_audit": [
        "freight",
        "shipping invoice",
        "shipment invoice",
        "فواتير الشحن",
        "فاتورة الشحن",
        "تدقيق الشحن",
        "shipping audit"
    ],

    "invoice_collection": [
        "invoice collection",
        "collect invoices",
        "تحصيل الفواتير",
        "invoice chasing"
    ],

    "rfq_comparison": [
        "rfq comparison",
        "compare quotations",
        "مقارنة عروض الأسعار",
        "مقارنة rfq"
    ],

    "tender_rfp": [
        "tender ai",
        "rfp ai",
        "tender analysis",
        "تحليل المناقصات"
    ],

    "warranty": [
        "warranty management",
        "warranty workflow",
        "إدارة الضمان",
        "ضمان المنتجات"
    ],

    "seo_content": [
        "seo content",
        "content refresh",
        "تحديث المحتوى",
        "seo refresh"
    ],

    "saas_monitoring": [
        "subscription monitoring",
        "saas monitoring",
        "مراقبة الاشتراكات"
    ],

    "scope_creep": [
        "scope creep",
        "تغير نطاق المشروع",
        "تجاوز نطاق المشروع"
    ]
}


# =========================================================
# NORMALIZATION
# =========================================================

def normalize_text(text):

    if not text:
        return ""

    text = text.lower()

    text = re.sub(
        r"[^a-z0-9\u0600-\u06ff\s]",
        " ",
        text
    )

    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text.strip()


# =========================================================
# LOCAL BLACKLIST CHECK
# =========================================================

def local_blacklist_check(idea):

    combined = normalize_text(
        " ".join(
            [
                idea.get("name", ""),
                idea.get("one_liner", ""),
                idea.get("product", ""),
                idea.get("problem", "")
            ]
        )
    )


    # -----------------------------------------
    # exact-ish similarity
    # -----------------------------------------

    for rejected in REJECTED_IDEAS:

        rejected_normalized = normalize_text(
            rejected
        )

        ratio = difflib.SequenceMatcher(
            None,
            combined,
            rejected_normalized
        ).ratio()


        if (
            rejected_normalized in combined
            or ratio >= 0.72
        ):

            return {
                "blocked": True,
                "reason": (
                    f"تشابه مباشر مع فكرة مرفوضة: "
                    f"{rejected}"
                )
            }


    # -----------------------------------------
    # concept detection
    # -----------------------------------------

    for concept, phrases in REJECTED_CONCEPTS.items():

        hits = []

        for phrase in phrases:

            if normalize_text(phrase) in combined:

                hits.append(
                    phrase
                )


        if hits:

            return {
                "blocked": True,
                "reason": (
                    f"تشابه مفاهيمي مع قائمة مرفوضة "
                    f"({concept}): "
                    + ", ".join(hits)
                )
            }


    return {
        "blocked": False,
        "reason": ""
    }


# =========================================================
# RATE LIMIT
# =========================================================

def retry_seconds(error_text):

    patterns = [
        r"try again in\s+([0-9.]+)s",
        r"retry after\s+([0-9.]+)",
        r"retry-after[^0-9]*([0-9.]+)"
    ]


    for pattern in patterns:

        match = re.search(
            pattern,
            error_text,
            re.IGNORECASE
        )


        if match:

            try:

                return max(
                    3,
                    math.ceil(
                        float(
                            match.group(1)
                        )
                    ) + 3
                )

            except:
                pass


    return 12


# =========================================================
# MODEL COOLDOWN
# =========================================================

def wait_for_model(
    model,
    stage,
    status
):

    last = MODEL_LAST_USED.get(
        model
    )


    if last is None:
        return


    gap = MIN_MODEL_GAP.get(
        model,
        10
    )


    elapsed = (
        time.time()
        - last
    )


    remaining = (
        gap - elapsed
    )


    if remaining > 0:

        seconds = math.ceil(
            remaining
        )


        status.warning(
            f"⏳ {stage}: "
            f"انتظار {seconds} ثانية "
            f"لحماية الحد المجاني..."
        )


        time.sleep(
            seconds
        )


# =========================================================
# JSON CALL
#
# Structured Outputs strict:true
# =========================================================

def call_json(
    model,
    system_prompt,
    user_prompt,
    schema_name,
    schema,
    max_tokens,
    stage,
    status,
    reasoning="none",
    retries=4
):

    wait_for_model(
        model,
        stage,
        status
    )


    errors = []


    for attempt in range(
        retries
    ):

        try:

            kwargs = {

                "model": model,

                "messages": [
                    {
                        "role": "system",
                        "content": system_prompt
                    },
                    {
                        "role": "user",
                        "content": user_prompt
                    }
                ],

                "temperature": 0.3,

                "max_completion_tokens": (
                    max_tokens
                ),

                "response_format": {
                    "type": "json_schema",

                    "json_schema": {

                        "name": schema_name,

                        "strict": True,

                        "schema": schema
                    }
                }
            }


            if reasoning:

                kwargs[
                    "reasoning_effort"
                ] = reasoning


            response = (
                client
                .chat
                .completions
                .create(**kwargs)
            )


            MODEL_LAST_USED[
                model
            ] = time.time()


            content = (
                response
                .choices[0]
                .message
                .content
            )


            if not content:

                errors.append(
                    f"{model}: empty JSON"
                )

                continue


            return {
                "ok": True,
                "data": json.loads(content),
                "error": None
            }


        except Exception as e:

            error_text = str(e)

            errors.append(
                error_text
            )


            if (
                "429" in error_text
                or "rate_limit" in error_text.lower()
            ):

                seconds = retry_seconds(
                    error_text
                )


                status.warning(
                    f"⏳ {stage}: "
                    f"Groq طلب انتظار "
                    f"{seconds} ثانية..."
                )


                time.sleep(
                    seconds
                )

                continue


            if (
                "413" in error_text
                or "too large" in error_text.lower()
            ):

                break


            time.sleep(2)


    return {
        "ok": False,
        "data": None,
        "error": "\n\n".join(errors)
    }


# =========================================================
# NORMAL TEXT CALL
# =========================================================

def call_text(
    model,
    system_prompt,
    user_prompt,
    max_tokens,
    stage,
    status,
    reasoning="none",
    retries=4
):

    wait_for_model(
        model,
        stage,
        status
    )


    errors = []


    for attempt in range(
        retries
    ):

        try:

            kwargs = {

                "model": model,

                "messages": [
                    {
                        "role": "system",
                        "content": system_prompt
                    },
                    {
                        "role": "user",
                        "content": user_prompt
                    }
                ],

                "temperature": 0.3,

                "max_completion_tokens": (
                    max_tokens
                )
            }


            if reasoning:

                kwargs[
                    "reasoning_effort"
                ] = reasoning


            response = (
                client
                .chat
                .completions
                .create(**kwargs)
            )


            MODEL_LAST_USED[
                model
            ] = time.time()


            content = (
                response
                .choices[0]
                .message
                .content
            )


            if content:

                return {
                    "ok": True,
                    "text": content.strip(),
                    "error": None
                }


            errors.append(
                f"{model}: empty response"
            )


        except Exception as e:

            error_text = str(e)

            errors.append(
                error_text
            )


            if (
                "429" in error_text
                or "rate_limit" in error_text.lower()
            ):

                seconds = retry_seconds(
                    error_text
                )


                status.warning(
                    f"⏳ {stage}: "
                    f"انتظار {seconds} ثانية..."
                )


                time.sleep(
                    seconds
                )

                continue


            time.sleep(2)


    return {
        "ok": False,
        "text": "",
        "error": "\n\n".join(errors)
    }


# =========================================================
# PREPARE BRIEF LOCALLY
# =========================================================

def prepare_case_brief(
    original
):

    text = original.strip()


    # البروتوكول موجود في الكود،
    # فلا داعي لإرساله للنماذج.
    protocol_markers = [
        "\n# الوكلاء",
        "\n## الوكيل الأول",
        "\n# قواعد المناظرة",
        "\n# نظام التقييم",
        "\n# شرط النجاح",
        "\n# اختبار الحقيقة",
        "\n# المرحلة النهائية"
    ]


    cuts = []


    for marker in protocol_markers:

        position = text.find(
            marker
        )

        if position != -1:

            cuts.append(
                position
            )


    if cuts:

        text = text[
            :min(cuts)
        ]


    # safety cap
    if len(text) > 13000:

        text = (
            text[:10000]
            + "\n\n[...]\n\n"
            + text[-2500:]
        )


    return text


# =========================================================
# HUNTER SCHEMA
# =========================================================

HUNTER_SCHEMA = {

    "type": "object",

    "properties": {

        "ideas": {

            "type": "array",

            "minItems": 3,

            "maxItems": 3,

            "items": {

                "type": "object",

                "properties": {

                    "id": {
                        "type": "string"
                    },

                    "name": {
                        "type": "string"
                    },

                    "one_liner": {
                        "type": "string"
                    },

                    "buyer": {
                        "type": "string"
                    },

                    "problem": {
                        "type": "string"
                    },

                    "product": {
                        "type": "string"
                    },

                    "why_pay": {
                        "type": "string"
                    },

                    "price": {
                        "type": "string"
                    },

                    "current_alternative": {
                        "type": "string"
                    },

                    "distribution": {
                        "type": "string"
                    },

                    "first_10_customers": {
                        "type": "string"
                    },

                    "automation": {
                        "type": "integer",
                        "minimum": 0,
                        "maximum": 100
                    },

                    "human_work": {
                        "type": "string"
                    },

                    "why_now": {
                        "type": "string"
                    },

                    "similar_to_rejected": {
                        "type": "boolean"
                    },

                    "rejected_similarity_explanation": {
                        "type": "string"
                    }
                },

                "required": [
                    "id",
                    "name",
                    "one_liner",
                    "buyer",
                    "problem",
                    "product",
                    "why_pay",
                    "price",
                    "current_alternative",
                    "distribution",
                    "first_10_customers",
                    "automation",
                    "human_work",
                    "why_now",
                    "similar_to_rejected",
                    "rejected_similarity_explanation"
                ],

                "additionalProperties": False
            }
        }
    },

    "required": [
        "ideas"
    ],

    "additionalProperties": False
}


# =========================================================
# HUNTER
# =========================================================

def run_hunter(
    case_brief,
    status
):

    system = """
أنت THE HUNTER.

ابحث عن أماكن تتحرك فيها الأموال فعلياً.

الأولوية:
- ألم مالي.
- willingness to pay.
- distribution.
- automation.
- recurring usage.
- speed to revenue.

لا تعتبر AI ميزة بحد ذاته.

ممنوع:
- AI wrapper.
- dashboard عام.
- أداة يمكن لـChatGPT تنفيذها بما يكفي.
- إعادة تغليف فكرة رفضها المستخدم.

أخرج 3 أفكار فقط.

إذا الفكرة قريبة من إحدى الأفكار المرفوضة،
يجب أن تجعل similar_to_rejected=true.

لا تختر WINNER.
"""


    prompt = f"""
هذه حالة المستخدم:

====================

{case_brief}

====================

أنشئ 3 أفكار مختلفة اقتصادياً.

لا تعتمد على ادعاءات غير مؤكدة كأنها حقائق.

هذه القائمة مرفوضة صراحة:

{json.dumps(REJECTED_IDEAS, ensure_ascii=False)}
"""


    return call_json(
        HUNTER_MODEL,
        system,
        prompt,
        "hunter_ideas",
        HUNTER_SCHEMA,
        max_tokens=1600,
        stage="THE HUNTER",
        status=status,
        reasoning="low"
    )


# =========================================================
# FILTER
# =========================================================

def filter_ideas(
    hunter_data
):

    passed = []

    blocked = []


    for idea in hunter_data[
        "ideas"
    ]:

        local_result = (
            local_blacklist_check(
                idea
            )
        )


        model_blocked = idea[
            "similar_to_rejected"
        ]


        if (
            local_result["blocked"]
            or model_blocked
        ):

            reason = (
                local_result["reason"]
                if local_result["blocked"]
                else idea[
                    "rejected_similarity_explanation"
                ]
            )


            blocked.append(
                {
                    "idea": idea,
                    "reason": reason
                }
            )


        else:

            passed.append(
                idea
            )


    return (
        passed,
        blocked
    )


# =========================================================
# WEB RESEARCH
#
# Compound Mini فقط لبحث صغير.
# لا نرسل الـbrief الضخم.
# =========================================================

def extract_search_sources(
    response
):

    sources = []


    try:

        tools = (
            response
            .choices[0]
            .message
            .executed_tools
        )


        if not tools:
            return sources


        for tool in tools:

            search_results = getattr(
                tool,
                "search_results",
                None
            )


            if not search_results:
                continue


            results = getattr(
                search_results,
                "results",
                None
            )


            if results is None:

                if isinstance(
                    search_results,
                    dict
                ):

                    results = (
                        search_results
                        .get(
                            "results",
                            []
                        )
                    )


            if not results:
                continue


            for item in results:

                if isinstance(
                    item,
                    dict
                ):

                    title = item.get(
                        "title",
                        ""
                    )

                    url = item.get(
                        "url",
                        ""
                    )

                    snippet = item.get(
                        "content",
                        ""
                    )


                else:

                    title = getattr(
                        item,
                        "title",
                        ""
                    )

                    url = getattr(
                        item,
                        "url",
                        ""
                    )

                    snippet = getattr(
                        item,
                        "content",
                        ""
                    )


                if url:

                    sources.append(
                        {
                            "title": title,
                            "url": url,
                            "snippet": snippet
                        }
                    )


    except Exception:
        pass


    # deduplicate
    unique = []

    seen = set()


    for source in sources:

        url = source[
            "url"
        ]


        if url not in seen:

            seen.add(
                url
            )

            unique.append(
                source
            )


    return unique[:8]


# =========================================================
# RESEARCH ONE IDEA
# =========================================================

def research_idea(
    idea,
    status
):

    wait_for_model(
        RESEARCH_MODEL,
        f"بحث {idea['name']}",
        status
    )


    prompt = f"""
Research this business idea using current web sources.

IDEA:
{idea["name"]}

BUYER:
{idea["buyer"]}

PRODUCT:
{idea["product"]}

PRICE HYPOTHESIS:
{idea["price"]}

DISTRIBUTION HYPOTHESIS:
{idea["distribution"]}

Search specifically for:

1. Direct competitors doing the same job-to-be-done.
2. Competitor pricing where publicly visible.
3. Existing free/platform-native alternatives.
4. Evidence that this buyer actually pays for this problem.
5. Evidence for or against urgency / why now.
6. Concrete distribution channels to reach buyers.
7. Regulatory/platform/API risks.
8. Evidence contradicting the idea.

Be skeptical.

Do not invent facts.

Separate:
VERIFIED
UNKNOWN
CONTRADICTED

Keep the answer concise.
"""


    errors = []


    for attempt in range(4):

        try:

            response = (
                client
                .chat
                .completions
                .create(
                    model=RESEARCH_MODEL,

                    messages=[
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ],

                    max_completion_tokens=1300,

                    compound_custom={
                        "tools": {
                            "enabled_tools": [
                                "web_search"
                            ]
                        }
                    }
                )
            )


            MODEL_LAST_USED[
                RESEARCH_MODEL
            ] = time.time()


            content = (
                response
                .choices[0]
                .message
                .content
            )


            if content:

                return {
                    "ok": True,
                    "text": content,
                    "sources": (
                        extract_search_sources(
                            response
                        )
                    ),
                    "error": None
                }


        except Exception as e:

            error_text = str(e)

            errors.append(
                error_text
            )


            if (
                "429" in error_text
                or "rate_limit" in error_text.lower()
            ):

                seconds = retry_seconds(
                    error_text
                )


                status.warning(
                    f"⏳ البحث عن "
                    f"{idea['name']}: "
                    f"انتظار {seconds} ثانية..."
                )


                time.sleep(
                    seconds
                )

                continue


            time.sleep(2)


    return {
        "ok": False,
        "text": "",
        "sources": [],
        "error": "\n\n".join(errors)
    }


# =========================================================
# KILLER SCHEMA
# =========================================================

KILLER_SCHEMA = {

    "type": "object",

    "properties": {

        "reviews": {

            "type": "array",

            "items": {

                "type": "object",

                "properties": {

                    "idea_id": {
                        "type": "string"
                    },

                    "top_failure_reason_1": {
                        "type": "string"
                    },

                    "top_failure_reason_2": {
                        "type": "string"
                    },

                    "top_failure_reason_3": {
                        "type": "string"
                    },

                    "kill_shot": {
                        "type": "string"
                    },

                    "immediate_rejection_evidence": {
                        "type": "string"
                    },

                    "research_supports_idea": {
                        "type": "string"
                    },

                    "research_hurts_idea": {
                        "type": "string"
                    },

                    "score_out_of_10": {
                        "type": "integer",
                        "minimum": 0,
                        "maximum": 10
                    },

                    "decision": {
                        "type": "string",
                        "enum": [
                            "SURVIVES",
                            "KILL IT"
                        ]
                    }
                },

                "required": [
                    "idea_id",
                    "top_failure_reason_1",
                    "top_failure_reason_2",
                    "top_failure_reason_3",
                    "kill_shot",
                    "immediate_rejection_evidence",
                    "research_supports_idea",
                    "research_hurts_idea",
                    "score_out_of_10",
                    "decision"
                ],

                "additionalProperties": False
            }
        }
    },

    "required": [
        "reviews"
    ],

    "additionalProperties": False
}


# =========================================================
# KILLER FIRST ATTACK
# =========================================================

def run_killer(
    ideas,
    research,
    status
):

    system = """
أنت THE KILLER.

أنت مستثمر متشائم.

لا تقترح أفكاراً جديدة.

حاول قتل كل فكرة بناءً على:
- competition
- WTP
- CAC
- churn
- distribution
- liability
- regulation
- platform risk
- security/privacy
- support
- moat
- ChatGPT substitution
- Feature vs Company

الأهم:
استخدم نتائج البحث كمصدر للحكم.

لا تعتبر ادعاء Hunter حقيقة إذا لم يدعمه البحث.

إذا الدليل غير موجود، قل UNKNOWN.

ممنوع المجاملة.
"""


    payload = []


    for idea in ideas:

        item = {
            "idea": idea,
            "research": research.get(
                idea["id"],
                {}
            ).get(
                "text",
                "NO RESEARCH"
            )
        }

        payload.append(
            item
        )


    prompt = (
        "هاجم هذه الأفكار:\n\n"
        + json.dumps(
            payload,
            ensure_ascii=False
        )
    )


    return call_json(
        KILLER_MODEL,
        system,
        prompt,
        "killer_reviews",
        KILLER_SCHEMA,
        max_tokens=1500,
        stage="THE KILLER",
        status=status,
        reasoning="none"
    )


# =========================================================
# HUNTER REBUTTAL SCHEMA
# =========================================================

REBUTTAL_SCHEMA = {

    "type": "object",

    "properties": {

        "responses": {

            "type": "array",

            "items": {

                "type": "object",

                "properties": {

                    "idea_id": {
                        "type": "string"
                    },

                    "valid_objection": {
                        "type": "string"
                    },

                    "disputed_objection": {
                        "type": "string"
                    },

                    "evidence_needed": {
                        "type": "string"
                    },

                    "position": {
                        "type": "string",
                        "enum": [
                            "DEFEND",
                            "DROP"
                        ]
                    }
                },

                "required": [
                    "idea_id",
                    "valid_objection",
                    "disputed_objection",
                    "evidence_needed",
                    "position"
                ],

                "additionalProperties": False
            }
        }
    },

    "required": [
        "responses"
    ],

    "additionalProperties": False
}


# =========================================================
# HUNTER REBUTTAL
# =========================================================

def run_rebuttal(
    ideas,
    killer_data,
    research,
    status
):

    system = """
أنت THE HUNTER.

هذه فرصتك الوحيدة للرد.

لا تضف أفكاراً جديدة.

إذا كشف البحث أو Killer عيباً حقيقياً،
اعترف به.

لا تحاول إنقاذ فكرة سيئة.

DROP أفضل من ترقيع فكرة ميتة.
"""


    payload = {
        "ideas": ideas,
        "killer": killer_data,
        "research": {
            idea["id"]:
                research.get(
                    idea["id"],
                    {}
                ).get(
                    "text",
                    ""
                )

            for idea in ideas
        }
    }


    return call_json(
        HUNTER_MODEL,
        system,
        json.dumps(
            payload,
            ensure_ascii=False
        ),
        "hunter_rebuttal",
        REBUTTAL_SCHEMA,
        max_tokens=900,
        stage="HUNTER REBUTTAL",
        status=status,
        reasoning="low"
    )


# =========================================================
# FINAL KILLER SCHEMA
# =========================================================

FINAL_KILLER_SCHEMA = {

    "type": "object",

    "properties": {

        "decisions": {

            "type": "array",

            "items": {

                "type": "object",

                "properties": {

                    "idea_id": {
                        "type": "string"
                    },

                    "decision": {
                        "type": "string",
                        "enum": [
                            "SURVIVES",
                            "KILL IT"
                        ]
                    },

                    "remaining_problem": {
                        "type": "string"
                    },

                    "wtp_real": {
                        "type": "boolean"
                    },

                    "distribution_real": {
                        "type": "boolean"
                    },

                    "feature_or_company": {
                        "type": "string",
                        "enum": [
                            "FEATURE",
                            "COMPANY",
                            "UNCLEAR"
                        ]
                    },

                    "final_score_out_of_10": {
                        "type": "integer",
                        "minimum": 0,
                        "maximum": 10
                    }
                },

                "required": [
                    "idea_id",
                    "decision",
                    "remaining_problem",
                    "wtp_real",
                    "distribution_real",
                    "feature_or_company",
                    "final_score_out_of_10"
                ],

                "additionalProperties": False
            }
        }
    },

    "required": [
        "decisions"
    ],

    "additionalProperties": False
}


# =========================================================
# KILLER FINAL
# =========================================================

def run_killer_final(
    ideas,
    first_killer,
    rebuttal,
    research,
    status
):

    system = """
أنت THE KILLER.

هذه آخر فرصة للهجوم.

لا تولد أفكاراً.

لا تعيد إحياء فكرة ضعيفة.

احكم بناءً على الأدلة الموجودة.

إذا WTP أو Distribution غير مثبتة،
لا تتعامل معها كحقائق.
"""


    payload = {
        "ideas": ideas,
        "first_attack": first_killer,
        "hunter_rebuttal": rebuttal,
        "research": {
            idea["id"]:
                research.get(
                    idea["id"],
                    {}
                ).get(
                    "text",
                    ""
                )

            for idea in ideas
        }
    }


    return call_json(
        KILLER_MODEL,
        system,
        json.dumps(
            payload,
            ensure_ascii=False
        ),
        "killer_final",
        FINAL_KILLER_SCHEMA,
        max_tokens=1000,
        stage="KILLER FINAL",
        status=status,
        reasoning="none"
    )


# =========================================================
# OPERATOR SCHEMA
# =========================================================

OPERATOR_SCHEMA = {

    "type": "object",

    "properties": {

        "evaluations": {

            "type": "array",

            "items": {

                "type": "object",

                "properties": {

                    "idea_id": {
                        "type": "string"
                    },

                    "idea_name": {
                        "type": "string"
                    },

                    "severity": {
                        "type": "integer",
                        "minimum": 0,
                        "maximum": 15
                    },

                    "willingness_to_pay": {
                        "type": "integer",
                        "minimum": 0,
                        "maximum": 15
                    },

                    "distribution": {
                        "type": "integer",
                        "minimum": 0,
                        "maximum": 15
                    },

                    "automation": {
                        "type": "integer",
                        "minimum": 0,
                        "maximum": 15
                    },

                    "recurring": {
                        "type": "integer",
                        "minimum": 0,
                        "maximum": 10
                    },

                    "competition": {
                        "type": "integer",
                        "minimum": 0,
                        "maximum": 10
                    },

                    "moat": {
                        "type": "integer",
                        "minimum": 0,
                        "maximum": 5
                    },

                    "speed_to_revenue": {
                        "type": "integer",
                        "minimum": 0,
                        "maximum": 10
                    },

                    "stack_fit": {
                        "type": "integer",
                        "minimum": 0,
                        "maximum": 5
                    },

                    "total_score": {
                        "type": "integer",
                        "minimum": 0,
                        "maximum": 100
                    },

                    "price": {
                        "type": "string"
                    },

                    "gross_margin": {
                        "type": "string"
                    },

                    "ltv": {
                        "type": "string"
                    },

                    "cac": {
                        "type": "string"
                    },

                    "customers_for_1k_mrr": {
                        "type": "string"
                    },

                    "customers_for_5k_mrr": {
                        "type": "string"
                    },

                    "customers_for_10k_mrr": {
                        "type": "string"
                    },

                    "automation_percent": {
                        "type": "integer",
                        "minimum": 0,
                        "maximum": 100
                    },

                    "first_buyer": {
                        "type": "string"
                    },

                    "fastest_test": {
                        "type": "string"
                    },

                    "biggest_risk": {
                        "type": "string"
                    },

                    "truth_test": {
                        "type": "string"
                    }
                },

                "required": [
                    "idea_id",
                    "idea_name",
                    "severity",
                    "willingness_to_pay",
                    "distribution",
                    "automation",
                    "recurring",
                    "competition",
                    "moat",
                    "speed_to_revenue",
                    "stack_fit",
                    "total_score",
                    "price",
                    "gross_margin",
                    "ltv",
                    "cac",
                    "customers_for_1k_mrr",
                    "customers_for_5k_mrr",
                    "customers_for_10k_mrr",
                    "automation_percent",
                    "first_buyer",
                    "fastest_test",
                    "biggest_risk",
                    "truth_test"
                ],

                "additionalProperties": False
            }
        },

        "winner_exists": {
            "type": "boolean"
        },

        "winner_idea_id": {
            "type": "string"
        },

        "winner_reason": {
            "type": "string"
        }
    },

    "required": [
        "evaluations",
        "winner_exists",
        "winner_idea_id",
        "winner_reason"
    ],

    "additionalProperties": False
}


# =========================================================
# OPERATOR
# =========================================================

def run_operator(
    surviving_ideas,
    research,
    killer_final,
    status
):

    system = """
أنت THE OPERATOR / ECONOMIST.

أنت CTO + CFO + Growth Operator.

قيّم فقط الأفكار التي نجت من Killer.

نظام الـ100:

Severity 15
WTP 15
Distribution 15
Automation 15
Recurring 10
Competition 10
Moat 5
Speed to Revenue 10
Stack Fit 5

لا ترفع الدرجة للوصول إلى 85.

WINNER فقط إذا total_score > 85.

إذا لا توجد فكرة تستحق ذلك:
winner_exists=false.

لا تخترع أرقاماً.
إذا CAC/LTV غير مثبتة، استخدم تقديراً واضحاً
ولا تتعامل معه كحقيقة.
"""


    payload = {

        "surviving_ideas":
            surviving_ideas,

        "research": {

            idea["id"]:
                research.get(
                    idea["id"],
                    {}
                ).get(
                    "text",
                    ""
                )

            for idea in surviving_ideas
        },

        "killer_final":
            killer_final
    }


    return call_json(
        OPERATOR_MODEL,
        system,
        json.dumps(
            payload,
            ensure_ascii=False
        ),
        "operator_evaluation",
        OPERATOR_SCHEMA,
        max_tokens=1800,
        stage="THE OPERATOR",
        status=status,
        reasoning="low"
    )


# =========================================================
# FINAL OBJECTION
# =========================================================

def final_objection(
    operator_data,
    surviving_ideas,
    research,
    status
):

    if not operator_data[
        "winner_exists"
    ]:

        return {
            "ok": True,
            "text": (
                "## FINAL OBJECTION\n\n"
                "لا يوجد Winner فوق 85/100؛ "
                "إجبار النظام على اختيار مشروع "
                "سيخالف قواعد التقييم."
            )
        }


    winner_id = operator_data[
        "winner_idea_id"
    ]


    winner = next(
        (
            idea
            for idea in surviving_ideas
            if idea["id"] == winner_id
        ),
        None
    )


    system = """
أنت THE KILLER.

هذه آخر فرصة لقتل الـWinner.

اكتب أقوى حجة واحدة ممكنة
تجعل المشروع ينتهي عند $0 MRR.

استخدم البحث الموجود.

لا تقترح فكرة جديدة.
"""


    prompt = json.dumps(
        {
            "winner": winner,
            "operator": operator_data,
            "research": (
                research.get(
                    winner_id,
                    {}
                ).get(
                    "text",
                    ""
                )
            )
        },
        ensure_ascii=False
    )


    return call_text(
        KILLER_MODEL,
        system,
        prompt,
        max_tokens=450,
        stage="FINAL OBJECTION",
        status=status,
        reasoning="none"
    )


# =========================================================
# FINAL VERDICT
# =========================================================

def final_verdict(
    operator_data,
    objection,
    surviving_ideas,
    status
):

    system = """
أنت THE OPERATOR.

اتخذ القرار النهائي.

ممنوع تغيير الدرجات عشوائياً.

إذا لا يوجد Winner فوق 85:
القرار KILL.

إذا يوجد Winner:
راجع FINAL OBJECTION.

أخرج فقط:

## FINAL VERDICT

BUILD
أو
KILL

ثم:
- السبب.
- اختبار 7 أيام.
- BUILD criterion.
- KILL criterion.
"""


    prompt = json.dumps(
        {
            "operator":
                operator_data,

            "final_objection":
                objection,

            "surviving_ideas":
                surviving_ideas
        },
        ensure_ascii=False
    )


    return call_text(
        OPERATOR_MODEL,
        system,
        prompt,
        max_tokens=600,
        stage="FINAL VERDICT",
        status=status,
        reasoning="low"
    )


# =========================================================
# RENDER IDEA
# =========================================================

def render_idea(
    idea
):

    st.subheader(
        idea["name"]
    )

    st.write(
        idea["one_liner"]
    )

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


# =========================================================
# FINAL REPORT
# =========================================================

def build_final_report(
    ideas,
    blocked,
    research,
    killer,
    rebuttal,
    killer_final,
    operator,
    objection,
    verdict
):

    text = """
MD INVESTMENT RESEARCH COUNCIL

========================================
IDEAS
========================================
"""


    for idea in ideas:

        text += f"""

{idea["name"]}

Buyer:
{idea["buyer"]}

Problem:
{idea["problem"]}

Price:
{idea["price"]}

Distribution:
{idea["distribution"]}

Automation:
{idea["automation"]}%

"""


    if blocked:

        text += """

========================================
BLOCKED IDEAS
========================================
"""


        for item in blocked:

            text += f"""

{item["idea"]["name"]}

Reason:
{item["reason"]}
"""


    text += """

========================================
WEB RESEARCH
========================================
"""


    for idea in ideas:

        result = research.get(
            idea["id"],
            {}
        )


        text += f"""

{idea["name"]}

{result.get("text", "")}

SOURCES:
"""


        for source in result.get(
            "sources",
            []
        ):

            text += (
                f"\n- "
                f"{source.get('title', '')}: "
                f"{source.get('url', '')}"
            )


    text += f"""

========================================
KILLER FIRST ATTACK
========================================

{json.dumps(killer, ensure_ascii=False, indent=2)}


========================================
HUNTER REBUTTAL
========================================

{json.dumps(rebuttal, ensure_ascii=False, indent=2)}


========================================
KILLER FINAL
========================================

{json.dumps(killer_final, ensure_ascii=False, indent=2)}


========================================
OPERATOR
========================================

{json.dumps(operator, ensure_ascii=False, indent=2)}


========================================
FINAL OBJECTION
========================================

{objection}


========================================
FINAL VERDICT
========================================

{verdict}
"""


    return text.strip()


# =========================================================
# PDF HELPERS
# =========================================================

def clean_pdf_text(
    text
):

    text = re.sub(
        r"#{1,6}\s*",
        "",
        text
    )

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
        "🎯"
    ]


    for emoji in emojis:

        text = text.replace(
            emoji,
            ""
        )


    return text


def find_font():

    options = [
        "/usr/share/fonts/truetype/noto/NotoNaskhArabic-Regular.ttf",
        "/usr/share/fonts/truetype/noto/NotoSansArabic-Regular.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
    ]


    for option in options:

        if os.path.exists(
            option
        ):

            return option


    matches = glob.glob(
        "/usr/share/fonts/**/*.ttf",
        recursive=True
    )


    for font in matches:

        lower = font.lower()

        if (
            "naskh" in lower
            or "arabic" in lower
            or "dejavusans" in lower
        ):

            return font


    return None


def create_pdf(
    report
):

    font = find_font()


    if not font:

        raise RuntimeError(
            "Arabic font not found."
        )


    pdf = FPDF(
        orientation="P",
        unit="mm",
        format="A4"
    )


    pdf.set_auto_page_break(
        True,
        15
    )

    pdf.set_margins(
        15,
        15,
        15
    )

    pdf.add_page()


    pdf.add_font(
        "Arabic",
        fname=font
    )

    pdf.set_font(
        "Arabic",
        size=10
    )


    try:

        pdf.set_text_shaping(
            use_shaping_engine=True,
            direction="rtl",
            script="arab",
            language="ara"
        )

    except Exception:
        pass


    report = clean_pdf_text(
        report
    )


    for line in report.split(
        "\n"
    ):

        line = line.strip()


        if not line:

            pdf.ln(3)
            continue


        if line.startswith(
            "================"
        ):

            continue


        pdf.multi_cell(
            0,
            6,
            line,
            align="R",
            new_x="LMARGIN",
            new_y="NEXT"
        )


    return bytes(
        pdf.output()
    )


# =========================================================
# STATE
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
    "operator": None,
    "objection": None,
    "verdict": None,
    "error": None
}


for key, value in (
    DEFAULT_STATE.items()
):

    if key not in st.session_state:

        st.session_state[
            key
        ] = value


# =========================================================
# UI
# =========================================================

st.title(
    "🧠 MD Investment Research Council"
)


st.caption(
    f"Build: {BUILD_ID}"
)


st.write(
    """
**THE HUNTER** يولد 3 فرص فقط.

ثم يقوم النظام بإسقاط الأفكار التي تشبه
قائمة المشاريع المرفوضة.

بعدها يتم إجراء **بحث ويب حقيقي**
على كل فكرة نجت.

ثم:

**THE KILLER → HUNTER REBUTTAL → THE KILLER → THE OPERATOR**
"""
)


original = st.text_area(
    "اكتب الحالة أو الصق البرومبت السابق:",
    height=330,
    value=st.session_state.original
)


start = st.button(
    "🚀 ابدأ Research Council",
    type="primary",
    use_container_width=True
)


# =========================================================
# RUN
# =========================================================

if start:

    if not original.strip():

        st.warning(
            "اكتب الحالة أولاً."
        )


    else:

        # reset
        for key, value in (
            DEFAULT_STATE.items()
        ):

            st.session_state[
                key
            ] = value


        st.session_state[
            "original"
        ] = original


        status = st.empty()


        # -----------------------------------------
        # PREPARE
        # -----------------------------------------

        status.info(
            "📋 تجهيز الحالة..."
        )


        case_brief = (
            prepare_case_brief(
                original
            )
        )


        st.session_state[
            "case_brief"
        ] = case_brief


        # -----------------------------------------
        # HUNTER
        # -----------------------------------------

        status.info(
            "🎯 THE HUNTER يولد 3 أفكار..."
        )


        hunter_result = run_hunter(
            case_brief,
            status
        )


        if not hunter_result["ok"]:

            st.session_state[
                "error"
            ] = hunter_result[
                "error"
            ]


        else:

            hunter_data = (
                hunter_result[
                    "data"
                ]
            )


            # -------------------------------------
            # FILTER
            # -------------------------------------

            ideas, blocked = (
                filter_ideas(
                    hunter_data
                )
            )


            st.session_state[
                "ideas"
            ] = ideas

            st.session_state[
                "blocked"
            ] = blocked


            if not ideas:

                st.session_state[
                    "verdict"
                ] = (
                    "## FINAL VERDICT\n\n"
                    "KILL\n\n"
                    "كل الأفكار التي ولدها Hunter "
                    "كانت قريبة من أفكار مرفوضة مسبقاً."
                )


                status.error(
                    "❌ لم تنج أي فكرة من فلتر المرفوضات."
                )


            else:

                # ---------------------------------
                # WEB RESEARCH
                # ---------------------------------

                research = {}


                for index, idea in enumerate(
                    ideas,
                    start=1
                ):

                    status.info(
                        f"🔎 بحث ويب "
                        f"{index}/{len(ideas)}: "
                        f"{idea['name']}"
                    )


                    result = research_idea(
                        idea,
                        status
                    )


                    research[
                        idea["id"]
                    ] = result


                st.session_state[
                    "research"
                ] = research


                # لا نسمح بالاستمرار إذا فشل
                # البحث على كل الأفكار.
                failed_research = [

                    idea["id"]

                    for idea in ideas

                    if not research[
                        idea["id"]
                    ]["ok"]
                ]


                if failed_research:

                    st.session_state[
                        "error"
                    ] = (
                        "فشل بحث الويب للأفكار: "
                        + ", ".join(
                            failed_research
                        )
                    )


                else:

                    # -----------------------------
                    # KILLER
                    # -----------------------------

                    status.info(
                        "🔪 THE KILLER يراجع "
                        "الأفكار والأدلة..."
                    )


                    result = run_killer(
                        ideas,
                        research,
                        status
                    )


                    if not result["ok"]:

                        st.session_state[
                            "error"
                        ] = result[
                            "error"
                        ]


                    else:

                        killer_data = (
                            result[
                                "data"
                            ]
                        )


                        st.session_state[
                            "killer"
                        ] = killer_data


                        # -------------------------
                        # REBUTTAL
                        # -------------------------

                        status.info(
                            "🎯 Hunter يرد مرة واحدة..."
                        )


                        result = run_rebuttal(
                            ideas,
                            killer_data,
                            research,
                            status
                        )


                        if not result["ok"]:

                            st.session_state[
                                "error"
                            ] = result[
                                "error"
                            ]


                        else:

                            rebuttal_data = (
                                result[
                                    "data"
                                ]
                            )


                            st.session_state[
                                "rebuttal"
                            ] = rebuttal_data


                            # ---------------------
                            # KILLER FINAL
                            # ---------------------

                            status.info(
                                "🔪 Killer يصدر الحكم الأخير..."
                            )


                            result = run_killer_final(
                                ideas,
                                killer_data,
                                rebuttal_data,
                                research,
                                status
                            )


                            if not result["ok"]:

                                st.session_state[
                                    "error"
                                ] = result[
                                    "error"
                                ]


                            else:

                                final_killer = (
                                    result[
                                        "data"
                                    ]
                                )


                                st.session_state[
                                    "killer_final"
                                ] = final_killer


                                surviving_ids = [

                                    item[
                                        "idea_id"
                                    ]

                                    for item in final_killer[
                                        "decisions"
                                    ]

                                    if item[
                                        "decision"
                                    ] == "SURVIVES"
                                ]


                                surviving = [

                                    idea

                                    for idea in ideas

                                    if idea[
                                        "id"
                                    ] in surviving_ids
                                ]


                                # -----------------
                                # NONE SURVIVED
                                # -----------------

                                if not surviving:

                                    st.session_state[
                                        "operator"
                                    ] = {
                                        "evaluations": [],
                                        "winner_exists": False,
                                        "winner_idea_id": "",
                                        "winner_reason": (
                                            "لم تنج أي فكرة "
                                            "من Red Team."
                                        )
                                    }


                                    st.session_state[
                                        "objection"
                                    ] = (
                                        "لا توجد فكرة ناجية "
                                        "يمكن الاعتراض على اختيارها."
                                    )


                                    st.session_state[
                                        "verdict"
                                    ] = (
                                        "## FINAL VERDICT\n\n"
                                        "KILL\n\n"
                                        "لم تنج أي فكرة من "
                                        "THE KILLER بعد البحث."
                                    )


                                    status.success(
                                        "✅ انتهت المناظرة: NO WINNER"
                                    )


                                else:

                                    # -------------
                                    # OPERATOR
                                    # -------------

                                    status.info(
                                        "📊 THE OPERATOR "
                                        "يحسب الاقتصاديات..."
                                    )


                                    result = run_operator(
                                        surviving,
                                        research,
                                        final_killer,
                                        status
                                    )


                                    if not result["ok"]:

                                        st.session_state[
                                            "error"
                                        ] = result[
                                            "error"
                                        ]


                                    else:

                                        operator_data = (
                                            result[
                                                "data"
                                            ]
                                        )


                                        st.session_state[
                                            "operator"
                                        ] = operator_data


                                        # ---------
                                        # OBJECTION
                                        # ---------

                                        status.info(
                                            "🔪 FINAL OBJECTION..."
                                        )


                                        result = final_objection(
                                            operator_data,
                                            surviving,
                                            research,
                                            status
                                        )


                                        if not result["ok"]:

                                            st.session_state[
                                                "error"
                                            ] = result[
                                                "error"
                                            ]


                                        else:

                                            objection = (
                                                result[
                                                    "text"
                                                ]
                                            )


                                            st.session_state[
                                                "objection"
                                            ] = objection


                                            # -----
                                            # VERDICT
                                            # -----

                                            status.info(
                                                "🏛️ FINAL VERDICT..."
                                            )


                                            result = final_verdict(
                                                operator_data,
                                                objection,
                                                surviving,
                                                status
                                            )


                                            if not result["ok"]:

                                                st.session_state[
                                                    "error"
                                                ] = result[
                                                    "error"
                                                ]


                                            else:

                                                st.session_state[
                                                    "verdict"
                                                ] = result[
                                                    "text"
                                                ]


                                                status.success(
                                                    "✅ انتهى Research Council"
                                                )


# =========================================================
# ERROR
# =========================================================

if st.session_state.error:

    st.error(
        "حدث خطأ ولم يصدر المجلس حكماً ناقصاً."
    )


    with st.expander(
        "🔧 التفاصيل التقنية",
        expanded=True
    ):

        st.code(
            st.session_state.error
        )


# =========================================================
# BLOCKED
# =========================================================

if st.session_state.blocked:

    st.divider()

    st.header(
        "🚫 أفكار أسقطها الفلتر"
    )


    for item in (
        st.session_state.blocked
    ):

        st.warning(
            f"**{item['idea']['name']}**\n\n"
            f"{item['reason']}"
        )


# =========================================================
# IDEAS
# =========================================================

if st.session_state.ideas:

    st.divider()

    st.header(
        "🎯 أفكار THE HUNTER"
    )


    for idea in (
        st.session_state.ideas
    ):

        with st.expander(
            idea["name"],
            expanded=True
        ):

            render_idea(
                idea
            )


# =========================================================
# RESEARCH
# =========================================================

if st.session_state.research:

    st.divider()

    st.header(
        "🔎 Web Research"
    )


    for idea in (
        st.session_state.ideas
        or []
    ):

        result = (
            st.session_state
            .research
            .get(
                idea["id"],
                {}
            )
        )


        with st.expander(
            f"🔎 {idea['name']}",
            expanded=False
        ):

            st.markdown(
                result.get(
                    "text",
                    ""
                )
            )


            sources = result.get(
                "sources",
                []
            )


            if sources:

                st.subheader(
                    "المصادر"
                )


                for source in sources:

                    st.markdown(
                        f"- [{source.get('title', 'Source')}]"
                        f"({source.get('url', '')})"
                    )


# =========================================================
# KILLER
# =========================================================

if st.session_state.killer:

    st.divider()

    st.header(
        "🔪 THE KILLER"
    )


    for review in (
        st.session_state
        .killer[
            "reviews"
        ]
    ):

        st.subheader(
            review[
                "idea_id"
            ]
        )


        st.markdown(
            f"""
**سبب الفشل 1:** {review["top_failure_reason_1"]}

**سبب الفشل 2:** {review["top_failure_reason_2"]}

**سبب الفشل 3:** {review["top_failure_reason_3"]}

**Kill Shot:** {review["kill_shot"]}

**الدليل الذي يقتلها فوراً:** {review["immediate_rejection_evidence"]}

**الدليل المؤيد:** {review["research_supports_idea"]}

**الدليل المضاد:** {review["research_hurts_idea"]}

**Score:** {review["score_out_of_10"]}/10

**Decision:** `{review["decision"]}`
"""
        )


# =========================================================
# REBUTTAL
# =========================================================

if st.session_state.rebuttal:

    st.divider()

    st.header(
        "🎯 HUNTER REBUTTAL"
    )


    for item in (
        st.session_state
        .rebuttal[
            "responses"
        ]
    ):

        st.markdown(
            f"""
### {item["idea_id"]}

**اعتراض صحيح:** {item["valid_objection"]}

**اعتراض يرفضه Hunter:** {item["disputed_objection"]}

**الدليل المطلوب:** {item["evidence_needed"]}

**Position:** `{item["position"]}`
"""
        )


# =========================================================
# KILLER FINAL
# =========================================================

if st.session_state.killer_final:

    st.divider()

    st.header(
        "🔪 KILLER FINAL"
    )


    for item in (
        st.session_state
        .killer_final[
            "decisions"
        ]
    ):

        st.markdown(
            f"""
### {item["idea_id"]}

**Decision:** `{item["decision"]}`

**المشكلة المتبقية:** {item["remaining_problem"]}

**WTP مثبتة؟** {item["wtp_real"]}

**Distribution واقعية؟** {item["distribution_real"]}

**Feature / Company:** {item["feature_or_company"]}

**Red-Team Score:** {item["final_score_out_of_10"]}/10
"""
        )


# =========================================================
# OPERATOR
# =========================================================

if st.session_state.operator:

    st.divider()

    st.header(
        "📊 THE OPERATOR"
    )


    operator = (
        st.session_state.operator
    )


    evaluations = operator.get(
        "evaluations",
        []
    )


    if evaluations:

        table_rows = []


        for item in evaluations:

            table_rows.append(
                {
                    "Idea": item[
                        "idea_name"
                    ],

                    "Score": item[
                        "total_score"
                    ],

                    "First Buyer": item[
                        "first_buyer"
                    ],

                    "Price": item[
                        "price"
                    ],

                    "Automation": (
                        f"{item['automation_percent']}%"
                    ),

                    "Fastest Test": item[
                        "fastest_test"
                    ],

                    "Biggest Risk": item[
                        "biggest_risk"
                    ]
                }
            )


        st.dataframe(
            table_rows,
            use_container_width=True,
            hide_index=True
        )


        for item in evaluations:

            with st.expander(
                f"{item['idea_name']} — "
                f"{item['total_score']}/100"
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

**Speed to Revenue:** {item["speed_to_revenue"]}/10

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


    if operator[
        "winner_exists"
    ]:

        st.success(
            f"WINNER: "
            f"{operator['winner_idea_id']}\n\n"
            f"{operator['winner_reason']}"
        )

    else:

        st.warning(
            "NO WINNER\n\n"
            + operator[
                "winner_reason"
            ]
        )


# =========================================================
# FINAL
# =========================================================

if st.session_state.objection:

    st.divider()

    st.header(
        "🔪 FINAL OBJECTION"
    )

    st.markdown(
        st.session_state.objection
    )


if st.session_state.verdict:

    st.divider()

    st.header(
        "🏛️ FINAL VERDICT"
    )

    st.markdown(
        st.session_state.verdict
    )


    report = build_final_report(
        st.session_state.ideas or [],
        st.session_state.blocked or [],
        st.session_state.research or {},
        st.session_state.killer or {},
        st.session_state.rebuttal or {},
        st.session_state.killer_final or {},
        st.session_state.operator or {},
        st.session_state.objection or "",
        st.session_state.verdict or ""
    )


    st.divider()

    st.header(
        "📥 التقرير"
    )


    st.download_button(
        "📝 تحميل التقرير TXT",
        data=report.encode(
            "utf-8"
        ),
        file_name=(
            "MD_Investment_Research.txt"
        ),
        mime="text/plain",
        use_container_width=True
    )


    try:

        pdf = create_pdf(
            report
        )


        st.download_button(
            "📄 تحميل التقرير PDF",
            data=pdf,
            file_name=(
                "MD_Investment_Research.pdf"
            ),
            mime="application/pdf",
            use_container_width=True
        )


    except Exception as e:

        with st.expander(
            "🔧 مشكلة PDF"
        ):

            st.code(
                str(e)
            )


# =========================================================
# RESET
# =========================================================

if (
    st.session_state.ideas
    or st.session_state.error
    or st.session_state.verdict
):

    st.divider()


    if st.button(
        "🗑️ مسح كل شيء وبدء بحث جديد"
    ):

        for key, value in (
            DEFAULT_STATE.items()
        ):

            st.session_state[
                key
            ] = value


        st.rerun()
