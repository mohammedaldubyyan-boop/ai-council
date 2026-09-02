import os, re, json, glob, time, math, difflib
from urllib.parse import urlparse

import streamlit as st
from groq import Groq
from ddgs import DDGS
from fpdf import FPDF

BUILD_ID = "V6-DDGS-RESEARCH-COUNCIL"

st.set_page_config(
    page_title="MD Investment Council",
    page_icon="🧠",
    layout="wide"
)

st.markdown("""
<style>
.stApp{direction:rtl}
h1,h2,h3,h4,h5,p{text-align:right}
div[data-testid="stMarkdownContainer"]{direction:rtl;text-align:right}
textarea,div[data-baseweb="textarea"] textarea{
    direction:rtl!important;
    text-align:right!important
}
</style>
""", unsafe_allow_html=True)


# =========================================================
# GROQ
# =========================================================

client = Groq(
    api_key=st.secrets["GROQ_API_KEY"],
    timeout=120.0
)

HUNTER_MODEL = "openai/gpt-oss-120b"
KILLER_MODEL = "qwen/qwen3.8-27b"
OPERATOR_MODEL = "openai/gpt-oss-20b"

MODEL_LAST_USED = {}

MIN_MODEL_GAP = {
    HUNTER_MODEL: 25,
    KILLER_MODEL: 25,
    OPERATOR_MODEL: 25
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
        "تدقيق الشحن"
    ],

    "invoice_collection": [
        "invoice collection",
        "collect invoices",
        "invoice chasing",
        "تحصيل الفواتير"
    ],

    "rfq_comparison": [
        "rfq comparison",
        "compare quotations",
        "مقارنة عروض الأسعار"
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
        "إدارة الضمان"
    ],

    "seo_content": [
        "seo content refresh",
        "content refresh",
        "تحديث المحتوى"
    ],

    "saas_monitoring": [
        "subscription monitoring",
        "saas monitoring",
        "مراقبة الاشتراكات"
    ],

    "scope_creep": [
        "scope creep",
        "تجاوز نطاق المشروع",
        "تغير نطاق المشروع"
    ],
}


# =========================================================
# HELPERS
# =========================================================

def norm(text):
    text = (text or "").lower()

    text = re.sub(
        r"[^a-z0-9\u0600-\u06ff\s]",
        " ",
        text
    )

    return re.sub(
        r"\s+",
        " ",
        text
    ).strip()


def cut(text, n):

    text = str(
        text or ""
    ).strip()

    if len(text) <= n:
        return text

    return (
        text[:n]
        + "\n[تم اختصار الباقي لتقليل الـtokens]"
    )


def retry_seconds(msg):

    match = re.search(
        r"try again in\s+([0-9.]+)s",
        msg,
        re.I
    )

    if match:

        return max(
            3,
            math.ceil(
                float(
                    match.group(1)
                )
            ) + 3
        )


    match = re.search(
        r"try again in\s+([0-9.]+)ms",
        msg,
        re.I
    )

    if match:

        return max(
            2,
            math.ceil(
                float(
                    match.group(1)
                ) / 1000
            ) + 2
        )


    return 12


def wait_model(
    model,
    stage,
    status
):

    last = MODEL_LAST_USED.get(
        model
    )

    if not last:
        return


    remaining = (
        MIN_MODEL_GAP.get(
            model,
            10
        )
        - (
            time.time()
            - last
        )
    )


    if remaining > 0:

        seconds = math.ceil(
            remaining
        )

        status.warning(
            f"⏳ {stage}: "
            f"انتظار {seconds} ثانية "
            f"لحماية الحد المجاني لـGroq..."
        )

        time.sleep(
            seconds
        )


# =========================================================
# GROQ JSON
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
    retries=4
):

    wait_model(
        model,
        stage,
        status
    )

    errors = []


    for _ in range(
        retries
    ):

        try:

            kwargs = {

                "model": model,

                "messages": [
                    {
                        "role": "system",
                        "content": system
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],

                "temperature": 0.3,

                "max_completion_tokens":
                    max_tokens,

                "response_format": {

                    "type":
                        "json_schema",

                    "json_schema": {

                        "name":
                            schema_name,

                        "strict":
                            True,

                        "schema":
                            schema
                    }
                }
            }


            if reasoning:

                kwargs[
                    "reasoning_effort"
                ] = reasoning

                kwargs[
                    "reasoning_format"
                ] = "hidden"


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

                    "data":
                        json.loads(
                            content
                        ),

                    "error":
                        None
                }


            errors.append(
                f"{model}: empty JSON response"
            )


        except Exception as e:

            msg = str(e)

            errors.append(
                msg
            )


            if (
                "429" in msg
                or
                "rate_limit"
                in msg.lower()
            ):

                seconds = (
                    retry_seconds(
                        msg
                    )
                )

                status.warning(
                    f"⏳ {stage}: "
                    f"انتظار "
                    f"{seconds} ثانية..."
                )

                time.sleep(
                    seconds
                )

                continue


            if (
                "413" in msg
                or
                "request_too_large"
                in msg.lower()
            ):

                break


            time.sleep(2)


    return {

        "ok": False,

        "data": None,

        "error":
            "\n\n".join(
                errors
            )
    }


# =========================================================
# GROQ TEXT
# =========================================================

def call_text(
    model,
    system,
    prompt,
    max_tokens,
    stage,
    status,
    reasoning="none",
    retries=4
):

    wait_model(
        model,
        stage,
        status
    )

    errors = []


    for _ in range(
        retries
    ):

        try:

            kwargs = {

                "model": model,

                "messages": [
                    {
                        "role": "system",
                        "content": system
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],

                "temperature": 0.3,

                "max_completion_tokens":
                    max_tokens
            }


            if reasoning:

                kwargs[
                    "reasoning_effort"
                ] = reasoning

                kwargs[
                    "reasoning_format"
                ] = "hidden"


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

                    "ok":
                        True,

                    "text":
                        content.strip(),

                    "error":
                        None
                }


            errors.append(
                f"{model}: empty response"
            )


        except Exception as e:

            msg = str(e)

            errors.append(
                msg
            )


            if (
                "429" in msg
                or
                "rate_limit"
                in msg.lower()
            ):

                seconds = (
                    retry_seconds(
                        msg
                    )
                )

                status.warning(
                    f"⏳ {stage}: "
                    f"انتظار "
                    f"{seconds} ثانية..."
                )

                time.sleep(
                    seconds
                )

                continue


            time.sleep(2)


    return {

        "ok":
            False,

        "text":
            "",

        "error":
            "\n\n".join(
                errors
            )
    }


# =========================================================
# PREPARE USER BRIEF
# =========================================================

def prepare_case_brief(
    text
):

    text = text.strip()


    markers = [

        "\n# الوكلاء",

        "\n## الوكيل الأول",

        "\n# قواعد المناظرة",

        "\n# نظام التقييم",

        "\n# شرط النجاح",

        "\n# اختبار الحقيقة",

        "\n# المرحلة النهائية"
    ]


    cuts = [

        text.find(
            marker
        )

        for marker
        in markers

        if text.find(
            marker
        ) != -1
    ]


    if cuts:

        text = text[
            :min(cuts)
        ]


    if len(text) > 13000:

        text = (
            text[:10000]
            + "\n[...اختصار محلي...]\n"
            + text[-2500:]
        )


    return text


# =========================================================
# BLACKLIST
# =========================================================

def blacklist_check(
    idea
):

    combined = norm(
        " ".join(
            [
                idea.get(
                    "name",
                    ""
                ),

                idea.get(
                    "one_liner",
                    ""
                ),

                idea.get(
                    "product",
                    ""
                ),

                idea.get(
                    "problem",
                    ""
                )
            ]
        )
    )


    short = norm(
        f"{idea.get('name','')} "
        f"{idea.get('one_liner','')}"
    )


    for rejected in (
        REJECTED_IDEAS
    ):

        rejected_norm = norm(
            rejected
        )


        if (
            rejected_norm

            and
            (
                rejected_norm
                in combined

                or
                difflib
                .SequenceMatcher(
                    None,
                    short,
                    rejected_norm
                )
                .ratio()
                >= 0.72
            )
        ):

            return (
                True,
                f"تشابه مع فكرة مرفوضة: "
                f"{rejected}"
            )


    for (
        concept,
        phrases
    ) in (
        REJECTED_CONCEPTS
        .items()
    ):

        hits = [

            phrase

            for phrase
            in phrases

            if norm(
                phrase
            )

            and
            norm(
                phrase
            )
            in combined
        ]


        if hits:

            return (
                True,

                f"تشابه مفاهيمي "
                f"مع ({concept}): "
                + ", ".join(
                    hits
                )
            )


    return (
        False,
        ""
    )


def filter_ideas(
    data
):

    passed = []

    blocked = []


    for idea in data[
        "ideas"
    ]:

        local_block, reason = (
            blacklist_check(
                idea
            )
        )


        if (
            local_block

            or
            idea.get(
                "similar_to_rejected"
            )
        ):

            blocked.append(
                {
                    "idea":
                        idea,

                    "reason":
                        reason
                        or
                        idea.get(
                            "rejected_similarity_explanation",
                            "قريبة من فكرة مرفوضة"
                        )
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
# JSON SCHEMA HELPERS
# =========================================================

def obj(
    props
):

    return {

        "type":
            "object",

        "properties":
            props,

        "required":
            list(
                props.keys()
            ),

        "additionalProperties":
            False
    }


def arr(
    items,
    min_items=None,
    max_items=None
):

    result = {

        "type":
            "array",

        "items":
            items
    }


    if (
        min_items
        is not None
    ):

        result[
            "minItems"
        ] = min_items


    if (
        max_items
        is not None
    ):

        result[
            "maxItems"
        ] = max_items


    return result


# =========================================================
# HUNTER SCHEMA
# =========================================================

idea_schema = obj(
    {

        "id":
            {
                "type":
                    "string"
            },

        "name":
            {
                "type":
                    "string"
            },

        "one_liner":
            {
                "type":
                    "string"
            },

        "buyer":
            {
                "type":
                    "string"
            },

        "problem":
            {
                "type":
                    "string"
            },

        "product":
            {
                "type":
                    "string"
            },

        "why_pay":
            {
                "type":
                    "string"
            },

        "price":
            {
                "type":
                    "string"
            },

        "current_alternative":
            {
                "type":
                    "string"
            },

        "distribution":
            {
                "type":
                    "string"
            },

        "first_10_customers":
            {
                "type":
                    "string"
            },

        "automation":
            {
                "type":
                    "integer",

                "minimum":
                    0,

                "maximum":
                    100
            },

        "human_work":
            {
                "type":
                    "string"
            },

        "why_now":
            {
                "type":
                    "string"
            },

        "similar_to_rejected":
            {
                "type":
                    "boolean"
            },

        "rejected_similarity_explanation":
            {
                "type":
                    "string"
            }
    }
)


HUNTER_SCHEMA = obj(
    {
        "ideas":
            arr(
                idea_schema,
                3,
                3
            )
    }
)


# =========================================================
# KILLER SCHEMA
# =========================================================

killer_review_schema = obj(
    {

        "idea_id":
            {
                "type":
                    "string"
            },

        "top_failure_reason_1":
            {
                "type":
                    "string"
            },

        "top_failure_reason_2":
            {
                "type":
                    "string"
            },

        "top_failure_reason_3":
            {
                "type":
                    "string"
            },

        "kill_shot":
            {
                "type":
                    "string"
            },

        "immediate_rejection_evidence":
            {
                "type":
                    "string"
            },

        "research_supports_idea":
            {
                "type":
                    "string"
            },

        "research_hurts_idea":
            {
                "type":
                    "string"
            },

        "score_out_of_10":
            {
                "type":
                    "integer",

                "minimum":
                    0,

                "maximum":
                    10
            },

        "decision":
            {
                "type":
                    "string",

                "enum":
                    [
                        "SURVIVES",
                        "KILL IT"
                    ]
            }
    }
)


KILLER_SCHEMA = obj(
    {
        "reviews":
            arr(
                killer_review_schema
            )
    }
)


# =========================================================
# REBUTTAL SCHEMA
# =========================================================

rebuttal_item_schema = obj(
    {

        "idea_id":
            {
                "type":
                    "string"
            },

        "valid_objection":
            {
                "type":
                    "string"
            },

        "disputed_objection":
            {
                "type":
                    "string"
            },

        "evidence_needed":
            {
                "type":
                    "string"
            },

        "position":
            {
                "type":
                    "string",

                "enum":
                    [
                        "DEFEND",
                        "DROP"
                    ]
            }
    }
)


REBUTTAL_SCHEMA = obj(
    {
        "responses":
            arr(
                rebuttal_item_schema
            )
    }
)


# =========================================================
# KILLER FINAL SCHEMA
# =========================================================

final_killer_item_schema = obj(
    {

        "idea_id":
            {
                "type":
                    "string"
            },

        "decision":
            {
                "type":
                    "string",

                "enum":
                    [
                        "SURVIVES",
                        "KILL IT"
                    ]
            },

        "remaining_problem":
            {
                "type":
                    "string"
            },

        "wtp_real":
            {
                "type":
                    "boolean"
            },

        "distribution_real":
            {
                "type":
                    "boolean"
            },

        "feature_or_company":
            {
                "type":
                    "string",

                "enum":
                    [
                        "FEATURE",
                        "COMPANY",
                        "UNCLEAR"
                    ]
            },

        "final_score_out_of_10":
            {
                "type":
                    "integer",

                "minimum":
                    0,

                "maximum":
                    10
            }
    }
)


FINAL_KILLER_SCHEMA = obj(
    {
        "decisions":
            arr(
                final_killer_item_schema
            )
    }
)


# =========================================================
# OPERATOR SCHEMA
# =========================================================

operator_item_schema = obj(
    {

        "idea_id":
            {
                "type":
                    "string"
            },

        "idea_name":
            {
                "type":
                    "string"
            },

        "severity":
            {
                "type":
                    "integer",

                "minimum":
                    0,

                "maximum":
                    15
            },

        "willingness_to_pay":
            {
                "type":
                    "integer",

                "minimum":
                    0,

                "maximum":
                    15
            },

        "distribution":
            {
                "type":
                    "integer",

                "minimum":
                    0,

                "maximum":
                    15
            },

        "automation":
            {
                "type":
                    "integer",

                "minimum":
                    0,

                "maximum":
                    15
            },

        "recurring":
            {
                "type":
                    "integer",

                "minimum":
                    0,

                "maximum":
                    10
            },

        "competition":
            {
                "type":
                    "integer",

                "minimum":
                    0,

                "maximum":
                    10
            },

        "moat":
            {
                "type":
                    "integer",

                "minimum":
                    0,

                "maximum":
                    5
            },

        "speed_to_revenue":
            {
                "type":
                    "integer",

                "minimum":
                    0,

                "maximum":
                    10
            },

        "stack_fit":
            {
                "type":
                    "integer",

                "minimum":
                    0,

                "maximum":
                    5
            },

        "price":
            {
                "type":
                    "string"
            },

        "gross_margin":
            {
                "type":
                    "string"
            },

        "ltv":
            {
                "type":
                    "string"
            },

        "cac":
            {
                "type":
                    "string"
            },

        "customers_for_1k_mrr":
            {
                "type":
                    "string"
            },

        "customers_for_5k_mrr":
            {
                "type":
                    "string"
            },

        "customers_for_10k_mrr":
            {
                "type":
                    "string"
            },

        "automation_percent":
            {
                "type":
                    "integer",

                "minimum":
                    0,

                "maximum":
                    100
            },

        "first_buyer":
            {
                "type":
                    "string"
            },

        "fastest_test":
            {
                "type":
                    "string"
            },

        "biggest_risk":
            {
                "type":
                    "string"
            },

        "truth_test":
            {
                "type":
                    "string"
            }
    }
)


OPERATOR_SCHEMA = obj(
    {
        "evaluations":
            arr(
                operator_item_schema
            )
    }
)


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
- willingness-to-pay.
- distribution.
- automation.
- recurring usage.
- speed to revenue.

لا تعتبر AI ميزة بحد ذاته.

ممنوع:
- AI wrapper بلا قيمة مستقلة.
- dashboard عام.
- أداة يستطيع ChatGPT تنفيذها بما يكفي.
- إعادة تغليف فكرة مرفوضة.

أخرج 3 أفكار فقط مختلفة اقتصادياً.

إذا كانت قريبة من فكرة مرفوضة:
similar_to_rejected=true.

لا تختر WINNER.
لا تعامل الادعاءات غير المثبتة كحقائق.
"""


    prompt = f"""
حالة المستخدم:

{case_brief}


القائمة المرفوضة:

{json.dumps(
    REJECTED_IDEAS,
    ensure_ascii=False
)}
"""


    return call_json(

        HUNTER_MODEL,

        system,

        prompt,

        "hunter_ideas",

        HUNTER_SCHEMA,

        1500,

        "THE HUNTER",

        status,

        "low"
    )


# =========================================================
# DDGS WEB SEARCH
# =========================================================

def search_item(
    raw
):

    if not isinstance(
        raw,
        dict
    ):

        return None


    url = (
        raw.get("href")
        or
        raw.get("url")
        or
        ""
    )


    if not url:
        return None


    return {

        "title":
            str(
                raw.get(
                    "title"
                )
                or
                ""
            ).strip(),

        "url":
            str(
                url
            ).strip(),

        "snippet":
            cut(
                str(
                    raw.get(
                        "body"
                    )
                    or
                    raw.get(
                        "snippet"
                    )
                    or
                    raw.get(
                        "content"
                    )
                    or
                    ""
                ).strip(),

                500
            )
    }


def domain(
    url
):

    try:

        return (
            urlparse(
                url
            )
            .netloc
            .lower()
            .replace(
                "www.",
                ""
            )
        )

    except Exception:

        return ""


def queries_for(
    idea
):

    return [

        f'"{idea["name"]}" '
        f'competitors pricing alternatives',

        f'{cut(idea["buyer"],180)} '
        f'{cut(idea["problem"],220)} '
        f'software service pricing',

        f'{cut(idea["product"],220)} '
        f'regulation compliance '
        f'market demand competitors'
    ]


def research_idea(
    idea,
    status
):

    errors = []

    found = []


    try:

        ddgs = DDGS(
            timeout=12
        )

    except Exception as e:

        return {

            "ok":
                False,

            "text":
                "",

            "sources":
                [],

            "error":
                f"DDGS init failed: {e}"
        }


    queries = queries_for(
        idea
    )


    for index, query in enumerate(
        queries,
        1
    ):

        status.info(
            f"🔎 {idea['name']}: "
            f"بحث "
            f"{index}/{len(queries)}"
        )


        try:

            results = (
                ddgs.text(
                    query,
                    max_results=4
                )
                or
                []
            )


            for raw in results:

                item = (
                    search_item(
                        raw
                    )
                )


                if item:

                    found.append(
                        item
                    )


        except Exception as e:

            errors.append(
                f"Search failed "
                f"[{query}]: {e}"
            )


        time.sleep(1)


    unique = []

    seen = set()

    domains = {}


    for item in found:

        if item[
            "url"
        ] in seen:

            continue


        dm = domain(
            item[
                "url"
            ]
        )


        if (
            dm
            and
            domains.get(
                dm,
                0
            ) >= 2
        ):

            continue


        seen.add(
            item[
                "url"
            ]
        )


        domains[
            dm
        ] = (
            domains.get(
                dm,
                0
            )
            + 1
        )


        unique.append(
            item
        )


        if len(
            unique
        ) >= 6:

            break


    # اقرأ أفضل مصدرين فقط
    for index, source in enumerate(
        unique[:2],
        1
    ):

        status.info(
            f"📄 قراءة مصدر "
            f"{index}/2: "
            f"{idea['name']}"
        )


        try:

            extracted = (
                ddgs.extract(
                    source[
                        "url"
                    ],
                    fmt="text_plain"
                )
            )


            content = (

                extracted.get(
                    "content",
                    ""
                )

                if isinstance(
                    extracted,
                    dict
                )

                else
                    ""
            )


            source[
                "page_excerpt"
            ] = cut(
                str(
                    content
                ).strip(),
                1200
            )


        except Exception as e:

            source[
                "page_excerpt"
            ] = ""


            errors.append(
                f"Extract failed "
                f"{source['url']}: "
                f"{e}"
            )


        time.sleep(0.7)


    if len(
        unique
    ) < 2:

        return {

            "ok":
                False,

            "text":
                "",

            "sources":
                unique,

            "error":
                (
                    f"مصادر غير كافية "
                    f"({len(unique)})\n"
                    +
                    "\n".join(
                        errors[-6:]
                    )
                )
        }


    packets = []


    for number, source in enumerate(
        unique,
        1
    ):

        packets.append(
            f"""
SOURCE {number}

Title:
{source["title"]}

URL:
{source["url"]}

Search snippet:
{source["snippet"]}

Page excerpt:
{source.get("page_excerpt","")}
"""
        )


    return {

        "ok":
            True,

        "text":
            cut(
                "\n\n---\n\n".join(
                    packets
                ),
                4800
            ),

        "sources":
            unique,

        "error":
            "\n".join(
                errors[-6:]
            )
    }


def research_all(
    ideas,
    status
):

    research = {}

    failures = []


    for index, idea in enumerate(
        ideas,
        1
    ):

        status.info(
            f"🌐 Web Research "
            f"{index}/{len(ideas)}: "
            f"{idea['name']}"
        )


        result = (
            research_idea(
                idea,
                status
            )
        )


        research[
            idea[
                "id"
            ]
        ] = result


        if not result[
            "ok"
        ]:

            failures.append(
                f"{idea['id']} — "
                f"{idea['name']}: "
                f"{result['error']}"
            )


        time.sleep(1)


    return {

        "ok":
            not failures,

        "research":
            research,

        "error":
            (
                "\n\n".join(
                    failures
                )

                if failures

                else
                    None
            )
    }


# =========================================================
# KILLER
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

هاجم:
- Competition
- WTP
- CAC
- churn
- Distribution
- Liability
- Regulation
- Platform/API risk
- Security/privacy
- Support
- Moat
- ChatGPT substitution
- Feature vs Company

استخدم Web Research كأدلة.

لا تعامل ادعاء Hunter كحقيقة
إذا لم تدعمه المصادر.

UNKNOWN ليس مثبتاً.

ممنوع المجاملة.
"""


    payload = [

        {

            "idea":
                idea,

            "web_research":
                cut(
                    research[
                        idea[
                            "id"
                        ]
                    ][
                        "text"
                    ],
                    4300
                )
        }

        for idea
        in ideas
    ]


    return call_json(

        KILLER_MODEL,

        system,

        json.dumps(
            payload,
            ensure_ascii=False
        ),

        "killer_reviews",

        KILLER_SCHEMA,

        1400,

        "THE KILLER",

        status,

        "none"
    )


# =========================================================
# HUNTER REBUTTAL
# =========================================================

def run_rebuttal(
    ideas,
    killer,
    research,
    status
):

    system = """
أنت THE HUNTER.

هذه فرصتك الوحيدة للرد.

لا تضف أفكاراً جديدة.

اعترف بالاعتراض الصحيح.

DROP أفضل من ترقيع فكرة ميتة.
"""


    payload = {

        "ideas":
            ideas,

        "killer":
            killer,

        "research": {

            idea[
                "id"
            ]:
                cut(
                    research[
                        idea[
                            "id"
                        ]
                    ][
                        "text"
                    ],
                    2500
                )

            for idea
            in ideas
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

        800,

        "HUNTER REBUTTAL",

        status,

        "low"
    )


# =========================================================
# KILLER FINAL
# =========================================================

def run_killer_final(
    ideas,
    killer,
    rebuttal,
    research,
    status
):

    system = """
أنت THE KILLER.

هذه آخر فرصة للهجوم.

لا تولد أفكاراً جديدة.

إذا WTP أو Distribution غير مثبتة،
فلا تعاملها كحقائق.
"""


    payload = {

        "ideas":
            ideas,

        "first_attack":
            killer,

        "hunter_rebuttal":
            rebuttal,

        "research": {

            idea[
                "id"
            ]:
                cut(
                    research[
                        idea[
                            "id"
                        ]
                    ][
                        "text"
                    ],
                    1800
                )

            for idea
            in ideas
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

        900,

        "KILLER FINAL",

        status,

        "none"
    )


# =========================================================
# OPERATOR
# =========================================================

def recalc_operator(
    data
):

    evaluations = data.get(
        "evaluations",
        []
    )


    for item in evaluations:

        item[
            "total_score"
        ] = (

            item[
                "severity"
            ]

            +
            item[
                "willingness_to_pay"
            ]

            +
            item[
                "distribution"
            ]

            +
            item[
                "automation"
            ]

            +
            item[
                "recurring"
            ]

            +
            item[
                "competition"
            ]

            +
            item[
                "moat"
            ]

            +
            item[
                "speed_to_revenue"
            ]

            +
            item[
                "stack_fit"
            ]
        )


    evaluations.sort(
        key=lambda x:
            x[
                "total_score"
            ],
        reverse=True
    )


    winner = (

        evaluations[0]

        if (
            evaluations
            and
            evaluations[0][
                "total_score"
            ] > 85
        )

        else
            None
    )


    highest = (

        evaluations[0][
            "total_score"
        ]

        if evaluations

        else
            None
    )


    return {

        "evaluations":
            evaluations,

        "winner_exists":
            bool(
                winner
            ),

        "winner_idea_id":
            (
                winner[
                    "idea_id"
                ]

                if winner

                else
                    ""
            ),

        "winner_reason":
            (
                f"أعلى نتيجة بعد "
                f"إعادة الحساب محلياً: "
                f"{winner['total_score']}/100"

                if winner

                else
                (
                    f"لا توجد فكرة "
                    f"تجاوزت 85/100. "
                    f"أعلى نتيجة: "
                    f"{highest}/100"

                    if highest
                    is not None

                    else
                        "لا توجد فكرة "
                        "ناجية للتقييم."
                )
            )
    }


def run_operator(
    survivors,
    research,
    killer_final,
    status
):

    system = """
أنت THE OPERATOR / ECONOMIST.

أنت CTO + CFO + Growth Operator.

قيّم فقط الناجين.

النقاط:

Severity = 15
WTP = 15
Distribution = 15
Automation = 15
Recurring = 10
Competition = 10
Moat = 5
Speed = 10
Stack Fit = 5

لا ترفع الدرجة للوصول إلى 85.

لا تخترع أرقاماً.

CAC/LTV غير المثبتة:
تقديرات فقط.

ركز على أول عميل يدفع.
"""


    payload = {

        "surviving_ideas":
            survivors,

        "research": {

            idea[
                "id"
            ]:
                cut(
                    research[
                        idea[
                            "id"
                        ]
                    ][
                        "text"
                    ],
                    2300
                )

            for idea
            in survivors
        },

        "killer_final":
            killer_final
    }


    result = call_json(

        OPERATOR_MODEL,

        system,

        json.dumps(
            payload,
            ensure_ascii=False
        ),

        "operator_evaluation",

        OPERATOR_SCHEMA,

        1500,

        "THE OPERATOR",

        status,

        "low"
    )


    if result[
        "ok"
    ]:

        result[
            "data"
        ] = (
            recalc_operator(
                result[
                    "data"
                ]
            )
        )


    return result


# =========================================================
# FINAL OBJECTION
# =========================================================

def final_objection(
    operator,
    survivors,
    research,
    status
):

    if not operator[
        "winner_exists"
    ]:

        return {

            "ok":
                True,

            "text":
                (
                    "## FINAL OBJECTION\n\n"
                    "لا يوجد Winner فوق 85/100؛ "
                    "إجبار النظام على الاختيار "
                    "يخالف قواعد التقييم."
                )
        }


    winner_id = operator[
        "winner_idea_id"
    ]


    winner = next(

        (
            idea

            for idea
            in survivors

            if idea[
                "id"
            ] == winner_id
        ),

        None
    )


    system = """
أنت THE KILLER.

هذه آخر فرصة لقتل الـWinner.

اكتب أقوى حجة واحدة
يمكن أن تجعله ينتهي عند $0 MRR.

استخدم البحث.

لا تقترح فكرة جديدة.
"""


    prompt = json.dumps(

        {

            "winner":
                winner,

            "operator":
                operator,

            "research":
                cut(
                    research[
                        winner_id
                    ][
                        "text"
                    ],
                    3200
                )
        },

        ensure_ascii=False
    )


    return call_text(

        KILLER_MODEL,

        system,

        prompt,

        400,

        "FINAL OBJECTION",

        status,

        "none"
    )


# =========================================================
# FINAL VERDICT
# =========================================================

def final_verdict(
    operator,
    objection,
    survivors,
    status
):

    if not operator[
        "winner_exists"
    ]:

        return {

            "ok":
                True,

            "text":
                (
                    "## FINAL VERDICT\n\n"
                    "**KILL**\n\n"
                    f"{operator['winner_reason']}\n\n"
                    "لا نبني مشروعاً فقط لأننا "
                    "نريد الخروج بفكرة."
                )
        }


    system = """
أنت THE OPERATOR.

اتخذ القرار النهائي:

BUILD
أو
KILL

لا تغير الدرجات عشوائياً.

راجع FINAL OBJECTION بجدية.

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

            "operator":
                operator,

            "final_objection":
                objection,

            "surviving_ideas":
                survivors
        },

        ensure_ascii=False
    )


    return call_text(

        OPERATOR_MODEL,

        system,

        prompt,

        550,

        "FINAL VERDICT",

        status,

        "low"
    )


# =========================================================
# DISPLAY IDEA
# =========================================================

def render_idea(
    idea
):

    st.subheader(
        idea[
            "name"
        ]
    )


    st.write(
        idea[
            "one_liner"
        ]
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
# REPORT
# =========================================================

def build_report(
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

    output = [

        "MD INVESTMENT RESEARCH COUNCIL",

        "\nIDEAS"
    ]


    for idea in ideas:

        output += [

            f"\n{idea['name']}",

            f"Buyer: "
            f"{idea['buyer']}",

            f"Problem: "
            f"{idea['problem']}",

            f"Product: "
            f"{idea['product']}",

            f"Price: "
            f"{idea['price']}",

            f"Distribution: "
            f"{idea['distribution']}",

            f"Automation: "
            f"{idea['automation']}%"
        ]


    if blocked:

        output.append(
            "\nBLOCKED IDEAS"
        )


        for item in blocked:

            output += [

                item[
                    "idea"
                ][
                    "name"
                ],

                f"Reason: "
                f"{item['reason']}"
            ]


    output.append(
        "\nWEB RESEARCH"
    )


    for idea in ideas:

        result = research.get(
            idea[
                "id"
            ],
            {}
        )


        output += [

            f"\n{idea['name']}",

            result.get(
                "text",
                ""
            ),

            "SOURCES:"
        ]


        for source in result.get(
            "sources",
            []
        ):

            output.append(
                f"- "
                f"{source.get('title','')}: "
                f"{source.get('url','')}"
            )


    output += [

        "\nKILLER FIRST ATTACK",

        json.dumps(
            killer,
            ensure_ascii=False,
            indent=2
        ),

        "\nHUNTER REBUTTAL",

        json.dumps(
            rebuttal,
            ensure_ascii=False,
            indent=2
        ),

        "\nKILLER FINAL",

        json.dumps(
            killer_final,
            ensure_ascii=False,
            indent=2
        ),

        "\nOPERATOR",

        json.dumps(
            operator,
            ensure_ascii=False,
            indent=2
        ),

        "\nFINAL OBJECTION",

        objection,

        "\nFINAL VERDICT",

        verdict
    ]


    return "\n".join(
        output
    )


# =========================================================
# PDF
# =========================================================

def find_font():

    paths = [

        "/usr/share/fonts/truetype/noto/NotoNaskhArabic-Regular.ttf",

        "/usr/share/fonts/truetype/noto/NotoSansArabic-Regular.ttf",

        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
    ]


    for path in paths:

        if os.path.exists(
            path
        ):

            return path


    for path in glob.glob(
        "/usr/share/fonts/**/*.ttf",
        recursive=True
    ):

        if any(
            word
            in path.lower()

            for word in
            [
                "naskh",
                "arabic",
                "dejavusans"
            ]
        ):

            return path


    return None


def create_pdf(
    report
):

    font = find_font()


    if not font:

        raise RuntimeError(
            "Arabic font not found"
        )


    pdf = FPDF(
        "P",
        "mm",
        "A4"
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


    text = re.sub(
        r"#{1,6}\s*",
        "",
        report
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
        "🎯",
        "🌐",
        "📋",
        "📊"
    ]


    for emoji in emojis:

        text = text.replace(
            emoji,
            ""
        )


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

            new_y="NEXT"
        )


    return bytes(
        pdf.output()
    )


# =========================================================
# SESSION STATE
# =========================================================

DEFAULT = {

    "original":
        "",

    "ideas":
        None,

    "blocked":
        None,

    "research":
        None,

    "killer":
        None,

    "rebuttal":
        None,

    "killer_final":
        None,

    "operator":
        None,

    "objection":
        None,

    "verdict":
        None,

    "error":
        None
}


for key, value in DEFAULT.items():

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


st.info(
    """
**V6**

Groq يستخدم فقط للوكلاء:

Hunter / Killer / Operator.

البحث الحي أصبح عبر DDGS مباشرة.

لا يوجد `groq/compound`.
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

        for (
            key,
            value
        ) in DEFAULT.items():

            st.session_state[
                key
            ] = value


        st.session_state.original = (
            original
        )


        status = st.empty()


        case = prepare_case_brief(
            original
        )


        # HUNTER
        status.info(
            "🎯 THE HUNTER يولد 3 أفكار..."
        )


        result = run_hunter(
            case,
            status
        )


        if not result[
            "ok"
        ]:

            st.session_state.error = (
                result[
                    "error"
                ]
            )


        else:

            ideas, blocked = (
                filter_ideas(
                    result[
                        "data"
                    ]
                )
            )


            st.session_state.ideas = (
                ideas
            )


            st.session_state.blocked = (
                blocked
            )


            if not ideas:

                st.session_state.verdict = (
                    "## FINAL VERDICT\n\n"
                    "**KILL**\n\n"
                    "كل الأفكار تشبه "
                    "أفكاراً مرفوضة مسبقاً."
                )


                status.error(
                    "❌ لا توجد فكرة بعد الفلتر."
                )


            else:

                # WEB RESEARCH
                status.info(
                    "🌐 Web Research عبر DDGS..."
                )


                research_result = (
                    research_all(
                        ideas,
                        status
                    )
                )


                st.session_state.research = (
                    research_result[
                        "research"
                    ]
                )


                if not research_result[
                    "ok"
                ]:

                    st.session_state.error = (
                        "فشل Web Research "
                        "عبر DDGS.\n\n"
                        +
                        research_result[
                            "error"
                        ]
                    )


                else:

                    research = (
                        research_result[
                            "research"
                        ]
                    )


                    # KILLER
                    status.info(
                        "🔪 THE KILLER يراجع "
                        "الأفكار والأدلة..."
                    )


                    result = run_killer(
                        ideas,
                        research,
                        status
                    )


                    if not result[
                        "ok"
                    ]:

                        st.session_state.error = (
                            result[
                                "error"
                            ]
                        )


                    else:

                        killer = result[
                            "data"
                        ]


                        st.session_state.killer = (
                            killer
                        )


                        # REBUTTAL
                        status.info(
                            "🎯 Hunter يرد مرة واحدة..."
                        )


                        result = run_rebuttal(
                            ideas,
                            killer,
                            research,
                            status
                        )


                        if not result[
                            "ok"
                        ]:

                            st.session_state.error = (
                                result[
                                    "error"
                                ]
                            )


                        else:

                            rebuttal = result[
                                "data"
                            ]


                            st.session_state.rebuttal = (
                                rebuttal
                            )


                            # KILLER FINAL
                            status.info(
                                "🔪 Killer يصدر "
                                "الحكم الأخير..."
                            )


                            result = run_killer_final(

                                ideas,

                                killer,

                                rebuttal,

                                research,

                                status
                            )


                            if not result[
                                "ok"
                            ]:

                                st.session_state.error = (
                                    result[
                                        "error"
                                    ]
                                )


                            else:

                                killer_final = (
                                    result[
                                        "data"
                                    ]
                                )


                                st.session_state.killer_final = (
                                    killer_final
                                )


                                survivor_ids = [

                                    item[
                                        "idea_id"
                                    ]

                                    for item in
                                    killer_final[
                                        "decisions"
                                    ]

                                    if item[
                                        "decision"
                                    ] == "SURVIVES"
                                ]


                                survivors = [

                                    idea

                                    for idea in ideas

                                    if idea[
                                        "id"
                                    ] in survivor_ids
                                ]


                                if not survivors:

                                    st.session_state.operator = {

                                        "evaluations":
                                            [],

                                        "winner_exists":
                                            False,

                                        "winner_idea_id":
                                            "",

                                        "winner_reason":
                                            "لم تنج أي فكرة "
                                            "من Red Team."
                                    }


                                    st.session_state.objection = (
                                        "لا توجد فكرة ناجية "
                                        "يمكن الاعتراض على اختيارها."
                                    )


                                    st.session_state.verdict = (
                                        "## FINAL VERDICT\n\n"
                                        "**KILL**\n\n"
                                        "لم تنج أي فكرة من "
                                        "THE KILLER بعد البحث."
                                    )


                                    status.success(
                                        "✅ انتهى المجلس: "
                                        "NO WINNER"
                                    )


                                else:

                                    # OPERATOR
                                    status.info(
                                        "📊 THE OPERATOR "
                                        "يحسب الاقتصاديات..."
                                    )


                                    result = run_operator(

                                        survivors,

                                        research,

                                        killer_final,

                                        status
                                    )


                                    if not result[
                                        "ok"
                                    ]:

                                        st.session_state.error = (
                                            result[
                                                "error"
                                            ]
                                        )


                                    else:

                                        operator = (
                                            result[
                                                "data"
                                            ]
                                        )


                                        st.session_state.operator = (
                                            operator
                                        )


                                        # FINAL OBJECTION
                                        status.info(
                                            "🔪 FINAL OBJECTION..."
                                        )


                                        result = final_objection(

                                            operator,

                                            survivors,

                                            research,

                                            status
                                        )


                                        if not result[
                                            "ok"
                                        ]:

                                            st.session_state.error = (
                                                result[
                                                    "error"
                                                ]
                                            )


                                        else:

                                            objection = (
                                                result[
                                                    "text"
                                                ]
                                            )


                                            st.session_state.objection = (
                                                objection
                                            )


                                            # FINAL VERDICT
                                            status.info(
                                                "🏛️ FINAL VERDICT..."
                                            )


                                            result = final_verdict(

                                                operator,

                                                objection,

                                                survivors,

                                                status
                                            )


                                            if not result[
                                                "ok"
                                            ]:

                                                st.session_state.error = (
                                                    result[
                                                        "error"
                                                    ]
                                                )


                                            else:

                                                st.session_state.verdict = (
                                                    result[
                                                        "text"
                                                    ]
                                                )


                                                status.success(
                                                    "✅ انتهى "
                                                    "Research Council"
                                                )


# =========================================================
# ERROR
# =========================================================

if st.session_state.error:

    st.error(
        "حدث خطأ ولم يسمح التطبيق "
        "بإصدار حكم ناقص."
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
            f"**{item['idea']['name']}**"
            f"\n\n"
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
            idea[
                "name"
            ],
            expanded=True
        ):

            render_idea(
                idea
            )


# =========================================================
# WEB RESEARCH
# =========================================================

if st.session_state.research:

    st.divider()

    st.header(
        "🌐 Web Research — DDGS"
    )


    for idea in (
        st.session_state.ideas
        or
        []
    ):

        result = (
            st.session_state
            .research
            .get(
                idea[
                    "id"
                ],
                {}
            )
        )


        with st.expander(
            f"🔎 {idea['name']}"
        ):

            if result.get(
                "ok"
            ):

                st.code(
                    result.get(
                        "text",
                        ""
                    ),
                    language="text"
                )


            else:

                st.warning(
                    "البحث غير مكتمل."
                )


                st.code(
                    result.get(
                        "error",
                        ""
                    )
                )


            if result.get(
                "sources"
            ):

                st.markdown(
                    "### المصادر"
                )


                for source in (
                    result[
                        "sources"
                    ]
                ):

                    title = (

                        source.get(
                            "title"
                        )

                        or
                        source.get(
                            "url"
                        )
                    )


                    url = source.get(
                        "url",
                        ""
                    )


                    st.markdown(
                        f"- [{title}]({url})"
                    )


# =========================================================
# KILLER
# =========================================================

if st.session_state.killer:

    st.divider()

    st.header(
        "🔪 THE KILLER"
    )


    for item in (
        st.session_state
        .killer[
            "reviews"
        ]
    ):

        st.markdown(
            f"""
### {item["idea_id"]}

**سبب 1:** {item["top_failure_reason_1"]}

**سبب 2:** {item["top_failure_reason_2"]}

**سبب 3:** {item["top_failure_reason_3"]}

**Kill Shot:** {item["kill_shot"]}

**الدليل القاتل:** {item["immediate_rejection_evidence"]}

**الدليل المؤيد:** {item["research_supports_idea"]}

**الدليل المضاد:** {item["research_hurts_idea"]}

**Score:** {item["score_out_of_10"]}/10

**Decision:** `{item["decision"]}`
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

        table = []


        for item in evaluations:

            table.append(
                {

                    "Idea":
                        item[
                            "idea_name"
                        ],

                    "Score":
                        item[
                            "total_score"
                        ],

                    "First Buyer":
                        item[
                            "first_buyer"
                        ],

                    "Price":
                        item[
                            "price"
                        ],

                    "Automation":
                        f"{item['automation_percent']}%",

                    "Fastest Test":
                        item[
                            "fastest_test"
                        ],

                    "Biggest Risk":
                        item[
                            "biggest_risk"
                        ]
                }
            )


        st.dataframe(
            table,
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


    if operator.get(
        "winner_exists"
    ):

        st.success(
            f"WINNER: "
            f"{operator['winner_idea_id']}"
            f"\n\n"
            f"{operator['winner_reason']}"
        )


    else:

        st.warning(
            "NO WINNER"
            "\n\n"
            +
            operator.get(
                "winner_reason",
                ""
            )
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


    report = build_report(

        st.session_state.ideas
        or
        [],

        st.session_state.blocked
        or
        [],

        st.session_state.research
        or
        {},

        st.session_state.killer
        or
        {},

        st.session_state.rebuttal
        or
        {},

        st.session_state.killer_final
        or
        {},

        st.session_state.operator
        or
        {},

        st.session_state.objection
        or
        "",

        st.session_state.verdict
        or
        ""
    )


    st.divider()

    st.header(
        "📥 التقرير"
    )


    st.download_button(

        "📝 تحميل التقرير TXT",

        report.encode(
            "utf-8"
        ),

        "MD_Investment_Research_V6.txt",

        "text/plain",

        use_container_width=True
    )


    try:

        st.download_button(

            "📄 تحميل التقرير PDF",

            create_pdf(
                report
            ),

            "MD_Investment_Research_V6.pdf",

            "application/pdf",

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
    or
    st.session_state.error
    or
    st.session_state.verdict
):

    st.divider()


    if st.button(
        "🗑️ مسح كل شيء وبدء بحث جديد"
    ):

        for (
            key,
            value
        ) in DEFAULT.items():

            st.session_state[
                key
            ] = value


        st.rerun()
