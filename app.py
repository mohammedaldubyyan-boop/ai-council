import os
import re
import glob
import time
import math
import streamlit as st

from openai import OpenAI
from fpdf import FPDF


# =========================================================
# BUILD
# =========================================================

BUILD_ID = "GROQ-DEBATE-4-NO-COMPOUND"


# =========================================================
# PAGE
# =========================================================

st.set_page_config(
    page_title="MD AI Council",
    page_icon="🧠",
    layout="wide"
)


# =========================================================
# ARABIC RTL
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

client = OpenAI(
    api_key=st.secrets["GROQ_API_KEY"],
    base_url="https://api.groq.com/openai/v1",
    timeout=120.0,
    max_retries=0
)


# =========================================================
# MODELS
# =========================================================

HUNTER_MODEL = {
    "id": "openai/gpt-oss-120b",
    "reasoning": "low"
}

KILLER_MODEL = {
    "id": "qwen/qwen3.8-27b",
    "reasoning": "none"
}

OPERATOR_MODEL = {
    "id": "qwen/qwen3.6-27b",
    "reasoning": "none"
}


# =========================================================
# FREE TIER THROTTLING
#
# نفس الموديل لا نعيد استخدامه فوراً
# =========================================================

MODEL_LAST_USED = {}

MIN_SECONDS_BETWEEN_SAME_MODEL = 28


# =========================================================
# HUNTER INSTRUCTIONS
# =========================================================

HUNTER_INSTRUCTIONS = """
أنت THE HUNTER.

أنت رائد أعمال وباحث فرص اقتصادية.

الهدف ليس العثور على فكرة AI مثيرة.
الهدف العثور على مشروع يمكن أن يولد مالاً حقيقياً
بأقل اعتماد ممكن على صاحبه.

الأولوية هي:

1. مشكلة تكلف العميل مالاً.
2. willingness-to-pay واضح.
3. عملية إلزامية أو متكررة.
4. وصول واضح للمشتري.
5. أتمتة عالية.
6. سرعة الوصول لأول عملية دفع.
7. أقل مصروف قبل إثبات الدفع.
8. عدم الحاجة لفريق في البداية.
9. عدم الحاجة لجمهور كبير.
10. عدم بناء شيء لأشهر قبل اختبار الدفع.

لا تعتبر AI ميزة تنافسية بحد ذاته.

تجنب:
- AI wrappers.
- dashboards العامة.
- أدوات يمكن لـChatGPT العادي تنفيذها بما يكفي.
- micro-SaaS بلا سبب قوي للدفع.
- أفكار تحتاج مبيعات بشرية ثقيلة.
- أفكار تحتاج دعماً يومياً من صاحب المشروع.

قدم بحد أقصى 5 أفكار.

لكل فكرة يجب أن تذكر باختصار:

1. من يدفع؟
2. لماذا يدفع؟
3. كم يدفع تقريباً؟
4. ماذا يحدث اليوم بدون المنتج؟
5. البديل الحالي؟
6. لماذا نحن أفضل؟
7. كيف نصل لأول 10 عملاء؟
8. ماذا يمكن أتمتته؟
9. العمل البشري المتبقي؟
10. لماذا الآن؟

لا تختر WINNER.

لا تقم بتقييم 100 نقطة بعد.

مهم:
كن كثيفاً ومباشراً.
لا تتجاوز تقريباً 900 كلمة.
"""


# =========================================================
# KILLER INSTRUCTIONS
# =========================================================

KILLER_INSTRUCTIONS = """
أنت THE KILLER.

أنت مستثمر متشائم ومدير منافس.
هدفك منعنا من بناء المشروع الخطأ.

لا تقترح أفكاراً جديدة.

هاجم أفكار Hunter من ناحية:

- المنافسون المباشرون.
- المنتجات المجانية.
- الحلول الموجودة داخل المنصة نفسها.
- هل ChatGPT/Claude يستطيع أداء المهمة بما يكفي؟
- CAC.
- willingness-to-pay.
- churn.
- صعوبة الوصول للمشتري.
- legal/regulatory risk.
- data availability.
- platform/API dependency.
- privacy/security.
- liability.
- support burden.
- هل هي Feature وليست Company؟
- سهولة التقليد.
- ضعف moat.
- الحاجة لمبيعات بشرية.
- الحاجة لخبرة غير موجودة.
- صغر السوق.
- ضعف التكرار.

لكل فكرة أعط:

1. أقوى 3 أسباب للفشل.
2. Kill Shot واحد.
3. ما الدليل الذي لو وجدناه نرفضها فوراً؟
4. تقييم من 10 بعد الهجوم.

إذا ماتت الفكرة اكتب بوضوح:

KILL IT

ممنوع المجاملة.
"""


# =========================================================
# OPERATOR INSTRUCTIONS
# =========================================================

OPERATOR_INSTRUCTIONS = """
أنت THE OPERATOR / ECONOMIST.

أنت CTO + CFO + Growth Operator.

وظيفتك الحكم فقط على الأفكار التي نجت.

لا تولد أفكاراً جديدة.

نظام التقييم من 100:

1. Severity of Problem — 15
2. Willingness to Pay — 15
3. Distribution — 15
4. Automation — 15
5. Recurring / Repeat Usage — 10
6. Competition — 10
7. Moat / Defensibility — 5
8. Speed to Revenue — 10
9. Stack Fit — 5

المجموع = 100.

لا ترفع الدرجة بسبب:
- TAM كبير.
- AI ترند.
- سهولة البرمجة.
- وجود API.
- وجود منافسين مربحين فقط.

WINNER ممنوع إلا إذا تجاوز 85/100 فعلاً بعد Red Team.

إذا أفضل فكرة حصلت على 78:
قل 78.

إذا لا توجد فكرة تستحق 85:
قل NO WINNER.

لكل فكرة نجت قيّم:

ECONOMICS:
- Price.
- Gross margin.
- recurring revenue.
- LTV.
- CAC.
- عدد العملاء المطلوب لـ:
  $1k MRR
  $5k MRR
  $10k MRR

BUILD:
- ماذا نبني؟
- ماذا نربط؟
- ماذا لا نحتاج بناءه؟
- تكلفة التشغيل.
- هل MVP خلال أيام؟

AUTOMATION:
- نسبة الأتمتة %.
- أين يحتاج تدخل محمد؟

DISTRIBUTION:
لا تقبل إجابات عامة مثل:
SEO
social media
content
ads

يجب تحديد قناة واضحة للوصول لأول عميل يدفع.

اختبار الحقيقة لكل فكرة 80+:

1. لماذا ليست Micro-SaaS سيموت عند $0 MRR؟
2. لماذا يعطي عميل غريب بطاقته لنا؟
3. لماذا لا يستخدم البديل الموجود؟
4. لماذا يحتاجها الآن؟
5. أقصر طريق لأول payment_succeeded؟

الجدول النهائي:

| Rank | Idea | Score | First Buyer | Price | Automation | Fastest Test | Biggest Risk |

إذا يوجد WINNER فوق 85، أعط:

1. الجملة الواحدة التي تشرح المشروع.
2. العميل المحدد جداً.
3. المشكلة المالية.
4. لماذا الآن.
5. المنتج.
6. كيف يدخل المال.
7. السعر الأولي.
8. acquisition channel.
9. automation architecture.
10. دور DeepSeek.
11. دور Replit.
12. دور Stripe.
13. دور Composio إن احتجناه.
14. العمل المتبقي على محمد.
15. المنافسون المباشرون.
16. لماذا لا يقتلوننا.
17. Kill Shot.
18. اختبار 7 أيام.
19. معيار BUILD.
20. معيار KILL.

لا تستخدم لغة تحفيزية.
"""


# =========================================================
# HELPERS
# =========================================================

def extract_content(content):

    if content is None:
        return ""

    if isinstance(content, str):
        return content.strip()

    if isinstance(content, list):

        pieces = []

        for part in content:

            if isinstance(part, str):
                pieces.append(part)

            elif isinstance(part, dict):

                text = part.get("text")

                if text:
                    pieces.append(str(text))

            else:

                text = getattr(part, "text", None)

                if text:
                    pieces.append(str(text))

        return "\n".join(pieces).strip()

    return str(content).strip()


# =========================================================
# LOCAL BRIEF PREPARATION
#
# لا نستخدم AI هنا.
#
# إذا المستخدم لصق البرومبت القديم،
# نحذف قسم الوكلاء لأن التطبيق يحتوي عليه أصلاً.
# =========================================================

def prepare_case_brief(text):

    text = text.strip()

    original_length = len(text)

    markers = [
        "\n# الوكلاء",
        "\n## الوكيل الأول",
        "\n# قواعد المناظرة",
        "\n# نظام التقييم",
        "\n# شرط النجاح",
        "\n# اختبار الحقيقة",
        "\n# المرحلة النهائية"
    ]

    cut_positions = []

    for marker in markers:

        position = text.find(marker)

        if position != -1:
            cut_positions.append(position)

    removed_protocol = False

    if cut_positions:

        first_cut = min(cut_positions)

        text = text[:first_cut]

        removed_protocol = True


    # إزالة فراغات مبالغ فيها
    text = re.sub(
        r"\n{4,}",
        "\n\n\n",
        text
    )


    # حد أمان
    MAX_CHARS = 14000

    trimmed = False

    if len(text) > MAX_CHARS:

        # نحافظ على البداية والنهاية
        text = (
            text[:10500]
            + "\n\n[تم اختصار جزء من النص تلقائياً]\n\n"
            + text[-3000:]
        )

        trimmed = True


    return {
        "text": text,
        "removed_protocol": removed_protocol,
        "trimmed": trimmed,
        "original_length": original_length,
        "final_length": len(text)
    }


# =========================================================
# TRUNCATE BETWEEN AGENTS
# =========================================================

def compact_text(text, max_chars):

    if not text:
        return ""

    text = text.strip()

    if len(text) <= max_chars:
        return text

    return (
        text[:max_chars]
        + "\n\n[تم اختصار بقية الرد لتقليل استهلاك الـtokens]"
    )


# =========================================================
# RATE LIMIT PARSER
# =========================================================

def get_retry_seconds(error_text):

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

                seconds = float(
                    match.group(1)
                )

                return max(
                    3,
                    math.ceil(seconds) + 3
                )

            except:
                pass

    return 12


# =========================================================
# COOLDOWN FOR SAME MODEL
# =========================================================

def wait_for_model(
    model,
    stage_name,
    status_box
):

    previous = MODEL_LAST_USED.get(model)

    if previous is None:
        return

    elapsed = time.time() - previous

    remaining = (
        MIN_SECONDS_BETWEEN_SAME_MODEL
        - elapsed
    )

    if remaining > 0:

        wait_seconds = math.ceil(
            remaining
        )

        status_box.warning(
            f"⏳ {stage_name}: "
            f"انتظار {wait_seconds} ثانية "
            f"لحماية الحد المجاني لـGroq..."
        )

        time.sleep(
            wait_seconds
        )


# =========================================================
# MODEL CALL
# =========================================================

def call_model(
    model_config,
    instructions,
    task,
    max_tokens,
    stage_name,
    status_box,
    retries=4
):

    model = model_config["id"]

    reasoning = model_config.get(
        "reasoning"
    )


    wait_for_model(
        model,
        stage_name,
        status_box
    )


    errors = []


    for attempt in range(retries):

        try:

            full_prompt = f"""
ROLE / RULES:

{instructions}

====================================

CURRENT TASK:

{task}
"""


            extra_body = {}


            if reasoning:

                extra_body[
                    "reasoning_effort"
                ] = reasoning

                extra_body[
                    "reasoning_format"
                ] = "hidden"


            kwargs = {

                "model": model,

                # Groq ينصح بتبسيط الـprompt،
                # لذلك نجمع الدور والمهمة في رسالة واحدة.
                "messages": [
                    {
                        "role": "user",
                        "content": full_prompt
                    }
                ],

                "max_completion_tokens": max_tokens,

                "temperature": 0.5
            }


            if extra_body:

                kwargs[
                    "extra_body"
                ] = extra_body


            response = (
                client
                .chat
                .completions
                .create(**kwargs)
            )


            if not response.choices:

                errors.append(
                    f"{model}: no choices returned"
                )

                continue


            choice = response.choices[0]


            content = extract_content(
                choice.message.content
            )


            finish_reason = getattr(
                choice,
                "finish_reason",
                None
            )


            MODEL_LAST_USED[
                model
            ] = time.time()


            if content:

                return {
                    "ok": True,
                    "text": content,
                    "model": model,
                    "finish_reason": finish_reason,
                    "error": None
                }


            errors.append(
                f"{model}: empty response | "
                f"finish_reason={finish_reason}"
            )


        except Exception as e:

            error_text = str(e)

            errors.append(
                f"{model}: {error_text}"
            )


            # 429
            if (
                "429" in error_text
                or "rate_limit" in error_text.lower()
            ):

                wait_seconds = get_retry_seconds(
                    error_text
                )

                status_box.warning(
                    f"⏳ {stage_name}: "
                    f"Groq طلب انتظار "
                    f"{wait_seconds} ثانية. "
                    f"سيكمل التطبيق تلقائياً..."
                )

                time.sleep(
                    wait_seconds
                )

                continue


            # 413
            if (
                "413" in error_text
                or "request_too_large" in error_text.lower()
            ):

                return {
                    "ok": False,
                    "text": "",
                    "model": model,
                    "error": (
                        "الطلب أكبر من المسموح. "
                        "هذا يعني أن إحدى مراحل "
                        "التطبيق أرسلت سياقاً كبيراً جداً.\n\n"
                        + error_text
                    )
                }


            time.sleep(2)


    return {
        "ok": False,
        "text": "",
        "model": model,
        "error": "\n\n".join(errors)
    }


# =========================================================
# 1. HUNTER
# =========================================================

def hunter_generate(
    case_brief,
    status
):

    task = f"""
هذه معلومات الحالة والقيود الخاصة بالمستخدم:

================ CASE BRIEF ================

{case_brief}

============================================


اقترح الآن بحد أقصى 5 مشاريع فقط.

احترم جميع الأفكار المرفوضة المذكورة
في CASE BRIEF.

لا تعيد تغليف فكرة مرفوضة.

الهدف المال وليس الابتكار.

استخدم تنسيقاً كثيفاً حتى تتسع الأفكار
بدون شرح مطول.
"""


    return call_model(
        HUNTER_MODEL,
        HUNTER_INSTRUCTIONS,
        task,
        max_tokens=1200,
        stage_name="THE HUNTER",
        status_box=status
    )


# =========================================================
# 2. KILLER FIRST ATTACK
# =========================================================

def killer_attack(
    hunter_output,
    status
):

    hunter_short = compact_text(
        hunter_output,
        7000
    )


    task = f"""
THE HUNTER اقترح المشاريع التالية:

================ HUNTER =================

{hunter_short}

=========================================


هاجم كل مشروع.

لا تحتاج إعادة وصف المشروع بالكامل.

لكل مشروع:

- 3 أسباب للفشل.
- Kill Shot.
- الدليل الذي يجعلنا نرفضه فوراً.
- Score /10 بعد الهجوم.

إذا مات:
KILL IT.
"""


    return call_model(
        KILLER_MODEL,
        KILLER_INSTRUCTIONS,
        task,
        max_tokens=1050,
        stage_name="THE KILLER - الهجوم الأول",
        status_box=status
    )


# =========================================================
# 3. HUNTER REBUTTAL
# =========================================================

def hunter_rebuttal(
    hunter_output,
    killer_output,
    status
):

    hunter_short = compact_text(
        hunter_output,
        4500
    )

    killer_short = compact_text(
        killer_output,
        4500
    )


    task = f"""
رأيك السابق:

{hunter_short}


هجوم THE KILLER:

{killer_short}


لديك رد واحد فقط.

ممنوع إضافة أفكار جديدة.

لكل فكرة ما زلت تدافع عنها:

- ما اعتراض Killer الصحيح؟
- ما اعتراضه الذي ترفضه؟
- لماذا؟
- ما الدليل المطلوب؟
- هل تتمسك بالفكرة أم تتخلى عنها؟

إذا اقتنعت أنها سيئة:
تخل عنها.

لا تتجاوز 500 كلمة.
"""


    return call_model(
        HUNTER_MODEL,
        HUNTER_INSTRUCTIONS,
        task,
        max_tokens=600,
        stage_name="THE HUNTER - الرد",
        status_box=status
    )


# =========================================================
# 4. KILLER FINAL
# =========================================================

def killer_final(
    hunter_output,
    killer_output,
    hunter_rebuttal_output,
    status
):

    task = f"""
Hunter Original:

{compact_text(hunter_output, 3000)}


Killer First Attack:

{compact_text(killer_output, 3000)}


Hunter Rebuttal:

{compact_text(hunter_rebuttal_output, 3000)}


أصدر حكمك الأخير.

لكل فكرة:

- SURVIVES أو KILL IT.
- أهم مشكلة متبقية.
- هل WTP حقيقية؟
- هل Distribution واقعية؟
- Feature أم Company؟
- Final Red-Team Score /10.

لا تضف أفكاراً.
لا تتجاوز 500 كلمة.
"""


    return call_model(
        KILLER_MODEL,
        KILLER_INSTRUCTIONS,
        task,
        max_tokens=600,
        stage_name="THE KILLER - الحكم الأخير",
        status_box=status
    )


# =========================================================
# LOCAL DEBATE PACKET
#
# لا يوجد Compound ولا summarizer.
# =========================================================

def make_debate_packet(
    hunter1,
    killer1,
    hunter2,
    killer2
):

    return f"""
=== HUNTER IDEAS ===

{compact_text(hunter1, 3300)}


=== KILLER FIRST ATTACK ===

{compact_text(killer1, 3300)}


=== HUNTER REBUTTAL ===

{compact_text(hunter2, 2600)}


=== KILLER FINAL ATTACK ===

{compact_text(killer2, 2600)}
"""


# =========================================================
# 5. OPERATOR
# =========================================================

def operator_evaluate(
    debate_packet,
    status
):

    task = f"""
هذا هو محضر Red Team المضغوط محلياً:

====================================

{debate_packet}

====================================


قيّم فقط الأفكار التي نجت من Killer.

طبق نظام 100 نقطة حرفياً.

لا ترفع الدرجات.

قدم:

1. Economics.
2. Build.
3. Automation.
4. Distribution.
5. Truth Test للأفكار 80+.

ثم الجدول:

| Rank | Idea | Score | First Buyer | Price | Automation | Fastest Test | Biggest Risk |

إذا لا توجد فكرة فوق 85:
اكتب NO WINNER.

إذا يوجد WINNER فوق 85:
قدم البنود العشرين المطلوبة في تعليماتك.

لا تكتب FINAL VERDICT بعد.
"""


    return call_model(
        OPERATOR_MODEL,
        OPERATOR_INSTRUCTIONS,
        task,
        max_tokens=1600,
        stage_name="THE OPERATOR",
        status_box=status
    )


# =========================================================
# 6. FINAL OBJECTION
# =========================================================

def killer_final_objection(
    operator_output,
    status
):

    operator_short = compact_text(
        operator_output,
        7000
    )


    task = f"""
THE OPERATOR أصدر التقييم التالي:

====================================

{operator_short}

====================================


اكتب الآن فقط:

## FINAL OBJECTION

إذا يوجد WINNER:

قدم أقوى حجة ممكنة تجعل هذا المشروع
ينتهي عند $0 MRR.

لا تجامل.

إذا NO WINNER:

اشرح باختصار لماذا عدم البناء
أفضل من إجبارنا على اختيار فكرة ضعيفة.

لا تقترح أي مشروع جديد.
"""


    return call_model(
        KILLER_MODEL,
        KILLER_INSTRUCTIONS,
        task,
        max_tokens=400,
        stage_name="THE KILLER - FINAL OBJECTION",
        status_box=status
    )


# =========================================================
# 7. FINAL VERDICT
# =========================================================

def operator_final_verdict(
    operator_output,
    objection_output,
    status
):

    task = f"""
تقييمك السابق:

{compact_text(operator_output, 6500)}


FINAL OBJECTION من Killer:

{compact_text(objection_output, 2500)}


اكتب الآن فقط:

## FINAL VERDICT

ثم واحد فقط:

BUILD

أو

KILL


بعدها تفسير مختصر.

إذا BUILD:
حدد أول اختبار مدفوع خلال 7 أيام.

إذا KILL:
حدد الافتراض الذي فشل.

لا تغير الدرجات السابقة
إلا إذا اعتراض Killer كشف خطأ جوهرياً.
"""


    return call_model(
        OPERATOR_MODEL,
        OPERATOR_INSTRUCTIONS,
        task,
        max_tokens=450,
        stage_name="THE OPERATOR - FINAL VERDICT",
        status_box=status
    )


# =========================================================
# REPORT
# =========================================================

def build_report(
    original_question,
    case_brief,
    hunter1,
    killer1,
    hunter2,
    killer2,
    operator,
    objection,
    verdict
):

    return f"""
MD AI COUNCIL
INVESTMENT RED TEAM

========================================
ORIGINAL INPUT
========================================

{original_question}


========================================
CASE BRIEF USED BY THE COUNCIL
========================================

{case_brief}


========================================
1. THE HUNTER
========================================

{hunter1}


========================================
2. THE KILLER
========================================

{killer1}


========================================
3. THE HUNTER - REBUTTAL
========================================

{hunter2}


========================================
4. THE KILLER - FINAL ATTACK
========================================

{killer2}


========================================
5. THE OPERATOR
========================================

{operator}


========================================
FINAL OBJECTION
========================================

{objection}


========================================
FINAL VERDICT
========================================

{verdict}
""".strip()


# =========================================================
# PDF
# =========================================================

def clean_pdf_text(text):

    text = re.sub(
        r"#{1,6}\s*",
        "",
        text
    )

    text = text.replace("**", "")
    text = text.replace("__", "")
    text = text.replace("`", "")

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
        "🗑️"
    ]

    for emoji in emojis:

        text = text.replace(
            emoji,
            ""
        )

    return text


def find_arabic_font():

    candidates = [
        "/usr/share/fonts/truetype/noto/NotoNaskhArabic-Regular.ttf",
        "/usr/share/fonts/truetype/noto/NotoSansArabic-Regular.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
    ]

    for path in candidates:

        if os.path.exists(path):
            return path


    fonts = glob.glob(
        "/usr/share/fonts/**/*.ttf",
        recursive=True
    )


    for font in fonts:

        lower = font.lower()

        if (
            "naskh" in lower
            or "arabic" in lower
            or "dejavusans" in lower
        ):
            return font


    return None


def create_pdf(report_text):

    font_path = find_arabic_font()

    if not font_path:

        raise RuntimeError(
            "لم يتم العثور على خط عربي."
        )


    pdf = FPDF(
        orientation="P",
        unit="mm",
        format="A4"
    )


    pdf.set_margins(
        15,
        15,
        15
    )


    pdf.set_auto_page_break(
        auto=True,
        margin=15
    )


    pdf.add_page()


    pdf.add_font(
        "Arabic",
        fname=font_path
    )


    pdf.set_font(
        "Arabic",
        size=11
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


    text = clean_pdf_text(
        report_text
    )


    for paragraph in text.split("\n"):

        paragraph = paragraph.strip()

        if not paragraph:

            pdf.ln(3)
            continue


        if paragraph.startswith(
            "================================"
        ):
            continue


        pdf.multi_cell(
            0,
            7,
            paragraph,
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

DEFAULT_STATE = {
    "original_question": "",
    "case_brief": None,
    "brief_info": None,
    "hunter1": None,
    "killer1": None,
    "hunter2": None,
    "killer2": None,
    "operator": None,
    "objection": None,
    "verdict": None,
    "error": None
}


for key, value in DEFAULT_STATE.items():

    if key not in st.session_state:

        st.session_state[key] = value


# =========================================================
# UI
# =========================================================

st.title(
    "🧠 مجلس MD الاستثماري"
)


st.caption(
    f"Build: {BUILD_ID}"
)


st.info(
    """
قواعد Hunter / Killer / Operator ونظام الـ100 نقطة
أصبحت مدمجة داخل التطبيق.

إذا لصقت البرومبت القديم الطويل،
سيحذف التطبيق قسم الوكلاء والقواعد تلقائياً
ويحتفظ بالهدف والأصول والأفكار المرفوضة.
"""
)


question = st.text_area(
    "اكتب الحالة أو الصق البرومبت السابق:",
    height=350,
    value=st.session_state.original_question,
    placeholder="""
مثال:

نريد مشروعاً عالي الأتمتة وسريع الوصول للدخل.

الأصول:
- Stripe
- DeepSeek API
- Replit
- Composio

الأفكار المرفوضة:
- ...
"""
)


start = st.button(
    "🚀 ابدأ المناظرة",
    type="primary",
    use_container_width=True
)


# =========================================================
# RUN
# =========================================================

if start:

    if not question.strip():

        st.warning(
            "اكتب الحالة أولاً."
        )

    else:

        # Reset
        for key, value in DEFAULT_STATE.items():

            st.session_state[key] = value


        st.session_state.original_question = (
            question
        )


        status = st.empty()


        # =================================================
        # PREPARE LOCALLY
        # =================================================

        status.info(
            "📋 تجهيز الحالة محلياً بدون AI..."
        )


        brief_info = prepare_case_brief(
            question
        )


        case_brief = brief_info["text"]


        st.session_state.case_brief = (
            case_brief
        )

        st.session_state.brief_info = (
            brief_info
        )


        # =================================================
        # HUNTER
        # =================================================

        status.info(
            "🎯 THE HUNTER يبحث عن المشاريع..."
        )


        result = hunter_generate(
            case_brief,
            status
        )


        if not result["ok"]:

            st.session_state.error = (
                result["error"]
            )

        else:

            st.session_state.hunter1 = (
                result["text"]
            )


            # =============================================
            # KILLER
            # =============================================

            status.info(
                "🔪 THE KILLER يهاجم المشاريع..."
            )


            result = killer_attack(
                st.session_state.hunter1,
                status
            )


            if not result["ok"]:

                st.session_state.error = (
                    result["error"]
                )

            else:

                st.session_state.killer1 = (
                    result["text"]
                )


                # =========================================
                # HUNTER REBUTTAL
                # =========================================

                status.info(
                    "🎯 THE HUNTER يرد مرة واحدة..."
                )


                result = hunter_rebuttal(
                    st.session_state.hunter1,
                    st.session_state.killer1,
                    status
                )


                if not result["ok"]:

                    st.session_state.error = (
                        result["error"]
                    )

                else:

                    st.session_state.hunter2 = (
                        result["text"]
                    )


                    # =====================================
                    # KILLER FINAL
                    # =====================================

                    status.info(
                        "🔪 THE KILLER يصدر حكمه النهائي..."
                    )


                    result = killer_final(
                        st.session_state.hunter1,
                        st.session_state.killer1,
                        st.session_state.hunter2,
                        status
                    )


                    if not result["ok"]:

                        st.session_state.error = (
                            result["error"]
                        )

                    else:

                        st.session_state.killer2 = (
                            result["text"]
                        )


                        # ================================
                        # LOCAL PACKET
                        # ================================

                        packet = make_debate_packet(
                            st.session_state.hunter1,
                            st.session_state.killer1,
                            st.session_state.hunter2,
                            st.session_state.killer2
                        )


                        # ================================
                        # OPERATOR
                        # ================================

                        status.info(
                            "📊 THE OPERATOR يحسب الاقتصاديات..."
                        )


                        result = operator_evaluate(
                            packet,
                            status
                        )


                        if not result["ok"]:

                            st.session_state.error = (
                                result["error"]
                            )

                        else:

                            st.session_state.operator = (
                                result["text"]
                            )


                            # ============================
                            # FINAL OBJECTION
                            # ============================

                            status.info(
                                "🔪 FINAL OBJECTION..."
                            )


                            result = killer_final_objection(
                                st.session_state.operator,
                                status
                            )


                            if not result["ok"]:

                                st.session_state.error = (
                                    result["error"]
                                )

                            else:

                                st.session_state.objection = (
                                    result["text"]
                                )


                                # ========================
                                # VERDICT
                                # ========================

                                status.info(
                                    "🏛️ FINAL VERDICT..."
                                )


                                result = operator_final_verdict(
                                    st.session_state.operator,
                                    st.session_state.objection,
                                    status
                                )


                                if not result["ok"]:

                                    st.session_state.error = (
                                        result["error"]
                                    )

                                else:

                                    st.session_state.verdict = (
                                        result["text"]
                                    )


                                    status.success(
                                        "✅ انتهت المناظرة"
                                    )


# =========================================================
# ERROR
# =========================================================

if st.session_state.error:

    st.error(
        "حدث خطأ أثناء المناظرة."
    )


    with st.expander(
        "🔧 التفاصيل التقنية",
        expanded=True
    ):

        st.code(
            st.session_state.error
        )


# =========================================================
# BRIEF INFO
# =========================================================

if st.session_state.case_brief:

    info = st.session_state.brief_info


    if info:

        if info["removed_protocol"]:

            st.success(
                "✅ تم حذف قسم الوكلاء والقواعد "
                "من البرومبت تلقائياً لأنهم مدمجون في التطبيق."
            )


        if info["trimmed"]:

            st.warning(
                "⚠️ كان النص طويلاً جداً، "
                "فتم اختصاره محلياً قبل إرساله."
            )


    with st.expander(
        "📋 CASE BRIEF الذي استخدمه المجلس"
    ):

        st.markdown(
            st.session_state.case_brief
        )


# =========================================================
# RESULTS
# =========================================================

if st.session_state.hunter1:

    st.divider()

    st.header(
        "🎯 1. THE HUNTER"
    )

    st.markdown(
        st.session_state.hunter1
    )


if st.session_state.killer1:

    st.divider()

    st.header(
        "🔪 2. THE KILLER"
    )

    st.markdown(
        st.session_state.killer1
    )


if st.session_state.hunter2:

    st.divider()

    st.header(
        "🎯 3. HUNTER REBUTTAL"
    )

    st.markdown(
        st.session_state.hunter2
    )


if st.session_state.killer2:

    st.divider()

    st.header(
        "🔪 4. KILLER FINAL ATTACK"
    )

    st.markdown(
        st.session_state.killer2
    )


if st.session_state.operator:

    st.divider()

    st.header(
        "📊 5. THE OPERATOR"
    )

    st.markdown(
        st.session_state.operator
    )


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
        st.session_state.original_question,
        st.session_state.case_brief,
        st.session_state.hunter1,
        st.session_state.killer1,
        st.session_state.hunter2,
        st.session_state.killer2,
        st.session_state.operator,
        st.session_state.objection,
        st.session_state.verdict
    )


    st.divider()

    st.header(
        "📥 تحميل التقرير"
    )


    st.download_button(
        label="📝 تحميل التقرير TXT",
        data=report.encode("utf-8"),
        file_name="MD_Investment_Debate.txt",
        mime="text/plain",
        use_container_width=True
    )


    try:

        pdf_bytes = create_pdf(
            report
        )


        st.download_button(
            label="📄 تحميل التقرير PDF",
            data=pdf_bytes,
            file_name="MD_Investment_Debate.pdf",
            mime="application/pdf",
            use_container_width=True
        )


    except Exception as e:

        st.warning(
            "تعذر إنشاء PDF."
        )


        with st.expander(
            "🔧 سبب مشكلة PDF"
        ):

            st.code(
                str(e)
            )


# =========================================================
# RESET
# =========================================================

if (
    st.session_state.case_brief
    or st.session_state.error
):

    st.divider()


    if st.button(
        "🗑️ مسح المناظرة والبدء من جديد"
    ):

        for key, value in DEFAULT_STATE.items():

            st.session_state[key] = value


        st.rerun()
