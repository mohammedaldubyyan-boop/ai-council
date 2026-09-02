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

BUILD_ID = "GROQ-DEBATE-3"


# =========================================================
# STREAMLIT
# =========================================================

st.set_page_config(
    page_title="MD AI Council",
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
# GROQ CLIENT
# =========================================================

client = OpenAI(
    api_key=st.secrets["GROQ_API_KEY"],
    base_url="https://api.groq.com/openai/v1",
    timeout=120.0,
    max_retries=0
)


# =========================================================
# MODELS
#
# 3 أعضاء حقيقيين:
#
# Hunter   = GPT OSS 120B
# Killer   = Qwen 3.8 27B
# Operator = Qwen 3.6 27B
#
# Compound Mini ليس عضواً.
# نستخدمه فقط لضغط الـ prompt الطويل.
# =========================================================

COMPRESSOR_MODEL = {
    "id": "groq/compound-mini",
    "reasoning": None
}


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
# SYSTEM PROMPTS
# =========================================================

HUNTER_SYSTEM = """
أنت THE HUNTER.

أنت رائد أعمال وباحث فرص اقتصادية.

مهمتك البحث عن أماكن تتحرك فيها الأموال فعلياً.

لا تحاول إبهار المستخدم بفكرة AI.

الأولوية:
- ألم مالي حقيقي
- willingness to pay
- سرعة الوصول لأول عملية دفع
- سهولة التوزيع
- الأتمتة
- اقتصاديات الوحدة
- تقليل تدخل صاحب المشروع

التزم بقيود المستخدم حرفياً.

لا تعيد الأفكار التي رفضها المستخدم
إلا إذا كان التغيير الاقتصادي جوهرياً ويمكن إثباته.

قدم بحد أقصى 5 أفكار.

كن قاسياً وواقعياً.

لا تستخدم لغة تحفيزية.
"""


KILLER_SYSTEM = """
أنت THE KILLER.

أنت مستثمر متشائم ومنافس يريد منعنا
من خسارة الوقت والمال.

لا تقترح أفكاراً جديدة.

مهمتك مهاجمة ما يقدمه THE HUNTER.

ابحث عن:
- المنافسين
- البدائل المجانية
- إمكانية استبداله بـ ChatGPT
- ضعف willingness to pay
- CAC
- churn
- platform risk
- regulatory risk
- security
- privacy
- liability
- ضعف moat
- سهولة التقليد
- الحاجة لمبيعات بشرية
- عدم تكرار الاستخدام
- كون الفكرة Feature وليست Company

ممنوع المجاملة.

إذا الفكرة سيئة قل KILL IT.
"""


OPERATOR_SYSTEM = """
أنت THE OPERATOR / ECONOMIST.

أنت CTO + CFO + Growth Operator.

وظيفتك ليست توليد أفكار جديدة.

وظيفتك الحكم على الأفكار التي نجت
من Hunter وKiller.

ركز على:
- Economics
- Pricing
- CAC
- LTV
- Gross Margin
- Recurring Revenue
- Speed to Revenue
- Build Complexity
- Automation
- Distribution
- First Paying Customer
- Stack Fit

لا ترفع الدرجات لتحقيق شرط 85.

إذا أفضل فكرة 74/100 قل 74.

إذا لم توجد فكرة تستحق البناء
قل بوضوح لا توجد فكرة تستحق البناء الآن.

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

        parts = []

        for item in content:

            if isinstance(item, str):
                parts.append(item)

            elif isinstance(item, dict):

                text = item.get("text")

                if text:
                    parts.append(str(text))

            else:

                text = getattr(
                    item,
                    "text",
                    None
                )

                if text:
                    parts.append(str(text))

        return "\n".join(parts).strip()

    return str(content).strip()


# =========================================================
# استخراج وقت الانتظار من Groq 429
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
                    2,
                    math.ceil(seconds) + 2
                )

            except:
                pass

    return 10


# =========================================================
# MODEL CALL
#
# أهم تعديل:
# إذا Groq قال انتظر 39 ثانية،
# ما نعيد بعد 5 ثواني.
# ننتظر الوقت الحقيقي.
# =========================================================

def call_model(
    model_config,
    system_prompt,
    user_prompt,
    max_tokens,
    stage_name,
    status_box=None,
    retries=4
):

    model = model_config["id"]

    reasoning = model_config.get(
        "reasoning"
    )

    errors = []


    for attempt in range(retries):

        try:

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

                "max_completion_tokens": max_tokens,

                "temperature": 0.4
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
                    f"{model}: no choices"
                )

                continue


            choice = response.choices[0]

            content = extract_content(
                choice.message.content
            )


            if content:

                return {
                    "ok": True,
                    "text": content,
                    "model": model,
                    "error": None
                }


            finish_reason = getattr(
                choice,
                "finish_reason",
                None
            )


            errors.append(
                f"{model}: empty response | "
                f"finish_reason={finish_reason}"
            )


        except Exception as e:

            error_text = str(e)

            errors.append(
                f"{model}: {error_text}"
            )


            # ==========================================
            # RATE LIMIT
            # ==========================================

            if (
                "429" in error_text
                or "rate_limit" in error_text.lower()
            ):

                wait_seconds = get_retry_seconds(
                    error_text
                )


                if status_box:

                    status_box.warning(
                        f"⏳ {stage_name}: "
                        f"وصلنا حد Groq المجاني. "
                        f"انتظار {wait_seconds} ثانية "
                        f"ثم نكمل تلقائياً..."
                    )


                time.sleep(
                    wait_seconds
                )

                continue


            # خطأ عادي
            time.sleep(2)


    return {
        "ok": False,
        "text": "",
        "model": model,
        "error": "\n\n".join(errors)
    }


# =========================================================
# 0 - ضغط الـ Brief
#
# هذه أهم خطوة لتجنب 8000 TPM.
#
# الـ prompt الضخم يذهب مرة واحدة إلى Compound Mini.
# ثم المجلس يتعامل مع نسخة مركزة.
# =========================================================

def compress_brief(
    original_prompt,
    status_box
):

    system = """
أنت لست مستشاراً ولا تحاول حل السؤال.

وظيفتك فقط ضغط تعليمات المستخدم
إلى INVESTMENT BRIEF كثيف ودقيق.

ممنوع حذف العناصر المهمة.

يجب أن تحافظ على:

1. الهدف النهائي.
2. الأشياء الممنوعة.
3. الأصول الموجودة.
4. قائمة الأفكار المرفوضة.
5. تعريف Hunter.
6. تعريف Killer.
7. تعريف Operator.
8. قواعد المناظرة.
9. نظام التقييم وجميع الأوزان.
10. شرط 85/100.
11. اختبار الحقيقة.
12. صيغة المخرجات النهائية.
13. كل الأرقام والأسعار والشروط المهمة.

لا تحل المشكلة.

لا تقترح فكرة.

لا تقيّم.

فقط حوّل النص إلى brief منظم ومضغوط
يستطيع ثلاثة وكلاء استخدامه دون الحاجة
للنص الأصلي.

كن كثيفاً جداً.
"""


    prompt = f"""
حوّل النص التالي إلى Investment Brief
مع الحفاظ على كل القيود المهمة:

================ ORIGINAL ================

{original_prompt}

================ END ====================
"""


    return call_model(
        COMPRESSOR_MODEL,
        system,
        prompt,
        max_tokens=1300,
        stage_name="تجهيز الـ Investment Brief",
        status_box=status_box
    )


# =========================================================
# 1 - HUNTER
# =========================================================

def hunter_generate(
    brief,
    status_box
):

    prompt = f"""
هذا هو Investment Brief:

{brief}


ابدأ الآن المرحلة الأولى فقط.

قدم بحد أقصى 5 أفكار.

لكل فكرة وضح:

1. من يدفع؟
2. لماذا يدفع؟
3. السعر التقريبي.
4. ماذا يحدث اليوم بدون المنتج؟
5. البديل الحالي.
6. لماذا نحن أفضل؟
7. كيف نصل لأول 10 عملاء؟
8. ماذا يمكن أتمتته؟
9. العمل البشري المتبقي.
10. لماذا الآن؟

لا تحاول الحكم النهائي.

لا تكتب WINNER.

مهمتك فقط تقديم أفضل المرشحين
ليهاجمهم Killer.
"""


    return call_model(
        HUNTER_MODEL,
        HUNTER_SYSTEM,
        prompt,
        max_tokens=1600,
        stage_name="THE HUNTER",
        status_box=status_box
    )


# =========================================================
# 2 - KILLER FIRST ATTACK
# =========================================================

def killer_attack(
    brief,
    hunter_output,
    status_box
):

    prompt = f"""
INVESTMENT BRIEF:

{brief}


THE HUNTER اقترح:

================ HUNTER ================

{hunter_output}

========================================


هاجم كل فكرة بشكل مستقل.

لكل فكرة أعط:

- أقوى 3 أسباب للفشل.
- Kill Shot واحد.
- الدليل الذي لو وجدناه نرفضها فوراً.
- تقييم من 10 بعد الهجوم.

افحص خصوصاً:

المنافسة،
ChatGPT substitution،
CAC،
WTP،
churn،
distribution،
platform risk،
privacy،
legal،
liability،
moat،
human sales،
repeat usage.

لا تقترح فكرة جديدة.

إذا تستحق الموت قل:

KILL IT.
"""


    return call_model(
        KILLER_MODEL,
        KILLER_SYSTEM,
        prompt,
        max_tokens=1400,
        stage_name="THE KILLER - الهجوم الأول",
        status_box=status_box
    )


# =========================================================
# 3 - HUNTER REBUTTAL
# =========================================================

def hunter_rebuttal(
    brief,
    hunter_output,
    killer_output,
    status_box
):

    prompt = f"""
INVESTMENT BRIEF:

{brief}


رأيك الأول:

================ HUNTER ORIGINAL ================

{hunter_output}


هجوم THE KILLER:

================ KILLER ATTACK ==================

{killer_output}

=================================================


لديك رد واحد فقط.

لا تضف أفكاراً جديدة.

لكل فكرة لم تُقتل بوضوح:

- رد على اعتراضات Killer.
- قل ما الاعتراض الذي تقبله.
- قل ما الاعتراض الذي ترفضه ولماذا.
- ما الدليل الذي نحتاجه؟
- هل ما زلت تدافع عنها؟

إذا اقتنعت أنها سيئة،
تخل عنها ولا تحاول ترقيعها.

كن مختصراً.
"""


    return call_model(
        HUNTER_MODEL,
        HUNTER_SYSTEM,
        prompt,
        max_tokens=750,
        stage_name="THE HUNTER - الرد",
        status_box=status_box
    )


# =========================================================
# 4 - KILLER FINAL ATTACK
# =========================================================

def killer_final(
    brief,
    hunter_output,
    killer_output,
    hunter_rebuttal_output,
    status_box
):

    prompt = f"""
INVESTMENT BRIEF:

{brief}


HUNTER ORIGINAL:

{hunter_output}


YOUR FIRST ATTACK:

{killer_output}


HUNTER REBUTTAL:

{hunter_rebuttal_output}


هذه فرصتك الأخيرة قبل Operator.

لكل فكرة:

- هل نجت أم ماتت؟
- أقوى مشكلة متبقية.
- هل يوجد سبب حقيقي للدفع؟
- هل Distribution واقعية؟
- هل هي Company أم Feature؟
- التقييم النهائي من 10 بعد Red Team.

إذا لم تنج:

KILL IT.

لا تقترح أفكاراً جديدة.
"""


    return call_model(
        KILLER_MODEL,
        KILLER_SYSTEM,
        prompt,
        max_tokens=750,
        stage_name="THE KILLER - الحكم الأخير",
        status_box=status_box
    )


# =========================================================
# ضغط المناظرة قبل Operator
#
# Compound لديه مساحة TPM أكبر.
# =========================================================

def compress_debate(
    brief,
    hunter1,
    killer1,
    hunter2,
    killer2,
    status_box
):

    system = """
أنت محرر محضر مناظرة.

لا تضف رأياً جديداً.

اختصر المناظرة إلى DEBATE PACKET دقيق.

احتفظ لكل فكرة بـ:

- اسم الفكرة.
- من يدفع.
- السعر.
- سبب الدفع.
- distribution.
- automation.
- أقوى دفاع Hunter.
- أقوى اعتراض Killer.
- هل Killer قتلها أم أبقاها؟
- أي درجات أو أرقام ذكرت.
- أي دليل ناقص.

لا تختر Winner.

لا تغير الأرقام.

لا ترفع الدرجات.
"""


    prompt = f"""
BRIEF:

{brief}


HUNTER:

{hunter1}


KILLER ATTACK:

{killer1}


HUNTER REBUTTAL:

{hunter2}


KILLER FINAL:

{killer2}


أنشئ الآن DEBATE PACKET مضغوطاً.
"""


    return call_model(
        COMPRESSOR_MODEL,
        system,
        prompt,
        max_tokens=1300,
        stage_name="ضغط محضر المناظرة",
        status_box=status_box
    )


# =========================================================
# 5 - OPERATOR PRELIMINARY VERDICT
# =========================================================

def operator_evaluate(
    brief,
    debate_packet,
    status_box
):

    prompt = f"""
INVESTMENT BRIEF:

{brief}


DEBATE PACKET:

{debate_packet}


أنت الآن THE OPERATOR.

قيّم فقط الأفكار التي نجت.

طبق نظام الـ100 نقطة الموجود في Brief
حرفياً.

لا ترفع الدرجات لتصل إلى 85.

لكل فكرة ناجية احسب أو قدّر:

Economics:
- Price
- Gross margin
- recurring revenue
- LTV
- CAC
- العملاء اللازمون لـ:
  $1k MRR
  $5k MRR
  $10k MRR

Build:
- ماذا نبني؟
- ماذا نربط؟
- ماذا لا نحتاج؟
- التشغيل التقريبي.
- زمن MVP.

Automation:
- النسبة %
- أين يحتاج تدخل محمد؟

Distribution:
- قناة محددة.
- كيف نحصل على أول عميل يدفع؟

اختبار الحقيقة لكل فكرة 80+.

ثم أعط الجدول:

| Rank | Idea | Score | First Buyer | Price | Automation | Fastest Test | Biggest Risk |

اختر PROVISIONAL WINNER فقط إذا تجاوز 85/100.

إذا لا يوجد:
قل بوضوح NO WINNER.

إذا يوجد Winner، قدم البنود الـ20 المطلوبة
في الـBrief.

لكن لا تكتب FINAL VERDICT بعد.
سنسمح لـKiller باعتراض أخير.
"""


    return call_model(
        OPERATOR_MODEL,
        OPERATOR_SYSTEM,
        prompt,
        max_tokens=2100,
        stage_name="THE OPERATOR - التقييم",
        status_box=status_box
    )


# =========================================================
# 6 - FINAL OBJECTION
# =========================================================

def killer_objection(
    brief,
    operator_output,
    status_box
):

    prompt = f"""
INVESTMENT BRIEF:

{brief}


THE OPERATOR كتب:

================ OPERATOR =================

{operator_output}

===========================================


اكتب الآن فقط:

## FINAL OBJECTION

إذا اختار Operator Winner:

قدم أقوى حجة ممكنة لعدم بناء هذه الفكرة.

حاول قتلها للمرة الأخيرة.

ركز على السبب الذي يمكن أن يجعل
المشروع يصل فعلياً إلى $0 MRR
حتى لو كان التحليل السابق يبدو جيداً.

إذا Operator قال NO WINNER:

اشرح في فقرة قصيرة لماذا هذا القرار
أفضل من إجبارنا على بناء فكرة ضعيفة.

لا تقترح فكرة جديدة.
"""


    return call_model(
        KILLER_MODEL,
        KILLER_SYSTEM,
        prompt,
        max_tokens=500,
        stage_name="THE KILLER - FINAL OBJECTION",
        status_box=status_box
    )


# =========================================================
# 7 - FINAL VERDICT
# =========================================================

def operator_final_verdict(
    brief,
    operator_output,
    killer_objection_output,
    status_box
):

    prompt = f"""
INVESTMENT BRIEF:

{brief}


تقييمك السابق:

================ OPERATOR =================

{operator_output}


اعتراض Killer النهائي:

================ FINAL OBJECTION ==========

{killer_objection_output}

===========================================


أنت THE OPERATOR.

لا تعيد التقرير كله.

اكتب الآن فقط:

## FINAL VERDICT

ثم:

BUILD

أو

KILL


بعدها فسّر القرار باختصار شديد.

إذا BUILD:
حدد أول اختبار مدفوع يجب تنفيذه
خلال 7 أيام.

إذا KILL:
حدد بالضبط أي افتراض فشل
ولماذا لا يجب البناء الآن.

لا ترفع الدرجة.

لا تغير Winner إلا إذا اعتراض Killer
فعلاً كشف مشكلة جوهرية.
"""


    return call_model(
        OPERATOR_MODEL,
        OPERATOR_SYSTEM,
        prompt,
        max_tokens=550,
        stage_name="THE OPERATOR - FINAL VERDICT",
        status_box=status_box
    )


# =========================================================
# REPORT
# =========================================================

def build_report(
    original_question,
    brief,
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
ORIGINAL REQUEST
========================================

{original_question}


========================================
INVESTMENT BRIEF
========================================

{brief}


========================================
1. THE HUNTER
========================================

{hunter1}


========================================
2. THE KILLER - FIRST ATTACK
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
# PDF HELPERS
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


        if (
            paragraph.startswith("===")
            or paragraph in [
                "MD AI COUNCIL",
                "INVESTMENT RED TEAM"
            ]
        ):

            continue


        pdf.set_font(
            "Arabic",
            size=11
        )


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

STATE_DEFAULTS = {

    "question": "",

    "brief": None,

    "hunter1": None,

    "killer1": None,

    "hunter2": None,

    "killer2": None,

    "packet": None,

    "operator": None,

    "objection": None,

    "verdict": None,

    "error": None
}


for key, value in STATE_DEFAULTS.items():

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


st.write(
    """
THE HUNTER يبحث عن الفرص.

THE KILLER يحاول قتلها.

THE OPERATOR يحكم على ما تبقى.
"""
)


question = st.text_area(
    "اكتب الـInvestment Brief أو السؤال:",
    height=350,
    value=st.session_state.question
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
            "اكتب الطلب أولاً."
        )

    else:

        # reset
        for key in STATE_DEFAULTS:

            st.session_state[key] = (
                "" if key == "question"
                else None
            )


        st.session_state.question = (
            question
        )


        status = st.empty()


        # =============================================
        # BRIEF
        # =============================================

        status.info(
            "📋 تجهيز Investment Brief مضغوط..."
        )


        brief_result = compress_brief(
            question,
            status
        )


        if not brief_result["ok"]:

            st.session_state.error = (
                brief_result["error"]
            )

        else:

            st.session_state.brief = (
                brief_result["text"]
            )


            # =========================================
            # HUNTER
            # =========================================

            status.info(
                "🎯 THE HUNTER يبحث عن فرص..."
            )


            result = hunter_generate(
                st.session_state.brief,
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


                # =====================================
                # KILLER
                # =====================================

                status.info(
                    "🔪 THE KILLER يهاجم الأفكار..."
                )


                result = killer_attack(
                    st.session_state.brief,
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


                    # =================================
                    # HUNTER REPLY
                    # =================================

                    status.info(
                        "🎯 THE HUNTER يرد مرة واحدة..."
                    )


                    result = hunter_rebuttal(
                        st.session_state.brief,
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


                        # =============================
                        # KILLER FINAL
                        # =============================

                        status.info(
                            "🔪 THE KILLER يصدر حكمه الأخير..."
                        )


                        result = killer_final(
                            st.session_state.brief,
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


                            # =========================
                            # COMPRESS DEBATE
                            # =========================

                            status.info(
                                "🗜️ تجهيز محضر مختصر للـOperator..."
                            )


                            result = compress_debate(
                                st.session_state.brief,
                                st.session_state.hunter1,
                                st.session_state.killer1,
                                st.session_state.hunter2,
                                st.session_state.killer2,
                                status
                            )


                            if not result["ok"]:

                                st.session_state.error = (
                                    result["error"]
                                )

                            else:

                                st.session_state.packet = (
                                    result["text"]
                                )


                                # =====================
                                # OPERATOR
                                # =====================

                                status.info(
                                    "📊 THE OPERATOR يحسب الاقتصاديات..."
                                )


                                result = operator_evaluate(
                                    st.session_state.brief,
                                    st.session_state.packet,
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


                                    # =================
                                    # OBJECTION
                                    # =================

                                    status.info(
                                        "🔪 FINAL OBJECTION..."
                                    )


                                    result = killer_objection(
                                        st.session_state.brief,
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


                                        # =============
                                        # VERDICT
                                        # =============

                                        status.info(
                                            "🏛️ FINAL VERDICT..."
                                        )


                                        result = operator_final_verdict(
                                            st.session_state.brief,
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
# SHOW BRIEF
# =========================================================

if st.session_state.brief:

    with st.expander(
        "📋 Investment Brief المضغوط"
    ):

        st.markdown(
            st.session_state.brief
        )


# =========================================================
# SHOW DEBATE
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
        "🎯 3. THE HUNTER - REBUTTAL"
    )

    st.markdown(
        st.session_state.hunter2
    )


if st.session_state.killer2:

    st.divider()

    st.header(
        "🔪 4. THE KILLER - FINAL ATTACK"
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


    # =============================================
    # REPORT
    # =============================================

    report = build_report(
        st.session_state.question,
        st.session_state.brief,
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
        "📝 تحميل التقرير TXT",
        data=report.encode("utf-8"),
        file_name="MD_Investment_Debate.txt",
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
            file_name="MD_Investment_Debate.pdf",
            mime="application/pdf",
            use_container_width=True
        )


    except Exception as e:

        with st.expander(
            "مشكلة PDF"
        ):

            st.code(
                str(e)
            )


# =========================================================
# RESET
# =========================================================

if (
    st.session_state.brief
    or st.session_state.error
):

    st.divider()


    if st.button(
        "🗑️ مسح المناظرة والبدء من جديد"
    ):

        for key, value in STATE_DEFAULTS.items():

            st.session_state[key] = value


        st.rerun()
