import os
import re
import glob
import time
import streamlit as st

from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor, as_completed
from fpdf import FPDF


# =========================================================
# BUILD
# =========================================================

BUILD_ID = "GROQ-COUNCIL-1"


# =========================================================
# إعداد الصفحة
# =========================================================

st.set_page_config(
    page_title="MD AI Council",
    page_icon="🧠",
    layout="wide"
)


# =========================================================
# RTL عربي
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
    timeout=90.0,
    max_retries=0
)


# =========================================================
# المستشارون
#
# كل مستشار يبدأ بموديل مختلف فعلياً
# =========================================================

AGENTS = {

    "🧠 المستشار الاستراتيجي": {

        "models": [
            {
                "id": "openai/gpt-oss-120b",
                "reasoning": "low"
            },
            {
                "id": "openai/gpt-oss-20b",
                "reasoning": "low"
            }
        ],

        "role": """
أنت مستشار استراتيجي مستقل ومحترف.

حلل سؤال المستخدم من منظور استراتيجي.

ركز على:
- الهدف الحقيقي
- الخيارات المتاحة
- المزايا والعيوب
- المخاطر
- العواقب قصيرة المدى
- العواقب طويلة المدى
- المعلومات التي تنقصنا
- أفضل قرار في الوضع الحالي

لا توافق على المستخدم لمجرد إرضائه.

إذا كانت الفكرة ضعيفة قل ذلك بوضوح.

قدم توصية عملية ومباشرة.

لا تتجاوز 600 كلمة.
"""
    },


    "😈 الناقد": {

        "models": [
            {
                "id": "qwen/qwen3.8-27b",
                "reasoning": "none"
            },
            {
                "id": "openai/gpt-oss-20b",
                "reasoning": "low"
            }
        ],

        "role": """
أنت Devil's Advocate وناقد مستقل.

اختبر الفكرة بقوة.

ابحث عن:
- الأخطاء
- نقاط الضعف
- المخاطر
- الافتراضات غير المثبتة
- التحيزات
- السيناريوهات التي تؤدي للفشل
- الأشياء التي قد يكون المستخدم يتجاهلها

لا تعارض لمجرد المعارضة.

كل اعتراض يجب أن يكون له سبب منطقي.

في النهاية وضح كيف يمكن تحسين الفكرة.

لا تتجاوز 600 كلمة.
"""
    },


    "💡 المستشار المبتكر": {

        "models": [
            {
                "id": "qwen/qwen3.6-27b",
                "reasoning": "none"
            },
            {
                "id": "openai/gpt-oss-20b",
                "reasoning": "low"
            }
        ],

        "role": """
أنت مستشار ابتكار وحلول عملية.

ابحث عن:
- حلول مختلفة
- بدائل لم يفكر بها المستخدم
- طرق أبسط
- طرق أقل تكلفة
- فرص مخفية
- طرق اختبار الفكرة قبل المخاطرة
- سيناريوهات بديلة

كن واقعياً وعملياً.

لا تقدم أفكاراً خيالية غير قابلة للتنفيذ.

لا تتجاوز 600 كلمة.
"""
    }
}


# =========================================================
# رئيس المجلس
# موديل رابع مختلف
# =========================================================

JUDGE_MODELS = [

    {
        "id": "openai/gpt-oss-20b",
        "reasoning": "low"
    },

    {
        "id": "qwen/qwen3.8-27b",
        "reasoning": "none"
    }
]


# =========================================================
# استخراج الرد
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

                text = getattr(
                    part,
                    "text",
                    None
                )

                if text:
                    pieces.append(str(text))

        return "\n".join(pieces).strip()

    return str(content).strip()


# =========================================================
# اختصار النصوص الطويلة قبل إرسالها للجولة التالية
# =========================================================

def truncate_text(
    text,
    max_chars=6500
):

    if not text:
        return ""

    if len(text) <= max_chars:
        return text

    return (
        text[:max_chars]
        + "\n\n[تم اختصار بقية الرد لتوفير السياق]"
    )


# =========================================================
# استدعاء موديل Groq
# =========================================================

def call_one_model(
    model_config,
    system_prompt,
    user_prompt,
    max_tokens=1200
):

    model = model_config["id"]

    reasoning = model_config.get(
        "reasoning"
    )

    errors = []


    # محاولتان فقط
    for attempt in range(2):

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

                "temperature": 0.4,

                "max_completion_tokens": max_tokens
            }


            # Groq يدعم reasoning_effort
            # لهذه الموديلات
            if reasoning:

                kwargs["reasoning_effort"] = reasoning


            response = (
                client
                .chat
                .completions
                .create(**kwargs)
            )


            if not response.choices:

                errors.append(
                    f"{model}: لم يرجع choices"
                )

                continue


            choice = response.choices[0]

            message = choice.message


            content = extract_content(
                message.content
            )


            finish_reason = getattr(
                choice,
                "finish_reason",
                None
            )


            if content:

                return {
                    "ok": True,
                    "text": content,
                    "model": model,
                    "finish_reason": finish_reason,
                    "error": None
                }


            errors.append(
                f"{model}: رد فارغ | "
                f"finish_reason={finish_reason}"
            )


        except Exception as e:

            error_text = str(e)

            errors.append(
                f"{model}: {error_text}"
            )


            # Rate limit
            if (
                "429" in error_text
                or "rate limit" in error_text.lower()
            ):

                time.sleep(5)

            else:

                time.sleep(1)


    return {
        "ok": False,
        "text": "",
        "model": model,
        "finish_reason": None,
        "error": " || ".join(errors)
    }


# =========================================================
# Fallback بين الموديلات
# =========================================================

def ask_models(
    models,
    system_prompt,
    user_prompt,
    max_tokens=1200
):

    all_errors = []


    for model_config in models:

        result = call_one_model(

            model_config,

            system_prompt,

            user_prompt,

            max_tokens=max_tokens
        )


        if result["ok"]:

            return result


        all_errors.append(
            result.get(
                "error",
                "Unknown error"
            )
        )


    return {
        "ok": False,
        "text": "",
        "model": None,
        "finish_reason": None,
        "error": "\n\n".join(all_errors)
    }


# =========================================================
# النتائج الناجحة
# =========================================================

def valid_results(results):

    if not results:
        return []


    return [

        result

        for result
        in results.values()

        if result.get("ok") is True
    ]


# =========================================================
# الجولة الأولى
# =========================================================

def first_round(question):

    results = {}


    def run_agent(
        name,
        config
    ):

        result = ask_models(

            config["models"],

            config["role"],

            f"""
السؤال:

{question}

قدم تحليلك المستقل الآن.
""",

            max_tokens=1100
        )


        return name, result


    # المستشارون الثلاثة يعملون بنفس الوقت
    with ThreadPoolExecutor(
        max_workers=3
    ) as executor:

        future_map = {}


        for name, config in AGENTS.items():

            future = executor.submit(
                run_agent,
                name,
                config
            )

            future_map[future] = name


        for future in as_completed(
            future_map
        ):

            name = future_map[future]

            try:

                agent_name, result = (
                    future.result()
                )

                results[agent_name] = result


            except Exception as e:

                results[name] = {

                    "ok": False,

                    "text": "",

                    "model": None,

                    "error": str(e)
                }


    # المحافظة على الترتيب
    ordered = {}


    for name in AGENTS:

        ordered[name] = results.get(

            name,

            {
                "ok": False,
                "text": "",
                "model": None,
                "error": "لم يصل رد."
            }
        )


    return ordered


# =========================================================
# الجولة الثانية
# كل واحد يقرأ الآخرين ويرد عليهم
# =========================================================

def debate_round(
    question,
    round1
):

    results = {}


    successful_answers = {

        name: result

        for name, result
        in round1.items()

        if result.get("ok")
    }


    def run_agent(
        name,
        config
    ):

        others = ""


        for (
            other_name,
            result
        ) in successful_answers.items():

            if other_name == name:
                continue


            others += f"""

====================================

رأي {other_name}:

{truncate_text(result["text"], 5000)}

"""


        previous = round1.get(
            name,
            {}
        )


        if previous.get("ok"):

            previous_text = truncate_text(
                previous["text"],
                5000
            )

        else:

            previous_text = (
                "لم يصل رأي منك في الجولة الأولى."
            )


        prompt = f"""
السؤال الأصلي:

{question}


رأيك الأول:

{previous_text}


آراء بقية المستشارين:

{others}


أنت الآن في الجولة الثانية من الاجتماع.

لا تكرر كلامك فقط.

رد على حجج الآخرين مباشرة.

أجب باختصار:

1. أين تتفق معهم؟

2. أين تختلف معهم؟

3. ما أقوى حجة طرحها أحدهم؟

4. ما أضعف افتراض في النقاش؟

5. هل غيرت رأيك؟

6. ما توصيتك المحدثة؟

لا تتجاوز 500 كلمة.
"""


        result = ask_models(

            config["models"],

            config["role"],

            prompt,

            max_tokens=1000
        )


        return name, result


    # الجولة الثانية أيضاً بالتوازي
    with ThreadPoolExecutor(
        max_workers=3
    ) as executor:

        future_map = {}


        for name, config in AGENTS.items():

            future = executor.submit(
                run_agent,
                name,
                config
            )

            future_map[future] = name


        for future in as_completed(
            future_map
        ):

            name = future_map[future]

            try:

                agent_name, result = (
                    future.result()
                )

                results[agent_name] = result


            except Exception as e:

                results[name] = {

                    "ok": False,

                    "text": "",

                    "model": None,

                    "error": str(e)
                }


    ordered = {}


    for name in AGENTS:

        ordered[name] = results.get(

            name,

            {
                "ok": False,
                "text": "",
                "model": None,
                "error": "لم يصل رد."
            }
        )


    return ordered


# =========================================================
# رئيس المجلس
# =========================================================

def final_judge(
    question,
    round1,
    round2
):

    meeting = ""


    for name in AGENTS:

        first = round1.get(
            name,
            {}
        )

        second = round2.get(
            name,
            {}
        )


        first_text = (

            truncate_text(
                first.get("text", ""),
                4500
            )

            if first.get("ok")

            else "لم يتوفر رأي أولي."
        )


        second_text = (

            truncate_text(
                second.get("text", ""),
                4500
            )

            if second.get("ok")

            else "لم يتوفر رد في جولة النقاش."
        )


        meeting += f"""

====================================

{name}

الرأي الأول:

{first_text}


الرأي بعد النقاش:

{second_text}

"""


    prompt = f"""
السؤال:

{question}


محضر مجلس المستشارين:

{meeting}


أنت رئيس المجلس.

لا تلخص كلامهم فقط.

احكم بين الحجج.

لا تنحز للأغلبية تلقائياً.

إذا كان رأي مستشار واحد أقوى من الآخرين،
اختر حجته.

لا تخترع معلومات.


أصدر تقريراً بهذا الترتيب:


# الخلاصة التنفيذية


# نقاط الاتفاق


# نقاط الخلاف


# أقوى الحجج


# الافتراضات غير المثبتة


# أهم المخاطر


# أفضل خيار


# توصية المجلس النهائية


# ماذا أفعل الآن؟

قدم خطوات عملية مرتبة.


# درجة الثقة

درجة من 0 إلى 100 مع تفسير مختصر.


لا تتجاوز 900 كلمة.
"""


    return ask_models(

        JUDGE_MODELS,

        """
أنت رئيس مجلس استشاري مستقل ومحايد.

مهمتك تقييم حجج ثلاثة مستشارين
والوصول إلى أفضل قرار.

احكم على جودة المنطق،
وليس اسم المستشار أو اسم الموديل.
""",

        prompt,

        max_tokens=1500
    )


# =========================================================
# التقرير الكامل
# =========================================================

def build_report(
    question,
    round1,
    round2,
    final
):

    report = f"""
MD AI COUNCIL
المجلس الاستشاري للذكاء الاصطناعي

========================================

السؤال

{question}


========================================

القرار النهائي

{final.get("text", "لم يتوفر تقرير.")}


========================================

الآراء الأولية
"""


    for name, result in round1.items():

        report += f"""

{name}

"""


        if result.get("ok"):

            report += result["text"]

        else:

            report += (
                "لم يتوفر رد من هذا المستشار."
            )


        report += """

----------------------------------------
"""


    report += """

========================================

جولة النقاش
"""


    for name, result in round2.items():

        report += f"""

{name}

"""


        if result.get("ok"):

            report += result["text"]

        else:

            report += (
                "لم يتوفر رد من هذا المستشار."
            )


        report += """

----------------------------------------
"""


    return report.strip()


# =========================================================
# تنظيف PDF
# =========================================================

def clean_pdf_text(text):

    text = re.sub(
        r"#{1,6}\s*",
        "",
        text
    )

    text = text.replace(
        "**",
        ""
    )

    text = text.replace(
        "__",
        ""
    )

    text = text.replace(
        "`",
        ""
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
        "1️⃣",
        "2️⃣",
        "3️⃣"
    ]


    for emoji in emojis:

        text = text.replace(
            emoji,
            ""
        )


    return text


# =========================================================
# إيجاد خط عربي
# =========================================================

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


# =========================================================
# PDF عربي
# =========================================================

def create_pdf(
    report_text
):

    font_path = find_arabic_font()


    if not font_path:

        raise RuntimeError(
            "لم أجد خطاً عربياً على الخادم."
        )


    text = clean_pdf_text(
        report_text
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
        family="Arabic",
        style="",
        fname=font_path
    )


    pdf.set_font(
        "Arabic",
        size=11
    )


    # تشكيل العربية و RTL
    pdf.set_text_shaping(
        use_shaping_engine=True,
        direction="rtl",
        script="arab",
        language="ara"
    )


    pdf.set_title(
        "MD AI Council"
    )


    for paragraph in text.split("\n"):

        paragraph = paragraph.strip()


        if not paragraph:

            pdf.ln(3)

            continue


        headings = [

            "MD AI COUNCIL",

            "المجلس الاستشاري للذكاء الاصطناعي",

            "السؤال",

            "القرار النهائي",

            "الآراء الأولية",

            "جولة النقاش"
        ]


        if paragraph in headings:

            pdf.set_font(
                "Arabic",
                size=16
            )

            height = 10

        else:

            pdf.set_font(
                "Arabic",
                size=11
            )

            height = 7


        pdf.multi_cell(

            w=0,

            h=height,

            text=paragraph,

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

    "question_saved": "",

    "round1": None,

    "round2": None,

    "final": None,

    "meeting_error": None
}


for key, value in DEFAULT_STATE.items():

    if key not in st.session_state:

        st.session_state[key] = value


# =========================================================
# الواجهة
# =========================================================

st.title(
    "🧠 مجلس MD للذكاء الاصطناعي"
)


st.caption(
    f"Build: {BUILD_ID}"
)


st.write(
    """
ثلاثة نماذج ذكاء اصطناعي مختلفة
تحلل سؤالك بشكل مستقل.

بعدها يقرأ كل مستشار آراء الآخرين
ويرد عليها.

ثم يصدر رئيس المجلس القرار النهائي.
"""
)


question = st.text_area(

    "وش تبي المجلس يناقش؟",

    height=180,

    value=st.session_state.question_saved,

    placeholder=(
        "مثال:\n\n"
        "هل الأفضل أبدأ مشروع صغير "
        "أو أستثمر 50 ألف ريال؟"
    )
)


start = st.button(

    "🚀 ابدأ الاجتماع",

    type="primary",

    use_container_width=True
)


# =========================================================
# تشغيل المجلس
# =========================================================

if start:

    if not question.strip():

        st.warning(
            "اكتب الموضوع أولاً."
        )


    else:

        st.session_state.question_saved = (
            question
        )

        st.session_state.round1 = None

        st.session_state.round2 = None

        st.session_state.final = None

        st.session_state.meeting_error = None


        progress = st.empty()


        # =================================================
        # الجولة الأولى
        # =================================================

        progress.info(
            "🧠 الجولة الأولى: "
            "3 مستشارين يفكرون بنفس الوقت..."
        )


        round1 = first_round(
            question
        )


        st.session_state.round1 = (
            round1
        )


        if len(
            valid_results(round1)
        ) < 2:

            st.session_state.meeting_error = (
                "لم ينجح عدد كافٍ من المستشارين "
                "في الجولة الأولى."
            )


            progress.error(
                "❌ فشلت الجولة الأولى."
            )


        else:

            # =============================================
            # النقاش
            # =============================================

            progress.info(
                "⚔️ المستشارون الآن "
                "يقرؤون آراء بعضهم..."
            )


            round2 = debate_round(

                question,

                round1
            )


            st.session_state.round2 = (
                round2
            )


            if len(
                valid_results(round2)
            ) < 2:

                st.session_state.meeting_error = (
                    "لم ينجح عدد كافٍ "
                    "في جولة النقاش."
                )


                progress.error(
                    "❌ فشلت جولة النقاش."
                )


            else:

                # =========================================
                # الحكم
                # =========================================

                progress.info(
                    "🏛️ رئيس المجلس "
                    "يراجع الاجتماع..."
                )


                final = final_judge(

                    question,

                    round1,

                    round2
                )


                if final.get("ok"):

                    st.session_state.final = (
                        final
                    )


                    progress.success(
                        "✅ انتهى الاجتماع بنجاح."
                    )


                else:

                    st.session_state.meeting_error = (
                        "نجح النقاش لكن رئيس المجلس "
                        "لم يستطع إصدار التقرير."
                    )


                    progress.error(
                        "❌ تعذر إصدار التقرير."
                    )


# =========================================================
# الخطأ
# =========================================================

if st.session_state.meeting_error:

    st.error(
        st.session_state.meeting_error
    )


# =========================================================
# عرض الجولة الأولى
# =========================================================

if st.session_state.round1:

    st.divider()


    st.header(
        "1️⃣ الآراء الأولية"
    )


    cols = st.columns(
        3
    )


    for col, (
        name,
        result
    ) in zip(
        cols,
        st.session_state.round1.items()
    ):

        with col:

            st.subheader(
                name
            )


            if result.get("ok"):

                st.markdown(
                    result["text"]
                )


                st.caption(
                    f'الموديل: {result["model"]}'
                )


            else:

                st.warning(
                    "تعذر الحصول على رد."
                )


                with st.expander(
                    "🔧 التفاصيل التقنية"
                ):

                    st.code(
                        result.get(
                            "error",
                            "Unknown error"
                        )
                    )


# =========================================================
# عرض النقاش
# =========================================================

if st.session_state.round2:

    st.divider()


    st.header(
        "2️⃣ ⚔️ النقاش"
    )


    for (
        name,
        result
    ) in st.session_state.round2.items():


        with st.expander(
            f"{name} يرد على المجلس",
            expanded=False
        ):


            if result.get("ok"):

                st.markdown(
                    result["text"]
                )


                st.caption(
                    f'الموديل: {result["model"]}'
                )


            else:

                st.warning(
                    "تعذر الحصول على الرد."
                )


                st.code(
                    result.get(
                        "error",
                        "Unknown error"
                    )
                )


# =========================================================
# التقرير النهائي
# =========================================================

if st.session_state.final:

    st.divider()


    st.header(
        "3️⃣ 🏛️ القرار النهائي"
    )


    st.markdown(
        st.session_state.final["text"]
    )


    st.caption(
        "رئيس المجلس: "
        f'{st.session_state.final["model"]}'
    )


    report = build_report(

        st.session_state.question_saved,

        st.session_state.round1,

        st.session_state.round2,

        st.session_state.final
    )


    st.divider()


    st.header(
        "📥 تحميل التقرير"
    )


    # =====================================================
    # TXT
    # =====================================================

    st.download_button(

        label="📝 تحميل التقرير كنص",

        data=report.encode(
            "utf-8"
        ),

        file_name=(
            "MD_AI_Council_Report.txt"
        ),

        mime="text/plain",

        use_container_width=True
    )


    # =====================================================
    # PDF
    # =====================================================

    try:

        pdf_bytes = create_pdf(
            report
        )


        st.download_button(

            label="📄 تحميل التقرير PDF",

            data=pdf_bytes,

            file_name=(
                "MD_AI_Council_Report.pdf"
            ),

            mime="application/pdf",

            use_container_width=True
        )


    except Exception as e:

        st.warning(
            "تعذر تجهيز PDF."
        )


        with st.expander(
            "🔧 سبب مشكلة PDF"
        ):

            st.code(
                str(e)
            )


# =========================================================
# مسح النتائج
# =========================================================

if (
    st.session_state.round1
    or st.session_state.final
):

    st.divider()


    if st.button(
        "🗑️ مسح النتيجة وبدء اجتماع جديد"
    ):

        for (
            key,
            value
        ) in DEFAULT_STATE.items():

            st.session_state[key] = value


        st.rerun()
