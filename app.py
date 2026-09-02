import os
import re
import glob
import time
import streamlit as st

from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor, as_completed
from fpdf import FPDF


# =========================================================
# BUILD ID
# إذا ظهر هذا الرقم في التطبيق نعرف أن Streamlit شغل النسخة الجديدة
# =========================================================

BUILD_ID = "2026-09-02-FIX-4"


# =========================================================
# إعداد الصفحة
# =========================================================

st.set_page_config(
    page_title="MD AI Council",
    page_icon="🧠",
    layout="wide"
)


# =========================================================
# تنسيق عربي RTL
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
# OpenRouter
# =========================================================

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=st.secrets["OPENROUTER_API_KEY"],
    timeout=90.0,
    max_retries=0
)


# =========================================================
# موديلات المجلس
#
# كل مستشار له موديل أساسي + موديلات احتياطية
# =========================================================

AGENTS = {

    "🧠 المستشار الاستراتيجي": {

        "models": [
            "inclusionai/ling-3.0-flash-fin:free",
            "openai/gpt-oss-20b:free",
            "openrouter/free"
        ],

        "role": """
أنت مستشار استراتيجي مستقل ومحترف.

حلل سؤال المستخدم بعمق.

ركز على:
- الهدف الحقيقي
- الخيارات المتاحة
- المزايا والعيوب
- المخاطر
- النتائج قصيرة المدى
- النتائج طويلة المدى
- الافتراضات التي يجب اختبارها
- المعلومات الناقصة

لا توافق على المستخدم لمجرد إرضائه.

إذا كانت الفكرة ضعيفة قل ذلك بوضوح.

أعط توصية عملية ومبررة.
"""
    },


    "😈 الناقد": {

        "models": [
            "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free",
            "openai/gpt-oss-20b:free",
            "openrouter/free"
        ],

        "role": """
أنت ناقد مستقل وDevil's Advocate.

مهمتك اختبار الفكرة بقوة.

ابحث عن:
- الأخطاء
- المخاطر
- الافتراضات غير المثبتة
- المعلومات الناقصة
- التحيزات
- أسباب الفشل المحتملة
- الحالات التي تجعل القرار خاطئاً

لا تعارض لمجرد المعارضة.

كل اعتراض يجب أن يكون له سبب منطقي.

في النهاية وضح كيف يمكن تحسين الفكرة.
"""
    },


    "💡 المستشار المبتكر": {

        "models": [
            "openai/gpt-oss-20b:free",
            "inclusionai/ling-3.0-flash-fin:free",
            "openrouter/free"
        ],

        "role": """
أنت مستشار ابتكار وحلول.

ابحث عن:
- حلول لم يفكر بها المستخدم
- بدائل أفضل
- طرق أبسط
- طرق أقل تكلفة
- فرص مخفية
- طرق اختبار الفكرة قبل الالتزام بها
- سيناريوهات بديلة

كن عملياً وواقعياً.

لا تقدم أفكاراً خيالية غير قابلة للتنفيذ.
"""
    }
}


JUDGE_MODELS = [
    "inclusionai/ling-3.0-flash-fin:free",
    "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free",
    "openrouter/free"
]


# =========================================================
# قراءة المحتوى مهما كان شكل الرد
# =========================================================

def extract_content(content):

    if content is None:
        return ""

    if isinstance(content, str):
        return content.strip()

    # بعض الـ APIs قد تعيد أجزاء متعددة
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
# استدعاء موديل واحد
# =========================================================

def call_one_model(
    model,
    system_prompt,
    user_prompt
):

    errors = []


    # نحاول مرتين قبل الانتقال للموديل التالي
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

                # نترك مساحة كافية للجواب النهائي
                "max_tokens": 3200
            }


            # الموديلات التي تستخدم reasoning
            # نخليه minimal حتى لا يأكل كل الـ output tokens
            if model != "openrouter/free":

                kwargs["extra_body"] = {
                    "reasoning": {
                        "effort": "minimal",
                        "exclude": True
                    }
                }


            response = client.chat.completions.create(
                **kwargs
            )


            if not response.choices:

                errors.append(
                    f"{model}: no choices returned"
                )

                continue


            choice = response.choices[0]

            message = choice.message


            content = extract_content(
                message.content
            )


            # نجاح حقيقي
            if content:

                actual_model = getattr(
                    response,
                    "model",
                    model
                )

                return {
                    "ok": True,
                    "text": content,
                    "model": actual_model,
                    "requested_model": model,
                    "error": None
                }


            # إذا OpenRouter نجح لكن لم يرجع جواب نهائي
            reasoning_exists = bool(
                getattr(
                    message,
                    "reasoning",
                    None
                )
            )


            finish_reason = getattr(
                choice,
                "finish_reason",
                None
            )


            errors.append(
                f"{model}: empty final content | "
                f"finish_reason={finish_reason} | "
                f"reasoning_present={reasoning_exists}"
            )


        except Exception as e:

            error_text = str(e)

            errors.append(
                f"{model}: {error_text}"
            )


            # إذا rate limit ننتظر ثم نجرب مرة ثانية
            if (
                "429" in error_text
                or "rate limit" in error_text.lower()
            ):

                time.sleep(3)

            else:

                # خطأ آخر: المحاولة الثانية بعد تأخير خفيف
                time.sleep(1)


    return {
        "ok": False,
        "text": "",
        "model": None,
        "requested_model": model,
        "error": " || ".join(errors)
    }


# =========================================================
# جرب عدة موديلات
# =========================================================

def ask_models(
    models,
    system_prompt,
    user_prompt
):

    all_errors = []


    for model in models:

        result = call_one_model(
            model,
            system_prompt,
            user_prompt
        )


        if result["ok"]:
            return result


        if result.get("error"):
            all_errors.append(
                result["error"]
            )


    combined_error = "\n\n".join(
        all_errors
    )


    return {
        "ok": False,
        "text": (
            "تعذر الحصول على رد من هذا المستشار حالياً."
        ),
        "model": None,
        "requested_model": None,
        "error": combined_error
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
            question
        )

        return name, result


    # الثلاثة يشتغلون بالتوازي
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
                    "text": (
                        "حدث خطأ أثناء تشغيل المستشار."
                    ),
                    "model": None,
                    "requested_model": None,
                    "error": str(e)
                }


    # نحافظ على ترتيب المستشارين
    ordered = {}


    for name in AGENTS:

        ordered[name] = results.get(
            name,
            {
                "ok": False,
                "text": "لم يصل رد.",
                "model": None,
                "requested_model": None,
                "error": "No result returned"
            }
        )


    return ordered


# =========================================================
# الجولة الثانية - النقاش
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


        for other_name, result in successful_answers.items():

            if other_name == name:
                continue


            others += f"""

====================================

رأي {other_name}:

{result["text"]}

"""


        previous = round1.get(
            name,
            {}
        )


        if previous.get("ok"):

            previous_text = previous["text"]

        else:

            previous_text = (
                "لم يصل منك رأي في الجولة الأولى."
            )


        prompt = f"""
السؤال الأصلي:

{question}


رأيك في الجولة الأولى:

{previous_text}


آراء المستشارين الآخرين:

{others}


أنت الآن في اجتماع حقيقي بين المستشارين.

لا تعيد إجابتك السابقة فقط.

اقرأ حجج الآخرين ورد عليها بشكل مباشر.

أجب عن:

1. ما النقاط التي تتفق معهم فيها؟

2. ما النقاط التي تختلف معهم فيها؟

3. ما الأخطاء أو الافتراضات الضعيفة في كلامهم؟

4. أي مستشار قدم أقوى حجة؟ ولماذا؟

5. هل غيرت موقفك بعد قراءة آرائهم؟

6. ما توصيتك المحدثة؟

إذا كانت حجة أحد المستشارين خاطئة،
اذكر الحجة واشرح سبب الخطأ.
"""


        result = ask_models(
            config["models"],
            config["role"],
            prompt
        )


        return name, result


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
                    "text": (
                        "حدث خطأ أثناء جولة النقاش."
                    ),
                    "model": None,
                    "requested_model": None,
                    "error": str(e)
                }


    ordered = {}


    for name in AGENTS:

        ordered[name] = results.get(
            name,
            {
                "ok": False,
                "text": "لم يصل رد.",
                "model": None,
                "requested_model": None,
                "error": "No result returned"
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
            first.get("text")
            if first.get("ok")
            else "لم يتوفر رأي أولي."
        )


        second_text = (
            second.get("text")
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
أنت رئيس مجلس استشاري مستقل ومحايد.

السؤال الأصلي:

{question}


هذا محضر الاجتماع الكامل:

{meeting}


مهمتك ليست تلخيص الكلام فقط.

يجب أن تحكم بين الحجج.

لا تنحز لأي مستشار بسبب اسمه أو الموديل المستخدم.

لا تخترع أرقاماً أو حقائق.

إذا كانت المعلومات غير كافية،
قل بوضوح ما المعلومات التي نحتاجها.


أصدر التقرير بهذا الترتيب:


# الخلاصة التنفيذية

اختصر الوضع والقرار.


# نقاط الاتفاق

اذكر الأشياء التي اتفق عليها المستشارون.


# نقاط الخلاف

اذكر أهم الخلافات.


# أقوى الحجج

أي الحجج كانت أقوى ولماذا؟


# الافتراضات غير المثبتة

ما الأشياء التي نفترضها بدون دليل؟


# أهم المخاطر

رتب أهم المخاطر.


# البدائل المتاحة

اذكر الخيارات الواقعية.


# توصية المجلس النهائية

اختر أفضل قرار حالياً وفسر السبب.


# ماذا أفعل الآن؟

قدم خطوات عملية واضحة ومرتبة.


# درجة الثقة

ضع درجة من 0 إلى 100.

فسر سبب الدرجة.
"""


    return ask_models(
        JUDGE_MODELS,

        """
أنت رئيس مجلس استشاري محايد.

احكم على قوة الحجج والمنطق.

لا توافق على الأغلبية تلقائياً.

إذا كان رأي الأقلية أقوى،
اختره.

لا تخترع معلومات غير موجودة.
""",

        prompt
    )


# =========================================================
# بناء التقرير الكامل
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

{final.get("text", "لم يتوفر قرار نهائي.")}


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
# تنظيف النص للـ PDF
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
# البحث عن خط عربي
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
# إنشاء PDF عربي
# =========================================================

def create_pdf(
    report_text
):

    font_path = find_arabic_font()


    if not font_path:

        raise RuntimeError(
            "لم يتم العثور على خط عربي على الخادم."
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


    # دعم RTL والعربية
    try:

        pdf.set_text_shaping(
            use_shaping_engine=True,
            direction="rtl",
            script="arab",
            language="ara"
        )

    except Exception:
        pass


    pdf.set_title(
        "MD AI Council Report"
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
# Session State
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
# واجهة التطبيق
# =========================================================

st.title(
    "🧠 مجلس MD للذكاء الاصطناعي"
)


st.caption(
    f"Build: {BUILD_ID}"
)


st.write(
    """
اكتب الموضوع الذي تريد مناقشته.

سيقوم ثلاثة مستشارين مختلفين بتحليله بشكل مستقل،
ثم يقرأ كل مستشار آراء الآخرين ويرد عليها،
ثم يصدر رئيس المجلس التقرير النهائي.
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
# تشغيل الاجتماع
# =========================================================

if start:

    if not question.strip():

        st.warning(
            "اكتب الموضوع أولاً."
        )

    else:

        # مسح الاجتماع السابق
        st.session_state.question_saved = question
        st.session_state.round1 = None
        st.session_state.round2 = None
        st.session_state.final = None
        st.session_state.meeting_error = None


        progress = st.empty()


        # -----------------------------------------
        # الجولة الأولى
        # -----------------------------------------

        progress.info(
            "🧠 الجولة الأولى: المستشارون يفكرون..."
        )


        round1 = first_round(
            question
        )


        st.session_state.round1 = round1


        successful_round1 = valid_results(
            round1
        )


        if len(successful_round1) < 2:

            st.session_state.meeting_error = (
                "لم ينجح عدد كافٍ من المستشارين "
                "في الجولة الأولى. "
                "افتح التفاصيل التقنية أدناه لمعرفة السبب."
            )


            progress.error(
                "❌ فشلت الجولة الأولى."
            )


        else:

            # -----------------------------------------
            # الجولة الثانية
            # -----------------------------------------

            progress.info(
                "⚔️ الجولة الثانية: "
                "المستشارون يقرأون آراء بعضهم..."
            )


            round2 = debate_round(
                question,
                round1
            )


            st.session_state.round2 = round2


            successful_round2 = valid_results(
                round2
            )


            if len(successful_round2) < 2:

                st.session_state.meeting_error = (
                    "لم ينجح عدد كافٍ من المستشارين "
                    "في جولة النقاش. "
                    "لن نصدر تقريراً ناقصاً."
                )


                progress.error(
                    "❌ فشلت جولة النقاش."
                )


            else:

                # -----------------------------------------
                # رئيس المجلس
                # -----------------------------------------

                progress.info(
                    "🏛️ رئيس المجلس يراجع النقاش..."
                )


                final = final_judge(
                    question,
                    round1,
                    round2
                )


                if final.get("ok"):

                    st.session_state.final = final


                    progress.success(
                        "✅ انتهى الاجتماع بنجاح."
                    )


                else:

                    st.session_state.meeting_error = (
                        "نجح اجتماع المستشارين، "
                        "لكن رئيس المجلس لم يتمكن "
                        "من إصدار التقرير."
                    )


                    progress.error(
                        "❌ تعذر إصدار التقرير النهائي."
                    )


# =========================================================
# الخطأ العام
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
                    f'النموذج: {result["model"]}'
                )


            else:

                st.warning(
                    "تعذر الحصول على رد "
                    "من هذا المستشار."
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
# عرض الجولة الثانية
# =========================================================

if st.session_state.round2:

    st.divider()


    st.header(
        "2️⃣ ⚔️ النقاش"
    )


    for name, result in st.session_state.round2.items():

        with st.expander(
            f"{name} يرد على المجلس",
            expanded=False
        ):


            if result.get("ok"):

                st.markdown(
                    result["text"]
                )


                st.caption(
                    f'النموذج: {result["model"]}'
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
# القرار النهائي
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
        "رئيس المجلس استخدم: "
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


    # -----------------------------------------
    # TXT
    # -----------------------------------------

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


    # -----------------------------------------
    # PDF
    # -----------------------------------------

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
            "تعذر إنشاء PDF حالياً."
        )


        with st.expander(
            "🔧 سبب مشكلة PDF"
        ):

            st.code(
                str(e)
            )


# =========================================================
# زر إعادة البداية
# =========================================================

if (
    st.session_state.round1
    or st.session_state.final
):

    st.divider()


    if st.button(
        "🗑️ مسح النتيجة وبدء اجتماع جديد"
    ):

        for key, value in DEFAULT_STATE.items():

            st.session_state[key] = value


        st.rerun()
