import os
import re
import glob
import streamlit as st

from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor, as_completed
from fpdf import FPDF


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

    html, body, [class*="css"] {
        direction: rtl;
    }

    .stApp {
        direction: rtl;
    }

    h1, h2, h3, h4, p {
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
    timeout=60.0,
    max_retries=0
)


# =========================================================
# المستشارون
# =========================================================

AGENTS = {

    "🧠 المستشار الاستراتيجي": {

        "models": [
            "z-ai/glm-5.2:free",
            "z-ai/glm-5.1:free"
        ],

        "role": """
أنت مستشار استراتيجي مستقل ومحترف.

حلل سؤال المستخدم من ناحية:

- الهدف الحقيقي
- الخيارات المتاحة
- المزايا والعيوب
- المخاطر
- النتائج قصيرة وطويلة المدى
- الافتراضات التي يجب اختبارها

لا توافق على المستخدم لمجرد إرضائه.

إذا كانت الفكرة ضعيفة قل ذلك بوضوح.

قدم رأياً عملياً يمكن استخدامه في اتخاذ القرار.
"""
    },


    "😈 الناقد": {

        "models": [
            "z-ai/glm-5.1:free",
            "z-ai/glm-4.5-air:free"
        ],

        "role": """
أنت Devil's Advocate وناقد مستقل.

مهمتك ليست الموافقة.

ابحث عن:

- نقاط الضعف
- المخاطر
- الافتراضات غير المثبتة
- المعلومات الناقصة
- أسباب الفشل المحتملة
- التحيز في طريقة التفكير
- الحالات التي تجعل القرار خاطئاً

لا تعترض لمجرد المعارضة.

كل اعتراض يجب أن يكون له سبب منطقي.

وفي النهاية قدم طريقة لتحسين الفكرة.
"""
    },


    "💡 المستشار المبتكر": {

        "models": [
            "z-ai/glm-4.5-air:free",
            "z-ai/glm-5.2:free"
        ],

        "role": """
أنت مستشار ابتكار وحلول.

ابحث عن:

- حلول غير تقليدية
- بدائل لم يفكر بها المستخدم
- طرق أبسط
- طرق أرخص
- فرص مخفية
- طرق اختبار الفكرة قبل الالتزام بها

كن عملياً.

لا تقدم أفكاراً خيالية غير قابلة للتنفيذ.
"""
    }
}


JUDGE_MODELS = [
    "z-ai/glm-5.2:free",
    "z-ai/glm-5.1:free"
]


# =========================================================
# أدوات مساعدة
# =========================================================

def privacy_error(error_text):

    text = str(error_text).lower()

    keywords = [
        "guardrail restrictions",
        "data policy",
        "no endpoints available",
        "privacy"
    ]

    return any(
        keyword in text
        for keyword in keywords
    )


def ask_models(
    models,
    system_prompt,
    user_prompt
):

    errors = []

    for model in models:

        try:

            response = client.chat.completions.create(
                model=model,

                messages=[
                    {
                        "role": "system",
                        "content": system_prompt
                    },
                    {
                        "role": "user",
                        "content": user_prompt
                    }
                ],

                max_tokens=1600
            )


            content = (
                response
                .choices[0]
                .message
                .content
            )


            # منع None والإجابات الفارغة
            if (
                isinstance(content, str)
                and content.strip()
            ):

                return {
                    "ok": True,
                    "text": content.strip(),
                    "model": model,
                    "error": None
                }


            errors.append(
                f"{model}: empty response"
            )


        except Exception as e:

            errors.append(
                f"{model}: {str(e)}"
            )


    combined_error = " | ".join(errors)


    if privacy_error(combined_error):

        message = (
            "تعذر تشغيل هذا المستشار بسبب "
            "إعدادات الخصوصية أو مزودي OpenRouter المتاحين."
        )

    else:

        message = (
            "تعذر الحصول على رد من هذا المستشار حالياً."
        )


    return {
        "ok": False,
        "text": message,
        "model": None,
        "error": combined_error
    }


def valid_results(results):

    if not results:
        return []

    return [
        result
        for result in results.values()
        if result.get("ok") is True
    ]


# =========================================================
# الجولة الأولى
# =========================================================

def first_round(question):

    results = {}


    def run_agent(name, config):

        result = ask_models(
            config["models"],
            config["role"],
            question
        )

        return name, result


    with ThreadPoolExecutor(
        max_workers=len(AGENTS)
    ) as executor:

        futures = [

            executor.submit(
                run_agent,
                name,
                config
            )

            for name, config
            in AGENTS.items()
        ]


        for future in as_completed(futures):

            try:

                name, result = future.result()

                results[name] = result

            except Exception as e:

                results["مستشار غير معروف"] = {
                    "ok": False,
                    "text": "حدث خطأ أثناء تشغيل المستشار.",
                    "model": None,
                    "error": str(e)
                }


    # إعادة الترتيب
    ordered = {}

    for name in AGENTS:

        ordered[name] = results.get(
            name,
            {
                "ok": False,
                "text": "لم يصل رد.",
                "model": None,
                "error": "No result"
            }
        )

    return ordered


# =========================================================
# الجولة الثانية
# =========================================================

def debate_round(
    question,
    round1
):

    results = {}


    successful_first_round = {

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


        for other_name, result in successful_first_round.items():

            if other_name == name:
                continue


            others += f"""

=================================

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
                "لم يتم الحصول على رأي سابق منك."
            )


        prompt = f"""
السؤال الأصلي:

{question}


رأيك السابق:

{previous_text}


هذه آراء المستشارين الآخرين:

{others}


أنت الآن في الجولة الثانية.

ناقش المستشارين الآخرين فعلياً.

أجب عن التالي:

1. ما الذي تتفق معهم فيه؟

2. ما الذي تختلف معهم فيه؟

3. ما الأخطاء أو الافتراضات الضعيفة؟

4. من قدم أقوى حجة؟ ولماذا؟

5. هل غيرت رأيك بعد قراءة الآخرين؟

6. ما توصيتك المحدثة الآن؟

لا تجامل المستشارين الآخرين.

إذا كانت إحدى حججهم خاطئة،
اشرح سبب الخطأ بوضوح.
"""


        result = ask_models(
            config["models"],
            config["role"],
            prompt
        )


        return name, result


    with ThreadPoolExecutor(
        max_workers=len(AGENTS)
    ) as executor:

        futures = [

            executor.submit(
                run_agent,
                name,
                config
            )

            for name, config
            in AGENTS.items()
        ]


        for future in as_completed(futures):

            try:

                name, result = future.result()

                results[name] = result

            except Exception as e:

                results["مستشار غير معروف"] = {
                    "ok": False,
                    "text": "حدث خطأ أثناء النقاش.",
                    "model": None,
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
                "error": "No result"
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

=================================

{name}

الرأي الأول:

{first_text}


الرأي بعد النقاش:

{second_text}

"""


    prompt = f"""
أنت رئيس مجلس استشاري مستقل.

السؤال الأصلي:

{question}


هذا محضر اجتماع المستشارين:

{meeting}


مهم جداً:

لا تفترض أن المستشارين على حق.

لا تكتفِ بتلخيص كلامهم.

قيّم الحجج بنفسك.

إذا كانت المعلومات غير كافية،
قل ذلك بوضوح.

لا تخترع أرقاماً أو حقائق غير موجودة.


أصدر التقرير بالترتيب التالي:


# الخلاصة التنفيذية

اشرح الوضع في عدة أسطر واضحة.


# نقاط الاتفاق

ما الأشياء التي اتفق عليها المستشارون؟


# نقاط الخلاف

أين اختلفوا؟


# أقوى الحجج

ما أقوى الحجج ولماذا؟


# الافتراضات غير المثبتة

ما المعلومات التي نحتاج التأكد منها؟


# أهم المخاطر

رتب المخاطر حسب أهميتها.


# البدائل المتاحة

اذكر الخيارات الواقعية.


# توصية المجلس النهائية

اختر أفضل قرار بناءً على المعلومات الحالية.


# ماذا أفعل الآن؟

قدم خطوات عملية واضحة ومرتبة.


# درجة الثقة

ضع درجة من 0 إلى 100.

اشرح لماذا اخترت هذه الدرجة.
"""


    return ask_models(
        JUDGE_MODELS,

        """
أنت رئيس مجلس استشاري محايد.

وظيفتك الحكم بين المستشارين
وليس تكرار كلامهم.

احكم بناءً على:

- جودة المنطق
- قوة الأدلة
- المخاطر
- المعلومات المتاحة
- المعلومات الناقصة

لا تنحز لأي نموذج أو مستشار.
""",

        prompt
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

            report += "لم يتوفر رد من هذا المستشار."


        report += "\n\n----------------------------------------"


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

            report += "لم يتوفر رد من هذا المستشار."


        report += "\n\n----------------------------------------"


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


    # إزالة إيموجي قد لا يدعمه الخط
    emojis = [
        "🧠",
        "😈",
        "💡",
        "🏛️",
        "⚔️",
        "✅",
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

        (
            "/usr/share/fonts/"
            "truetype/noto/"
            "NotoSansArabic-Regular.ttf"
        ),

        (
            "/usr/share/fonts/"
            "truetype/noto/"
            "NotoNaskhArabic-Regular.ttf"
        ),

        (
            "/usr/share/fonts/"
            "truetype/dejavu/"
            "DejaVuSans.ttf"
        )
    ]


    for path in candidates:

        if os.path.exists(path):

            return path


    fonts = glob.glob(
        "/usr/share/fonts/**/*.ttf",
        recursive=True
    )


    preferred_words = [
        "arabic",
        "naskh",
        "dejavusans"
    ]


    for font in fonts:

        lower = font.lower()

        if any(
            word in lower
            for word in preferred_words
        ):

            return font


    return None


# =========================================================
# إنشاء PDF عربي
# =========================================================

def create_pdf(report_text):

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


    # دعم عربي و RTL حقيقي
    pdf.set_text_shaping(
        use_shaping_engine=True,
        direction="rtl",
        script="arab",
        language="ara"
    )


    pdf.set_title(
        "MD AI Council Report"
    )


    paragraphs = text.split("\n")


    for paragraph in paragraphs:

        paragraph = paragraph.strip()


        if not paragraph:

            pdf.ln(3)

            continue


        # عناوين بسيطة
        is_heading = (
            paragraph in [
                "MD AI COUNCIL",
                "المجلس الاستشاري للذكاء الاصطناعي",
                "السؤال",
                "القرار النهائي",
                "الآراء الأولية",
                "جولة النقاش"
            ]
        )


        if is_heading:

            pdf.set_font(
                "Arabic",
                size=16
            )

            line_height = 10

        else:

            pdf.set_font(
                "Arabic",
                size=11
            )

            line_height = 7


        pdf.multi_cell(
            w=0,
            h=line_height,
            text=paragraph,
            align="R",
            new_x="LMARGIN",
            new_y="NEXT"
        )


    output = pdf.output()

    return bytes(output)


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
# الواجهة
# =========================================================

st.title(
    "🧠 مجلس MD للذكاء الاصطناعي"
)


st.write(
    """
اكتب الموضوع الذي تريد مناقشته.

سيقوم ثلاثة مستشارين بتحليله بشكل مستقل،
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
        "عندي فكرة مشروع وأريد تقييمها "
        "من ناحية الجدوى والمخاطر والبدائل."
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

        # مسح نتائج الاجتماع السابق
        st.session_state.question_saved = question

        st.session_state.round1 = None

        st.session_state.round2 = None

        st.session_state.final = None

        st.session_state.meeting_error = None


        progress = st.empty()


        # -----------------------------------------
        # ROUND 1
        # -----------------------------------------

        progress.info(
            "🧠 الجولة الأولى: المستشارون يفكرون..."
        )


        round1 = first_round(
            question
        )


        st.session_state.round1 = round1


        if len(valid_results(round1)) < 2:

            st.session_state.meeting_error = (
                "لم ينجح عدد كافٍ من المستشارين "
                "في الجولة الأولى. "
                "لن نصدر قراراً ناقصاً."
            )

            progress.error(
                "❌ فشلت الجولة الأولى."
            )


        else:

            # -----------------------------------------
            # ROUND 2
            # -----------------------------------------

            progress.info(
                "⚔️ الجولة الثانية: "
                "المستشارون يراجعون آراء بعضهم..."
            )


            round2 = debate_round(
                question,
                round1
            )


            st.session_state.round2 = round2


            if len(valid_results(round2)) < 2:

                st.session_state.meeting_error = (
                    "لم ينجح عدد كافٍ من المستشارين "
                    "في جولة النقاش. "
                    "لن نصدر قراراً نهائياً ناقصاً."
                )

                progress.error(
                    "❌ فشلت جولة النقاش."
                )


            else:

                # -----------------------------------------
                # JUDGE
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
                        "✅ انتهى الاجتماع بنجاح"
                    )

                else:

                    st.session_state.meeting_error = (
                        "نجح النقاش ولكن رئيس المجلس "
                        "لم يتمكن من إصدار التقرير."
                    )

                    progress.error(
                        "❌ تعذر إصدار التقرير النهائي."
                    )


# =========================================================
# عرض الخطأ العام
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
        len(AGENTS)
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
                    result["text"]
                )


# =========================================================
# عرض النقاش
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
                    result["text"]
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


    # =====================================================
    # التقرير
    # =====================================================

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
            "تعذر تجهيز PDF حالياً.\n\n"
            f"السبب: {str(e)}"
        )


# =========================================================
# مسح الاجتماع
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
