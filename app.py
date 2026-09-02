import streamlit as st
from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor, as_completed
from fpdf import FPDF
import arabic_reshaper
from bidi.algorithm import get_display
import os
import re
import glob


# =====================================
# إعداد الصفحة
# =====================================

st.set_page_config(
    page_title="MD AI Council",
    page_icon="🧠",
    layout="wide"
)


# =====================================
# OpenRouter
# =====================================

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=st.secrets["OPENROUTER_API_KEY"],
    timeout=45.0,
    max_retries=1
)


# =====================================
# المستشارون
# =====================================

AGENTS = {
    "🧠 المستشار الاستراتيجي": """
أنت مستشار استراتيجي مستقل.

حلل السؤال بعمق.
ركز على:
- الخيارات
- الفرص
- العواقب طويلة المدى
- أفضل طريقة لاتخاذ القرار

لا توافق على المستخدم لمجرد إرضائه.
إذا كانت الفكرة ضعيفة فقل ذلك بوضوح.
""",

    "😈 الناقد": """
أنت Devil's Advocate.

مهمتك نقد الأفكار وكشف:
- المخاطر
- الأخطاء
- الافتراضات غير المثبتة
- السيناريوهات التي قد تسبب الفشل

لا تعارض لمجرد المعارضة.
اعترض فقط بحجج منطقية.
""",

    "💡 المستشار المبتكر": """
أنت مستشار ابتكار وحلول.

ابحث عن:
- بدائل لم ينتبه لها الآخرون
- حلول أبسط
- فرص جديدة
- طرق مختلفة للوصول إلى الهدف

فكر بشكل عملي وليس خيالياً فقط.
"""
}


# =====================================
# سؤال AI
# =====================================

def ask_ai(system_prompt, user_prompt):

    try:

        response = client.chat.completions.create(
            model="openrouter/free",
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
            max_tokens=1400
        )

        return response.choices[0].message.content

    except Exception as e:

        return (
            "⚠️ تعذر الحصول على رد من هذا المستشار حالياً.\n\n"
            f"السبب التقني: {str(e)[:250]}"
        )


# =====================================
# الجولة الأولى
# =====================================

def first_round(question):

    results = {}

    with ThreadPoolExecutor(max_workers=3) as executor:

        futures = {}

        for name, role in AGENTS.items():

            future = executor.submit(
                ask_ai,
                role,
                question
            )

            futures[future] = name

        for future in as_completed(futures):

            name = futures[future]

            try:
                results[name] = future.result()

            except Exception as e:
                results[name] = f"⚠️ حدث خطأ: {e}"

    # ترتيب النتائج بنفس ترتيب المستشارين
    return {
        name: results.get(name, "لم يصل رد.")
        for name in AGENTS
    }


# =====================================
# جولة النقاش
# =====================================

def debate_round(question, first_answers):

    results = {}

    with ThreadPoolExecutor(max_workers=3) as executor:

        futures = {}

        for name, role in AGENTS.items():

            others = ""

            for other_name, answer in first_answers.items():

                if other_name != name:

                    others += f"""

=======================

رأي {other_name}:

{answer}

"""

            prompt = f"""
السؤال الأصلي:

{question}


أنت الآن في الجولة الثانية من اجتماع استشاري.

هذه آراء المستشارين الآخرين:

{others}


اقرأ حججهم ورد عليها مباشرة.

أريد منك:

1. أين تتفق معهم؟
2. أين تختلف معهم؟
3. ما الافتراضات الضعيفة؟
4. من قدم حجة قوية؟
5. هل غيرت رأيك بعد قراءة كلامهم؟
6. ما توصيتك المحدثة؟

لا تجامل بقية المستشارين.
"""

            future = executor.submit(
                ask_ai,
                role,
                prompt
            )

            futures[future] = name

        for future in as_completed(futures):

            name = futures[future]

            try:
                results[name] = future.result()

            except Exception as e:
                results[name] = f"⚠️ حدث خطأ: {e}"

    return {
        name: results.get(name, "لم يصل رد.")
        for name in AGENTS
    }


# =====================================
# رئيس المجلس
# =====================================

def final_judge(question, round1, round2):

    meeting = ""

    for name in AGENTS:

        meeting += f"""

=================================

{name}

الرأي الأول:

{round1[name]}


الرأي بعد النقاش:

{round2[name]}

"""


    prompt = f"""
أنت رئيس مجلس استشاري مستقل.

السؤال الأصلي:

{question}


هذا محضر اجتماع المستشارين:

{meeting}


لا تكتفِ بتلخيص كلامهم.

قيم الحجج بنفسك وحدد أين كانت الحجج قوية أو ضعيفة.

أصدر تقريراً نهائياً بالعربية يحتوي على:

## الخلاصة التنفيذية

## نقاط الاتفاق

## نقاط الخلاف

## أقوى الحجج

## الافتراضات غير المثبتة

## أهم المخاطر

## البدائل المتاحة

## توصية المجلس النهائية

## ماذا أفعل الآن؟

قدم خطوات عملية واضحة.

## درجة الثقة

ضع درجة من 0 إلى 100 وفسر سبب الدرجة.
"""

    return ask_ai(
        """
أنت رئيس مجلس استشاري.

لا تنحز لأي مستشار.
احكم فقط على جودة المنطق والحجج.

إذا لم تكن المعلومات كافية،
وضح ذلك ولا تخترع معلومات.
""",
        prompt
    )


# =====================================
# إنشاء التقرير
# =====================================

def build_report(question, round1, round2, final):

    text = f"""
MD AI COUNCIL
=================================

السؤال:

{question}


=================================
القرار النهائي
=================================

{final}


=================================
الجولة الأولى
=================================
"""

    for name, answer in round1.items():

        text += f"""

{name}

{answer}

---------------------------------
"""

    text += """

=================================
النقاش بين المستشارين
=================================
"""

    for name, answer in round2.items():

        text += f"""

{name}

{answer}

---------------------------------
"""

    return text


# =====================================
# تنظيف Markdown
# =====================================

def clean_markdown(text):

    text = re.sub(r"[*#>`_]", "", text)
    text = text.replace("✅", "")
    text = text.replace("⚠️", "")
    text = text.replace("🏛️", "")
    text = text.replace("🧠", "")
    text = text.replace("💡", "")
    text = text.replace("😈", "")

    return text


# =====================================
# تجهيز العربي للـ PDF
# =====================================

def rtl_text(text):

    try:
        reshaped = arabic_reshaper.reshape(text)
        return get_display(reshaped)

    except:
        return text


# =====================================
# العثور على خط عربي
# =====================================

def find_arabic_font():

    possible_fonts = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/freefont/FreeSans.ttf"
    ]

    for font in possible_fonts:

        if os.path.exists(font):
            return font

    fonts = glob.glob(
        "/usr/share/fonts/**/*.ttf",
        recursive=True
    )

    for font in fonts:

        if "DejaVuSans" in font:
            return font

    return None


# =====================================
# إنشاء PDF
# =====================================

def create_pdf(report_text):

    font_path = find_arabic_font()

    if not font_path:
        raise Exception(
            "لم أجد خطاً يدعم العربية على الخادم."
        )


    pdf = FPDF(
        orientation="P",
        unit="mm",
        format="A4"
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


    # الشعار إذا رفعته إلى GitHub
    if os.path.exists("logo.png"):

        try:

            pdf.image(
                "logo.png",
                x=75,
                w=60
            )

            pdf.ln(5)

        except:
            pass


    pdf.set_font(
        "Arabic",
        size=18
    )

    pdf.multi_cell(
        0,
        10,
        rtl_text("MD للاستشارات والخدمات العامة"),
        align="R"
    )

    pdf.ln(5)


    clean_text = clean_markdown(
        report_text
    )


    pdf.set_font(
        "Arabic",
        size=11
    )


    # كتابة التقرير سطراً سطراً
    for paragraph in clean_text.split("\n"):

        if paragraph.strip():

            pdf.multi_cell(
                0,
                7,
                rtl_text(paragraph),
                align="R"
            )

        else:

            pdf.ln(3)


    output = pdf.output()

    return bytes(output)


# =====================================
# Session State
# =====================================

if "question_saved" not in st.session_state:
    st.session_state.question_saved = ""

if "round1" not in st.session_state:
    st.session_state.round1 = None

if "round2" not in st.session_state:
    st.session_state.round2 = None

if "final" not in st.session_state:
    st.session_state.final = None


# =====================================
# الواجهة
# =====================================

st.markdown(
    """
<style>

.main {
    direction: rtl;
}

h1, h2, h3, p {
    text-align: right;
}

</style>
""",
    unsafe_allow_html=True
)


# الشعار
if os.path.exists("logo.png"):

    st.image(
        "logo.png",
        width=180
    )


st.title("🧠 مجلس MD للذكاء الاصطناعي")

st.write(
    """
اكتب الموضوع الذي تريد مناقشته.

سيقوم ثلاثة مستشارين بتحليله بشكل مستقل،
ثم يقرأ كل واحد آراء الآخرين ويرد عليها،
ثم يصدر رئيس المجلس تقريراً نهائياً.
"""
)


question = st.text_area(
    "وش تبي المجلس يناقش؟",
    height=170,
    value=st.session_state.question_saved,
    placeholder="""
مثال:

أفكر أبدأ مشروع جديد.
حلل الفكرة والمخاطر والخيارات البديلة.
"""
)


start = st.button(
    "🚀 ابدأ الاجتماع",
    type="primary",
    use_container_width=True
)


# =====================================
# تشغيل المجلس
# =====================================

if start:

    if not question.strip():

        st.warning(
            "اكتب السؤال أو الموضوع أولاً."
        )

    else:

        st.session_state.question_saved = question

        progress = st.empty()


        progress.info(
            "🧠 الجولة الأولى: المستشارون يفكرون..."
        )

        round1 = first_round(question)

        st.session_state.round1 = round1


        progress.info(
            "⚔️ الجولة الثانية: المستشارون يناقشون آراء بعض..."
        )

        round2 = debate_round(
            question,
            round1
        )

        st.session_state.round2 = round2


        progress.info(
            "🏛️ رئيس المجلس يراجع النقاش..."
        )

        final = final_judge(
            question,
            round1,
            round2
        )

        st.session_state.final = final


        progress.success(
            "✅ انتهى الاجتماع"
        )


# =====================================
# عرض النتائج المحفوظة
# =====================================

if st.session_state.round1:

    st.divider()

    st.header("1️⃣ الآراء الأولية")

    cols = st.columns(3)

    for col, (name, answer) in zip(
        cols,
        st.session_state.round1.items()
    ):

        with col:

            st.subheader(name)

            st.markdown(answer)


if st.session_state.round2:

    st.divider()

    st.header("2️⃣ ⚔️ النقاش")

    for name, answer in st.session_state.round2.items():

        with st.expander(
            f"{name} يرد على المجلس",
            expanded=False
        ):

            st.markdown(answer)


if st.session_state.final:

    st.divider()

    st.header(
        "3️⃣ 🏛️ القرار النهائي"
    )

    st.markdown(
        st.session_state.final
    )


    # التقرير الكامل
    report = build_report(
        st.session_state.question_saved,
        st.session_state.round1,
        st.session_state.round2,
        st.session_state.final
    )


    st.divider()

    st.subheader(
        "📥 تحميل التقرير"
    )


    # تحميل Markdown
    st.download_button(
        label="📝 تحميل التقرير كنص",
        data=report.encode("utf-8"),
        file_name="MD_AI_Council_Report.md",
        mime="text/markdown",
        use_container_width=True
    )


    # تحميل PDF
    try:

        pdf_file = create_pdf(report)

        st.download_button(
            label="📄 تحميل التقرير PDF",
            data=pdf_file,
            file_name="MD_AI_Council_Report.pdf",
            mime="application/pdf",
            use_container_width=True
        )

    except Exception as pdf_error:

        st.warning(
            f"تعذر إنشاء PDF حالياً: {pdf_error}"
        )


    if st.button(
        "🗑️ مسح النتيجة وبدء اجتماع جديد"
    ):

        st.session_state.question_saved = ""
        st.session_state.round1 = None
        st.session_state.round2 = None
        st.session_state.final = None

        st.rerun()
