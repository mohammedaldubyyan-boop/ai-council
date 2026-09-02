import streamlit as st
from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=st.secrets["OPENROUTER_API_KEY"]
)

AGENTS = {
    "🧠 المستشار الاستراتيجي": """
أنت مستشار استراتيجي.
حلل السؤال بعمق.
ركز على الفرص والخيارات والقرار طويل المدى.
لا توافق على المستخدم تلقائياً.
""",

    "😈 الناقد": """
أنت ناقد وDevil's Advocate.
ابحث عن الأخطاء والمخاطر والافتراضات الضعيفة.
اعترض عندما توجد أسباب منطقية.
""",

    "💡 المستشار المبتكر": """
أنت مستشار مبتكر.
ابحث عن حلول بديلة وأفكار جديدة
وزوايا قد لا ينتبه لها الآخرون.
"""
}


def ask_ai(system_prompt, user_prompt):

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
        ]
    )

    return response.choices[0].message.content


def first_round(question):

    def run_agent(item):
        name, role = item

        answer = ask_ai(
            role,
            question
        )

        return name, answer

    with ThreadPoolExecutor(max_workers=3) as executor:
        results = list(
            executor.map(
                run_agent,
                AGENTS.items()
            )
        )

    return dict(results)


def debate_round(question, first_answers):

    def run_agent(item):

        name, role = item

        others = ""

        for other_name, answer in first_answers.items():

            if other_name != name:

                others += f"""
                
رأي {other_name}:

{answer}

---------------------

"""

        prompt = f"""
السؤال الأصلي:

{question}

هذه آراء بقية أعضاء المجلس:

{others}

الآن رد عليهم.

حدد:
- ماذا أصابوا فيه؟
- ماذا أخطأوا فيه؟
- ما الذي تختلف معهم فيه؟
- هل غيرت رأيك؟
- ما توصيتك الآن؟
"""

        answer = ask_ai(
            role,
            prompt
        )

        return name, answer

    with ThreadPoolExecutor(max_workers=3) as executor:

        results = list(
            executor.map(
                run_agent,
                AGENTS.items()
            )
        )

    return dict(results)


def final_judge(question, round1, round2):

    meeting = ""

    for name in AGENTS:

        meeting += f"""

{name}

الرأي الأول:
{round1[name]}

الرأي بعد النقاش:
{round2[name]}

============================

"""

    prompt = f"""
أنت رئيس مجلس استشاري.

السؤال الأصلي:

{question}

هذا نقاش المستشارين:

{meeting}

قيم الحجج بنفسك.

أعطني:

## الخلاصة

## نقاط الاتفاق

## نقاط الخلاف

## أقوى حجة

## المخاطر

## التوصية النهائية

## ماذا أفعل الآن؟

## درجة الثقة
من 0 إلى 100
"""

    return ask_ai(
        """
أنت رئيس مجلس استشاري مستقل.
اقرأ آراء المستشارين واحكم بينهم بناءً على قوة المنطق.
""",
        prompt
    )


st.set_page_config(
    page_title="AI Council",
    page_icon="🧠",
    layout="wide"
)

st.title("🧠 مجلس الذكاء الاصطناعي")

st.write("""
اكتب أي موضوع تريد مناقشته.

سيقوم 3 مستشارين بتحليله،
ثم سيقرأ كل واحد آراء الآخرين ويرد عليها،
ثم يصدر رئيس المجلس القرار النهائي.
""")

question = st.text_area(
    "وش تبي المجلس يناقش؟",
    height=150,
    placeholder="مثال: هل أبدأ مشروع إلكتروني أو أستثمر المبلغ؟"
)

if st.button(
    "🚀 ابدأ الاجتماع",
    type="primary",
    use_container_width=True
):

    if question:

        with st.spinner("🧠 المستشارون يفكرون..."):
            round1 = first_round(question)

        st.header("1️⃣ الآراء الأولية")

        columns = st.columns(3)

        for col, (name, answer) in zip(
            columns,
            round1.items()
        ):

            with col:
                st.subheader(name)
                st.markdown(answer)

        with st.spinner(
            "⚔️ المستشارون يقرأون آراء بعض..."
        ):
            round2 = debate_round(
                question,
                round1
            )

        st.divider()

        st.header("2️⃣ النقاش")

        for name, answer in round2.items():

            with st.expander(
                f"{name} يرد على الآخرين",
                expanded=True
            ):
                st.markdown(answer)

        with st.spinner(
            "🏛️ رئيس المجلس يتخذ القرار..."
        ):

            final = final_judge(
                question,
                round1,
                round2
            )

        st.divider()

        st.header("3️⃣ 🏛️ القرار النهائي")

        st.markdown(final)
