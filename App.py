from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st
from PyPDF2 import PdfReader

st.title("AI Interview Preparation Assistant")

uploaded_file = st.file_uploader(
    "Upload your Resume (PDF)",
    type=["pdf"]
)

if uploaded_file is not None:

    # Read PDF
    reader = PdfReader(uploaded_file)

    text = ""

    for page in reader.pages:
        page_text = page.extract_text()

        if page_text:
            text += page_text

    # Display Resume Content
    st.subheader("Resume Content")
    st.text_area("Extracted Text", text, height=300)

    # Skill List
    skills = [
        "Python",
        "SQL",
        "Machine Learning",
        "Data Analysis",
        "Java",
        "HTML",
        "CSS"
    ]

    # Detect Skills
    detected_skills = []

    for skill in skills:
        if skill.lower() in text.lower():
            detected_skills.append(skill)

    # Display Skills
    st.subheader("Detected Skills")

    if detected_skills:
        for skill in detected_skills:
            st.success(skill)
    else:
        st.warning("No skills detected")

    # Question Database
    questions = {
        "Python": [
            "What is Python?",
            "What is a list in Python?",
            "What is the difference between a list and a tuple?"
        ],

        "SQL": [
            "What is a primary key?",
            "What is a JOIN?",
            "What is normalization?"
        ],

        "Machine Learning": [
            "What is Machine Learning?",
            "What is overfitting?",
            "What is supervised learning?"
        ]
    }

    # Model Answers
    answers = {
        "What is Python?":
            "Python is a high level programming language.",

        "What is a list in Python?":
            "A list is an ordered and mutable collection.",

        "What is the difference between a list and a tuple?":
            "Lists are mutable while tuples are immutable.",

        "What is a primary key?":
            "A primary key uniquely identifies each record.",

        "What is a JOIN?":
            "A JOIN combines rows from multiple tables.",

        "What is normalization?":
            "Normalization reduces data redundancy.",

        "What is Machine Learning?":
            "Machine Learning enables systems to learn from data.",

        "What is overfitting?":
            "Overfitting occurs when a model learns training data too well.",

        "What is supervised learning?":
            "Supervised learning uses labeled training data."
    }

    st.subheader("Interview Questions")

    total_score = 0
    question_count = 0

    for skill in detected_skills:

        if skill in questions:

            st.markdown(f"### {skill}")

            for question in questions[skill]:

                st.write("•", question)

                answer = st.text_area(
                    f"Answer for: {question}",
                    key=question
                )

                if answer:

                    expected_answer = answers.get(question, "")

                    vectorizer = TfidfVectorizer()

                    vectors = vectorizer.fit_transform(
                        [answer, expected_answer]
                    )

                    similarity = cosine_similarity(
                        vectors[0],
                        vectors[1]
                    )[0][0]

                    score = round(similarity * 100)

                    total_score += score
                    question_count += 1

                    st.write(f"Score: {score}%")

    # Overall Score
    if question_count > 0:

        overall_score = round(
            total_score / question_count
        )

        st.subheader("Interview Summary")

        st.metric(
            "Overall Interview Score",
            f"{overall_score}%"
        )

        if overall_score >= 80:
            st.success(
                "Excellent performance. You are interview ready."
            )

        elif overall_score >= 60:
            st.warning(
                "Good performance. Practice a few more concepts."
            )

        else:
            st.error(
                "Needs improvement. Review the fundamentals."
            )