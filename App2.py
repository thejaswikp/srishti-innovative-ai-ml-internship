import streamlit as st

st.set_page_config(
    page_title="AI Interview Assistant",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>

/* Main page */
.main {
    background-color: #F4F7FC;
}

/* Title */
h1{
    color:#2563EB;
    text-align:center;
}

/* Section headings */
h2,h3{
    color:#1E3A8A;
}

/* Buttons */
.stButton>button{
    background:#2563EB;
    color:white;
    border:none;
    border-radius:10px;
    padding:12px;
    font-size:18px;
    font-weight:bold;
    width:100%;
}

.stButton>button:hover{
    background:#1D4ED8;
}

/* Text area */
textarea{
    border-radius:12px !important;
    border:2px solid #2563EB !important;
}

/* Success boxes */
div[data-testid="stAlert"]{
    border-radius:12px;
}

</style>
""", unsafe_allow_html=True)
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import streamlit as st
import pandas as pd
from PyPDF2 import PdfReader

@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

model = load_model()

# Feedback Function
def generate_feedback(score):

    if score >= 90:
        return (
            " Excellent!",
            "Your answer is very accurate and well explained."
        )

    elif score >= 75:
        return (
            " Good!",
            "Your answer is mostly correct. Try adding a little more detail."
        )

    elif score >= 50:
        return (
            " Average",
            "You understand the basics but should explain the concept more clearly."
        )

    else:
        return (
            " Needs Improvement",
            "Review this topic and practice again."
        )

# Page Title
st.title(" AI Interview Preparation Assistant")

st.caption(
    "Practice Technical Interviews using Artificial Intelligence"
)

st.divider()

# Load Dataset
@st.cache_data
def load_dataset():
    return pd.read_csv(
        "Software Questions.csv",
        encoding="latin1"
    )

df = load_dataset()

st.success("Dataset Loaded Successfully!")

# Session State
if "interview_questions" not in st.session_state:
    st.session_state.interview_questions = None

# Upload Resume
uploaded_file = st.file_uploader(
    "Upload Your Resume (PDF)",
    type=["pdf"]
)

resume_text = ""

if uploaded_file is not None:

    try:
        
        reader = PdfReader(uploaded_file)
        
        for page in reader.pages:
            
            page_text = page.extract_text()
            
            if page_text:
                
                resume_text += page_text
                
    except Exception:
        
        st.error("Unable to read the uploaded PDF.")

    st.subheader("Extracted Resume")

    st.text_area(
        "Resume Text",
        resume_text,
        height=250
    )

# Skill Detection Keywords
skills = {

    "Python": [
        "python",
        "numpy",
        "pandas",
        "flask",
        "django",
        "streamlit"
    ],

    "SQL": [
        "sql",
        "mysql",
        "postgresql",
        "database"
    ],

    "Machine Learning": [
        "machine learning",
        "tensorflow",
        "keras",
        "scikit",
        "scikit-learn",
        "pytorch"
    ],

    "Artificial Intelligence": [
        "artificial intelligence",
        "ai"
    ],

    "Web Development": [
        "html",
        "css",
        "javascript",
        "react"
    ],

    "Java": [
        "java"
    ]
}

# Detect Skills
detected_skills = []

if resume_text != "":

    for skill, keywords in skills.items():

        for keyword in keywords:

            if keyword.lower() in resume_text.lower():

                detected_skills.append(skill)

                break

# Display Skills
if detected_skills:

    st.subheader("Detected Skills")

    for skill in detected_skills:

        st.success(skill)

else:

    if uploaded_file is not None:

        st.warning("No skills detected.")

# Skill -> Category Mapping
skill_category_map = {

    "Python": [
        "General Programming",
        "Languages and Frameworks"
    ],

    "SQL": [
        "Database and SQL",
        "Database Systems"
    ],

    "Machine Learning": [
        "Machine Learning",
        "Artificial Intelligence"
    ],

    "Artificial Intelligence": [
        "Artificial Intelligence"
    ],

    "Web Development": [
        "Web Development",
        "Front-end",
        "Back-end",
        "Full-stack"
    ],

    "Java": [
        "General Programming",
        "Languages and Frameworks"
    ]
}

# Generate Categories
selected_categories = []

for skill in detected_skills:

    if skill in skill_category_map:

        selected_categories.extend(
            skill_category_map[skill]
        )

selected_categories = list(set(selected_categories))

# Display Categories
if selected_categories:

    st.subheader("Interview Categories")

    for category in selected_categories:

        st.success(category)

# Filter Dataset
filtered_questions = df[
    df["Category"].isin(selected_categories)
]

# Generate Questions
if len(filtered_questions) > 0:

    if st.button("Start Interview"):

        st.session_state.interview_questions = filtered_questions.sample(
            min(5, len(filtered_questions))
        )

    if st.session_state.interview_questions is not None:
        
        st.subheader("Interview Questions")
        
        with st.form("interview_form"):
            
            answers = {}
            
            
            for i, row in enumerate(
                st.session_state.interview_questions.itertuples(),
                start=1
            ):
                
                st.markdown(f"## Question {i}")
                
                st.write(row.Question)
                
                
                answers[i] = st.text_area(
                    "Your Answer",
                    key=f"answer_{i}"
                )

            submitted = st.form_submit_button("Submit Answers")
            if submitted:
                
                total_score = 0
                question_count = 0
                
                for i, row in enumerate(
                    st.session_state.interview_questions.itertuples(),
                    start=1
                ):

                    user_answer = answers[i]
                    
                    if user_answer.strip():
                        
                        expected_answer = row.Answer
                        
                        user_embedding = model.encode(user_answer)

                        expected_embedding = model.encode(expected_answer)
                        
                        similarity = cosine_similarity(
                            [user_embedding],
                            [expected_embedding]
                        )[0][0]
                        
                        score = round(similarity * 100, 2)
                        total_score += score
                        question_count += 1
                        
                        st.markdown(f"## Result for Question {i}")
                        
                        st.metric("Score", f"{score}%")
                        
                        title, feedback = generate_feedback(score)
                        
                        if score >= 90:
                            st.success(title)
                        
                        elif score >= 75:
                            st.info(title)
                        
                        elif score >= 50:
                            st.warning(title)
                        
                        else:
                            st.error(title)
                            
                        st.write(feedback)
                        
                        with st.expander("View Expected Answer"):
                            
                            st.write(expected_answer)
                            
                        st.divider()

                if question_count > 0:
                    average_score = round(total_score / question_count, 2)
                    
                    st.subheader("Overall Interview Score")
                    st.metric("Average Score", f"{average_score}%")
                    st.progress(float(average_score) / 100)
                    
                else:
                    st.warning("Please answer at least one question.")
                
                st.subheader("Overall Interview Score")
                
                st.metric("Average Score", f"{average_score}%")

                st.progress(float(average_score) / 100)

            
else:

    if uploaded_file is not None:

        st.error("No interview questions found.")

if st.session_state.interview_questions is not None:
    
    if st.button("Restart Interview"):
        
        st.session_state.interview_questions = None
        st.rerun()