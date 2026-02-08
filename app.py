import streamlit as st
import pickle
import docx
import PyPDF2
import re
import numpy as np

# -------------------------------
# Load trained model and encoders
# -------------------------------
svc_model = pickle.load(open('clf.pkl', 'rb'))
tfidf = pickle.load(open('tfidf.pkl', 'rb'))
le = pickle.load(open('encoder.pkl', 'rb'))


# -------------------------------
# Helper Functions
# -------------------------------
def clean_resume(txt):
    clean_text = re.sub(r'http\S+\s', ' ', txt)
    clean_text = re.sub(r'RT|cc', ' ', clean_text)
    clean_text = re.sub(r'#\S+\s', ' ', clean_text)
    clean_text = re.sub(r'@\S+', ' ', clean_text)
    clean_text = re.sub(r'[%s]' % re.escape("""!"#$%&'()*+,-./:;<=>?@[]^_`{|}~"""), ' ', clean_text)
    clean_text = re.sub(r'[^\x00-\x7f]', ' ', clean_text)
    clean_text = re.sub(r'\s+', ' ', clean_text)
    return clean_text


def extract_text_from_pdf(file):
    pdf_reader = PyPDF2.PdfReader(file)
    text = ''
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text


def extract_text_from_docx(file):
    doc = docx.Document(file)
    text = ''
    for paragraph in doc.paragraphs:
        text += paragraph.text + '\n'
    return text


def extract_text_from_txt(file):
    try:
        text = file.read().decode('utf-8')
    except UnicodeDecodeError:
        text = file.read().decode('latin-1')
    return text


def handle_file_upload(uploaded_file):
    file_extension = uploaded_file.name.split('.')[-1].lower()
    if file_extension == 'pdf':
        text = extract_text_from_pdf(uploaded_file)
    elif file_extension == 'docx':
        text = extract_text_from_docx(uploaded_file)
    elif file_extension == 'txt':
        text = extract_text_from_txt(uploaded_file)
    else:
        raise ValueError("Unsupported file type. Please upload a PDF, DOCX, or TXT file.")
    return text


# -------------------------------
# Prediction Function (Fixed)
# -------------------------------
def pred(input_resume):
    cleaned_text = clean_resume(input_resume)
    vectorized_text = tfidf.transform([cleaned_text])

    # Fix sparse/dense mismatch
    try:
        predicted_category = svc_model.predict(vectorized_text)
    except ValueError:
        predicted_category = svc_model.predict(vectorized_text.toarray())

    predicted_category_name = le.inverse_transform(predicted_category)
    return predicted_category_name[0]


# -------------------------------
# Resume Guidelines
# -------------------------------
def get_resume_guidelines(category):
    guidelines = {
        "AI Engineer": [
            "✅ Include Machine Learning & Deep Learning projects.",
            "✅ Mention frameworks like TensorFlow, PyTorch, or Scikit-learn.",
            "✅ Highlight experience in Python, Data Preprocessing, and Model Deployment.",
            "✅ Add skills: NLP, Computer Vision, Model Optimization.",
            "✅ Certifications from Coursera, Kaggle, or Google AI are a plus."
        ],
        "Data Analyst": [
            "✅ Showcase skills in SQL, Power BI, or Tableau.",
            "✅ Mention data cleaning, visualization, and EDA projects.",
            "✅ Include tools like Excel, Python (pandas, matplotlib, seaborn).",
            "✅ Highlight business understanding and storytelling skills."
        ],
        "Software Developer": [
            "✅ Add coding skills in Python, Java, or C++.",
            "✅ Include GitHub link with personal or team projects.",
            "✅ Mention frameworks (Django, Flask, React, etc.).",
            "✅ Highlight problem-solving and DSA knowledge."
        ]
    }

    return guidelines.get(category, [
        "✅ Highlight key skills relevant to your role.",
        "✅ Include measurable achievements.",
        "✅ Keep resume concise and structured."
    ])


# -------------------------------
# Learning Resources
# -------------------------------
def get_learning_resources(category):
    resources = {
        "AI Engineer": [
            "📘 *Deep Learning Specialization* by Andrew Ng (Coursera)",
            "📗 *Hands-On Machine Learning* by Aurélien Géron",
            "🌐 Kaggle competitions for practice"
        ],
        "Data Analyst": [
            "📘 *Google Data Analytics Certificate* (Coursera)",
            "📗 *Storytelling with Data* by Cole Nussbaumer",
            "🌐 Learn Power BI / Tableau on YouTube (free resources)"
        ],
        "Software Developer": [
            "📘 *CS50x: Introduction to Computer Science* (Harvard EdX)",
            "📗 *Clean Code* by Robert C. Martin",
            "🌐 LeetCode and HackerRank for practice"
        ]
    }

    return resources.get(category, [
        "📘 Explore free learning materials on Coursera, YouTube, or Kaggle.",
        "📗 Focus on building projects and showcasing them on GitHub."
    ])


# -------------------------------
# New Feature 3: Resume Strength Score
# -------------------------------
def calculate_resume_strength(resume_text, category):
    category_keywords = {
        "AI Engineer": ["machine learning", "deep learning", "pytorch", "tensorflow", "model", "ai", "neural network"],
        "Data Analyst": ["excel", "sql", "tableau", "power bi", "visualization", "data", "pandas"],
        "Software Developer": ["python", "java", "c++", "github", "flask", "django", "project"]
    }

    text = resume_text.lower()
    keywords = category_keywords.get(category, [])
    score = sum(1 for word in keywords if word in text)

    # Normalize to strength levels
    if score >= 5:
        return "🟢 Strong Resume (Excellent)"
    elif 3 <= score < 5:
        return "🟡 Average Resume (Good)"
    else:
        return "🔴 Weak Resume (Needs improvement)"


# -------------------------------
# Streamlit App
# -------------------------------
def main():
    st.set_page_config(page_title="Resume Category Prediction", page_icon="📄", layout="wide")

    st.markdown(
        """
        <h1 style='text-align: center; color: #4CAF50; font-family: Arial, sans-serif;'>
            📄 Resume Screening Application
        </h1>
        <p style='text-align: center; font-size:18px; color: #ccc;'>
            Upload your resume and get the predicted job category instantly 🚀
        </p>
        """,
        unsafe_allow_html=True
    )

    uploaded_file = st.file_uploader("Upload a Resume", type=["pdf", "docx", "txt"])

    if uploaded_file is not None:
        try:
            resume_text = handle_file_upload(uploaded_file)
            st.success("✅ Successfully extracted text from the uploaded resume.")

            if st.checkbox("Show extracted text", False):
                st.text_area("Extracted Resume Text", resume_text, height=300)

            # Predict Category
            st.subheader("Predicted Category")
            category = pred(resume_text)
            st.write(f"🎯 The predicted category of the uploaded resume is: **{category}**")

            # Resume Strength
            st.markdown("---")
            st.subheader("💪 Resume Strength Analysis")
            strength = calculate_resume_strength(resume_text, category)
            st.write(strength)

            # Guidelines
            st.markdown("---")
            st.subheader("🧭 Resume Building Guidelines")
            guidelines = get_resume_guidelines(category)
            for g in guidelines:
                st.markdown(f"- {g}")

            # Resources
            st.markdown("---")
            st.subheader("📚 Recommended Learning Resources")
            resources = get_learning_resources(category)
            for r in resources:
                st.markdown(f"- {r}")

        except Exception as e:
            st.error(f"❌ Error processing the file: {str(e)}")


if __name__ == "__main__":
    main()
