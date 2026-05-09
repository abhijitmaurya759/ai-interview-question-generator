import streamlit as st
import pdfplumber
import requests
import faiss
import pickle
import numpy as np

from sentence_transformers import SentenceTransformer

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(
    page_title="AI Interview Generator",
    layout="wide"
)

st.title("🧠 AI Resume Interview Question Generator")
st.write("RAG + Mistral + Ollama")

# =========================
# LOAD EMBEDDING MODEL
# =========================
@st.cache_resource
def load_embedding_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

embed_model = load_embedding_model()

# =========================
# LOAD FAISS INDEX
# =========================
index = faiss.read_index('faiss_index.bin')

with open('questions.pkl', 'rb') as f:
    questions = pickle.load(f)

# =========================
# EXTRACT PDF TEXT
# =========================
def extract_text_from_pdf(pdf_file):

    text = ""

    with pdfplumber.open(pdf_file) as pdf:

        for page in pdf.pages:

            extracted = page.extract_text()

            if extracted:
                text += extracted + "\n"

    return text


# =========================
# RETRIEVE RELEVANT QUESTIONS
# =========================
def retrieve_questions(resume_text, top_k=5):

    # Convert resume into embedding
    resume_embedding = embed_model.encode([resume_text])

    # Search similar questions
    distances, indices = index.search(
        np.array(resume_embedding),
        top_k
    )

    # Get retrieved questions
    retrieved = [questions[i] for i in indices[0]]

    return retrieved


# =========================
# GENERATE QUESTIONS USING OLLAMA
# =========================
def generate_questions(resume_text, retrieved_questions):

    # Convert retrieved list into text
    context = "\n".join(retrieved_questions)

    prompt = f"""
You are a senior technical interviewer.

Candidate Resume:
{resume_text}

Relevant Interview Knowledge:
{context}

Instructions:
1. Analyze candidate skills and domain.
2. Use retrieved interview knowledge.
3. Generate realistic and professional questions.
4. Avoid generic questions.
5. Adjust difficulty according to experience.

Generate:

## Technical Questions
- 5 technical questions

## Scenario-Based Questions
- 3 scenario-based questions

## Problem-Solving Questions
- 2 analytical questions

## HR Questions
- 2 behavioral questions

Return output in markdown format.
"""

    response = requests.post(
        "http://localhost:11434/api/generate",
        json={
            "model": "mistral",
            "prompt": prompt,
            "stream": False
        }
    )

    return response.json()["response"]


# =========================
# FILE UPLOAD
# =========================
uploaded_file = st.file_uploader(
    "📄 Upload Resume PDF",
    type=["pdf"]
)

# =========================
# MAIN FLOW
# =========================
if uploaded_file:

    st.success("✅ Resume uploaded successfully!")

    # Extract resume text
    resume_text = extract_text_from_pdf(uploaded_file)

    # Show extracted text
    with st.expander("📄 View Extracted Resume Text"):
        st.write(resume_text[:4000])

    # Retrieve relevant interview questions
    retrieved = retrieve_questions(resume_text)

    # Show retrieved questions
    with st.expander("🔍 Retrieved Interview Knowledge"):

        for q in retrieved:
            st.write("-", q)

    # Generate final questions
    if st.button("🚀 Generate Interview Questions"):

        with st.spinner("Generating AI Interview Questions..."):

            final_output = generate_questions(
                resume_text,
                retrieved
            )

        st.subheader("🎯 Personalized Interview Questions")

        st.markdown(final_output)