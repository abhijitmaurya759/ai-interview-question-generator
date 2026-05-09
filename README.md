# 🧠 AI Interview Question Generator (RAG + LLM)

An AI-powered interview preparation system that generates personalized interview questions from uploaded resumes using Retrieval-Augmented Generation (RAG).

## 🚀 Features

- Resume PDF upload
- Resume text extraction
- RAG-based retrieval using FAISS
- Personalized interview question generation
- Streamlit UI
- Local LLM inference using Ollama + Mistral

---

## 🛠 Tech Stack

- Python
- Streamlit
- FAISS
- Sentence Transformers
- Ollama
- Mistral
- PDFPlumber

---

## 📸 Screenshots

### Home Page

![Home](screenshots/home.png)

### Generated Questions

![Output](screenshots/output.png)

---

## ⚙️ Installation

### Clone Repository

```bash
git clone https://github.com/abhijitmaurya759/ai-interview-question-generator.git
```

### Create Virtual Environment

```bash
python -m venv venv
```

### Activate Environment

#### Windows

```bash
venv\Scripts\activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

---

## ▶️ Run Ollama

```bash
ollama run mistral
```

---

## ▶️ Run Application

```bash
streamlit run app.py
```

---

## 🔥 Future Improvements

- Voice interviews
- ATS resume scoring
- Company-specific interview preparation
- Multi-agent AI workflow
- Cloud deployment

---

## 👨‍💻 Author

Abhijit