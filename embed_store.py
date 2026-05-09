from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import pickle

# Load embedding model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Load questions
with open('data/questions.txt', 'r', encoding='utf-8') as f:
    questions = f.readlines()

questions = [q.strip() for q in questions if q.strip()]

# Convert questions into embeddings
embeddings = model.encode(questions)

# Create FAISS index
dimension = embeddings.shape[1]

index = faiss.IndexFlatL2(dimension)

index.add(np.array(embeddings))

# Save FAISS index
faiss.write_index(index, 'faiss_index.bin')

# Save questions
with open('questions.pkl', 'wb') as f:
    pickle.dump(questions, f)

print("RAG database created successfully!")