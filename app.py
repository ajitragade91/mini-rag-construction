import streamlit as st
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

# Load PDFs (make sure PDFs are in root)
def load_pdf(path):
    reader = PdfReader(path)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text

doc1 = load_pdf("policy.pdf")
doc2 = load_pdf("faq.pdf")
doc3 = load_pdf("specs.pdf")

documents = doc1 + "\n" + doc2 + "\n" + doc3

# Chunking
def chunk_text(text, chunk_size=500, overlap=100):
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunks.append(" ".join(words[start:end]))
        start += chunk_size - overlap
    return chunks

chunks = chunk_text(documents)

# Embeddings
model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = model.encode(chunks)

# FAISS
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(np.array(embeddings))

def retrieve_chunks(query, k=3):
    query_embedding = model.encode([query])
    _, indices = index.search(np.array(query_embedding), k)
    return [chunks[i] for i in indices[0]]

# Grounded Answer
def generate_answer(context):
    if not context.strip():
        return "I don't know based on the provided documents."
    return "Based on the retrieved documents:\n\n" + context

# Streamlit UI
st.set_page_config(page_title="Mini RAG – Construction Assistant")
st.title("🏗️ Mini RAG – Construction Assistant")

query = st.text_input("Ask a question")

if query:
    retrieved = retrieve_chunks(query)
    context = "\n\n".join(retrieved)
    answer = generate_answer(context)

    st.subheader("📄 Retrieved Document Chunks")
    for i, chunk in enumerate(retrieved, 1):
        st.write(f"**Chunk {i}:** {chunk}")

    st.subheader("🤖 Final Answer")
    st.write(answer)
