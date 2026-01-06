#  Mini RAG – Construction Assistant

This project implements a **Mini Retrieval-Augmented Generation (RAG)** system for a construction marketplace.
The assistant answers user queries **strictly using internal documents** (policies, FAQs, specifications).

The focus is on **retrieval accuracy, grounding, transparency, and explainability**.


## 1️ Embedding Model and LLM Used

###  Embedding Model
**Model:** `sentence-transformers/all-MiniLM-L6-v2`

Why this model?
- Lightweight and fast (384-dimension vectors)
- Strong semantic similarity performance
- Runs fully offline (no API key required)
- Ideal for local FAISS-based RAG systems

The model converts both document chunks and user queries into vector embeddings for similarity search.

---

###  Large Language Model (LLM)
**LLM Used:** ❌ None (retrieval-only mode)

**Why no LLM?**
- Guarantees **zero hallucination**
- Answers are **100% grounded** in retrieved documents
- Fully satisfies the assignment requirement for grounded responses

> Optional enhancement: A local LLM (Phi / Mistral via Ollama) can be added later for summarization.

---

## 2️ Document Chunking & Retrieval

###  Document Processing
- PDF documents are loaded using **PyPDF**
- Text from all documents is merged into a single corpus

---

###  Chunking Strategy
Documents are split into overlapping chunks:

- **Chunk size:** 500 words
- **Overlap:** 100 words

**Why chunking is required**
- Improves semantic retrieval accuracy
- Prevents loss of contextual information
- Enables efficient vector indexing

```python
def chunk_text(text, chunk_size=500, overlap=100):
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunks.append(" ".join(words[start:end]))
        start += chunk_size - overlap
    return chunks
