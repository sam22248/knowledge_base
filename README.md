# 🧠 Local Knowledge Base with Pinecone and MiniLM

This project demonstrates how to build a small semantic search knowledge base using:

- [MiniLM](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) (local embedding model)
- [Pinecone](https://www.pinecone.io) (vector database for similarity search)

Ask questions in natural language and get relevant answers from your indexed documents.

---

## 🚀 Features

- Runs a local model (no GPU required)
- Uploads and indexes vector embeddings in Pinecone
- Interactive query interface
- Uses latest Pinecone SDK (`pinecone>=2.2.0`)

---

## 📦 Requirements

- Python 3.8+
- Pinecone account + API key

Install dependencies:

```bash
pip install sentence-transformers pinecone dotenv
