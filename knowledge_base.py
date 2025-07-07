# knowledge_base.py

import os
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone, ServerlessSpec
from dotenv import load_dotenv

load_dotenv()

# Step 1: Sample knowledge base
docs = [
    "Pinecone is a vector database for similarity search.",
    "MiniLM is a small transformer model useful for embeddings.",
    "Vector databases help in storing and retrieving embeddings.",
    "Embeddings convert text into numerical format for ML models.",
]

# Step 2: Load embedding model
print("Loading MiniLM model...")
model = SentenceTransformer('all-MiniLM-L6-v2')
embeddings = model.encode(docs).tolist()

# Step 3: Setup Pinecone
PINECONE_API_KEY = os.getenv('PC_DB_KEY')
pc = Pinecone(api_key=PINECONE_API_KEY)

index_name = "local-knowledge-base"

# Check and create index if it doesn't exist
if index_name not in pc.list_indexes().names():
    print("Creating Pinecone index...")
    pc.create_index(
        name=index_name,
        dimension=384,  # output size
        metric='cosine',
        spec=ServerlessSpec(
            cloud='aws',
            region='us-east-1'
        )
    )

# Connect to the index
index = pc.Index(index_name)

# Step 4: Upload vectors
print("Uploading embeddings...")
vectors = [
    {
        "id": str(i),
        "values": embeddings[i],
        "metadata": {"text": docs[i]}
    }
    for i in range(len(docs))
]
index.upsert(vectors=vectors)

# Step 5: Query loop
print("\n--- Knowledge Base Ready ---")
while True:
    query = input("\nAsk a question (or type 'exit'): ")
    if query.lower() == "exit":
        break

    query_vector = model.encode(query).tolist()
    result = index.query(vector=query_vector, top_k=3, include_metadata=True)

    print("\nTop Matches:")
    for match in result['matches']:
        print(f"- Score: {match['score']:.2f}")
        print(f"  Answer: {match['metadata']['text']}")
