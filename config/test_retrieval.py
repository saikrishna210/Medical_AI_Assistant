import faiss
import pandas as pd
from sentence_transformers import SentenceTransformer

# Load model
model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2"
)

# Load FAISS index
index = faiss.read_index(
    "ai_engine/data/medquad_faiss.index"
)

# Load metadata
metadata = pd.read_csv(
    "ai_engine/data/medquad_metadata.csv"
)

# Test question
query = "I have headache, fever and bodypain ?"

# Generate embedding
query_embedding = model.encode(
    [query],
    normalize_embeddings=True,
    convert_to_numpy=True
).astype("float32")

# Search top 5 results
scores, indices = index.search(query_embedding, 5)

print("\nMEDICAL RETRIEVAL RESULTS\n")

for score, idx in zip(scores[0], indices[0]):
    row = metadata.iloc[int(idx)]
    print("Similarity:", round(float(score), 4))
    print("Focus:", row["focus"])
    print("Question:", row["question"])
    print("Chunk:", row["chunk_text"])
    print("-" * 60)