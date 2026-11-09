import faiss
import pandas as pd
index = faiss.read_index(
    "ai_engine/data/medquad_faiss.index"
)
metadata = pd.read_csv(
    "ai_engine/data/medquad_metadata.csv"
)
print("Total FAISS vectors:", index.ntotal)
print("Vector dimensions:", index.d)
print("Metadata rows:", len(metadata))
print("Metadata columns:", metadata.columns.tolist())