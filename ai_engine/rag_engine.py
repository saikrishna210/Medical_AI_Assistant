
import os
import re
from pathlib import Path

import faiss
import pandas as pd
from groq import Groq
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

from .safety_engine import check_emergency


# --------------------------------------------------
# 1. PROJECT PATHS
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"

INDEX_PATH = DATA_DIR / "medquad_faiss.index"
METADATA_PATH = DATA_DIR / "medquad_metadata.csv"


# --------------------------------------------------
# 2. LOAD ENVIRONMENT VARIABLES
# --------------------------------------------------

PROJECT_ROOT = BASE_DIR.parent

load_dotenv(PROJECT_ROOT / ".env", override=True)

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY is missing. Check your .env file.")


# --------------------------------------------------
# 3. CONFIGURATION
# --------------------------------------------------

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

SIMILARITY_THRESHOLD = 0.50
TOP_K = 5


# --------------------------------------------------
# 4. LOAD EMBEDDING MODEL
# --------------------------------------------------

print("Loading embedding model...", flush=True)

embedding_model = SentenceTransformer(MODEL_NAME)

print("Embedding model loaded successfully.", flush=True)


# --------------------------------------------------
# 5. LOAD FAISS INDEX AND METADATA
# --------------------------------------------------

if not INDEX_PATH.exists():
    raise FileNotFoundError(f"FAISS index not found: {INDEX_PATH}")

if not METADATA_PATH.exists():
    raise FileNotFoundError(f"Metadata file not found: {METADATA_PATH}")

faiss_index = faiss.read_index(str(INDEX_PATH))

metadata = pd.read_csv(METADATA_PATH)

required_columns = ["focus", "question", "chunk_text"]

for column in required_columns:
    if column not in metadata.columns:
        raise ValueError(f"Missing metadata column: {column}")

if faiss_index.ntotal != len(metadata):
    raise ValueError("FAISS index and metadata row counts do not match.")

if faiss_index.d != embedding_model.get_sentence_embedding_dimension():
    raise ValueError("FAISS index and embedding model dimensions do not match.")

if faiss_index.ntotal < TOP_K:
    raise ValueError("FAISS index contains fewer vectors than TOP_K.")


print("FAISS index loaded:", faiss_index.ntotal, flush=True)
print("Metadata rows loaded:", len(metadata), flush=True)


# --------------------------------------------------
# 6. INITIALIZE GROQ
# --------------------------------------------------

client = Groq(api_key=GROQ_API_KEY)

print("Groq client initialized successfully.", flush=True)


# --------------------------------------------------
# 7. MEDICAL SYMPTOM RAG FUNCTION
# --------------------------------------------------

def medical_symptom_rag(user_symptoms):

    print("\n========== RAG STARTED ==========", flush=True)
    print("RAG INPUT:", repr(user_symptoms), flush=True)

    if not user_symptoms or not user_symptoms.strip():
        return "Please enter your symptoms before submitting."

    # ----------------------------------------------
    # STEP 1: EMERGENCY CHECK
    # ----------------------------------------------

    if check_emergency(user_symptoms):
        return (
            "Your symptoms may indicate a medical emergency. "
            "Please seek immediate medical care or contact "
            "local emergency services. Do not delay care while "
            "using this application."
        )

    # ----------------------------------------------
    # STEP 2: GENERATE SYMPTOM EMBEDDING
    # ----------------------------------------------

    query_embedding = embedding_model.encode(
        [user_symptoms],
        convert_to_numpy=True,
        normalize_embeddings=True
    ).astype("float32")

    print("Query embedding shape:", query_embedding.shape, flush=True)

    # ----------------------------------------------
    # STEP 3: SEARCH FAISS INDEX
    # ----------------------------------------------

    scores, indices = faiss_index.search(
        query_embedding,
        TOP_K
    )

    print("FAISS scores:", scores.tolist(), flush=True)
    print("FAISS indices:", indices.tolist(), flush=True)

    # ----------------------------------------------
    # STEP 4: RETRIEVE RELEVANT DOCUMENTS
    # ----------------------------------------------

    retrieved_documents = []

    for score, idx in zip(scores[0], indices[0]):

        if idx < 0:
            continue

        if float(score) >= SIMILARITY_THRESHOLD:

            row = metadata.iloc[int(idx)]

            retrieved_documents.append({
                "focus": str(row["focus"]),
                "question": str(row["question"]),
                "chunk_text": str(row["chunk_text"]),
                "similarity": float(score)
            })

    print(
        "Retrieved document count:",
        len(retrieved_documents),
        flush=True
    )

    for doc in retrieved_documents:
        print(
            "Retrieved topic:",
            doc["focus"],
            "| Similarity:",
            round(doc["similarity"], 4),
            flush=True
        )

    # ----------------------------------------------
    # STEP 5: CHECK RETRIEVAL RESULTS
    # ----------------------------------------------

    if not retrieved_documents:
        return (
            "I could not find sufficiently relevant information "
            "in the medical knowledge base for the symptoms provided. "
            "Please consult a qualified healthcare professional. "
            "This application cannot rule out a medical condition."
        )

    # ----------------------------------------------
    # STEP 6: PREPARE MEDICAL CONTEXT
    # ----------------------------------------------

    context = "\n\n".join(
        f"Topic: {doc['focus']}\n"
        f"Question: {doc['question']}\n"
        f"Information: {doc['chunk_text']}"
        for doc in retrieved_documents
    )

    print("Retrieved context prepared.", flush=True)

    # ----------------------------------------------
    # STEP 7: GENERATE RESPONSE USING GROQ
    # ----------------------------------------------

    print("Sending symptoms and context to Groq...", flush=True)

    response = client.chat.completions.create(
        model="openai/gpt-oss-120b",

        messages=[
              {
    "role": "system",
    "content": """
You are an educational medical information assistant.

Analyze the user's symptoms using the retrieved MedQuAD information.

Generate the response using the following structure:

1. Important Note
Explain that the information is educational and not a diagnosis.

2. Possible Health-Related Reasons
Explain the most relevant medical information retrieved from
the knowledge base. Mention only conditions supported by the
retrieved information. Do not assume a condition is confirmed.

3. Why These Symptoms May Occur
Explain how the relevant retrieved information relates to
the symptoms reported by the user.

4. Recommended Next Steps
Provide safe general guidance and recommend appropriate
professional medical evaluation.

5. When to Seek Emergency Help
Explain important warning signs that require immediate
medical attention.

FORMATTING RULES:

- Use the five section headings exactly as provided.
- Use numbered lists where appropriate.
- Use simple and understandable English.
- Keep each section clearly separated.
- Do not use Markdown tables.
- Do not use unnecessary information.
- Do not invent medical facts.
- Do not prescribe medicines or dosages.
- Never provide a definite diagnosis.
- Do not treat similarity scores as diagnostic confidence.
- If retrieved information is insufficient, clearly explain it.
- End with an educational disclaimer.
"""

            },
            {
                "role": "user",
                "content": f"""
User symptoms and details:

{user_symptoms}

Medical information retrieved from the MedQuAD
knowledge base using FAISS:

{context}

Explain the retrieved information in relation to
the user's symptoms.

Clearly mention if the retrieved information is
insufficient or unrelated.
"""
            }
        ],

        max_completion_tokens=1600
    )

    # ----------------------------------------------
    # STEP 8: EXTRACT AND CLEAN RESPONSE
    # ----------------------------------------------

    answer = response.choices[0].message.content

    if not answer:
        return "Unable to generate a response. Please try again."

    answer = answer.strip()

    return answer.strip()
