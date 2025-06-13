# utils.py
import os
import json
import faiss
import numpy as np
import openai
from dotenv import load_dotenv
load_dotenv()  # Ensure this is done before reading the env vars

# Set HF cache dir (safe for Hugging Face Spaces)
os.environ['HF_HOME'] = '/tmp/huggingface'
os.makedirs('/tmp/huggingface', exist_ok=True)

# Configure AI‑Pipe (OpenAI proxy)
openai.api_key = os.getenv("OPENAI_API_KEY")
openai.api_base = os.getenv("OPENAI_BASE_URL", "https://aipipe.org/openai/v1")

print("OpenAI key from env:", os.getenv("OPENAI_API_KEY"))


# Load embedding data
with open("app/embedding_data.json", "r", encoding="utf-8") as f:
    embedding_data = json.load(f)

# Load FAISS index
index = faiss.read_index("app/my_index.faiss")

# Load sentence transformer (used for vector search)
from sentence_transformers import SentenceTransformer
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# Retrieve top_k results using vector search + LLM call via AI‑Pipe
def retrieve_answer(query, top_k=3):
    query_emb = model.encode(query, convert_to_numpy=True)
    query_emb = query_emb / np.linalg.norm(query_emb)
    D, I = index.search(np.array([query_emb.astype("float32")]), top_k)

    results = []
    for score, idx in zip(D[0], I[0]):
        result = embedding_data[idx]
        snippet = result.get("combined_text", "")[:500]
        results.append((score, snippet, result.get("url", "https://discourse.onlinedegree.iitm.ac.in/")))

    # Combine top snippets
    context = "\n\n".join([f"{text}" for _, text, _ in results])
    prompt = f"""You are a helpful virtual teaching assistant for an IIT Madras course. Based on the following forum snippets, answer the question concisely.\n\nContext:\n{context}\n\nQuestion: {query}"""

    # AI-Pipe model response
    try:
        response = openai.Completion.create(
            model="text-davinci-003",  # Or whatever AI‑Pipe supports
            prompt=prompt,
            temperature=0.7,
            max_tokens=500
        )
        answer = response.choices[0].text.strip()
    except Exception as e:
        answer = f"LLM error: {str(e)}"

    links = [{"url": url, "text": "Related discussion"} for _, _, url in results]
    return answer, links
