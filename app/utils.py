import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# Load model, embeddings, index on startup
with open("app/embedding_data.json", "r", encoding="utf-8") as f:
    embedding_data = json.load(f)

index = faiss.read_index("app/my_index.faiss")
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

def retrieve_answer(query, top_k=3):
    query_emb = model.encode(query, convert_to_numpy=True)
    query_emb = query_emb / np.linalg.norm(query_emb)
    D, I = index.search(np.array([query_emb.astype("float32")]), top_k)

    results = []
    for score, idx in zip(D[0], I[0]):
        result = embedding_data[idx]
        snippet = result.get("combined_text", "")[:500]
        results.append((score, snippet, result.get("url", "https://discourse.onlinedegree.iitm.ac.in/")))

    answer_text = "\n\n".join([f"[Score: {s:.4f}]\n{text}" for s, text, _ in results])
    links = [{"url": url, "text": "Related discussion"} for _, _, url in results]
    return answer_text, links
