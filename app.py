import gradio as gr
import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# Load embeddings and index
with open("embedding_data.json", "r", encoding="utf-8") as f:
    embedding_data = json.load(f)

index = faiss.read_index("my_index.faiss")
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

def retrieve_answer(query, top_k=3):
    query_emb = model.encode(query, convert_to_numpy=True)
    query_emb = query_emb / np.linalg.norm(query_emb)
    query_emb = query_emb.astype("float32")
    D, I = index.search(np.array([query_emb]), top_k)

    answers = []
    for score, idx in zip(D[0], I[0]):
        result = embedding_data[idx]
        snippet = result['combined_text'][:500]
        answers.append(f"[Score: {score:.4f}]\n{snippet}\n---")
    return "\n\n".join(answers)

iface = gr.Interface(fn=retrieve_answer,
                     inputs="text",
                     outputs="text",
                     title="TDS Virtual TA",
                     description="Ask questions related to Tools in Data Science")

if __name__ == "__main__":
    iface.launch()
