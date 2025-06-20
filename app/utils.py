# utils.py
import os
import json
import faiss
import numpy as np
import openai
from dotenv import load_dotenv
import base64
import pytesseract
from PIL import Image
from io import BytesIO
from openai import OpenAI

# Set Hugging Face cache dirs
os.environ['HF_HOME'] = '/tmp/huggingface'
os.environ['TRANSFORMERS_CACHE'] = '/tmp/huggingface/transformers'
os.environ['HF_DATASETS_CACHE'] = '/tmp/huggingface/datasets'
os.environ['HUGGINGFACE_HUB_CACHE'] = '/tmp/huggingface/hub'
os.makedirs('/tmp/huggingface', exist_ok=True)

# Load keys
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_API_BASE = os.getenv("OPENAI_BASE_URL", "https://aipipe.org/openai/v1")

client = OpenAI(
    api_key=OPENAI_API_KEY,
    base_url=OPENAI_API_BASE
)

print("OpenAI key loaded:", bool(OPENAI_API_KEY))

from sentence_transformers import SentenceTransformer

# Load sentence embedding model
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# Load embedding data and FAISS index
with open("app/embedding_data.json", "r", encoding="utf-8") as f:
    embedding_data = json.load(f)

index = faiss.read_index("app/my_index.faiss")

# OCR helper
def extract_text_from_image(image_base64):
    try:
        image_data = base64.b64decode(image_base64)
        image = Image.open(BytesIO(image_data))
        return pytesseract.image_to_string(image)
    except Exception as e:
        return f"[Image OCR failed: {e}]"

# Main answer engine
def retrieve_answer(query, image=None, top_k=3):
    if image:
        ocr_text = extract_text_from_image(image)
        query = query + "\n\nOCR Extracted Text: " + ocr_text

    # Embed query and search
    query_emb = model.encode(query, convert_to_numpy=True)
    query_emb = query_emb / np.linalg.norm(query_emb)
    D, I = index.search(np.array([query_emb.astype("float32")]), top_k)

    results = []
    for score, idx in zip(D[0], I[0]):
        result = embedding_data[idx]
        snippet = result.get("combined_text", "")[:500]
        results.append((score, snippet, result.get("url", "https://discourse.onlinedegree.iitm.ac.in/")))

    context = "\n\n".join([f"{text}" for _, text, _ in results])

    # Chat-style prompt
    messages = [
        {"role": "system", "content": "You are a helpful virtual teaching assistant for the IIT Madras Online BSc degree. Use the context snippets to answer the question as accurately and concisely as possible."},
        {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"}
    ]

    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=messages,
            temperature=0.7,
            max_tokens=500
        )
        answer = response.choices[0].message.content.strip()
    except Exception as e:
        answer = f"LLM error: {str(e)}"

    links = [{"url": url, "text": "Related discussion"} for _, _, url in results]
    return answer, links
