import os
import json
import base64
from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from app.utils import retrieve_answer

# Load environment variables
load_dotenv()

# Set Hugging Face cache directory (for Spaces compatibility)
os.environ['HF_HOME'] = '/tmp/huggingface'
os.makedirs('/tmp/huggingface', exist_ok=True)

# Check if OpenAI API key is loaded
openai_key = os.getenv("OPENAI_API_KEY")
print("OpenAI key found:", bool(openai_key))

# Initialize FastAPI app
app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Optional Pydantic model
class Query(BaseModel):
    question: str
    image: str = None  # Optional base64-encoded image

# API endpoint
@app.post("/api/")
async def ask_question(request: Request):
    try:
        body = await request.json()
    except Exception:
        body_text = await request.body()
        try:
            body = json.loads(body_text)
        except Exception as e:
            return {"error": f"Invalid request body: {str(e)}"}

    question = body.get("question", "")
    image_base64 = body.get("image")

    if not question:
        return {"error": "Missing 'question' field in request."}

    # Forward question + image to utils
    answer, links = retrieve_answer(question, image_base64)

    return {
        "answer": answer,
        "links": links
    }

# Health check endpoint
@app.get("/")
def root():
    return {"message": "TDS Virtual TA API is running!"}
