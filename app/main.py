import os
import json
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
print("OpenAI key found:", "OPENAI_API_KEY" in os.environ)

# Initialize FastAPI app
app = FastAPI()

# Enable CORS (optional for Hugging Face Spaces)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Optional Pydantic model (if needed by other clients)
class Query(BaseModel):
    question: str

# Main API endpoint
@app.post("/api/")
async def ask_question(request: Request):
    try:
        # Try reading properly formatted JSON
        body = await request.json()
    except Exception:
        # Fallback if the body is a stringified JSON
        body_text = await request.body()
        try:
            body = json.loads(body_text)
        except Exception as e:
            return {"error": f"Invalid request body: {str(e)}"}

    question = body.get("question", "")
    if not question:
        return {"error": "Missing 'question' field in request."}

    # Get answer using your vector search + LLM logic
    answer, links = retrieve_answer(question)

    return {
        "answer": answer,
        "links": links
    }

# Health check or homepage endpoint
@app.get("/")
def root():
    return {"message": "TDS Virtual TA API is running on main web!"}
