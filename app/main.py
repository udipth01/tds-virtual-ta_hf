import os
from dotenv import load_dotenv
load_dotenv()

# Set Hugging Face cache directory (especially for Hugging Face Spaces)
os.environ['HF_HOME'] = '/tmp/huggingface'

from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional
from fastapi.middleware.cors import CORSMiddleware
from app.utils import retrieve_answer

app = FastAPI()

# Enable CORS (useful if called from frontend or Promptfoo)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Updated model to include optional image field
class Query(BaseModel):
    question: str
    image: Optional[str] = None  # Promptfoo may send this even if unused

@app.post("/api/")
def ask_question(query: Query):
    print("Received question:", query.question)
    if query.image:
        print("Received image (length):", len(query.image))  # Log if present
    answer, links = retrieve_answer(query.question)
    return {
        "answer": answer,
        "links": links
    }

@app.get("/")
def root():
    return {"message": "TDS Virtual TA API is running on main web!"}
