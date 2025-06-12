import os
os.environ['HF_HOME'] = '/tmp/huggingface'  # <--- Add this line at the very top
openai_key = os.getenv("OPENAI_API_KEY")
print("OpenAI key found:", "OPENAI_API_KEY" in os.environ)

from fastapi import FastAPI, Request
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from app.utils import retrieve_answer

app = FastAPI()

# Enable CORS (optional for Hugging Face Spaces)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class Query(BaseModel):
    question: str

@app.post("/api/")
def ask_question(query: Query):
    answer, links = retrieve_answer(query.question)
    return {
        "answer": answer,
        "links": links
    }

@app.get("/")
def root():
    print("OpenAI key found:", "OPENAI_API_KEY" in os.environ)
    return {"message": "TDS Virtual TA API is running on main web!"}
