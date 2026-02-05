from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from src.search import RAGSearch
import uvicorn

from fastapi.middleware.cors import CORSMiddleware
# -------------------------
# FastAPI App
# -------------------------
app = FastAPI(
    title="RAG Question Answering API",
    description="FAISS + SentenceTransformers + Groq LLM",
    version="1.0.0"
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Allows your React app to connect
    allow_credentials=True,
    allow_methods=["*"], 
    allow_headers=["*"],
)
# -------------------------
# Load RAG ONCE (startup)
# -------------------------
rag_search: RAGSearch | None = None


@app.on_event("startup")
def load_rag():
    global rag_search
    try:
        rag_search = RAGSearch(
            persist_dir="faiss_store",
            embedding_model="all-MiniLM-L6-v2",
            llm_model="llama-3.1-8b-instant"
        )
        print("[INFO] RAG system loaded successfully")
    except Exception as e:
        print(f"[ERROR] Failed to load RAG system: {e}")
        raise


# -------------------------
# Request / Response Models
# -------------------------
class QueryRequest(BaseModel):
    query: str
    top_k: int = 3


class QueryResponse(BaseModel):
    query: str
    answer: str


# -------------------------
# Routes
# -------------------------
@app.get("/")
def root():
    return {"message": "RAG API is running. Go to /docs"}


@app.post("/query", response_model=QueryResponse)
def query_rag(payload: QueryRequest):
    if not rag_search:
        raise HTTPException(status_code=503, detail="RAG system not ready")

    try:
        answer = rag_search.search_and_summarize(
            query=payload.query,
            top_k=payload.top_k
        )
        return QueryResponse(
            query=payload.query,
            answer=answer
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# -------------------------
# Run locally
# -------------------------
if __name__ == "__main__":
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )
