import os
from dataclasses import dataclass
from typing import List, Optional

from dotenv import load_dotenv
from langchain_groq import ChatGroq

from src.vectorstore import FaissVectorStore

load_dotenv()

@dataclass
class RetrievalResult:
    index: int
    distance: float
    text: Optional[str]

class RAGSearch:
    def __init__(
        self,
        persist_dir: str = "faiss_store",
        embedding_model: str = "all-MiniLM-L6-v2",
        llm_model: str = "llama-3.1-8b-instant",
    ):
        # Vector store setup
        self.vectorstore = FaissVectorStore(persist_dir=persist_dir, embedding_model=embedding_model)

        faiss_path = os.path.join(persist_dir, "faiss.index")
        meta_path = os.path.join(persist_dir, "metadata.pkl")

        if not (os.path.exists(faiss_path) and os.path.exists(meta_path)):
            # Build from local 'data' directory if index doesn't exist
            from src.data_loader import load_all_documents
            docs = load_all_documents("data")
            self.vectorstore.build_from_documents(docs)
        else:
            self.vectorstore.load()

        # LLM setup
        groq_api_key = os.getenv("GROQ_API_KEY")
        if not groq_api_key:
            raise ValueError("GROQ_API_KEY missing in environment")
        self.llm_model = llm_model
        self.embedding_model = embedding_model
        self.llm = ChatGroq(groq_api_key=groq_api_key, model_name=llm_model)
        print(f"[INFO] Groq LLM initialized: {llm_model}")

    def retrieve(self, query: str, top_k: int = 5) -> List[RetrievalResult]:
        results = self.vectorstore.query(query_text=query, top_k=top_k)
        out: List[RetrievalResult] = []
        for r in results:
            text = r["metadata"]["texts"] if r.get("metadata") and r["metadata"].get("texts") else None
            out.append(RetrievalResult(index=int(r["index"]), distance=float(r["distance"]), text=text))
        return out

    def summarize(self, query: str, retrieved: List[RetrievalResult]) -> str:
        texts = [r.text for r in retrieved if r.text]
        context = "\n\n".join(texts)
        if not context:
            return "No relevant documents found."
        prompt = (
            f"Summarize the following context for the query: '{query}'. "
            "Quote relevant lines, keep the explanation concise, and include a short final answer.\n\n"
            f"Context:\n{context}\n\nSummary:"
        )
        response = self.llm.invoke(prompt)
        return response.content

    def search_and_summarize(self, query: str, top_k: int = 5) -> str:
        retrieved = self.retrieve(query, top_k=top_k)
        return self.summarize(query, retrieved)

if __name__ == "__main__":
    rag_search = RAGSearch()
    query = "What is Database Management System?"
    summary = rag_search.search_and_summarize(query, top_k=3)
    print("Summary:", summary)