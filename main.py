"""
Simple CLI test runner for RAG retrieval + summarization.
Run: python main.py
"""

from src.search import RAGSearch

if __name__ == "__main__":
    rag_search = RAGSearch(
        persist_dir="faiss_store",
        embedding_model="all-MiniLM-L6-v2",
        llm_model="llama-3.1-8b-instant"
    )
    query = "What is Database Management System?"
    answer = rag_search.search_and_summarize(query=query, top_k=3)
    print("Answer:\n", answer)