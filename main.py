from src.data_loader import load_all_documents
from src.vectorstore import FaissVectorStore
from src.search import RAGSearch

if __name__ == "__main__":
    
    docs = load_all_documents("Research/data/pdf")
    store = FaissVectorStore("faiss_Store")
    #store.build_from_documents(docs)
    store.load()
    # print(store.query("What is DataBase Management System?"))
    rag_search = RAGSearch()
    query = "What is DataBase Management System?"
    summary = rag_search.search_and_summarize(query=query, top_k=3)
    print("Summary:" , summary)
    