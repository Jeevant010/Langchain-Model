from src.search import RAGSearch

print("Initializing RAG...")
rag = RAGSearch()
query = "What is the capital of France?"
print(f"Querying: {query}")
result = rag.search_and_summarize(query)
print("-" * 20)
print(f"Result:\n{result}")
print("-" * 20)
