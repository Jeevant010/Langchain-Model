import os
import faiss
import numpy as np
import pickle
from typing import List, Any
from sentence_transformers import SentenceTransformer
from src.embedding import EmbeddingPipeline

class FaissVectorStore:
    def __init__(self, persist_dir : str = "faiss_Store", embedding_model : str = "all-MiniLM-L6-v2", chunk_size : int = 1000, chunk_overlap : int = 200 ):
        self.persist_dir = persist_dir
        os.makedirs(self.persist, exist_ok=True)
        self.index = None
        self.metadata = []
        self.embedding_model = embedding_model
        self.model = SentenceTransformer(embedding_model)
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        print(f"[INFO] Loaded embedding model: {embedding_model}")
        
    def build_from_documents(self, documents: List[Any]):
        print(f"[INFO] Building vector store {len(documents)} raw document...")
        emp_pipe = EmbeddingPipeline(model_name=self.embedding_model, chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap)
        chunks = emb_pipe.embed_chunks(chunks)
        embeddings = emb_pipe.embed_chunks(chunks)
        metadatas = [{"texts": chunk.page_content} for chunk in chunks]
        self.add_embeddings(np.array(embeddings).astype('float32'), metadatas)
        self.save()
        print(f"[INFO] Vector Store built and saved to {self.persist_dir}")
        
    def add_embedding(self, embedding : np.ndarray, metadatas : List[Any] = None):
        dim = embedding.shape[1]
        if self.index is None:
            self.index = faiss.IndexFlatL2(dim)
        self.index.add(embedding)
        