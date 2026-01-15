import json
import numpy as np
from sentence_transformers import CrossEncoder
from typing import List, Dict
import pickle
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import chromadb
from langchain_openai import OpenAIEmbeddings
import os
load_dotenv()
os.environ["TOKENIZERS_PARALLELISM"] = "false"

class RAG:
    def __init__(self, model_name: str):
        self.embedding_model = model_name
        self.openai_embeddings = OpenAIEmbeddings(model=model_name)
        self.chunks = None
        self.metadata = None
        self.client = None
        self.collection = None
        self.cross_encoder = None
        self.cross_encoder_name = None
        
    def load(self, filepath: str, model_name: str = None, persist_path: str = 'vector_store'):
        """Load the vector store from disk."""
        with open(f"{filepath}.pkl", 'rb') as f:
            data = pickle.load(f)
        
        self.chunks = data['chunks']
        self.metadata = data['metadata']
        self.client = chromadb.PersistentClient(path=persist_path)

        print(f"Using OpenAI embeddings model: {model_name}")
        self.embedding_model = OpenAIEmbeddings(model=model_name)
        self.collection = self.client.get_or_create_collection(name=f"courses_openai")

        print(f"Vector store loaded from Chroma at '{persist_path}' with embeddings='{model_name}' and metadata from {filepath}.pkl")


    def retrieve(self, query: str, top_k: int = 10, rerank_top_n: int | None = None, rerank_min_score: float | None = None, rerank_model_name: str | None = None) -> List[Dict]:
        """Retrieve relevant information."""
        query_embedding = [self.embedding_model.embed_query(query)]

        query_embedding = np.array(query_embedding)
        query_embedding = query_embedding / np.linalg.norm(query_embedding, axis=1, keepdims=True)

        combined: List[Dict] = []

        cb_q = self.collection.query(
            query_embeddings=query_embedding.tolist(),
            n_results=max(1, top_k),
            include=["documents", "metadatas", "distances"]
        )          
        cb_docs = cb_q.get('documents', [[]])[0]
        cb_metas = cb_q.get('metadatas', [[]])[0]
        cb_dists = cb_q.get('distances', [[]])[0]            

        for i in range(len(cb_docs)):
            distance = float(cb_dists[i]) if i < len(cb_dists) else 0.0
            similarity = 1.0 - distance
            combined.append({
                'text': cb_docs[i],
                'metadata': cb_metas[i] if i < len(cb_metas) else {},
                'similarity_score': similarity,
                'rank': len(combined) + 1,
            })

        if rerank_top_n is not None or rerank_min_score is not None or rerank_model_name is not None:
            combined = self.rerank(
                query = query,
                items = combined,
                model_name = rerank_model_name,
                top_n = rerank_top_n,
                min_score = rerank_min_score,
            )

        return combined
    
    def rerank(self, query: str, items: List[Dict], model_name: str | None = None, top_n: int | None = None, min_score: float | None = None) -> List[Dict]: 
        """Rerank retrieved items with a CrossEncoder.

        - model_name: Hugging Face cross-encoder model, defaults to 'cross-encoder/ms-marco-MiniLM-L-6-v2'
        - top_n: if provided, keep only the top N items by rerank score
        - min_score: if provided, filter out items with score < min_score
        """
        if not items:
            return items

        model_to_use = model_name or self.cross_encoder_name or 'cross-encoder/ms-marco-MiniLM-L-6-v2'
        if self.cross_encoder is None or self.cross_encoder_name != model_to_use:
            self.cross_encoder = CrossEncoder(model_to_use)
            self.cross_encoder_name = model_to_use

        pairs = [(query, it.get('text', '')) for it in items]
        scores = self.cross_encoder.predict(pairs)

        for it, sc in zip(items, scores):
            try:
                score_float = float(sc)
            except Exception:
                score_float = 0.0
            it['rerank_score'] = score_float

        items.sort(key=lambda x: x.get('rerank_score', 0.0), reverse=True)
        for idx, it in enumerate(items, start=1):
            it['rerank_rank'] = idx

        if min_score is not None:
            items = [it for it in items if it.get('rerank_score', 0.0) >= float(min_score)]
        if top_n is not None:
            items = items[: int(top_n)]

        return items
    
    def generate(self, user_query: str, context: str, max_tokens: int = 1000): 
        model = ChatOpenAI(model="gpt-4o-mini", temperature=0.2, max_tokens=max_tokens)
        prompt = PromptTemplate.from_template(
            """
            You are a helpful assistant that helps students
            with their academic, educational, and general questions about their education and program.
            
            User question: {user_query}
            Context: {context}
            
            Provide a good answer in an organized and structured way. Don't make it shorter or longer than necessary.
            """
        )
        rendered = prompt.format(context=context, user_query=user_query)
        
        for chunk in model.stream(rendered): 
            if chunk.content: 
                yield chunk.content