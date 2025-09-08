# Attention: Update this function in such a way that it can 
# take the departments/selected keys from the user via UI.
import json
import numpy as np
from sentence_transformers import SentenceTransformer, CrossEncoder
from typing import List, Dict
from utils import *
import pickle
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import chromadb
from chromadb.config import Settings
from langchain_openai import OpenAIEmbeddings
import os
load_dotenv()
os.environ["TOKENIZERS_PARALLELISM"] = "false"

class RAG:
    def __init__(self, model_name: str):
        self.embedding_model = model_name
        self.backend = 'openai' if model_name.startswith('text-embedding') else 'st'
        if self.backend == 'st':
            self.model = SentenceTransformer(model_name)
            self.openai_embeddings = None
        else:
            self.model = None
            self.openai_embeddings = OpenAIEmbeddings(model=model_name)
        self.chunks = None
        self.metadata = None
        self.client = None
        self.collection = None
        self.cross_encoder = None
        self.cross_encoder_name = None

    def get_available_keys(self) -> List[str]:
        """Get all available bulletin keys."""
        keys = set(meta['department'] for meta in self.metadata if isinstance(meta, dict) and 'department' in meta and meta.get('department'))
        return sorted(list(keys))
        
    def load(self, filepath: str, model_name: str = None, persist_path: str = 'vector_store'):
        """Load the vector store from disk."""
        # Load metadata/chunks
        with open(f"{filepath}.pkl", 'rb') as f:
            data = pickle.load(f)
        
        self.chunks = data['chunks']
        self.metadata = data['metadata']
        self.client = chromadb.PersistentClient(path=persist_path)
        self.backend = 'openai' if model_name.startswith('text-embedding') else 'st'

        if self.backend == 'st':
            print(f"Using SentenceTransformer model: {model_name}")
            self.embedding_model = SentenceTransformer(model_name)
            self.collection = self.client.get_or_create_collection(name=f"courses_st")
        else:
            print(f"Using OpenAI embeddings model: {model_name}")
            self.embedding_model = OpenAIEmbeddings(model=model_name)
            self.collection = self.client.get_or_create_collection(name=f"courses_openai")

        
        print(f"Vector store loaded from Chroma at '{persist_path}' with embeddings='{model_name}' and metadata from {filepath}.pkl")


    def retrieve(self, query: str, bulletin_department: str = None, cab_department: str = None, top_k_bulletin: int = 5, top_k_cab: int = 5, rerank_top_n: int | None = None, rerank_min_score: float | None = None, rerank_model_name: str | None = None) -> List[Dict]:
        """Retrieve from bulletin and cab separately using provided department codes and merge results."""
        # Embed once
        if self.backend == 'st':
            query_embedding = self.embedding_model.encode([query], convert_to_tensor=False)
        else:
            query_embedding = [self.embedding_model.embed_query(query)]

        query_embedding = np.array(query_embedding)
        query_embedding = query_embedding / np.linalg.norm(query_embedding, axis=1, keepdims=True)

        combined: List[Dict] = []

        if bulletin_department:
            b_q = self.collection.query(
                query_embeddings=query_embedding.tolist(),
                n_results=max(1, top_k_bulletin),
                include=["documents", "metadatas", "distances"],
                where={"$and": [
                    {"department": {"$eq": bulletin_department}},
                    {"source": {"$eq": "bulletin"}}
                ]},
            )
            b_docs = b_q.get('documents', [[]])[0]
            b_metas = b_q.get('metadatas', [[]])[0]
            b_dists = b_q.get('distances', [[]])[0]
            
            for i in range(len(b_docs)):
                distance = float(b_dists[i]) if i < len(b_dists) else 0.0
                similarity = 1.0 - distance
                combined.append({
                    'text': b_docs[i],
                    'metadata': b_metas[i] if i < len(b_metas) else {},
                    'similarity_score': similarity,
                    'rank': len(combined) + 1,
                })

        if cab_department:
            c_q = self.collection.query(
                query_embeddings=query_embedding.tolist(),
                n_results=max(1, top_k_cab),
                include=["documents", "metadatas", "distances"],
                where={"$and": [
                    {"department": {"$eq": cab_department}},
                    {"source": {"$eq": "cab"}}
                ]},
            )
            c_docs = c_q.get('documents', [[]])[0]
            c_metas = c_q.get('metadatas', [[]])[0]
            c_dists = c_q.get('distances', [[]])[0]

            for i in range(len(c_docs)):
                distance = float(c_dists[i]) if i < len(c_dists) else 0.0
                similarity = 1.0 - distance
                combined.append({
                    'text': c_docs[i],
                    'metadata': c_metas[i] if i < len(c_metas) else {},
                    'similarity_score': similarity,
                    'rank': len(combined) + 1,
                })

        # Optional cross-encoder reranking
        if rerank_top_n is not None or rerank_min_score is not None or rerank_model_name is not None:
            combined = self.rerank(
                query=query,
                items=combined,
                model_name=rerank_model_name,
                top_n=rerank_top_n,
                min_score=rerank_min_score,
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
            You are a helpful assistant that helps students at Brown University 
            with their academic, educational, and general questions about their education and program.
            
            User question: {user_query}
            Context: {context}
            
            Provide a good answer in an organized and structured way. Don't make it shorter or longer than necessary.
            """
        )
        rendered = prompt.format(context=context, user_query=user_query)
        resp = model.invoke(rendered)
        return resp.content