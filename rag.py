# Attention: Update this function in such a way that it can 
# take the departments/selected keys from the user via UI.
import json
import numpy as np
from sentence_transformers import SentenceTransformer
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

class RAG:
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        self.embedding_model = model_name
        self._backend = 'openai' if model_name.startswith('text-embedding') else 'st'
        if self._backend == 'st':
            self.model = SentenceTransformer(model_name)
            self._openai_embeddings = None
        else:
            self.model = None
            self._openai_embeddings = OpenAIEmbeddings(model=model_name)
        self.chunks = None
        self.metadata = None
        self.client = None
        self.collection = None

    def get_available_keys(self) -> List[str]:
        """Get all available bulletin keys."""
        keys = set(meta['key'] for meta in self.metadata if isinstance(meta, dict) and 'key' in meta and meta.get('key'))
        return sorted(list(keys))
        
    def load(self, filepath: str, model_name: str = None, persist_path: str = 'vector_store_chroma'):
        """Load the vector store from disk."""
        # Load metadata/chunks
        with open(f"{filepath}.pkl", 'rb') as f:
            data = pickle.load(f)
        
        self.chunks = data['chunks']
        self.metadata = data['metadata']
        # Choose embedding model based on persisted name unless overridden
        persisted_model = data.get('embedding_model')
        chosen_model = model_name or persisted_model or os.getenv('EMBEDDING_MODEL', 'all-MiniLM-L6-v2')
        self.embedding_model = chosen_model
        self._backend = 'openai' if chosen_model.startswith('text-embedding') else 'st'
        if self._backend == 'st':
            print(f"Using SentenceTransformer model: {chosen_model}")
            self.model = SentenceTransformer(chosen_model)
            self._openai_embeddings = None
        else:
            self.model = None
            print(f"Using OpenAI embeddings model: {chosen_model}")
            self._openai_embeddings = OpenAIEmbeddings(model=chosen_model)
        self.client = chromadb.PersistentClient(path=persist_path)
        safe_model = __import__('re').sub(r'[^a-zA-Z0-9_]+', '_', self.embedding_model).lower()
        self.collection_name = f"courses_{safe_model}"
        self.collection = self.client.get_or_create_collection(name=self.collection_name)
        
        print(f"Vector store loaded from Chroma at '{persist_path}' with embeddings='{self.embedding_model}' and metadata from {filepath}.pkl")


    def retrieve(self, query: str, bulletin_department: str = None, cab_department: str = None, top_k_bulletin: int = 5, top_k_cab: int = 5) -> List[Dict]:
        """Retrieve from bulletin and cab separately using provided department codes and merge results."""
        # Embed once
        if self._backend == 'st':
            query_embedding = self.model.encode([query], convert_to_tensor=False)
        else:
            query_embedding = [self._openai_embeddings.embed_query(query)]
        query_embedding = np.array(query_embedding)
        query_embedding = query_embedding / np.linalg.norm(query_embedding, axis=1, keepdims=True)

        combined: List[Dict] = []

        if bulletin_department:
            b_q = self.collection.query(
                query_embeddings=query_embedding.tolist(),
                n_results=max(1, top_k_bulletin),
                include=["documents", "metadatas", "distances"],
                where={"$and": [
                    {"key": {"$eq": bulletin_department}},
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
                    {"key": {"$eq": cab_department}},
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

        return combined
    
    def generate(self, user_query, context): 
        model = ChatOpenAI(model="gpt-4o-mini", temperature=0.2, max_tokens=1000,)
        prompt = PromptTemplate.from_template(
            """
            You are a helpful assistant that answers strictly using the provided context.
            
            User question: {user_query}
            Context: {context}
            
            Provide a concise, actionable answer.
            """
        )
        print("Context:")
        print(context)
        print()
        rendered = prompt.format(context=context, user_query=user_query)
        resp = model.invoke(rendered)
        return resp.content

def main(user_question, bulletin_department, cab_department, top_k_bulletin, top_k_cab):
    rag = RAG()
    rag.load("vector_store")
    vector_store = rag
    print(f"Query: {user_question}")
    print(f"Bulletin dept: {bulletin_department} | CAB dept: {cab_department}")
    print()        

    output_file = "vector_store_analysis_results.txt"
    context = ""
    with open(output_file, 'w', encoding='utf-8') as f:
        
        f.write(f"PROCESSING SUMMARY\n")
        f.write("-" * 20 + "\n")
        f.write(f"Total chunks processed: {len(rag.chunks)}\n")
        f.write(f"Available keys: {rag.get_available_keys()}\n\n")
                
        # Example searches
        f.write("SEARCH EXAMPLES\n")
        f.write("-" * 15 + "\n\n")
        f.write(f"USER QUESTION: {user_question}\n")
        f.write("=" * 60 + "\n\n")
        
        # This is how you'd use it in your RAG system
        f.write("RAG SYSTEM SEARCH:\n")
        f.write("-" * 18 + "\n")
        f.write(f"Query: '{user_question}'\n")
        f.write(f"Filters: bulletin={bulletin_department}, cab={cab_department}\n")
        f.write(f"Action: rag.retrieve_dual(query, bulletin_department, cab_department, {top_k_bulletin}, {top_k_cab})\n\n")
        
        # Retrieve separately and combine
        results = rag.retrieve(user_question, bulletin_department, cab_department, top_k_bulletin, top_k_cab)
        
        f.write(f"Retrieved {len(results)} relevant chunks to send to LLM:\n\n")
        
        for i, result in enumerate(results, 1):
            f.write(f"CHUNK {i} (Similarity: {result['similarity_score']:.4f}):\n")
            source_title = ''
            if isinstance(result['metadata'], dict):
                source_title = result['metadata'].get('full_title', result['metadata'].get('key', ''))
            f.write(f"Source: {source_title}\n")
            f.write(f"Content to send to LLM:\n")
            f.write(f"{'=' * 50}\n")
            f.write(f"{result['text']}\n")
            f.write(f"{'=' * 50}\n\n")
            context += result['text'] + "\n"
    
    print(f"Analysis complete! Results saved to '{output_file}'")
    print("You can now open the file to read all the outputs in detail.")
    return rag.generate(user_question, context)
    
# Example usage
if __name__ == "__main__":
    # Provide both department codes explicitly
    user_question = "I am studying Applied Mathematics. Does APMA 1650 count for my requirements?"
    bulletin_department = "apma"  # bulletin code
    cab_department = "APMA"       # cab code
    top_k_bulletin, top_k_cab = (3, 5)
    response = main(user_question, bulletin_department, cab_department, top_k_bulletin, top_k_cab)
    print("User question:")
    print(user_question)
    print()
    print("Response:")
    print(response)