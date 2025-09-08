import os
import re
from typing import Dict, List, Tuple
import numpy as np
from sentence_transformers import SentenceTransformer
import pickle
import json
import chromadb
from chromadb.config import Settings
from utils import format_course, count_tokens
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings

load_dotenv()

def chunk_bulletin(department: str, content: str) -> List[Tuple[str, Dict]]:
    """
    Split content by ### headers and create chunks with metadata.
    
    Args:
        department: The bulletin department (e.g., 'math', 'comp', etc.)
        content: The full content string
        
    Returns:
        List of tuples containing (chunk_text, metadata)
    """
    chunks = []
    metadata = []

    # Split by ### headers
    sections = re.split(r'^###\s*', content, flags=re.MULTILINE)
    
    # The first section is the main content (before any ### headers)
    if sections[0].strip():
        main_content = sections[0].strip()
        section_title =  'Main Content'
        chunk = f"{section_title}\n{main_content}"
        chunks.append(chunk)
        
        metadata.append({
            'department': department,
            'source': 'bulletin',
        })        
    
    # Process sections with ### headers
    for i, section in enumerate(sections[1:], 1):
        if section.strip():
            # Extract the header title (first line after ###)
            lines = section.strip().split('\n')
            header_title = lines[0].strip() if lines else f'Section {i}'
            section_content = '\n'.join(lines[1:]).strip() if len(lines) > 1 else ''
            
            # Only add if there's actual content
            if section_content:
                metadata.append({
                    'department': department,
                    'source': 'bulletin',
                })

                section_title = header_title
                chunk = f"{section_title}\n{section_content}"
                chunks.append(chunk)
    
    return chunks, metadata
        
class VectorStore:
    def __init__(self, model_name: str = None, persist_path: str = 'vector_store'):
        """
        Initialize a Chroma persistent collection and embedding model.
        Chooses backend based on model_name/env EMBEDDING_MODEL.
        """
        self.embedding_model = os.getenv('EMBEDDING_MODEL', '')
        self.backend = 'openai' if self.embedding_model.startswith('text-embedding') else 'st' if self.embedding_model else 'st'
        if self.backend == 'st':
            print(f"Using SentenceTransformer model: {self.embedding_model}")
            self.model = SentenceTransformer(self.embedding_model)
            self.openai_embeddings = None
        else:
            print(f"Using OpenAI embeddings model: {self.embedding_model}")
            self.model = None
            self.openai_embeddings = OpenAIEmbeddings(model=self.embedding_model)

        self.client = chromadb.PersistentClient(path=persist_path)
        safe_model = re.sub(r'[^a-zA-Z0-9_]+', '_', self.embedding_model).lower()
        self.collection_name = f"courses_{safe_model}"
        self.collection = self.client.get_or_create_collection(name=self.collection_name)
        self.chunks: List[str] = []
        self.metadata: List[Dict] = []
        self.tokenized_chunks = []
        
    def embed_documents(self, texts: List[str]) -> np.ndarray:
        """
        Embed a list of texts using the configured backend.
        For OpenAI, batch by approximate token count to avoid 300k tokens/request errors.
        """
        if not texts:
            return np.zeros((0, 0), dtype=float)

        if self.backend == 'st':
            embeds = self.model.encode(texts, convert_to_tensor=False)
            return np.array(embeds)

        # OpenAI: token-aware batching
        batches: List[List[str]] = []
        current: List[str] = []
        current_tokens = 0
        max_tokens_per_request = 280_000
        fallback_batch_size = 32
        try:
            for text in texts:
                # use utils.count_tokens for a safe estimate
                n_tokens = count_tokens(text, model=self.embedding_model or "text-embedding-3-large")
                if current and (current_tokens + n_tokens) > max_tokens_per_request:
                    batches.append(current)
                    current = [text]
                    current_tokens = n_tokens
                else:
                    current.append(text)
                    current_tokens += n_tokens
            if current:
                batches.append(current)
        except Exception:
            batches = [texts[i:i + fallback_batch_size] for i in range(0, len(texts), fallback_batch_size)]

        all_embeddings: List[List[float]] = []
        for batch in batches:
            all_embeddings.extend(self.openai_embeddings.embed_documents(batch))
        return np.array(all_embeddings)

    def process_bulletin(self, bulletin: Dict[str, str]):
        """ 
        Process the entire bulletin dictionary and add all chunks to the vector store.
        
        Args:
            bulletin: Dictionary with keys like 'MATH', 'COMP' and their content
        """                
        all_chunks = []
        all_metadata = []
        
        for department, content in bulletin.items():
            chunk, metadata = chunk_bulletin(department, content)

            all_chunks += chunk
            all_metadata += metadata
        
        # Generate embeddings for all texts
        embeddings = self.embed_documents(all_chunks)
        
        # Normalize embeddings for cosine similarity
        embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

        # Add to Chroma
        start_id = len(self.chunks)
        ids = [str(start_id + i) for i in range(len(all_chunks))]

        self.collection.add(ids=ids, documents=all_chunks, embeddings=embeddings.tolist(), metadatas=all_metadata)
        self.chunks.extend(all_chunks)
        self.metadata.extend(all_metadata)
        self.last_bulletin_id = start_id + len(all_chunks)

        print(f"Processed {len(all_chunks)} chunks from {len(bulletin)} bulletin entries")

    def process_cab(self, cab: Dict[str, str]):
        """
        Process the entire cab dictionary and add all chunks to the vector store.        
        """

        cab_chunks = format_course(cab)
        embeddings = self.embed_documents(cab_chunks)
        embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

        # Generate minimal metadata with source and id if present
        cab_metas: List[Dict] = []
        for term in cab.keys():
            for department in cab[term].keys():
                for course in cab[term][department]:
                    meta = {
                        'term': term,
                        'department': department,
                        'course_id': course.get('course_id', ''),
                        'source': 'cab',
                    }
                    cab_metas.append(meta)                    

        start_id = self.last_bulletin_id
        ids = [str(start_id + i) for i in range(len(cab_chunks))]
        self.collection.add(ids=ids, documents=cab_chunks, embeddings=embeddings.tolist(), metadatas=cab_metas)
        self.metadata.extend(cab_metas)
        self.chunks.extend(cab_chunks)
        
    def save(self, filepath: str):
        """Persist Chroma collection and write companion metadata pickle."""

        data = {
            'chunks': self.chunks,
            'metadata': self.metadata,
            'embedding_model': self.embedding_model,
        }
        with open(f"{filepath}.pkl", 'wb') as f:
            pickle.dump(data, f)
        print(f"Vector store saved to Chroma (persistent) and {filepath}.pkl")

def main():           
    with open('cab.json', 'r') as cab_file:
        cab = json.load(cab_file)

    with open('bulletin.json', 'r') as bulletin_file:
        bulletin = json.load(bulletin_file)
    
    vector_store = VectorStore(model_name=os.getenv('EMBEDDING_MODEL', ''), persist_path='vector_store')
    vector_store.process_bulletin(bulletin)
    vector_store.process_cab(cab)
    vector_store.save("vector_store")

if __name__ == "__main__":
    main()