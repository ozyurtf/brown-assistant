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
from langchain_openai import OpenAIEmbeddings

os.environ["TOKENIZERS_PARALLELISM"] = "false"


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
    def __init__(self, persist_path: str = 'vector_store'):
        """
        Initialize a Chroma persistent collection and embedding model.
        """
        self.sentence_transformer = SentenceTransformer('all-MiniLM-L6-v2')
        self.openai_model = OpenAIEmbeddings(model='text-embedding-3-large')
        self.client = chromadb.PersistentClient(path=persist_path)
        self.st_collection = self.client.get_or_create_collection(name=f"courses_st")
        self.openai_collection = self.client.get_or_create_collection(name=f"courses_openai")
        self.chunks: List[str] = []
        self.metadata: List[Dict] = []
        
    def embed_documents(self, texts: List[str]) -> Dict[str, np.ndarray]:
        """
        Embed a list of texts using both sentence transformer and OpenAI models.
        """
        if not texts:
            return {
                'sentence_embedding_model': np.zeros((0, 0), dtype=float),
                'openai_embedding_model': np.zeros((0, 0), dtype=float)
            }

        # Generate sentence transformer embeddings
        sentence_embeddings = self.sentence_transformer.encode(texts, convert_to_tensor=False)
        sentence_embeddings = np.array(sentence_embeddings)

        # Generate OpenAI embeddings with token-aware batching
        batches: List[List[str]] = []
        current: List[str] = []
        current_tokens = 0
        max_tokens_per_request = 280_000
        fallback_batch_size = 32
        
        try:
            for text in texts:
                n_tokens = count_tokens(text, model="text-embedding-3-large")  # Fixed: removed self.embedding_model
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

        openai_embeddings: List[List[float]] = []
        for batch in batches:
            openai_embeddings.extend(self.openai_model.embed_documents(batch))
        
        openai_embeddings = np.array(openai_embeddings)

        return {
            'sentence_embedding_model': sentence_embeddings,
            'openai_embedding_model': openai_embeddings,
        }
            
    def process_bulletin(self, bulletin: Dict[str, str]):
        """ 
        Process the entire bulletin dictionary and add all chunks to both vector stores.
        """                
        all_chunks = []
        all_metadata = []
        
        for department, content in bulletin.items():
            chunk, metadata = chunk_bulletin(department, content)
            all_chunks += chunk
            all_metadata += metadata
        
        # Generate embeddings for all texts
        embeddings_dict = self.embed_documents(all_chunks)

        # Add to sentence transformer collection
        st_embeddings = embeddings_dict['sentence_embedding_model']
        st_embeddings = st_embeddings / np.linalg.norm(st_embeddings, axis=1, keepdims=True)
        st_metadata = [{**meta, 'embedding_model': 'sentence_transformer'} for meta in all_metadata]
        st_ids = [f"st_{i}" for i in range(len(all_chunks))]
        
        self.st_collection.add(ids=st_ids, documents=all_chunks, embeddings=st_embeddings.tolist(), metadatas=st_metadata)
        
        # Add to OpenAI collection
        openai_embeddings = embeddings_dict['openai_embedding_model']
        openai_embeddings = openai_embeddings / np.linalg.norm(openai_embeddings, axis=1, keepdims=True)
        openai_metadata = [{**meta, 'embedding_model': 'openai'} for meta in all_metadata]
        openai_ids = [f"openai_{i}" for i in range(len(all_chunks))]
        
        self.openai_collection.add(ids=openai_ids, documents=all_chunks, embeddings=openai_embeddings.tolist(), metadatas=openai_metadata)
        
        # Store for later use
        self.chunks.extend(all_chunks)
        self.metadata.extend(all_metadata)
        self.bulletin_count = len(all_chunks)
        
        print(f"Processed {len(all_chunks)} chunks from {len(bulletin)} bulletin entries")

    def process_cab(self, cab: Dict[str, str]):
        """
        Process the entire cab dictionary and add all chunks to both vector stores.        
        """
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

        cab_chunks = format_course(cab)
        embeddings_dict = self.embed_documents(cab_chunks)

        # Add to sentence transformer collection
        st_embeddings = embeddings_dict['sentence_embedding_model']
        st_embeddings = st_embeddings / np.linalg.norm(st_embeddings, axis=1, keepdims=True)
        st_metadata = [{**meta, 'embedding_model': 'sentence_transformer'} for meta in cab_metas]
        st_ids = [f"st_{self.bulletin_count + i}" for i in range(len(cab_chunks))]
        
        self.st_collection.add(ids=st_ids, documents=cab_chunks, embeddings=st_embeddings.tolist(), metadatas=st_metadata)
        
        # Add to OpenAI collection
        openai_embeddings = embeddings_dict['openai_embedding_model']
        openai_embeddings = openai_embeddings / np.linalg.norm(openai_embeddings, axis=1, keepdims=True)
        openai_metadata = [{**meta, 'embedding_model': 'openai'} for meta in cab_metas]
        openai_ids = [f"openai_{self.bulletin_count + i}" for i in range(len(cab_chunks))]
        
        self.openai_collection.add(ids=openai_ids, documents=cab_chunks, embeddings=openai_embeddings.tolist(), metadatas=openai_metadata)
        
        # Store for later use
        self.chunks.extend(cab_chunks)
        self.metadata.extend(cab_metas)
        
        print(f"Processed {len(cab_chunks)} CAB chunks")
        
    def save(self, filepath: str):
        """Persist Chroma collection and write companion metadata pickle."""
        data = {
            'chunks': self.chunks,
            'metadata': self.metadata,
        }
        with open(f"{filepath}.pkl", 'wb') as f:
            pickle.dump(data, f)
        print(f"Vector store saved to Chroma (persistent) and {filepath}.pkl")

def main():           
    with open('cab.json', 'r') as cab_file:
        cab = json.load(cab_file)

    with open('bulletin.json', 'r') as bulletin_file:
        bulletin = json.load(bulletin_file)
    
    vector_store = VectorStore(persist_path='vector_store')
    vector_store.process_cab(cab)
    vector_store.process_bulletin(bulletin)
    vector_store.save("vector_store")

if __name__ == "__main__":
    main()