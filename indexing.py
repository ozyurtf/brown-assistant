import os
import re
from typing import Dict, List, Tuple
import numpy as np
import pickle
import json
import chromadb
from chromadb.config import Settings
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
load_dotenv()

os.environ["TOKENIZERS_PARALLELISM"] = "false"

def format_course(cab):
    course_info_all = []
    for term in cab.keys(): 
        for dept in cab[term].keys():
            for course in cab[term][dept]:
                output_lines = []
                output_lines.append(f"Term: {term}")
                output_lines.append(f"Department: {dept}")
                output_lines.append(f"Course ID: {course.get('course_id', '')}")
                output_lines.append(f"Course: {course.get('course', '')}")
                output_lines.append(f"Title: {course.get('title', '')}")
                output_lines.append(f"Total Sections: {course.get('total_sections', '')}")
                output_lines.append(f"Instructor Name: {course.get('instructor_name', '')}")
                output_lines.append(f"Instructor Email: {course.get('instructor_email', '')}")
                output_lines.append(f"Meeting Times: {course.get('meeting_times', '')}")
                output_lines.append(f"Description: {course.get('description', '')}")
                output_lines.append(f"Registration Restrictions: {course.get('registration_restrictions', '')}")
                output_lines.append(f"Course Attributes: {course.get('course_attributes', '')}")
                output_lines.append(f"Exam Info: {course.get('exam_info', '')}")
                output_lines.append(f"Class Notes: {course.get('class_notes', '')}")
                output_lines.append(f"Sections Text: {course.get('sections_text', '')}")
                course_info = "\n".join(output_lines)
                course_info_all.append(course_info)
    return course_info_all

def chunk_concentration(concentration: str, content: str) -> List[Tuple[str, Dict]]:
    """
    Split content by ### headers and create chunks with metadata.
    
    Args:
        concentration: The concentration (e.g., 'math', 'comp', etc.)
        content: The full content string
        
    Returns:
        List of tuples containing (chunk_text, metadata)
    """
    chunks = []
    metadata = []

    sections = re.split(r'^###\s*', content, flags=re.MULTILINE)
    
    if sections[0].strip():
        main_content = sections[0].strip()
        section_title =  'Main Content'
        chunk = f"{section_title}\n{main_content}"
        chunks.append(chunk)
        
        metadata.append({
            'concentration': concentration,
            'source': 'bulletin',
        })        
    
    for i, section in enumerate(sections[1:], 1):
        if section.strip():
            lines = section.strip().split('\n')
            header_title = lines[0].strip() if lines else f'Section {i}'
            section_content = '\n'.join(lines[1:]).strip() if len(lines) > 1 else ''
            
            if section_content:
                metadata.append({
                    'concentration': concentration,
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
        self.openai_model = OpenAIEmbeddings(model='text-embedding-3-large')
        self.client = chromadb.PersistentClient(path=persist_path)
        self.st_collection = self.client.get_or_create_collection(name=f"courses_st")
        self.openai_collection = self.client.get_or_create_collection(name=f"courses_openai")
        self.chunks: List[str] = []
        self.metadata: List[Dict] = []
        
    def embed_documents(self, texts: List[str]) -> Dict[str, np.ndarray]:
        """
        Embed a list of texts using OpenAI models.
        """
        if not texts:
            return {
                'openai_embedding_model': np.zeros((0, 0), dtype=float)
            }

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
            'openai_embedding_model': openai_embeddings,
        }
            
    def process_concentration(self, concentration_dict: Dict[str, str]):
        """ 
        Process the entire bulletin dictionary and add all chunks to both vector stores.
        """                
        all_chunks = []
        all_metadata = []
        
        for concentration, content in concentration_dict.items():
            chunk, metadata = chunk_concentration(concentration, content)
            all_chunks += chunk
            all_metadata += metadata
        
        if not all_chunks:
            print("No concentration chunks to process, skipping...")
            return
        
        embeddings_dict = self.embed_documents(all_chunks)
        
        openai_embeddings = embeddings_dict['openai_embedding_model']
        openai_embeddings = openai_embeddings / np.linalg.norm(openai_embeddings, axis=1, keepdims=True)
        openai_metadata = [{**meta, 'embedding_model': 'openai'} for meta in all_metadata]
        openai_offset = self.openai_collection.count()
        openai_ids = [f"openai_{openai_offset + i}" for i in range(len(all_chunks))]
        
        self.openai_collection.add(ids=openai_ids, documents=all_chunks, embeddings=openai_embeddings.tolist(), metadatas=openai_metadata)
        self.chunks.extend(all_chunks)
        self.metadata.extend(all_metadata)
        
        print(f"Processed {len(all_chunks)} chunks from {len(concentration_dict)} concentration entries")

    def process_cab(self, cab_dict: Dict[str, str]):
        """
        Process the entire cab dictionary and add all chunks to both vector stores.        
        """
        cab_metas: List[Dict] = []
        for term in cab_dict.keys():
            for department in cab_dict[term].keys():
                for course in cab_dict[term][department]:
                    meta = {
                        'term': term,
                        'department': department,
                        'course_id': course.get('course_id', ''),
                        'source': 'cab',
                    }
                    cab_metas.append(meta)                

        cab_chunks = format_course(cab_dict)
        
        if not cab_chunks:
            print("No CAB chunks to process, skipping...")
            return
        
        embeddings_dict = self.embed_documents(cab_chunks)

        openai_embeddings = embeddings_dict['openai_embedding_model']
        openai_embeddings = openai_embeddings / np.linalg.norm(openai_embeddings, axis=1, keepdims=True)
        openai_metadata = [{**meta, 'embedding_model': 'openai'} for meta in cab_metas]
        openai_offset = self.openai_collection.count()
        openai_ids = [f"openai_{openai_offset + i}" for i in range(len(cab_chunks))]
        
        self.openai_collection.add(ids=openai_ids, documents=cab_chunks, embeddings=openai_embeddings.tolist(), metadatas=openai_metadata)
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
    with open('files/cab.json', 'r') as cab_file:
        cab_dict = json.load(cab_file)

    with open('files/concentration.json', 'r') as concentration_file:
        concentration_dict = json.load(concentration_file)
    
    vector_store = VectorStore(persist_path='vector_store')
    vector_store.process_cab(cab_dict)
    vector_store.process_concentration(concentration_dict)
    vector_store.save("vector_store")

if __name__ == "__main__":
    main()