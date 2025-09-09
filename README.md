The goal of this project is implementing a Retrieval-Augmented Generation (RAG) system that helps all students at Brown University explore the courses that are open in the semester and understand their degree requirements by combining two sources:

- CAB (Courses @ Brown): Courses in different departments, their descriptions, schedules and meeting times, instructors, descriptions, etc. These can be found [here](https://cab.brown.edu)
- Bulletin: Concentration requirements, academic rules and policies. These can be found [here](https://bulletin.brown.edu/the-college/concentrations/)

A FastAPI backend is implemented to handle data processing, serve queries efficiently, and evaluate the performance of the generator. In addition, a Streamlit UI is built to provide a simple, interactive interface for users. 

**Note**: Currently, the concentration section is specific to undergraduate programs, and the course-related information can be taken only for the Fall 2025 semester.

## Demo

## End-to-End Workflow

1) **Data Acquisition**
- `bulletin.py`: Scrapes Bulletin concentration pages concurrently using `crawl4ai`, processes, cleans, and organizes it and writes the data into `files/bulletin.json`
- `cab.py`: Queries the CAB API for all available departments in parallel for Fall 2025 term, processes, cleans and organizes it and writes the results into `files/cab.json`

2) **Indexing and Vectorization**
- `indexing.py`: Reads the data in files/bulletin.json and `files/cab.json`
    - Data is chunked and embedded with two embedding models separately:
        - Sentence Transformer (`all-MiniLM-L6-v2`)
        - OpenAI Embedding Model (`text-embedding-3-large`)
    - Embeddings are normalized and stored in Chroma vector store along with metadata
    - vector_store.pkl stores human-readable chunks and metadata used by the RAG runtime

3) **Retrieval and generation**
- `rag.py`: Core RAG class with methods:
  - load: Loads the persisted vector store, selects the appropriate collection based on the chosen embedding model
  - retrieve: Filters the chunks in the vector store based on the department and/or concentration specified by the user, retrieves the most similar/relevant chunks based on the user query and selected embedding model, and reranks the retrieved chunks with CrossEncoder model
  - `generate`: Calls ChatOpenAI to produce final answer based on user query and the retrieved context

4) **Serving**
- `api.py`: Initializes and caches a RAG instance per embedding backend (via `rag_instances` and `get_or_create_rag`) so models and Chroma collections load once and are reused. This lets clients switch embedding backends per request without reloads and keeps both instances warm and ready for queries. Serves `/query` and `/evaluate`, logs requests/responses, and can precompute evaluation summaries at startup.
  - POST `/query`: Retrieves relevant chunks from vector store+ reranks them, and generates an answer
  - POST `/evaluate`: Computes BLEU and ROUGE-L scores based on the generated text and actual text in the `files/evaluation.json`. It computes and then caches results per model and a cached summary is returned after the first computation.
- `ui.py`: Allows users to pick departments and/or concentrations, ask questions, see the performance of the generator, retrieved context, and latency

In addition, 
- `utils.py` includes functions used in various parts of the project such as counting the number of tokens, calculating BLEU and ROUGE scores, etc.
- `models.py` includes Pydantic request/response models used by the API
- `startup.sh` orchestrates extraction (if missing), indexing (if missing), then starts API and UI with health checks.
- `docker-compose.yml` and `Dockerfile` containerize the full stack and run both services together.
- `deploy-aws.sh` sets up Docker and Docker Compose on a fresh Ubuntu EC2 instance.

## Models Used

1) **Bi-Encoder Embeddings**

- Sentence Transformer: `all-MiniLM-L6-v2`
    - Pros: Local, free, lightweight (~22.7 million parameters), fast, quick inference and efficient deployment
    - Cons: Might miss hard-to-catch patterns and important details that can be captured by larger models.

- OpenAI: `text-embedding-3-large`
    - Pros: Rich and high-quality embeddings, better context understanding, can capture subtle details
    - Cons: Dependency on API, computational overhead, higher cost


2) **Vector Database for Indexing and Retrieval**

- ChromaDB
    - Pros: Easy metadata filtering with `where` clause, simple Python API, works well for small/medium datasets, self-contained, easy to dockerize, no extra dependencies.
    - Cons: Operates on a single node architecture, slower for large-scale search, fewer indexing options

3) **Cross-Encoder Reranking for Retrieval**

- CrossEncoder: `BAAI/bge-reranker-base`
    - Pros: High accruacy for ranking, open source, free
    - Cons: Slow inference, compute intensive

4) **Generator**

- ChatOpenAI: `gpt-4o-mini`
    - Pros: High quality generation, reasoning ability, easy integration
    - Cons: API dependency, hallucination risk, cost per call

5) **Evaluation**
    
- Retriever 
    - `Precision` **(Not implemented yet):** The fraction of the retrieved chunks that overlap with the original chunks
    - `Recall` **(Not implemented yet):** The fraction of the original chunks that overlap with the retrieved chunks

- Generator
    - `BLEU`: Measures how much generated text overlaps with original text in terms of exact word sequences
    - `ROUGE-L (F-1)`: Finds the longest sequence of words in both generated text and original text (in the same order) and calculates F-1 score which is the balance of the fraction of the original text that is covered by the predicted text (Recall) and the fraction of the predicted text that is covered by the original text (Precision)

## How to Add New Datasets for Evaluation?

To add a new dataset to measure how well the generated text performs with it, you can enter your question, concentration for Bulletin, department for CAB, and answer into the `evaluation.json` file in the `files` folder, and click the `Evaluate` button in the UI.

## Observations and Future Development

Currently, the pipeline performs relatively well for most questions. Wrong answers usually occur when the question contains only technical words (e.g., `APMA 2230`, `CSCI 0320`, `ECON 2950`, etc.) without context and/or when the wrong department or concentration is selected in the UI. To solve the issue of poor performance with technical words, I tried to integrate a sparse retriever, BM25, along with the dense retriever and then re-ranked the retrieved chunks with a Cross Encoder, but this did not help much. Therefore, I removed it. 

In addition, retrieval + generation process usually takes between 5-20 seconds. Sometimes it may take up to 30 seconds for the chunks to be retrieved from the vector store and for an answer to be generated. Although significant amount of this time is spent for text generation, caching is still one of the important features to add and it will be integrated into the process in the next steps considering the fact that the same or similar questions can be asked by hundreds/thousands of other students as well. FAISS vector database can also be tried instead of ChromaDB to check if retrieval time reduces significantly.

Also, currently, performance is measured by calculating the BLEU and ROUGE-L scores between the generated answer and the original answer. When OpenAI's embedding model is used, BLEU score is calculated as around ~0.10 and ROUGE-L score is calculated around ~0.40. When all-MiniLM-L6-v2 sentence transformer is used, BLEU score is calculated as around ~0.05 and ROUGE-L score is calculated as around ~0.25.

Morevoer, we can measure the performance of the retriever by computing the precision and recall scores between the retrieved chunks and the original chunks in the next step.

Currently, users need to specify a department and/or concentration in the UI. In the future, the information from all courses in each department can be combined, summarized and keywords can be extracted. After aggregating these, we can use them to determine which department is most relevant to the user query based on similarity scores. A similar approach can be applied to concentrations as well. By following this method, users may no longer need to specify a department and concentration since these can be identified automatically.

## Running Locally

**1) Create .env** 

Copy example environment and edit:

```bash  
cp env.example .env
```

Open .env and set `OPENAI_API_KEY`

**2) Build and Run**

```bash
docker-compose up --build
```

**3) Access**

- UI: `http://localhost:8501`
- API: `http://localhost:8000/docs`

First run notes
- Data extraction and initial model loads can take several minutes
- Vector store is persisted under vector_store/; subsequent runs skip re-indexing unless files are missing

## Running on AWS EC2

**1) Create EC2 Instance**
- AMI: Ubuntu 24.04 LTS
- Instance type: t3.medium minimum (t3.large recommended)
- Storage: 20–30 GB
- Security group rules: allow ports 22 (SSH from your IP), 8000, 8501 (0.0.0.0/0 for testing), or front with an ALB/NGINX later

**2) Connect and Prepare the Machine**
  
```bash  
ssh -i your-key.pem ubuntu@your-public-ip
  
sudo apt update && sudo apt upgrade -y
sudo apt install -y docker.io git
sudo systemctl enable --now docker
sudo usermod -aG docker ubuntu
  
# Install Docker Compose
sudo curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
sudo chmod +x /usr/local/bin/docker-compose
  
exit
```

```bash
# Reconnect (group change takes effect)
ssh -i your-key.pem ubuntu@your-public-ip
```

**3) Deploy Code**

Clone your repository and configure environment:

```bash  
git clone <your-repo-url> rag
cd rag
cp env.example .env
nano .env   # set OPENAI_API_KEY and any other variables
```

**4) Launch Services**
  
```bash
docker-compose up -d --build
```

**5) Access**

- UI at `http://your-public-ip:8501`
- API docs at `http://your-public-ip:8000/docs`
