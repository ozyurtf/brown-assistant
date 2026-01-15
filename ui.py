import os
import time
import json
import requests
import streamlit as st
from dotenv import load_dotenv
load_dotenv()

host = os.getenv("API_HOST")
port = os.getenv("API_PORT")

if host and port:
    API_BASE = f"http://{host}:{port}"
else:
    API_BASE = "http://localhost:8000"

API_TOKEN = os.getenv("API_TOKEN")

if not API_TOKEN:
    try:
        with open(".env", 'r') as f:
            for line in f:
                line = line.strip()
                if line.startswith('API_TOKEN='):
                    API_TOKEN = line.split('=', 1)[1]
                    break
    except Exception as e:
        print(f"Error reading .env file: {e}")

REQUEST_TIMEOUT = 180

def get_auth_headers():
    """Get authorization headers if API token is configured."""
    if API_TOKEN:
        return {"Authorization": f"Bearer {API_TOKEN}"}
    return {}  

def create_model_display_name(model_key: str, model_value: str) -> str:
    """Create a user-friendly display name for the model."""
    model_name_map = {
        "openai": "OpenAI"
    }
    base_name = model_name_map.get(model_key, model_key.replace("_", " ").title())
    return f"{base_name} ({model_value})"

st.set_page_config(page_title="Brown Assisstant", layout="wide")

st.markdown("""
    <style>
    * {
        font-family: 'Avenir', 'Avenir Next', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif !important;
    }
    </style>
""", unsafe_allow_html=True)

st.title("Brown Assisstant")

question = st.text_area(
    "Ask a question", 
    placeholder="e.g., Does APMA 1650 satisfy my requirements?", 
    height=120
)

left_col, right_col, _spacer = st.columns([0.12, 0.12, 0.76], gap="small")

with left_col:
    search_clicked = st.button("Search", type="primary")

if search_clicked:
    if not question.strip():
        st.warning("Please enter a question.")
    else:
        payload = {
            "question": question.strip(),
            "rerank_model_name": "BAAI/bge-reranker-base",
            "embedding_model": "openai",
        }
        
        try:
            retrieval_time_seconds = [None] 
            with st.spinner(f"Retrieving relevant information and generating an answer."):
                start_time = time.perf_counter()
                response = requests.post(
                    f"{API_BASE}/query", 
                    json=payload, 
                    headers=get_auth_headers(),
                    stream=True,
                    timeout=REQUEST_TIMEOUT,
                )
                response.raise_for_status()
                
                retrieved_results = []
                time_measured = [False]
                def stream_generator():
                    for line in response.iter_lines():
                        if line:
                            try:
                                line_str = line.decode('utf-8')
                                if line_str.startswith('data: '):
                                    line_str = line_str[6:] 
                                data = json.loads(line_str)
                                if 'retrieved' in data: 
                                    retrieved_results.extend(data['retrieved'])
                                elif 'chunk' in data:
                                    if not time_measured[0]: 
                                        retrieval_time_seconds[0] = time.perf_counter() - start_time
                                        time_measured[0] = True
                                    yield data['chunk']
                                    time.sleep(0.05)
                            except (json.JSONDecodeError, UnicodeDecodeError) as e:
                                continue  
                
                st.write_stream(stream_generator())
                
        except requests.exceptions.Timeout:
            st.error("Request timed out. Please try again.")
        except requests.exceptions.ConnectionError:
            st.error("Could not connect to the API. Please check if the server is running.")
        except requests.exceptions.HTTPError as e:
            st.error(f"API error ({e.response.status_code}): {e.response.text}")
        except Exception as e:
            st.error(f"Request failed: {str(e)}")
            
        else:                
            if retrieval_time_seconds[0] is not None:
                st.subheader("Retrieval Time")
                st.write(f"Retrieval Time: **{retrieval_time_seconds[0]:.2f} s**")

            if retrieved_results == []:
                st.warning("No context was retrieved for this query.")